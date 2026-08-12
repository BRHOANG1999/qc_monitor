"""Off-process readers for the evoked ``*_evoked.mat`` (h5py) path -- the evoked
sibling of ``chunk_reader`` (which does the same for KMrecorder ``.mat`` chunks).

The dashboard's chronic-evoked / peri-ictal / waveforms views read ~600 MB evoked
HDF5 files via h5py and, on the feature paths, run a Morlet CWT on top
(~15 s/file). On a dashboard-owned thread that read+compute holds the GIL for its
whole duration and freezes every other request -- Overview, and a second user in
Video Review. Here the read (and, for the feature path, the whole feature compute)
happens in a warm child process; only the small result comes back:

    compute_rows_offproc   -> a list of small feature-row dicts (no arrays cross)
    read_file_offproc      -> per-channel dict; trace arrays via a local .npy
    gather_leadup_offproc  -> the final windowed ERP trials via a local .npy

Routed to automatically by ``evoked_output.read_file_evoked`` /
``compute_feature_rows`` when running on a dashboard-owned thread
(``QC_DASHBOARD_ROLE`` set, not already an off-proc worker) -- exactly as
``chunk_cache.get_chunk`` routes ``.mat`` reads to ``chunk_reader``. The daemon and
the offline warm tool (no ``QC_DASHBOARD_ROLE``) read in-process, unchanged.

On the dashboard an off-proc failure RAISES rather than silently re-reading
in-process (which would reintroduce the freeze); the peri-ictal build workers
already catch it and surface it in their progress poll.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading

# Ensure the repo root is importable in a spawned child (belt-and-suspenders;
# 'spawn' already propagates the parent's sys.path).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

_TMP_DIR = os.path.join(tempfile.gettempdir(), "qc_evoked_offproc")
# One ~600 MB read + CWT is ~15 s warm, but a COLD SMB read of the file can take
# minutes; a lead-up gather reads many such files, so it gets a larger budget.
_ONE_FILE_TIMEOUT_SEC = 300.0
_GATHER_TIMEOUT_SEC = 1200.0
# Evoked warming is inherently sequential (one animal, oldest-first) and each
# child holds a ~600 MB file; keep exactly one in flight to bound memory. This is
# a SEPARATE pool from chunk_reader's, so a long peri-ictal warm can never occupy
# the 2 interactive video-read workers (that isolation is the whole point).
_MAX_WORKERS = 1

_pool = None
_pool_lock = threading.Lock()


def _clean_tmp_dir() -> None:
    """Best-effort: drop stale temp arrays from a previous run (a crash mid-read
    can leave one behind; the normal path deletes its own)."""
    try:
        for name in os.listdir(_TMP_DIR):
            if name.startswith(("evk_", "erp_")) and name.endswith(".npy"):
                try:
                    os.remove(os.path.join(_TMP_DIR, name))
                except OSError:
                    pass
    except OSError:
        pass


def _get_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            _clean_tmp_dir()
            _pool = ProcessPoolExecutor(
                max_workers=_MAX_WORKERS, mp_context=mp.get_context("spawn"))
        return _pool


def _reset_pool() -> None:
    global _pool
    with _pool_lock:
        p, _pool = _pool, None
    if p is not None:
        try:
            p.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass


def _save_npy(arr, prefix: str) -> str:
    """CHILD: dump one array to a LOCAL temp .npy and return its path (the big
    array never crosses the process pipe)."""
    os.makedirs(_TMP_DIR, exist_ok=True)
    fd, npy = tempfile.mkstemp(prefix=prefix, suffix=".npy", dir=_TMP_DIR)
    os.close(fd)
    np.save(npy, arr)
    return npy


def _load_npy(path: str):
    """PARENT: load a child-written .npy into RAM (the memcpy is GIL-released)
    and delete it."""
    try:
        mm = np.load(path, mmap_mode="r")
        arr = np.array(mm)
        del mm
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return arr


# --------------------------------------------------------------------- #
#  Feature rows -- small payload, returned directly (no arrays cross)
# --------------------------------------------------------------------- #
def _rows_worker(path, animal, cfg, expensive, include_wavelet, trial_avg):
    from src.utils.offproc_guard import mark_worker
    mark_worker()                       # this IS an allowed off-proc reader
    from src.utils.evoked_output import _compute_feature_rows_impl
    return _compute_feature_rows_impl(
        path, animal, cfg, expensive, include_wavelet, trial_avg)


def compute_rows_offproc(path, animal, cfg=None, expensive: bool = False,
                         include_wavelet: bool = True,
                         trial_avg: int | None = None) -> list:
    """Compute *animal*'s feature rows for one ``*_evoked.mat`` in a child process
    (read + Morlet CWT off the caller's GIL). Only the small row dicts return."""
    from concurrent.futures import TimeoutError as _FTimeout
    try:
        fut = _get_pool().submit(_rows_worker, path, animal, cfg, expensive,
                                 include_wavelet, trial_avg)
    except Exception:  # noqa: BLE001 -- pool unusable
        _reset_pool()
        raise
    try:
        return fut.result(timeout=_ONE_FILE_TIMEOUT_SEC)
    except _FTimeout:
        _reset_pool()
        raise TimeoutError(f"off-proc evoked feature build timed out: {path}")
    except Exception:  # noqa: BLE001 -- broken pool / worker crash
        _reset_pool()
        raise


# --------------------------------------------------------------------- #
#  Raw per-channel read -- trace arrays handed back via .npy
# --------------------------------------------------------------------- #
def _read_file_worker(path, only_animals):
    from src.utils.offproc_guard import mark_worker
    mark_worker()
    from src.utils.evoked_output import _read_file_evoked_impl
    out = _read_file_evoked_impl(path, only_animals)
    meta: dict = {}
    for ch, rec in out.items():
        traces = rec.get("traces")
        time_ms = rec.get("time_ms")
        meta[ch] = {
            "times": rec.get("times"),
            "stim_peak": rec.get("stim_peak"),
            "stim_trough": rec.get("stim_trough"),
            "traces_npy": _save_npy(traces, "evk_") if traces is not None else None,
            "time_ms": time_ms.tolist() if time_ms is not None else None,
        }
    return meta


def read_file_offproc(path, only_animals=None) -> dict:
    """Per-channel evoked data for one file, read in a child process. Trace
    matrices come back via a local .npy; the small scalar lists cross inline."""
    from concurrent.futures import TimeoutError as _FTimeout
    keep = list(only_animals) if only_animals else None
    try:
        fut = _get_pool().submit(_read_file_worker, path, keep)
    except Exception:  # noqa: BLE001
        _reset_pool()
        raise
    try:
        meta = fut.result(timeout=_ONE_FILE_TIMEOUT_SEC)
    except _FTimeout:
        _reset_pool()
        raise TimeoutError(f"off-proc evoked read timed out: {path}")
    except Exception:  # noqa: BLE001
        _reset_pool()
        raise
    out: dict = {}
    for ch, rec in meta.items():
        tnpy = rec.get("traces_npy")
        tms = rec.get("time_ms")
        out[ch] = {
            "times": rec.get("times"),
            "stim_peak": rec.get("stim_peak"),
            "stim_trough": rec.get("stim_trough"),
            "traces": _load_npy(tnpy) if tnpy else None,
            "time_ms": (np.asarray(tms, dtype=np.float64)
                        if tms is not None else None),
        }
    return out


# --------------------------------------------------------------------- #
#  ERP lead-up gather -- the WHOLE multi-file gather runs in one child, so
#  only the final windowed trials return (not each file's full traces)
# --------------------------------------------------------------------- #
def _gather_worker(evoked_dir, animal, onset_epoch, lookback_sec, cfg, post_sec):
    from src.utils.offproc_guard import mark_worker
    mark_worker()
    from src.periictal.erpimage import gather_leadup_trials
    res = gather_leadup_trials(evoked_dir, animal, onset_epoch, lookback_sec,
                               cfg=cfg, post_sec=post_sec)
    trials = res.get("trials")
    has = trials is not None and getattr(trials, "size", 0) > 0
    return {
        "trials_npy": _save_npy(trials, "erp_") if has else None,
        "row_ms": np.asarray(res.get("row_ms")).ravel().tolist()
        if res.get("row_ms") is not None else [],
        "tto": np.asarray(res.get("tto")).ravel().tolist()
        if res.get("tto") is not None else [],
    }


def gather_leadup_offproc(evoked_dir, animal, onset_epoch, lookback_sec,
                          cfg=None, post_sec=None) -> dict:
    """Run ``erpimage.gather_leadup_trials`` in a child process and return its
    ``{trials, row_ms, tto}`` dict; the raw per-file traces never reach the
    caller, only the final windowed trial matrix (via a local .npy)."""
    from concurrent.futures import TimeoutError as _FTimeout
    try:
        fut = _get_pool().submit(
            _gather_worker, evoked_dir, animal, float(onset_epoch),
            float(lookback_sec), cfg, post_sec)
    except Exception:  # noqa: BLE001
        _reset_pool()
        raise
    try:
        meta = fut.result(timeout=_GATHER_TIMEOUT_SEC)
    except _FTimeout:
        _reset_pool()
        raise TimeoutError(f"off-proc ERP gather timed out (animal={animal})")
    except Exception:  # noqa: BLE001
        _reset_pool()
        raise
    tnpy = meta.get("trials_npy")
    trials = _load_npy(tnpy) if tnpy else np.empty((0, 0))
    return {
        "trials": trials,
        "row_ms": np.asarray(meta.get("row_ms") or [], dtype=np.float64),
        "tto": np.asarray(meta.get("tto") or [], dtype=np.float64),
    }
