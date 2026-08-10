"""Read a KMrecorder .mat in a CHILD PROCESS so the slow SMB read + parse never
hold the calling (dashboard) process's GIL.

The dashboard (LFP Browser / Video Review) reads 0.3-4.7 GB .mat chunks off the
Tailscale share on request threads via ``chunk_cache.get_chunk`` -> ``load_mat``.
Those C reads (scipy.io.loadmat / h5py) hold the GIL for MINUTES, freezing every
other dashboard thread (Overview, other users) -- the split fixed the daemon's
reads but not the dashboard's own.

Here the read happens in a warm child process: the child ``load_mat()``s and
``np.save()``s only the signal array to a LOCAL temp .npy; the parent waits on the
future (which RELEASES the GIL), then copies the array back from the fast local
file (numpy runs that memcpy GIL-released) and deletes it. Net: the minutes-long
share read is entirely off the dashboard's GIL; only a fast local copy remains.
Only small metadata + the (small) trdata timestamps cross the process pipe.

Gated to the dashboard process by the caller (``QC_DASHBOARD_ROLE``): the daemon's
own load_mat reads already live in a separate process and don't need this.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading

# Ensure the repo root is importable in a spawned child. multiprocessing 'spawn'
# propagates the parent's sys.path, but this is a cheap belt-and-suspenders that
# also runs when the child imports this module.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

_TMP_DIR = os.path.join(tempfile.gettempdir(), "qc_chunk_offproc")
_READ_TIMEOUT_SEC = 180.0     # kill a hung share read rather than pin a worker forever
_MAX_WORKERS = 2              # matches chunk_cache._MAX_CONCURRENT_LOADS (interactive)

_pool = None
_pool_lock = threading.Lock()


def _worker(path: str, tmp_dir: str) -> dict:
    """CHILD process: load the .mat, dump ONLY the signal to a local .npy, and
    return small metadata + that path. The big array never crosses the pipe."""
    sys.path.insert(0, _REPO_ROOT)                 # child import safety
    from src.utils.mat_loader import load_mat
    import numpy as _np
    cd = load_mat(path)
    os.makedirs(tmp_dir, exist_ok=True)
    fd, npy = tempfile.mkstemp(prefix="chunk_", suffix=".npy", dir=tmp_dir)
    os.close(fd)
    _np.save(npy, cd.signal)
    return {
        "npy": npy, "fs": cd.fs, "num_channels": cd.num_channels,
        "num_samples": cd.num_samples, "duration_sec": cd.duration_sec,
        "source_path": cd.source_path, "channel_names": cd.channel_names,
        "trdata": cd.trdata, "timestamps": cd.timestamps,
    }


def _clean_tmp_dir() -> None:
    """Best-effort: drop stale temp .npy from a previous run (a crash mid-read
    can leave one behind; the normal path deletes its own)."""
    try:
        for name in os.listdir(_TMP_DIR):
            if name.startswith("chunk_") and name.endswith(".npy"):
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


def read_chunk_offproc(path: str):
    """Load *path*'s ChunkData via a warm child process (off the caller's GIL).

    Raises ``TimeoutError`` if the child read exceeds the deadline (the hung
    worker is killed). Degrades gracefully to an in-process ``load_mat`` if the
    pool can't be created or a worker dies -- so the dashboard never hard-breaks
    on this optimization."""
    from concurrent.futures import TimeoutError as _FTimeout
    from src.utils.mat_loader import ChunkData, load_mat
    try:
        fut = _get_pool().submit(_worker, path, _TMP_DIR)
    except Exception:  # noqa: BLE001 -- pool unusable; degrade to in-process
        return load_mat(path)
    try:
        meta = fut.result(timeout=_READ_TIMEOUT_SEC)   # GIL released while waiting
    except _FTimeout:
        _reset_pool()                                  # kill the hung worker
        raise TimeoutError(f"off-proc chunk read timed out: {path}")
    except Exception:  # noqa: BLE001 -- broken pool / worker crash: fall back once
        _reset_pool()
        return load_mat(path)
    npy = meta["npy"]
    try:
        mm = np.load(npy, mmap_mode="r")
        signal = np.array(mm, dtype=np.float64)        # local->RAM copy (GIL-released)
        del mm
    finally:
        try:
            os.remove(npy)
        except OSError:
            pass
    return ChunkData(
        signal=signal, fs=meta["fs"], num_channels=meta["num_channels"],
        num_samples=meta["num_samples"], duration_sec=meta["duration_sec"],
        source_path=meta["source_path"], timestamps=meta["timestamps"],
        trdata=meta["trdata"], channel_names=meta["channel_names"])
