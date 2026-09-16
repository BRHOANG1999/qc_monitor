"""Per-recording stim-artifact trend across a FOLDER of stim recordings.

Self-contained: no DB, no MATLAB pipeline, no ``notifications`` import. For each
recording in a folder it triggers off the stimCopy channel, averages the
artifact channel around every pulse, and computes per-recording metrics
(peak-to-peak, positive/negative peak, access resistance Rₐ, slow-phase
steady-state Z_ss, saturation %) -- so an electrode stress/integrity test (e.g.
``electrodeTest-100nC``, 71 hourly files) can be trended over time even though
the folder was never ingested into the DB.

Reuses the same primitives as the daemon impedance path so the numbers match:
``stim_blank.detect_stim_onsets`` (pulse onsets), ``trace_average`` (per-recording
average on ONE shared grid), ``impedance.access_resistance``/``slow_steady_state``
(Rₐ/Z_ss), ``stim_parser`` (the ``*_STIM_REPORT.txt`` stim params -- the only
source of ``gain``, which Rₐ/Z_ss require).

Grid note: both channels are averaged with ``align=None`` on ONE grid.
``access_resistance`` locates transition sample indices in ``stim_mean_trace``
and indexes those same indices into ``mean_trace``; per-channel rising-edge
alignment would decouple the grids and break Rₐ. The epochs are already
onset-aligned (``detect_stim_onsets`` triggers on the stim-copy rising edge and
each epoch is a fixed-sample window around its onset).
"""

from __future__ import annotations

import glob
import logging
import os
import re
from datetime import datetime

import numpy as np

from src.utils.animal import stim_copy_indices
from src.utils.chunk_cache import get_chunk
from src.utils.impedance import (access_resistance, phase_currents,
                                 slow_steady_state)
from src.utils.stim_blank import detect_stim_onsets
from src.utils.stim_parser import parse_stim_report_file
from src.utils.trace_average import align_average_with_traces, average_traces

logger = logging.getLogger("qc_monitor.stim_artifact_trend")

WIN_PRE_MS, WIN_POST_MS = -1.0, 2.0        # epoch window around each pulse onset
MAX_EPOCHS_PER_REC = 200                   # subsample cap per recording
SAT_RAIL_FRAC = 0.98                       # |x| within 2% of the rail = saturated
_MAX_RECORDINGS = 100_000                  # NASA loop bound

# ``..._YYYY_MM_DD__HH_MM_SS.mat`` (evoked_output._DT_RX requires a _evoked.mat
# suffix, so a local pattern is needed for the raw recordings).
_DT_RX = re.compile(r"(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})\.mat$")


def recording_datetime(path: str) -> datetime | None:
    m = _DT_RX.search((path or "").replace("\\", "/"))
    if not m:
        return None
    try:
        return datetime(*[int(g) for g in m.groups()])
    except ValueError:
        return None


def list_recordings(folder: str) -> list[str]:
    """Every ``*.mat`` in *folder* (the ``.txt`` sidecars are excluded), sorted
    oldest-first by the timestamp in the filename (basename tie-break)."""
    files = glob.glob(os.path.join(folder, "*.mat"))
    return sorted(files, key=lambda p: (recording_datetime(p) or datetime.min,
                                        os.path.basename(p)))


def stim_params_for(mat_path: str) -> dict:
    """Stim parameters for one recording from its ``*_STIM_REPORT.txt`` sidecar
    -- the only source of ``gain`` (Rₐ/Z_ss need it). Returns
    ``{charge_nC, pulse_width_us, ratio, gain, freq_hz, source}``; all-None with
    ``source='none'`` when there is no readable report (Rₐ/Z_ss then skip; the
    peak/saturation metrics still compute)."""
    report = (mat_path[:-4] if mat_path.lower().endswith(".mat") else mat_path
              ) + "_STIM_REPORT.txt"
    if os.path.isfile(report):
        try:
            rep = parse_stim_report_file(report)
            ch = next((c for c in rep.channels if c.active), None)
            if ch is not None:
                return {"charge_nC": ch.charge_nC or None,
                        "pulse_width_us": ch.pulse_width_us or None,
                        "ratio": ch.neg_pulse_ratio or None,
                        "gain": ch.gain or None,
                        "freq_hz": ch.frequency_hz or None,
                        "source": "report"}
        except Exception as e:                          # noqa: BLE001
            logger.debug("stim report parse failed for %s: %s", report, e)
    return {"charge_nC": None, "pulse_width_us": None, "ratio": None,
            "gain": None, "freq_hz": None, "source": "none"}


def pick_channels(channel_names) -> tuple[int | None, int | None]:
    """``(stim_idx, artifact_idx)`` by NAME. The artifact channel is the first
    non-stimCopy channel -- deliberately NOT ``is_animal_channel`` (its
    NON_ANIMAL_KEYWORDS contains ``'test'``, which an ``electrodeTest`` channel
    name would trip). ``(None, None)`` when no channel name says stimCopy;
    :func:`analyze_recording` then falls back to the larger-transient channel."""
    names = list(channel_names or [])
    stim = stim_copy_indices(names)
    if not stim:
        return None, None
    stim_idx = min(stim)
    artifact_idx = next((i for i in range(len(names)) if i not in stim), None)
    return stim_idx, artifact_idx


def build_epochs(signal, fs, stim_idx, artifact_idx, onsets_sec,
                 max_epochs=MAX_EPOCHS_PER_REC) -> list[dict]:
    """Fixed ``[-1, 2] ms`` windows around each onset on BOTH channels, on one
    shared time grid, subsampled to *max_epochs*. Each dict: ``{time_ms, art,
    stim}``. Edge-truncated windows are dropped."""
    lo = int(round(WIN_PRE_MS * fs / 1000.0))
    hi = int(round(WIN_POST_MS * fs / 1000.0))
    tm = np.arange(lo, hi) / fs * 1000.0
    n = len(signal)
    centers = np.round(np.asarray(onsets_sec, dtype=float) * fs).astype(int)
    centers = centers[(centers + lo >= 0) & (centers + hi < n)]
    if len(centers) > max_epochs:
        centers = centers[np.linspace(0, len(centers) - 1, max_epochs).astype(int)]
    art = signal[:, artifact_idx]
    stim = signal[:, stim_idx]
    out = []
    for c in centers:
        out.append({"time_ms": tm,
                    "art": art[c + lo:c + hi],
                    "stim": stim[c + lo:c + hi]})
    return out


def average_recording(epochs):
    """``(time_ms, mean_trace, sem, stim_mean_trace)`` as lists -- both channels
    averaged on the SAME grid with ``align=None`` (see the module grid note)."""
    tm, mean, sem, _ = align_average_with_traces(
        epochs, value_key="art", time_key="time_ms", align=None)
    _tm2, stim_mean = average_traces(
        epochs, value_key="stim", time_key="time_ms", align=None)
    return tm, mean, sem, stim_mean


# --------------------------------------------------------- metric helpers ---- #
# Local (mirror notifications.evoked_weekly._peak_amplitude) so this util never
# imports notifications/db -- that dependency direction would be backwards.

def _win(time_ms, trace, x0, x1):
    return [v for t, v in zip(time_ms, trace) if x0 <= t <= x1]


def _peak_amplitude(time_ms, trace, x0, x1):
    vals = _win(time_ms, trace, x0, x1)
    return (max(vals) - min(vals)) if len(vals) >= 2 else None


def _baseline(time_ms, trace):
    pre = [v for t, v in zip(time_ms, trace) if t < WIN_PRE_MS / 2.0]
    return float(np.median(pre)) if pre else 0.0


def _pos_peak(time_ms, trace, x0, x1):
    vals = _win(time_ms, trace, x0, x1)
    return (max(vals) - _baseline(time_ms, trace)) if vals else None


def _neg_peak(time_ms, trace, x0, x1):
    vals = _win(time_ms, trace, x0, x1)
    return (min(vals) - _baseline(time_ms, trace)) if vals else None


def saturation_pct(col) -> float:
    """Fraction (%) of samples within 2% of the channel's rail -- a clipping
    flag. Computed on the RAW artifact channel (not the averaged trace)."""
    a = np.abs(np.asarray(col, dtype=np.float64))
    rail = float(a.max()) if a.size else 0.0
    if rail <= 0:
        return 0.0
    return float(100.0 * np.mean(a >= SAT_RAIL_FRAC * rail))


def compute_metrics(time_ms, mean_trace, stim_mean_trace, artifact_col,
                    params) -> dict:
    """``{ptp, pos, neg, ra, zss, sat}`` -- any may be None. Rₐ/Z_ss need the
    commanded currents (from charge/pulse-width) AND gain; when absent the peak
    and saturation metrics still compute."""
    pw = params.get("pulse_width_us")
    ratio = params.get("ratio")
    x0 = -0.2
    x1 = (float(pw) * (1.0 + float(ratio)) / 1000.0 + 0.3) if (pw and ratio) else 1.5
    m = {"ptp": _peak_amplitude(time_ms, mean_trace, x0, x1),
         "pos": _pos_peak(time_ms, mean_trace, x0, x1),
         "neg": _neg_peak(time_ms, mean_trace, x0, x1),
         "ra": None, "zss": None,
         "sat": saturation_pct(artifact_col)}
    gain = params.get("gain")
    i_pos, i_neg = phase_currents(params.get("charge_nC"), pw, ratio)
    if gain and i_pos and i_neg:
        ar = access_resistance(mean_trace, time_ms, stim_mean_trace,
                               gain, i_pos, i_neg)
        m["ra"] = ar.get("r_access_kohm")
        if m["ra"] is not None:                         # matches impedance_refresh
            zss = slow_steady_state(mean_trace, time_ms, stim_mean_trace,
                                    gain, i_neg)
            m["zss"] = zss.get("slow_ss_kohm")
    return m


def analyze_recording(mat_path: str, *, get_chunk_fn=get_chunk) -> dict:
    """One recording -> a record dict. *get_chunk_fn* is the injectable read seam
    (tests pass a fake returning a ChunkData). A single bad file yields
    ``{"ok": False, "error": ...}`` rather than aborting the folder."""
    rec = {"path": mat_path, "dt": recording_datetime(mat_path),
           "ok": False, "error": None, "n_pulses": 0, "time_ms": None,
           "mean_trace": None, "sem": None, "stim_mean_trace": None,
           "metrics": {}, "stim_params": {}}
    try:
        chunk = get_chunk_fn(mat_path)
        sig = chunk.signal
        stim_idx, art_idx = pick_channels(chunk.channel_names)
        if stim_idx is None or art_idx is None:
            if sig.ndim == 2 and sig.shape[1] == 2:     # larger-transient fallback
                ptps = [float(np.ptp(sig[:, c])) for c in range(2)]
                stim_idx = int(np.argmax(ptps))
                art_idx = 1 - stim_idx
            else:
                raise ValueError(
                    f"cannot identify stim/artifact channels in "
                    f"{chunk.channel_names}")
        onsets = detect_stim_onsets(sig[:, stim_idx], chunk.fs)
        if onsets.size == 0:
            raise ValueError("no stim onsets detected on the stim channel")
        epochs = build_epochs(sig, chunk.fs, stim_idx, art_idx, onsets)
        if not epochs:
            raise ValueError("no usable epochs (windows edge-truncated)")
        tm, mean, sem, stim_mean = average_recording(epochs)
        params = stim_params_for(mat_path)
        metrics = compute_metrics(tm, mean, stim_mean, sig[:, art_idx], params)
        rec.update({"ok": True, "n_pulses": int(onsets.size), "time_ms": tm,
                    "mean_trace": mean, "sem": sem, "stim_mean_trace": stim_mean,
                    "metrics": metrics, "stim_params": params})
    except Exception as e:                              # noqa: BLE001
        rec["error"] = str(e)
        logger.debug("analyze_recording failed for %s: %s", mat_path, e)
    return rec


def analyze_folder(folder: str, *, progress_cb=None, get_chunk_fn=get_chunk,
                   max_recordings=None) -> list[dict]:
    """Every recording in *folder* -> a list of record dicts, oldest-first.
    *progress_cb(i, n, path)* is called before each file (and once at the end
    with ``i == n``) so a UI can show a "Reading i/n" line."""
    files = list_recordings(folder)
    if max_recordings:
        files = files[:int(max_recordings)]
    n = len(files)
    assert n < _MAX_RECORDINGS, "recording count runaway"
    out = []
    for i, f in enumerate(files):
        if progress_cb:
            progress_cb(i, n, f)
        out.append(analyze_recording(f, get_chunk_fn=get_chunk_fn))
    if progress_cb:
        progress_cb(n, n, "")
    return out
