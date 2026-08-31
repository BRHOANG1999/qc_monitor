"""Per-trial stim-artifact metrics on the LFP record channel.

Metrics are computed on the RECORD channel's evoked trace over the artifact
window (default [-1, 2] ms, t=0 = stimulus sample) -- the stim artifact as it
couples onto the recording electrode. Three metrics per trial:

* polarity  -- sign of the first post-stim (t>0) extremum, by latency.
* amplitude -- peak-to-trough over the window.
* shape     -- deviation from a FIXED template (median artifact over the
               detected stable Rₐ run): Pearson correlation and span-normalised
               RMSE. corr->1 / nrmse->0 == identical to the stable reference.

Reading: the evoked HDF5 stores ``evokedData`` chunked along the sample axis,
so a windowed COLUMN slice (~61 of 20001 samples) reads ~170x faster than the
whole trace (3 ms vs 520 ms per file measured). We slice directly with h5py
rather than materialising the full array via ``evoked_output._read_channel``.
"""

from __future__ import annotations

import os
import time

import numpy as np

from src.utils import evoked_features as ef
from src.utils.animal import split_animal_electrode
from src.utils.evoked_output import parse_recording_dt

_MAX_FILES = 1_000_000      # NASA Rule 2: explicit scan bound.


def read_window_traces(path: str, animal: str, lo_ms: float,
                       hi_ms: float) -> dict:
    """Windowed per-channel read for one ``*_evoked.mat``. Returns
    ``{channel: {"seg": [E,W] float32, "win_ms": [W], "times": [E]}}`` for
    channels of *animal* only. Column-slices ``evokedData`` in the HDF5 so it
    never materialises the full ±500 ms trace. ``{}`` on any read error."""
    assert path and animal, "path and animal required"
    assert hi_ms > lo_ms, "window must have hi_ms > lo_ms"
    import h5py
    out: dict = {}
    try:
        with h5py.File(path, "r") as g:
            grp = g.get("allAnimalResults")
            if grp is None:
                return {}
            for ch in list(grp.keys()):
                if split_animal_electrode(str(ch))[0] != animal:
                    continue
                rec = _read_window_channel(grp.get(ch), lo_ms, hi_ms)
                if rec is not None:
                    out[str(ch)] = rec
    except (OSError, KeyError, ValueError):
        return {}
    return out


def _read_window_channel(node, lo_ms: float, hi_ms: float):
    """One channel's windowed segment, or None if it lacks the datasets."""
    if node is None or "evokedData" not in node or "timeAxis" not in node:
        return None
    tm = np.asarray(node["timeAxis"][()], dtype=np.float64).ravel()
    mask = (tm >= lo_ms) & (tm <= hi_ms)
    idx = np.where(mask)[0]
    if idx.size < 5:
        return None
    lo_i, hi_i = int(idx[0]), int(idx[-1] + 1)
    seg = np.asarray(node["evokedData"][:, lo_i:hi_i], dtype=np.float32)
    times = np.asarray(node["stimulusTimes"][()], dtype=np.float64).ravel()
    if seg.ndim != 2 or seg.shape[0] != times.size:
        return None
    return {"seg": seg, "win_ms": tm[lo_i:hi_i], "times": times}


def _in_epoch_range(path: str, lo: float, hi: float) -> bool:
    """True if the file's recording-start epoch lies in [lo, hi]."""
    base = parse_recording_dt(os.path.basename(path))
    return base is not None and lo <= base.timestamp() <= hi


def gather_trials(files: list[str], animal: str, lo_ms: float, hi_ms: float,
                  *, progress=None) -> dict:
    """Read the artifact window for every trial across *files*, tagged with its
    absolute epoch and record channel. Returns stacked arrays. Progress is
    reported via *progress(done, total, path)* (default: periodic stderr)."""
    assert isinstance(files, list), "files must be a list"
    files = sorted(files)
    segs, epochs, chans, fidx = [], [], [], []
    win_ms = None
    t0 = time.time()
    total = len(files)
    for k, fp in enumerate(files):
        assert k < _MAX_FILES, "file scan runaway"
        base = parse_recording_dt(os.path.basename(fp))
        if base is None:
            continue
        base_ep = base.timestamp()
        for ch, rec in read_window_traces(fp, animal, lo_ms, hi_ms).items():
            segs.append(rec["seg"])
            epochs.append(base_ep + rec["times"])
            chans.append(np.array([ch] * rec["seg"].shape[0]))
            fidx.append(np.full(rec["seg"].shape[0], k, dtype=int))
            win_ms = rec["win_ms"]
        _emit_progress(progress, k + 1, total, fp, t0)
    return _stack_trials(segs, epochs, chans, fidx, win_ms)


def _emit_progress(progress, done, total, fp, t0):
    """Loading feedback: caller callback, else a periodic stderr line."""
    if progress is not None:
        progress(done, total, fp)
        return
    if done == 1 or done == total or done % 50 == 0:
        el = time.time() - t0
        rate = done / el if el > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(f"  trials: {done}/{total} files  {rate*60:.0f}/min  "
              f"ETA {eta/60:.1f} min", flush=True)


def _stack_trials(segs, epochs, chans, fidx, win_ms) -> dict:
    """Concatenate per-file arrays into one trial-major struct."""
    if not segs:
        return {"seg": np.empty((0, 0), np.float32), "epoch": np.empty(0),
                "channel": np.empty(0, object), "file_idx": np.empty(0, int),
                "win_ms": np.empty(0)}
    order = np.argsort(np.concatenate(epochs))
    return {
        "seg": np.concatenate(segs, axis=0)[order],
        "epoch": np.concatenate(epochs)[order],
        "channel": np.concatenate(chans).astype(object)[order],
        "file_idx": np.concatenate(fidx)[order],
        "win_ms": win_ms,
    }


def collect_segments(files: list[str], animal: str, lo_ms: float, hi_ms: float,
                     epoch_lo: float, epoch_hi: float, *,
                     progress=None) -> dict:
    """Gather ONLY the trial segments whose absolute epoch falls in
    ``[epoch_lo, epoch_hi]`` (used to build the stable-window template without
    holding the whole record in memory)."""
    assert epoch_hi >= epoch_lo, "epoch window must be ordered"
    # Only files whose recording START can carry a trial in [epoch_lo, epoch_hi]
    # need opening: start <= epoch_hi and start >= epoch_lo - (max recording
    # length). 6 h margin comfortably covers a chunk's duration.
    margin = 6 * 3600.0
    cand = [f for f in sorted(files)
            if _in_epoch_range(f, epoch_lo - margin, epoch_hi)]
    segs = []
    win_ms = None
    t0 = time.time()
    total = len(cand)
    for k, fp in enumerate(cand):
        assert k < _MAX_FILES, "file scan runaway"
        base = parse_recording_dt(os.path.basename(fp))
        if base is None:
            continue
        base_ep = base.timestamp()
        for _ch, rec in read_window_traces(fp, animal, lo_ms, hi_ms).items():
            ep = base_ep + rec["times"]
            m = (ep >= epoch_lo) & (ep <= epoch_hi)
            if m.any():
                segs.append(rec["seg"][m])
                win_ms = rec["win_ms"]
        _emit_progress(progress, k + 1, total, fp, t0)
    seg = np.concatenate(segs, axis=0) if segs else np.empty((0, 0), np.float32)
    return {"seg": seg, "win_ms": win_ms}


def gather_sample_trials(files: list[str], animal: str, lo_ms: float,
                         hi_ms: float, *, max_trials: int = 8000,
                         progress=None) -> dict:
    """A representative sample of raw artifact WAVEFORMS across the whole record
    (for trace galleries / ERP-images). Takes evenly-spaced trials from every
    file so time coverage is uniform, then trims to ``max_trials``. Returns
    ``{seg[N,W], epoch[N], channel[N], win_ms[W]}`` sorted by epoch."""
    assert max_trials >= 1, "max_trials must be >= 1"
    files = sorted(files)
    per_file = max(1, int(np.ceil(max_trials / max(1, len(files)))))
    segs, eps, chans = [], [], []
    win_ms = None
    t0 = time.time()
    for k, fp in enumerate(files):
        assert k < _MAX_FILES, "file scan runaway"
        base = parse_recording_dt(os.path.basename(fp))
        if base is None:
            continue
        base_ep = base.timestamp()
        for ch, rec in read_window_traces(fp, animal, lo_ms, hi_ms).items():
            n = rec["seg"].shape[0]
            take = (np.linspace(0, n - 1, per_file).astype(int) if n > per_file
                    else np.arange(n))
            segs.append(rec["seg"][take])
            eps.append(base_ep + rec["times"][take])
            chans.append(np.array([ch] * take.size))
            win_ms = rec["win_ms"]
        _emit_progress(progress, k + 1, len(files), fp, t0)
    return _stack_sample(segs, eps, chans, win_ms, max_trials)


def _stack_sample(segs, eps, chans, win_ms, max_trials) -> dict:
    """Concatenate, sort by epoch, and evenly trim a waveform sample."""
    if not segs:
        return {"seg": np.empty((0, 0), np.float32), "epoch": np.empty(0),
                "channel": np.empty(0, object), "win_ms": np.empty(0)}
    seg = np.concatenate(segs, axis=0)
    ep = np.concatenate(eps)
    ch = np.concatenate(chans).astype(object)
    order = np.argsort(ep)
    seg, ep, ch = seg[order], ep[order], ch[order]
    if ep.size > max_trials:
        keep = np.linspace(0, ep.size - 1, max_trials).astype(int)
        seg, ep, ch = seg[keep], ep[keep], ch[keep]
    return {"seg": seg, "epoch": ep, "channel": ch, "win_ms": win_ms}


def stream_metrics(files: list[str], animal: str, lo_ms: float, hi_ms: float,
                   template, *, progress=None) -> dict:
    """Compute per-trial metrics against a FIXED *template* by streaming files
    one at a time -- segments are never all held in memory. Returns stacked,
    chronologically-sorted metric arrays plus each trial's epoch and channel."""
    assert template is not None and np.ndim(template) == 1, "1-D template req'd"
    cols: dict = {k: [] for k in
                  ("epoch", "channel", "file_idx", "polarity", "polarity_abs",
                   "amplitude", "corr", "nrmse")}
    t0 = time.time()
    total = len(files)
    for k, fp in enumerate(sorted(files)):
        assert k < _MAX_FILES, "file scan runaway"
        base = parse_recording_dt(os.path.basename(fp))
        if base is None:
            continue
        base_ep = base.timestamp()
        for ch, rec in read_window_traces(fp, animal, lo_ms, hi_ms).items():
            _accumulate_file_metrics(cols, rec, ch, base_ep, k, template)
        _emit_progress(progress, k + 1, total, fp, t0)
    return _finalize_metric_columns(cols)


def _accumulate_file_metrics(cols, rec, ch, base_ep, k, template):
    """Compute one file/channel's metric rows and append to the accumulators."""
    seg, win_ms = rec["seg"], rec["win_ms"]
    m = per_trial_metrics(seg, win_ms, template)
    n = seg.shape[0]
    cols["epoch"].append(base_ep + rec["times"])
    cols["channel"].append(np.array([ch] * n))
    cols["file_idx"].append(np.full(n, k, dtype=int))
    for key in ("polarity", "polarity_abs", "amplitude", "corr", "nrmse"):
        cols[key].append(np.asarray(m[key]))


def _finalize_metric_columns(cols) -> dict:
    """Concatenate + chronologically sort the streamed metric columns."""
    if not cols["epoch"]:
        return {k: np.empty(0) for k in cols}
    ep = np.concatenate(cols["epoch"])
    order = np.argsort(ep)
    out = {"epoch": ep[order]}
    out["channel"] = np.concatenate(cols["channel"]).astype(object)[order]
    for key in ("file_idx", "polarity", "polarity_abs", "amplitude",
                "corr", "nrmse"):
        out[key] = np.concatenate(cols[key])[order]
    return out


def stim_polarity(seg, win_ms) -> np.ndarray:
    """Sign (+1/-1) of the first post-stim (t>0) extremum, by latency: +1 if the
    window maximum precedes its minimum, else -1."""
    seg = np.asarray(seg, float)
    win_ms = np.asarray(win_ms, float)
    assert seg.ndim == 2 and seg.shape[1] == win_ms.size, "shape mismatch"
    post = win_ms > 0
    assert post.sum() >= 2, "need >=2 post-stim samples"
    sp = seg[:, post]
    imax = sp.argmax(axis=1)
    imin = sp.argmin(axis=1)
    return np.where(imax <= imin, 1, -1).astype(int)


def polarity_abs_sign(seg, win_ms) -> np.ndarray:
    """Robustness check: sign at the max-|amplitude| post-stim sample."""
    seg = np.asarray(seg, float)
    win_ms = np.asarray(win_ms, float)
    post = win_ms > 0
    sp = seg[:, post]
    iabs = np.abs(sp).argmax(axis=1)
    return np.sign(sp[np.arange(sp.shape[0]), iabs]).astype(int)


def build_template(seg_stable) -> np.ndarray:
    """Fixed shape template = element-wise MEDIAN artifact over the stable-run
    trials. Requires a reasonable sample (>=50) so it is not noise."""
    seg_stable = np.asarray(seg_stable, float)
    assert seg_stable.ndim == 2, "seg_stable must be 2-D [trials, samples]"
    assert seg_stable.shape[0] >= 50, "need >=50 stable trials for a template"
    return np.median(seg_stable, axis=0)


def template_correlation(seg, template) -> np.ndarray:
    """Pearson correlation of each trial vs the FIXED template."""
    seg = np.asarray(seg, float)
    t = np.asarray(template, float)
    assert seg.shape[1] == t.size, "template length mismatch"
    a = seg - seg.mean(axis=1, keepdims=True)
    b = t - t.mean()
    den = np.sqrt((a * a).sum(axis=1) * float((b * b).sum()))
    den = np.where(den == 0, np.nan, den)
    return (a @ b) / den


def normalized_rmse(seg, template) -> np.ndarray:
    """Span-normalised RMSE of each trial vs the FIXED template:
    ``sqrt(mean((x-T)^2)) / (max(T)-min(T))``."""
    seg = np.asarray(seg, float)
    t = np.asarray(template, float)
    assert seg.shape[1] == t.size, "template length mismatch"
    span = float(t.max() - t.min())
    assert span > 0, "degenerate template (zero span)"
    return np.sqrt(((seg - t) ** 2).mean(axis=1)) / span


def per_trial_metrics(seg, win_ms, template) -> dict:
    """All three metrics for a stack of trials, as equal-length arrays."""
    seg = np.asarray(seg, float)
    assert seg.ndim == 2 and seg.shape[0] >= 1, "seg must be [trials, samples]"
    return {
        "polarity": stim_polarity(seg, win_ms),
        "polarity_abs": polarity_abs_sign(seg, win_ms),
        "amplitude": ef.peak_to_trough(seg),
        "corr": template_correlation(seg, template),
        "nrmse": normalized_rmse(seg, template),
    }


def switch_count(polarity) -> int:
    """Number of consecutive-trial polarity sign changes (series must already
    be in chronological order)."""
    p = np.asarray(polarity, int)
    if p.size < 2:
        return 0
    return int((p[1:] != p[:-1]).sum())
