"""One-time comprehensive stim-artifact overlays for a FOLDER of raw KMrecorder
electrode-stress recordings (no DB, no seizures).

For every stimulated pole (e.g. ``stim11-2a``) AND for the within-electrode
differential ``A - B`` (the two poles of one electrode), each pulse's LFP artifact
is captured in a ``[-1, 2] ms`` window and rendered as comprehensively as the
peri-ictal / stim-stability reports -- and, per the user, NOTHING is downsampled:

  * per-hour overlay: every pulse of that recording (thin) under the mean +/- SEM;
  * session all-pulse overlay: every pulse of every recording, coloured on a
    continuous early->late time gradient, under the grand mean;
  * per-file mean overlay: each recording's mean artifact, time-gradient coloured;
  * metric trends across the session (peak-to-peak / +peak / -peak / saturation,
    plus access-resistance Rₐ and slow steady-state Z_ss for the real poles).

The differential removes the common-mode so a true bipolar artifact remains.

HEAVY + OFFLINE ONLY: each raw ``.mat`` is ~3.5 GB and takes ~35 s to load; the
signal is released as soon as its (small) epoch stacks are extracted, so peak RAM
stays near one file. Reuses the stim-artifact-trend primitives so the pulse
detection / windows / metrics match the daemon path.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from src.utils.animal import stim_copy_indices     # noqa: E402
from src.utils.impedance import (access_resistance,  # noqa: E402
                                 phase_currents, slow_steady_state)
from src.utils.mat_loader import load_mat          # noqa: E402
from src.utils.stim_artifact_trend import (        # noqa: E402
    WIN_PRE_MS, WIN_POST_MS, _neg_peak, _peak_amplitude, _pos_peak,
    recording_datetime, saturation_pct, stim_params_for)
from src.utils.stim_blank import detect_stim_onsets  # noqa: E402

logger = logging.getLogger("qc_monitor.electrode_test.artifact_figures")

_FIGSIZE = (9.0, 7.0)
_ACCENT = "#5e7ce2"
# The two poles of each electrode, by in-file channel name. The differential is
# a - b; the group's stimCopy (resolved as the nearest preceding stim-copy
# column) triggers the pulse onsets for both poles.
_ELECTRODES = [("11-2", "stim11-2a", "stim11-2b"),
               ("12-1", "stim12-1a", "stim12-1b")]


# --------------------------------------------------------------- reading --- #

def _resolve_group(names: list, a_name: str, b_name: str) -> tuple:
    """``(a_col, b_col, trigger_col)`` for one electrode: the two pole columns by
    name + the stimCopy column immediately preceding pole A as the onset trigger.
    ``(None, None, None)`` when either pole is absent from *names*."""
    if a_name not in names or b_name not in names:
        return None, None, None
    a_col, b_col = names.index(a_name), names.index(b_name)
    sc = sorted(stim_copy_indices(names))
    trig = max((s for s in sc if s < a_col), default=(sc[0] if sc else None))
    return a_col, b_col, trig


def _epoch_stacks(sig, fs: float, cols: list, onsets_sec) -> tuple:
    """``(time_ms, {col: stack[n_pulses, n_samp]}, n)`` -- fixed ``[-1, 2] ms``
    windows around EVERY onset for each requested column (never subsampled).
    Edge-truncated windows are dropped."""
    assert sig.ndim == 2 and cols, "2-D signal and columns required"
    lo = int(round(WIN_PRE_MS * fs / 1000.0))
    hi = int(round(WIN_POST_MS * fs / 1000.0))
    tm = np.arange(lo, hi) / fs * 1000.0
    centers = np.round(np.asarray(onsets_sec, dtype=float) * fs).astype(int)
    centers = centers[(centers + lo >= 0) & (centers + hi < len(sig))]
    stacks = {c: np.empty((centers.size, hi - lo), dtype=float) for c in cols}
    for i, c in enumerate(centers):
        seg = sig[c + lo:c + hi, :]
        for col in cols:
            stacks[col][i] = seg[:, col]
    return tm, stacks, int(centers.size)


def _series_rec(dt, tm, stack, n, sat, stim_mean) -> dict:
    """One recording's contribution to a series: the pulse stack + its mean/SEM,
    plus the scalars later metrics need. ``stim_mean`` (averaged stimCopy) and
    ``sat`` are None for a differential series."""
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0) if stack.size else np.zeros(tm.size)
        sem = (np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n)) if n >= 2 \
            else np.zeros(tm.size)
    return {"dt": dt, "tm": tm, "stack": stack, "n": n,
            "mean": mean, "sem": np.nan_to_num(sem),
            "sat": sat, "stim_mean": stim_mean}


def process_file(path: str) -> tuple:
    """Load one raw recording once, then extract per-pole + per-differential pulse
    stacks for both electrodes, releasing the ~3.5 GB signal before returning.
    Returns ``(dt, {series_key: rec})``; empty dict when nothing usable."""
    assert path, "path required"
    chunk = load_mat(path)
    sig, fs, names = chunk.signal, chunk.fs, list(chunk.channel_names or [])
    dt = recording_datetime(path)
    out = {}
    for elec, a_name, b_name in _ELECTRODES:
        a_col, b_col, trig = _resolve_group(names, a_name, b_name)
        if a_col is None or trig is None:
            continue
        onsets = detect_stim_onsets(sig[:, trig], fs)
        if onsets.size == 0:
            continue
        tm, stacks, n = _epoch_stacks(sig, fs, [a_col, b_col, trig], onsets)
        if n == 0:
            continue
        stim_mean = np.nanmean(stacks[trig], axis=0)
        out[a_name] = _series_rec(dt, tm, stacks[a_col], n,
                                  saturation_pct(sig[:, a_col]), stim_mean)
        out[b_name] = _series_rec(dt, tm, stacks[b_col], n,
                                  saturation_pct(sig[:, b_col]), stim_mean)
        out[f"diff_{elec}"] = _series_rec(
            dt, tm, stacks[a_col] - stacks[b_col], n, None, None)
    del sig, chunk
    return dt, out


# --------------------------------------------------------------- metrics --- #

def _metric_window(params: dict) -> tuple:
    """``(x0, x1)`` ms measurement window from the stim params (pulse width x
    (1+ratio)), matching stim_artifact_trend.compute_metrics; a safe default when
    the STIM_REPORT is missing."""
    pw, ratio = params.get("pulse_width_us"), params.get("ratio")
    x1 = (float(pw) * (1.0 + float(ratio)) / 1000.0 + 0.3) if (pw and ratio) else 1.5
    return -0.2, x1


def _rec_metrics(rec: dict, params: dict, is_real: bool) -> dict:
    """Per-recording metrics from its mean trace: peak-to-peak / +peak / -peak /
    saturation for any series, plus Rₐ / Z_ss for a real pole when gain + currents
    are available (best-effort; None otherwise)."""
    tm, mean = list(rec["tm"]), list(rec["mean"])
    x0, x1 = _metric_window(params)
    m = {"ptp": _peak_amplitude(tm, mean, x0, x1),
         "pos": _pos_peak(tm, mean, x0, x1),
         "neg": _neg_peak(tm, mean, x0, x1),
         "sat": rec.get("sat"), "ra": None, "zss": None}
    gain = params.get("gain")
    i_pos, i_neg = phase_currents(params.get("charge_nC"),
                                  params.get("pulse_width_us"), params.get("ratio"))
    if is_real and rec.get("stim_mean") is not None and gain and i_pos and i_neg:
        try:
            ar = access_resistance(mean, tm, list(rec["stim_mean"]),
                                   gain, i_pos, i_neg)
            m["ra"] = ar.get("r_access_kohm")
            if m["ra"] is not None:
                zss = slow_steady_state(mean, tm, list(rec["stim_mean"]),
                                        gain, i_neg)
                m["zss"] = zss.get("slow_ss_kohm")
        except Exception as e:                              # noqa: BLE001
            logger.debug("Rₐ/Z_ss failed: %s", e)
    return m


# --------------------------------------------------------------- figures --- #

def _fracs(recs: list) -> list:
    """Each recording's [0,1] position in the session (by datetime), for the
    early->late colour gradient. All-zero when the timestamps are unusable."""
    dts = [r["dt"] for r in recs]
    if any(d is None for d in dts) or len(recs) < 2:
        return [0.0] * len(recs)
    t0 = min(dts).timestamp()
    span = (max(dts).timestamp() - t0) or 1.0
    return [(d.timestamp() - t0) / span for d in dts]


def _add_pulses(ax, tm, stack, color, alpha, lw=0.4) -> None:
    """Overlay every row of *stack* as one LineCollection (light: one artist for
    thousands of pulses)."""
    if stack.size:
        segs = [np.column_stack([tm, row]) for row in stack]
        ax.add_collection(LineCollection(segs, colors=[color], linewidths=lw,
                                         alpha=alpha))


def plot_hour(rec: dict, title: str, out_png: str, color=_ACCENT) -> str:
    """Per-recording overlay: every pulse (thin) under the mean +/- SEM band."""
    assert rec and out_png, "rec and out_png required"
    tm = np.asarray(rec["tm"])
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    _add_pulses(ax, tm, rec["stack"], color, alpha=0.08)
    mean, sem = np.asarray(rec["mean"]), np.asarray(rec["sem"])
    ax.fill_between(tm, mean - sem, mean + sem, color=color, alpha=0.25,
                    linewidth=0, label="± SEM")
    ax.plot(tm, mean, color="#111111", lw=1.8, label=f"mean (n={rec['n']})")
    ax.axvline(0, color="#888", lw=0.6, ls="--")
    ax.axhline(0, color="#888", lw=0.6, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time from stim (ms)")
    ax.set_ylabel("artifact (V)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_session_allpulse(recs: list, title: str, out_png: str) -> str:
    """Every pulse of every recording, coloured on a continuous early->late time
    gradient, under the bold grand mean (of the per-recording means)."""
    assert recs and out_png, "recs and out_png required"
    cmap = plt.get_cmap("viridis")
    tm = np.asarray(recs[0]["tm"])
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for frac, r in zip(_fracs(recs), recs):
        _add_pulses(ax, np.asarray(r["tm"]), r["stack"], cmap(frac),
                    alpha=0.05, lw=0.3)
    grand = np.mean([np.asarray(r["mean"]) for r in recs], axis=0)
    ax.plot(tm, grand, color="#111111", lw=2.4, zorder=5, label="grand mean")
    ax.axvline(0, color="#888", lw=0.6, ls="--")
    ax.axhline(0, color="#888", lw=0.6, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time from stim (ms)")
    ax.set_ylabel("artifact (V)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, framealpha=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("recording time (early → late)", fontsize=8)
    cb.set_ticks([0.0, 1.0])
    cb.set_ticklabels(["start", "end"])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_filemean_overlay(recs: list, title: str, out_png: str) -> str:
    """Each recording's mean artifact, time-gradient coloured, + the grand mean --
    the drift of the average artifact across the stress test."""
    assert recs and out_png, "recs and out_png required"
    cmap = plt.get_cmap("viridis")
    tm = np.asarray(recs[0]["tm"])
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for frac, r in zip(_fracs(recs), recs):
        ax.plot(np.asarray(r["tm"]), np.asarray(r["mean"]),
                color=cmap(frac), lw=1.0, alpha=0.85)
    grand = np.mean([np.asarray(r["mean"]) for r in recs], axis=0)
    ax.plot(tm, grand, color="#111111", lw=2.4, zorder=5, label="grand mean")
    ax.axvline(0, color="#888", lw=0.6, ls="--")
    ax.axhline(0, color="#888", lw=0.6, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time from stim (ms)")
    ax.set_ylabel("artifact (V)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, framealpha=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("recording time (early → late)", fontsize=8)
    cb.set_ticks([0.0, 1.0])
    cb.set_ticklabels(["start", "end"])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


_TREND_SPECS = [("ptp", "peak-to-peak (V)", "#5e7ce2"),
                ("pos", "+peak (V)", "#2ca02c"),
                ("neg", "−peak (V)", "#d62728"),
                ("sat", "saturation (%)", "#9467bd"),
                ("ra", "Rₐ (kΩ)", "#ff7f0e"),
                ("zss", "Z_ss (kΩ)", "#17becf")]


def plot_metric_trends(recs: list, metrics: list, hours: list, title: str,
                       out_png: str) -> str:
    """Every metric that has data, stacked vs hours-into-test (dates in the
    x-label) -- one row per metric so drift over the stress test reads directly."""
    assert recs and out_png, "recs and out_png required"
    present = [s for s in _TREND_SPECS
               if any(m.get(s[0]) is not None for m in metrics)]
    if not present:
        present = [_TREND_SPECS[0]]
    fig, axes = plt.subplots(len(present), 1, figsize=(9.0, 1.7 * len(present)),
                             sharex=True, squeeze=False)
    for ax, (key, ylab, col) in zip(axes[:, 0], present):
        xy = [(h, m[key]) for h, m in zip(hours, metrics) if m.get(key) is not None]
        if xy:
            ax.plot([p[0] for p in xy], [p[1] for p in xy], "o-", color=col,
                    ms=3, lw=1.3)
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(True, alpha=0.2)
    d0 = next((r["dt"] for r in recs if r["dt"]), None)
    d1 = next((r["dt"] for r in reversed(recs) if r["dt"]), None)
    span = (f"  ({d0:%Y-%m-%d %H:%M} → {d1:%Y-%m-%d %H:%M})"
            if d0 and d1 else "")
    axes[0, 0].set_title(title + span, fontsize=10)
    axes[-1, 0].set_xlabel("hours into stress test")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# --------------------------------------------------------- orchestration --- #

def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-.") else "_" for c in str(s))


def _hours_axis(recs: list) -> list:
    dts = [r["dt"] for r in recs]
    t0 = min((d for d in dts if d), default=None)
    return [((d.timestamp() - t0.timestamp()) / 3600.0 if d and t0 else i)
            for i, d in enumerate(dts)]


def _render_series(key: str, recs: list, params: dict, out_dir: str,
                   progress) -> dict:
    """All figures for one series (pole or differential): per-hour overlays +
    session all-pulse + per-file mean + metric trends. Returns a small summary."""
    sdir = os.path.join(out_dir, _safe(key))
    hourly = os.path.join(sdir, "hourly")
    os.makedirs(hourly, exist_ok=True)
    is_real = not key.startswith("diff_")
    for r in recs:                                        # per-hour breakdowns
        stamp = r["dt"].strftime("%Y%m%d_%H%M") if r["dt"] else f"{recs.index(r):03d}"
        progress(f"{key}: hour {stamp}")
        plot_hour(r, f"{key} — {stamp} (n={r['n']})",
                  os.path.join(hourly, f"{_safe(key)}_{stamp}.png"),
                  color=(_ACCENT if is_real else "#d62728"))
    progress(f"{key}: session figures")
    plot_session_allpulse(recs, f"{key} — every pulse, all recordings",
                          os.path.join(sdir, f"{_safe(key)}_session_allpulse.png"))
    plot_filemean_overlay(recs, f"{key} — per-recording mean artifact",
                          os.path.join(sdir, f"{_safe(key)}_filemean_overlay.png"))
    metrics = [_rec_metrics(r, params, is_real) for r in recs]
    plot_metric_trends(recs, metrics, _hours_axis(recs),
                       f"{key} — metric trends",
                       os.path.join(sdir, f"{_safe(key)}_metric_trends.png"))
    return {"n_recordings": len(recs), "dir": sdir}


def run(folder: str, out_dir: str, *, max_files=None, progress=None) -> dict:
    """Render the full electrode-test figure set for *folder* into *out_dir*.
    Loads each recording once (heavy), accumulates the per-series pulse stacks,
    then renders every series. Returns ``{series: {n_recordings, dir}}``."""
    assert folder and out_dir, "folder and out_dir required"
    from src.utils.stim_artifact_trend import list_recordings
    prog = progress or (lambda *_a: None)
    files = list_recordings(folder)
    if max_files:
        files = files[:int(max_files)]
    assert files, "no .mat recordings in folder"
    os.makedirs(out_dir, exist_ok=True)
    series = {}
    params = stim_params_for(files[0])
    for i, path in enumerate(files):
        prog(f"reading {i + 1}/{len(files)}: {os.path.basename(path)[:48]}")
        _dt, per = process_file(path)
        for key, rec in per.items():
            series.setdefault(key, []).append(rec)
    out = {}
    for key in sorted(series):
        out[key] = _render_series(key, series[key], params, out_dir, prog)
    return {"folder": folder, "out_dir": out_dir, "series": out,
            "stim_params": params, "n_files": len(files)}


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="One-time electrode-test stim-artifact overlay figures")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-files", type=int, default=None,
                    help="only the first N recordings (for a quick preview)")
    args = ap.parse_args(argv)
    folder = args.folder.rstrip("\\/")
    out_dir = args.out_dir or os.path.join(
        "data", "electrode_test", _safe(os.path.basename(folder))[:60])
    res = run(folder, out_dir, max_files=args.max_files,
              progress=lambda m: print("  ", m, flush=True))
    print(f"\ndone: {res['n_files']} recordings -> {out_dir}")
    for key, d in res["series"].items():
        print(f"  {key}: {d['n_recordings']} recordings")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
