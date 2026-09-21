"""Peri-ictal stimulus-artifact stability: is a seizure destabilising the stim
artifact?

For each scored seizure of an animal, the LFP stim artifact (-1..2 ms deflection
to each pulse) is averaged over the stimuli that fall in three 30-minute windows
around onset:

    pre  = [onset-30min, onset]
    at   = [onset,       onset+30min]
    post = [onset+30min, onset+60min]

A forward window (``at``/``post``) is shortened so it never overlaps the NEXT
seizure onset; the ``pre`` window is likewise clamped at the PREVIOUS seizure so
it isn't contaminated. Per seizure we render (a) the three window-averaged artifact
waveforms overlaid (mean +/- SEM) and (b) each stimulus's artifact magnitude vs
minutes-from-onset with the windows shaded -- so a jump at/after onset reads as
instability. A pooled figure compares pre/at/post magnitudes across all seizures.

HEAVY: reads raw ``*_evoked.mat`` epochs (only near-seizure files, once each), so
run offline via the CLI -- never on a dashboard thread (``read_file_evoked``
guards that anyway).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from src.notifications.evoked_weekly import _peak_amplitude  # noqa: E402
from src.notifications.stim_figures import (  # noqa: E402
    plot_gradient_overlay, plot_stim_trace)
from src.preictal.isi import included_seizures, parse_chunk_datetime  # noqa: E402
from src.utils.animal import split_animal_electrode  # noqa: E402
from src.utils.evoked_output import read_file_evoked  # noqa: E402

logger = logging.getLogger("qc_monitor.periictal.peri_stim_artifact")

_WIDTH = 1800.0                    # 30-min window (seconds)
_ARTIFACT_MS = (-1.0, 2.0)         # LFP stim-artifact window
_FILE_SPAN_SEC = 3600.0           # nominal recording length for file/window overlap

WIN_ORDER = ("pre", "at", "post")
WIN_COLOR = {"pre": "#3b4cc0", "at": "#d62728", "post": "#2ca02c"}
WIN_LABEL = {"pre": "pre (−30..0 min)", "at": "at (0..+30 min)",
             "post": "post (+30..+60 min)"}
_FIGSIZE = (9.0, 7.0)


# --------------------------------------------------------------- windows --- #

def seizure_windows(onset: float, prev_onset, next_onset,
                    width: float = _WIDTH) -> dict:
    """{name: (lo_epoch, hi_epoch)} for the three windows, truncated so a forward
    window never overlaps the next seizure and ``pre`` never overlaps the previous
    one. Windows that collapse to <=0 length are omitted."""
    nxt = next_onset if next_onset is not None else float("inf")
    prv = prev_onset if prev_onset is not None else float("-inf")
    pre_lo = max(onset - width, prv)
    at_hi = min(onset + width, nxt)
    post_lo = onset + width
    post_hi = min(onset + 2 * width, nxt)
    out = {}
    if pre_lo < onset:
        out["pre"] = (pre_lo, onset)
    if onset < at_hi:
        out["at"] = (onset, at_hi)
    if post_lo < post_hi:
        out["post"] = (post_lo, post_hi)
    return out


# ------------------------------------------------------------- DB lookup --- #

def _animal_files(store, animal: str) -> list[tuple]:
    """Sorted ``[(file_id, chunk_start_epoch)]`` for every evoked recording that
    carries a channel of *animal*."""
    assert animal, "animal required"
    conn = store._connect()
    try:
        rows = conn.execute(
            """SELECT DISTINCT pf.id, pf.chunk_datetime
               FROM evoked_waveforms ew JOIN processed_files pf
                 ON pf.id = ew.file_id
               WHERE ew.channel_name LIKE ? AND pf.chunk_datetime IS NOT NULL""",
            (f"{animal}%",)).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        dt = parse_chunk_datetime(r["chunk_datetime"])
        if dt is not None:
            out.append((int(r["id"]), dt.timestamp()))
    out.sort(key=lambda t: t[1])
    return out


def _channels_for(store, animal: str, override) -> list[str]:
    """Stimulated channel(s) to analyse: *override* if given, else the animal's
    active-impedance (stimulated) channels, else every evoked channel it has."""
    if override:
        return [override]
    try:
        active = sorted(c for (a, c) in store.active_impedance_channel_keys()
                        if a == animal)
    except Exception:                                # noqa: BLE001
        active = []
    if active:
        return active
    conn = store._connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT channel_name FROM evoked_waveforms "
            "WHERE channel_name LIKE ?", (f"{animal}%",)).fetchall()
    finally:
        conn.close()
    return sorted(r["channel_name"] for r in rows if r["channel_name"])


# --------------------------------------------------------------- epochs --- #

def _match_channel(chans: dict, channel: str):
    """read_file_evoked keys can drift from channel_name; match on animal+electrode."""
    rec = chans.get(channel)
    if rec is not None:
        return rec
    want = split_animal_electrode(channel)
    for k, v in chans.items():
        if split_animal_electrode(k) == want:
            return v
    return None


def _read_epochs(store, file_id: int, start_epoch: float, animal: str,
                 channel: str, art_ms) -> tuple:
    """(abs_times[n], traces[n, k], time_ms[k]) for one file+channel, each epoch's
    LFP cropped to *art_ms*. Empty arrays when the .mat/channel is unusable. The
    full evokedData is sliced then released (no per-file array retained)."""
    path = store.evoked_output_path_for_file(int(file_id))
    chans = (read_file_evoked(path, only_animals=[animal])
             if path and os.path.exists(path) else {})
    rec = _match_channel(chans, channel)
    if not rec:
        return np.empty(0), np.empty((0, 0)), np.empty(0)
    traces, tms, times = rec.get("traces"), rec.get("time_ms"), rec.get("times")
    if traces is None or tms is None or getattr(traces, "ndim", 0) != 2:
        return np.empty(0), np.empty((0, 0)), np.empty(0)
    tm = np.asarray(tms, dtype=float)
    m = (tm >= art_ms[0]) & (tm <= art_ms[1])
    if m.sum() < 3:
        return np.empty(0), np.empty((0, 0)), np.empty(0)
    sub = np.asarray(traces, dtype=float)[:, m]
    abs_t = start_epoch + np.asarray(times, dtype=float)[:sub.shape[0]]
    return abs_t, sub, tm[m]


# --------------------------------------------------------------- compute --- #

def _window_stats(abs_t, traces, tm, lo, hi) -> dict | None:
    """Mean +/- SEM artifact waveform + per-stimulus peak-to-trough magnitudes for
    the epochs whose absolute time is in [lo, hi). Keeps EVERY constituent epoch
    trace (``traces``, never downsampled) so each window mean can be overlaid on
    all the recordings that make it up."""
    sel = (abs_t >= lo) & (abs_t < hi)
    n = int(sel.sum())
    if n < 1:
        return None
    w = traces[sel]
    mags = np.array([_peak_amplitude(tm.tolist(), row.tolist(), tm[0], tm[-1])
                     for row in w], dtype=float)
    mags = mags[np.isfinite(mags)]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(w, axis=0)
        sem = (np.nanstd(w, axis=0, ddof=1) / np.sqrt(n)) if n >= 2 \
            else np.zeros(w.shape[1])
    return {"n": n, "time_ms": tm, "mean": mean, "sem": np.nan_to_num(sem),
            "mags": mags, "abs_t": abs_t[sel],
            "traces": w, "trace_abs_t": abs_t[sel],
            "mag_mean": float(np.nanmean(mags)) if mags.size else float("nan"),
            "mag_sem": (float(np.nanstd(mags, ddof=1) / np.sqrt(mags.size))
                        if mags.size >= 2 else 0.0)}


def _process_seizure(store, animal, channel, files, sz, prev_on, next_on,
                     art_ms) -> dict | None:
    """All three window stats for one seizure. Reads each overlapping file once."""
    wins = seizure_windows(sz.onset_epoch, prev_on, next_on)
    if not wins:
        return None
    span_lo = min(lo for lo, _ in wins.values())
    span_hi = max(hi for _, hi in wins.values())
    eps = []
    for fid, fstart in files:                         # files overlapping the span
        if fstart < span_hi and fstart + _FILE_SPAN_SEC > span_lo:
            abs_t, tr, tm = _read_epochs(store, fid, fstart, animal, channel, art_ms)
            if abs_t.size:
                eps.append((abs_t, tr, tm))
    if not eps:
        return None
    abs_all = np.concatenate([e[0] for e in eps])
    tr_all = np.concatenate([e[1] for e in eps])
    tm = eps[0][2]
    per_win = {}
    for name, (lo, hi) in wins.items():
        st = _window_stats(abs_all, tr_all, tm, lo, hi)
        if st is not None:
            per_win[name] = st
    if not per_win:
        return None
    order = np.argsort(abs_all)                       # chronological for the gradient
    return {"onset_epoch": sz.onset_epoch, "racine": sz.racine,
            "chunk_datetime": sz.chunk_datetime, "windows": per_win,
            "span": (span_lo, span_hi), "tm": tm,
            "all_traces": tr_all[order], "all_abs_t": abs_all[order]}


# --------------------------------------------------------------- figures --- #

def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))


def plot_seizure_panel(res: dict, title: str, out_png: str) -> str:
    """Per-seizure panel: (top) the three window-averaged artifact waveforms
    overlaid (mean +/- SEM); (bottom) each stimulus's peak-to-trough magnitude vs
    minutes-from-onset, coloured by window with the per-window mean drawn."""
    onset = res["onset_epoch"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=_FIGSIZE)
    for name in WIN_ORDER:
        st = res["windows"].get(name)
        if not st:
            continue
        c, tm = WIN_COLOR[name], np.asarray(st["time_ms"])
        ax1.fill_between(tm, st["mean"] - st["sem"], st["mean"] + st["sem"],
                         color=c, alpha=0.2, linewidth=0)
        ax1.plot(tm, st["mean"], color=c, lw=2.0,
                 label=f"{WIN_LABEL[name]}  n={st['n']}")
    ax1.axvline(0, color="#888", lw=0.6, ls="--")
    ax1.axhline(0, color="#888", lw=0.6, alpha=0.5)
    ax1.set_title(title, fontsize=10)
    ax1.set_xlabel("time from stim (ms)")
    ax1.set_ylabel("LFP artifact (µV)")
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=8, framealpha=0.6)
    for name in WIN_ORDER:
        st = res["windows"].get(name)
        if not st or not st["mags"].size:
            continue
        c = WIN_COLOR[name]
        mins = (st["abs_t"] - onset) / 60.0
        ax2.scatter(mins, st["mags"], s=12, color=c, alpha=0.55, linewidths=0)
        if np.isfinite(st["mag_mean"]):
            ax2.hlines(st["mag_mean"], mins.min(), mins.max(), color=c, lw=2.5)
    ax2.axvline(0, color="#c0392b", lw=1.2, label="onset")
    ax2.set_xlabel("minutes from seizure onset")
    ax2.set_ylabel("artifact magnitude, p2p (µV)")
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=8, framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_window_overlay(res: dict, title: str, out_png: str) -> str:
    """Overlay version of the mean panel: every window's constituent stimulus
    artifacts drawn thin + window-coloured, with the three window MEANS bold on
    top -- so the spread of each window (and any pre->at->post shift) is visible,
    not just the averages."""
    assert res and out_png, "res and out_png required"
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    drew = False
    for name in WIN_ORDER:
        st = res["windows"].get(name)
        if not st:
            continue
        c, tm = WIN_COLOR[name], np.asarray(st["time_ms"])
        for row in st["traces"]:
            if row.size == tm.size:
                ax.plot(tm, row, color=c, lw=0.4, alpha=0.12, zorder=1)
        ax.plot(tm, st["mean"], color=c, lw=2.2, zorder=3,
                label=f"{WIN_LABEL[name]}  n={st['n']}")
        drew = True
    ax.axvline(0, color="#888", lw=0.6, ls="--")
    ax.axhline(0, color="#888", lw=0.6, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time from stim (ms)")
    ax.set_ylabel("LFP artifact (µV)")
    ax.grid(True, alpha=0.2)
    if drew:
        ax.legend(fontsize=8, framealpha=0.6, title="window mean over its stimuli")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_perwindow_overlays(res: dict, title_base: str, stem: str) -> list:
    """One dedicated overlay PNG per window: the window mean +/- SEM over its own
    constituent stimulus artifacts (thin, transparent) via the shared
    ``plot_stim_trace`` renderer -- the same look as the stim-stability report.
    Returns the paths written."""
    assert res and stem, "res and stem required"
    out = []
    for name in WIN_ORDER:
        st = res["windows"].get(name)
        if not st:
            continue
        tm = np.asarray(st["time_ms"]).tolist()
        overlay = [row.tolist() for row in st["traces"] if row.size == len(tm)]
        png = f"{stem}_{name}.png"
        p = plot_stim_trace(
            tm, np.asarray(st["mean"]).tolist(),
            f"{title_base} — {WIN_LABEL[name]} (n={st['n']})", png,
            ylabel="LFP artifact (µV)", color=WIN_COLOR[name],
            sem=np.asarray(st["sem"]).tolist(), overlay=overlay)
        if p:
            out.append(p)
    return out


def plot_seizure_gradient(res: dict, title: str, out_png: str) -> str | None:
    """Every peri-ictal stimulus artifact drawn on a CONTINUOUS pre->post time
    gradient (onset in the middle) under the bold grand mean -- so a smooth drift
    of the artifact THROUGH the seizure reads as colour. Reuses the report's
    ``plot_gradient_overlay``."""
    assert res and out_png, "res and out_png required"
    tr, at = res.get("all_traces"), res.get("all_abs_t")
    if tr is None or getattr(tr, "size", 0) == 0:
        return None
    lo, hi = res["span"]
    span = (hi - lo) or 1.0
    tm = np.asarray(res["tm"]).tolist()
    traces = [((float(t) - lo) / span, tm, row.tolist())
              for row, t in zip(tr, at) if row.size == len(tm)]
    base = (tm, np.nanmean(tr, axis=0).tolist()) if tr.shape[0] else None
    return plot_gradient_overlay(
        traces, base, title, out_png, ylabel="LFP artifact (µV)",
        cbar_label="time through peri-ictal window (pre → post)")


def plot_pooled_overlay(results: list, animal: str, channel: str,
                        out_png: str) -> str:
    """Pooled overlay: for each window, every seizure's window-mean waveform drawn
    thin + window-coloured with the across-seizure grand mean bold -- the
    seizure-to-seizure spread of the artifact per window in one view."""
    assert results and out_png, "results and out_png required"
    fig, axes = plt.subplots(1, len(WIN_ORDER), figsize=(13.5, 4.6), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), WIN_ORDER):
        means = [(np.asarray(r["windows"][name]["time_ms"]),
                  np.asarray(r["windows"][name]["mean"]))
                 for r in results if name in r["windows"]]
        for tm, y in means:
            if tm.size == y.size:
                ax.plot(tm, y, color=WIN_COLOR[name], lw=0.7, alpha=0.35)
        if means:
            k = min(y.size for _, y in means)
            tm0 = means[0][0][:k]
            ax.plot(tm0, np.mean([y[:k] for _, y in means], axis=0),
                    color=WIN_COLOR[name], lw=2.6,
                    label=f"grand mean ({len(means)} sz)")
            ax.legend(fontsize=8, framealpha=0.6)
        ax.axvline(0, color="#888", lw=0.6, ls="--")
        ax.axhline(0, color="#888", lw=0.6, alpha=0.5)
        ax.set_title(WIN_LABEL[name], fontsize=9)
        ax.set_xlabel("time from stim (ms)")
        ax.grid(True, alpha=0.2)
    np.atleast_1d(axes)[0].set_ylabel("LFP artifact (µV)")
    fig.suptitle(f"{animal} {channel} — per-seizure window means overlaid",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_pooled(results: list, animal: str, channel: str, out_png: str) -> str:
    """Pooled across seizures: (left) each seizure's pre/at/post magnitude means
    connected (faint) + the across-seizure mean +/- SEM per window; (right) the
    grand-average artifact waveform per window."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    xs = {n: i for i, n in enumerate(WIN_ORDER)}
    per = {n: [] for n in WIN_ORDER}
    for r in results:
        pts = [(xs[n], r["windows"][n]["mag_mean"]) for n in WIN_ORDER
               if n in r["windows"] and np.isfinite(r["windows"][n]["mag_mean"])]
        if len(pts) >= 2:
            ax1.plot([p[0] for p in pts], [p[1] for p in pts], color="#aaa",
                     lw=0.8, alpha=0.5, marker="o", ms=3)
        for n, _ in pts:
            per[WIN_ORDER[n]].append(r["windows"][WIN_ORDER[n]]["mag_mean"])
    for n in WIN_ORDER:
        if per[n]:
            m = float(np.mean(per[n]))
            s = (float(np.std(per[n], ddof=1) / np.sqrt(len(per[n])))
                 if len(per[n]) >= 2 else 0.0)
            ax1.errorbar(xs[n], m, yerr=s, color=WIN_COLOR[n], marker="o", ms=9,
                         capsize=4, lw=2, zorder=5)
    ax1.set_xticks(list(xs.values()))
    ax1.set_xticklabels([WIN_LABEL[n] for n in WIN_ORDER], fontsize=8)
    ax1.set_ylabel("artifact magnitude, p2p (µV)")
    ax1.set_title(f"{animal} {channel} — magnitude by window "
                  f"({len(results)} seizures)", fontsize=10)
    ax1.grid(True, alpha=0.2, axis="y")
    for n in WIN_ORDER:
        means = [np.asarray(r["windows"][n]["mean"]) for r in results
                 if n in r["windows"]]
        if not means:
            continue
        k = min(m.size for m in means)
        tm = np.asarray(next(r["windows"][n]["time_ms"] for r in results
                             if n in r["windows"]))[:k]
        ax2.plot(tm, np.mean([m[:k] for m in means], axis=0), color=WIN_COLOR[n],
                 lw=2, label=WIN_LABEL[n])
    ax2.axvline(0, color="#888", lw=0.6, ls="--")
    ax2.axhline(0, color="#888", lw=0.6, alpha=0.5)
    ax2.set_xlabel("time from stim (ms)")
    ax2.set_ylabel("LFP artifact (µV)")
    ax2.set_title("grand-average artifact per window", fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=8, framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# --------------------------------------------------------- orchestration --- #

def run(store, animal: str, *, out_dir: str, channel=None, art_ms=_ARTIFACT_MS,
        max_seizures=None, last_seizures=None, progress=None) -> dict:
    """Render per-seizure + pooled peri-ictal stim-artifact figures for *animal*.
    *max_seizures* = the earliest N, *last_seizures* = the most recent N (else all;
    a seizure with no stim epochs in its windows is skipped). Returns
    ``{animal, channels: {ch: {n_seizures, pooled}}, panels: [...]}``."""
    assert animal and out_dir, "animal and out_dir required"
    prog = progress or (lambda *_a: None)
    seizures = included_seizures(store, animal)
    if not seizures:
        return {"animal": animal, "reason": "no scored seizures",
                "channels": {}, "panels": []}
    onsets = [s.onset_epoch for s in seizures]
    files = _animal_files(store, animal)
    os.makedirs(out_dir, exist_ok=True)
    out = {"animal": animal, "channels": {}, "panels": []}
    if max_seizures is not None:
        idxs = list(range(min(max_seizures, len(seizures))))
    elif last_seizures is not None:
        idxs = list(range(max(0, len(seizures) - last_seizures), len(seizures)))
    else:
        idxs = list(range(len(seizures)))
    for ch in _channels_for(store, animal, channel):
        results = []
        for pos, i in enumerate(idxs):
            prog(f"{ch}: seizure {i + 1} [{pos + 1}/{len(idxs)}]")
            sz = seizures[i]
            res = _process_seizure(
                store, animal, ch, files, sz,
                onsets[i - 1] if i > 0 else None,
                onsets[i + 1] if i + 1 < len(onsets) else None, art_ms)
            if res is None:
                continue
            dt = datetime.fromtimestamp(sz.onset_epoch).strftime("%Y-%m-%d %H:%M")
            stem = os.path.join(out_dir,
                                f"{_safe(animal)}_{_safe(ch)}_sz{i+1:02d}")
            ttl = f"{animal} {ch} — seizure {i+1} ({dt}, R{sz.racine})"
            made = [plot_seizure_panel(res, ttl, f"{stem}.png"),
                    plot_window_overlay(res, ttl + "  · overlay",
                                        f"{stem}_overlay.png"),
                    plot_seizure_gradient(res, ttl + "  · time gradient",
                                          f"{stem}_gradient.png")]
            made.extend(plot_perwindow_overlays(res, ttl, stem))
            out["panels"].extend(p for p in made if p)
            results.append(res)
        if results:
            base = os.path.join(out_dir, f"{_safe(animal)}_{_safe(ch)}")
            plot_pooled(results, animal, ch, f"{base}_pooled.png")
            plot_pooled_overlay(results, animal, ch, f"{base}_pooled_overlay.png")
            out["channels"][ch] = {"n_seizures": len(results),
                                   "pooled": f"{base}_pooled.png",
                                   "pooled_overlay": f"{base}_pooled_overlay.png"}
    return out


def _main(argv=None) -> int:
    import argparse

    import yaml

    from src.db.store import Store

    ap = argparse.ArgumentParser(description="Peri-ictal stim-artifact figures")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", required=True)
    ap.add_argument("--channel", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-seizures", type=int, default=None,
                    help="only the earliest N seizures")
    ap.add_argument("--last", type=int, default=None,
                    help="only the most recent N seizures")
    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    store = Store(cfg["database"]["path"])
    out_dir = args.out_dir or os.path.join("data", "peri_stim_artifact", args.animal)
    res = run(store, args.animal, out_dir=out_dir, channel=args.channel,
              max_seizures=args.max_seizures, last_seizures=args.last,
              progress=lambda m: print("  ", m))
    print(f"{res.get('reason', '')} panels: {len(res['panels'])} -> {out_dir}")
    for ch, d in res.get("channels", {}).items():
        print(f"  {ch}: {d['n_seizures']} seizures, pooled={d['pooled']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
