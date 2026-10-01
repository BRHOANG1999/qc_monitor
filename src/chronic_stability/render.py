"""Figures for the chronic-stim stability map (matplotlib, headless-safe).

Design goals (for a reader who is NOT the analyst): every figure has a title
that states what it shows, axis labels with units, and a legend. Time axes are
in ELAPSED DAYS (easy to reason about) with calendar dates as a supplementary
top axis. Kept separate from the analysis so the numbers stay unit-testable.
"""

from __future__ import annotations

import datetime as _dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates                      # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.patches import Patch                  # noqa: E402
from matplotlib.ticker import FuncFormatter           # noqa: E402

from src.periictal.forecast import _ecdf              # noqa: E402

_DPI = 130
_DAY = 86400.0
_CFG_COLORS = ["#3182ce", "#dd6b20", "#38a169", "#805ad5"]


def _days(epochs, t0):
    return (np.asarray(epochs, float) - t0) / _DAY


def _date_top(ax, t0, label="calendar date"):
    """Supplementary top x-axis showing dates for a primary axis in days."""
    sec = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    sec.xaxis.set_major_formatter(FuncFormatter(
        lambda d, _p: (_dt.datetime.fromtimestamp(t0)
                       + _dt.timedelta(days=d)).strftime("%b %d")))
    sec.set_xlabel(label, fontsize=8)
    sec.tick_params(labelsize=7)
    return sec


def _cfg_letter(k: int) -> str:
    return chr(65 + k)


_BATTERY_COL = "#6b46c1"
_CAGE_COL = "#b7791f"


def _event_lines(axes, events, t0):
    """Vertical battery / cage change lines across one or more axes (x in days
    from *t0*). Returns legend handles. ``events`` = {'battery': [epoch...],
    'cage': [epoch...]}."""
    if not events:
        return []
    axlist = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
    handles = []
    for key, col, ls in (("battery", _BATTERY_COL, "-"),
                         ("cage", _CAGE_COL, ":")):
        eps = events.get(key) or []
        for ep in eps:
            for ax in axlist:
                ax.axvline((ep - t0) / _DAY, color=col, lw=0.7, ls=ls,
                           alpha=0.45, zorder=1)
        if eps:
            handles.append(Line2D([0], [0], color=col, lw=1.0, ls=ls,
                                  label=f"{key} change ({len(eps)})"))
    return handles


def ra_over_time_fig(ra: dict, intervals: list, run: dict, seizure_epochs,
                     out_png: str, *, t0: float | None = None,
                     events: dict | None = None) -> str:
    """Both electrodes' access-resistance trends on one wall-clock axis (days),
    so the SR→SLM hand-off at the swap and the flat stable plateau are visible.
    Each configuration's record electrode is one coloured line. Battery / cage
    change times are overlaid as vertical lines. ``t0`` sets day 0 (default: the
    earliest Rₐ point; pass the record start to match the other figures)."""
    if t0 is None:
        t0 = min(te[0] for te, _r in ra.values() if len(te))
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    for k, iv in enumerate(intervals):
        te, r = ra.get(k, (np.empty(0), np.empty(0)))
        if not len(te):
            continue
        ax.plot(_days(te, t0), r, lw=1.0, color=_CFG_COLORS[k % 4],
                label=f"Config {_cfg_letter(k)}, {iv['record_site']} "
                      f"record electrode Rₐ")
    handles = _shade_stable(ax, run, t0)
    ev = _event_lines(ax, events, t0)
    rug = _seizure_rug(ax, seizure_epochs, t0)
    ax.set_xlabel("days since recording start")
    ax.set_ylabel("access resistance Rₐ (kΩ)")
    ax.set_title("Access resistance over time, both record electrodes "
                 "(battery / cage changes marked)")
    _date_top(ax, t0)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + handles + ev + rug,
              loc="upper left", fontsize=7.5, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def _shade_stable(ax, run, t0):
    if not run.get("found"):
        return []
    ax.axvspan(_days(run["epoch_start"], t0), _days(run["epoch_end"], t0),
               color="#38a169", alpha=0.18, zorder=0)
    return [Patch(facecolor="#38a169", alpha=0.3, label="detected stable window")]


def _seizure_rug(ax, seizure_epochs, t0):
    e = np.asarray(seizure_epochs, float)
    lo, hi = ax.get_xlim()
    d = _days(e, t0)
    d = d[(d >= lo) & (d <= hi)]
    if d.size:
        y = ax.get_ylim()[1]
        ax.plot(d, [y] * d.size, "v", color="#e53e3e", ms=5, clip_on=False)
    return [Line2D([0], [0], marker="v", color="#e53e3e", ls="",
                   label=f"seizure ({d.size} shown)")]


def ra_aligned_fig(ra: dict, intervals: list, run: dict, out_png: str) -> str:
    """Both electrodes' Rₐ ALIGNED to each configuration's own start (day 0), so
    their flatness and drift are directly comparable side by side."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, iv in enumerate(intervals):
        te, r = ra.get(k, (np.empty(0), np.empty(0)))
        if not len(te):
            continue
        ax.plot((te - iv["start_epoch"]) / _DAY, r, lw=1.1,
                color=_CFG_COLORS[k % 4],
                label=f"Config {_cfg_letter(k)} — {iv['config_label']}")
    handles = []
    if run.get("found"):
        s = intervals[run_cfg(intervals, run)]["start_epoch"]
        ax.axvspan((run["epoch_start"] - s) / _DAY,
                   (run["epoch_end"] - s) / _DAY, color="#38a169", alpha=0.18)
        handles = [Patch(facecolor="#38a169", alpha=0.3,
                         label="stable window (in its config)")]
    ax.set_xlabel("days since THIS configuration began (aligned)")
    ax.set_ylabel("access resistance Rₐ (kΩ)")
    ax.set_title("Access resistance, configurations aligned to day 0 "
                 "(comparable)")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + handles,
              loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def run_cfg(intervals, run) -> int:
    """Index of the configuration the stable run falls in."""
    for k, iv in enumerate(intervals):
        if iv["start_epoch"] <= run["epoch_start"] < iv["end_epoch"]:
            return k
    return 0


def sensitivity_fig(sweep: list[dict], t0: float, default: dict,
                    out_png: str) -> str:
    """Robustness check: each row is ONE threshold choice; the bar is the
    stable window it detects (in days). If the bars line up, the window barely
    depends on the thresholds. The default choice is highlighted."""
    found = [s for s in sweep if s.get("found")]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.28 * len(found) + 1)))
    for i, s in enumerate(found):
        is_def = all(abs(s.get(k, -1) - default.get(k, -2)) < 1e-9
                     for k in ("win_h", "disp_k", "slope_k"))
        col = "#e53e3e" if is_def else "#3182ce"
        ax.plot([_days(s["epoch_start"], t0), _days(s["epoch_end"], t0)],
                [i, i], color=col, lw=3, solid_capstyle="butt",
                alpha=0.9 if is_def else 0.55)
    ax.set_yticks(range(len(found)))
    ax.set_yticklabels([f"win {int(s['win_h'])} h · disp×{s['disp_k']:g} · "
                        f"slope×{s['slope_k']:g}" for s in found], fontsize=7)
    ax.set_xlabel("days since recording start")
    ax.set_ylabel("threshold choice")
    ax.set_title("Stable-window detection is robust to the thresholds\n"
                 "(each bar = the window found for one setting)")
    ax.legend(handles=[Line2D([0], [0], color="#e53e3e", lw=3, label="default"),
                       Line2D([0], [0], color="#3182ce", lw=3, alpha=.55,
                              label="other settings")], loc="lower right",
              fontsize=8)
    _date_top(ax, t0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def metric_timeseries_fig(epoch, mets: dict, ra: dict, intervals: list,
                          run: dict, seizure_epochs, out_png: str, *,
                          events: dict | None = None) -> str:
    """Per-trial artifact metrics + both Rₐ trends over time. X axis is elapsed
    DAYS (calendar dates on top). Configuration bands and the stable window are
    shaded; battery / cage changes are marked; a legend labels every band and
    series."""
    t0 = float(np.min(epoch))
    x = _days(epoch, t0)
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    panels = [("amplitude", "peak-to-trough\namplitude (µV)", "#3182ce"),
              ("corr", "shape corr\nvs template", "#38a169"),
              ("nrmse", "shape nRMSE\nvs template", "#dd6b20")]
    for ax, (key, lab, col) in zip(axes[:3], panels):
        ax.plot(x, mets[key], ",", color=col, alpha=0.3)
        ax.plot([], [], "o", color=col, label=lab.replace("\n", " "))
        ax.set_ylabel(lab, fontsize=9)
        ax.legend(loc="upper right", fontsize=7, markerscale=1)
    _ra_panel(axes[3], ra, intervals, t0)
    _polarity_panel(axes[4], x, mets["polarity"])
    _overlay_bands(axes, epoch, intervals, run, seizure_epochs, t0)
    ev = _event_lines(axes, events, t0)
    axes[4].set_xlabel("days since recording start (Day 0 = first trial)")
    _date_top(axes[0], t0)
    if ev:
        axes[3].legend(handles=axes[3].get_legend_handles_labels()[0] + ev,
                       loc="upper right", fontsize=7, ncol=2)
    axes[0].set_title("BCH111 per-trial artifact metrics and Rₐ over time "
                      "(battery / cage changes marked)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def _ra_panel(ax, ra, intervals, t0):
    for k, iv in enumerate(intervals):
        te, r = ra.get(k, (np.empty(0), np.empty(0)))
        if len(te):
            ax.plot(_days(te, t0), r, lw=0.8, color=_CFG_COLORS[k % 4],
                    label=f"{iv['record_site']} Rₐ")
    ax.set_ylabel("Rₐ (kΩ)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)


def _polarity_panel(ax, x, polarity):
    ax.plot(x, polarity, ",", color="#2d3748", alpha=0.3)
    ax.plot([], [], "o", color="#2d3748", label="first-extremum sign")
    ax.set_ylabel("polarity", fontsize=9)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["− (trough)", "+ (peak)"], fontsize=8)
    ax.legend(loc="upper right", fontsize=7)


def _overlay_bands(axes, epoch, intervals, run, seizure_epochs, t0):
    """Shade config intervals + stable window; seizure rug; shared legend."""
    e0, e1 = float(np.min(epoch)), float(np.max(epoch))
    band_cols = ["#dbeafe", "#fef3c7", "#e6fffa", "#f3e8ff"]
    proxies = []
    for k, iv in enumerate(intervals):
        proxies.append(Patch(facecolor=band_cols[k % 4], alpha=0.7,
                             label=f"Config {_cfg_letter(k)} {iv['config_label']}"))
        for ax in axes:
            lo, hi = max(iv["start_epoch"], e0), min(iv["end_epoch"], e1)
            if hi > lo:
                ax.axvspan(_days(lo, t0), _days(hi, t0),
                           color=band_cols[k % 4], alpha=0.55, zorder=0)
    if run.get("found"):
        for ax in axes:
            ax.axvspan(_days(run["epoch_start"], t0), _days(run["epoch_end"], t0),
                       color="#38a169", alpha=0.22, zorder=0)
        proxies.append(Patch(facecolor="#38a169", alpha=0.35,
                             label="detected stable window"))
    d = _days(np.asarray(seizure_epochs, float), t0)
    d = d[(d >= _days(e0, t0)) & (d <= _days(e1, t0))]
    for s in d:
        axes[0].axvline(s, color="#e53e3e", lw=0.4, alpha=0.5)
    proxies.append(Line2D([0], [0], color="#e53e3e", lw=1,
                          label=f"seizure ({d.size})"))
    axes[0].legend(handles=proxies, loc="upper left", fontsize=7, ncol=2,
                   framealpha=0.9)


def erpimage_fig(sample: dict, intervals: list, run: dict, out_png: str, *,
                 total_trials: int | None = None) -> str:
    """ERP-image: a time-ordered SAMPLE of artifact waveforms, each row one real
    trial (oldest at the bottom; colour = µV). The record holds ~millions of
    trials; this shows an evenly-spaced sample of them so the whole stability
    history reads at once (polarity flip at the swap, flat stable band, later
    drift). Horizontal lines mark config boundaries and the stable window."""
    seg = np.asarray(sample["seg"], float)
    ep = np.asarray(sample["epoch"], float)
    win = np.asarray(sample["win_ms"], float)
    assert seg.ndim == 2 and seg.shape[0] > 1, "need a stacked waveform sample"
    n = seg.shape[0]
    v = float(np.percentile(np.abs(seg), 98))
    fig, ax = plt.subplots(figsize=(8.5, 9))
    im = ax.imshow(seg, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v, extent=[win[0], win[-1], 0, n],
                   interpolation="nearest")
    handles = _erp_markers(ax, ep, intervals, run, win)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    frac = f"1 of ~{round(total_trials / n):d} trials" if total_trials else ""
    ax.set_xlabel("time from stimulus (ms)")
    ax.set_ylabel("sampled trial (oldest at bottom; calendar dates at right)")
    ax.set_title(f"Artifact waveforms over the record ({n:,}-trial sample)\n"
                 f"each row is one real trial{'; ' + frac if frac else ''}")
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.colorbar(im, ax=ax, label="artifact amplitude (µV)", shrink=0.55)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def _erp_markers(ax, ep, intervals, run, win):
    """Date y-ticks + config-boundary / stable-window lines; returns legend."""
    n = ep.size
    ticks = np.linspace(0, n - 1, 8).astype(int)
    ax.set_yticks(ticks)
    ax.set_yticklabels([_dt.datetime.fromtimestamp(ep[i]).strftime("%b %d")
                        for i in ticks], fontsize=7)
    row = lambda e: int(np.searchsorted(ep, e))
    handles = []
    for k, iv in enumerate(intervals):
        if np.isfinite(iv["end_epoch"]) and row(iv["end_epoch"]) < n:
            ax.axhline(row(iv["end_epoch"]), color="k", lw=0.9, alpha=0.6)
    handles.append(Line2D([0], [0], color="k", lw=0.9, label="config boundary"))
    if run.get("found"):
        for e in (run["epoch_start"], run["epoch_end"]):
            ax.axhline(row(e), color="#2f855a", lw=1.6)
        handles.append(Line2D([0], [0], color="#2f855a", lw=1.6,
                              label="stable window"))
    return handles


def trace_gallery_fig(sample: dict, groups: list, template, out_png: str) -> str:
    """Grid of example artifact traces: for each named group, ~14 individual
    trials (thin), the group MEAN (thick), and the stable template (dashed)."""
    win = np.asarray(sample["win_ms"], float)
    seg = np.asarray(sample["seg"], float)
    ng = len(groups)
    fig, axes = plt.subplots(1, ng, figsize=(4.2 * ng, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, (label, mask, col) in zip(axes, groups):
        idx = np.where(mask)[0]
        if idx.size == 0:
            ax.set_title(f"{label}\n(no trials)", fontsize=9)
            continue
        pick = rng.choice(idx, size=min(14, idx.size), replace=False)
        for j in pick:
            ax.plot(win, seg[j], color=col, lw=0.5, alpha=0.35)
        ax.plot(win, seg[idx].mean(0), color=col, lw=2.4, label="group mean")
        ax.plot(win, template, color="k", lw=1.3, ls="--",
                label="stable template")
        ax.plot([], [], color=col, lw=0.5, alpha=0.6, label="single trials")
        ax.axvline(0, color="0.6", lw=0.6, ls=":")
        ax.set_title(f"{label}\n(n={idx.size})", fontsize=9)
        ax.set_xlabel("ms from stimulus")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("artifact amplitude (µV)")
    fig.suptitle("Example artifact waveforms per configuration "
                 "(thin = individual trials, thick = mean)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def mean_trace_overlay_fig(sample: dict, groups: list, template,
                           out_png: str) -> str:
    """Mean artifact ± SD per group, overlaid, with each group's individual
    trials drawn thin + transparent under its mean and the stable template."""
    win = np.asarray(sample["win_ms"], float)
    seg = np.asarray(sample["seg"], float)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, mask, col in groups:
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        pick = rng.choice(idx, size=min(14, idx.size), replace=False)
        for j in pick:                    # individual trials (the constituents)
            ax.plot(win, seg[j], color=col, lw=0.4, alpha=0.18)
        m, sd = seg[idx].mean(0), seg[idx].std(0)
        ax.plot(win, m, color=col, lw=2, label=f"{label} (n={idx.size})")
        ax.fill_between(win, m - sd, m + sd, color=col, alpha=0.12)
    ax.plot(win, template, color="k", lw=1.4, ls="--", label="stable template")
    ax.axvline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("time from stimulus (ms)")
    ax.set_ylabel("artifact amplitude (µV)")
    ax.set_title("Mean artifact waveform per configuration "
                 "(thin = individual trials, band = ±1 SD)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def _plot_sample(x, cap: int = 20000) -> np.ndarray:
    """Finite values, deterministically down-sampled to <= *cap* points for
    plotting only (KS/MWU/CIs elsewhere use the full data)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size <= cap:
        return x
    return x[::int(np.ceil(x.size / cap))]


def distribution_panel(groups: list, labels: list, metric_name: str,
                       out_png: str) -> str:
    """Overlaid ECDF + violin for 2+ metric samples (plot-sampled for speed).
    ECDF: fraction of trials at or below a value. Violin: the full shape."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for g, lab, c in zip(groups, labels, _CFG_COLORS):
        g = _plot_sample(g)
        if g.size == 0:
            continue
        xs, ys = _ecdf(g)
        a1.step(xs, ys, where="post", color=c, lw=1.6, label=lab)
    a1.set_xlabel(metric_name)
    a1.set_ylabel("cumulative fraction of trials (ECDF)")
    a1.set_title("Cumulative distribution")
    a1.legend(fontsize=8)
    clean = [g for g in (_plot_sample(x) for x in groups) if g.size]
    if clean:
        a2.violinplot(clean, showmedians=True)
        a2.set_xticks(range(1, len(clean) + 1))
        a2.set_xticklabels([lab for lab, g in zip(labels, clean)],
                           fontsize=7, rotation=12, ha="right")
    a2.set_ylabel(metric_name)
    a2.set_title("Distribution shape (violin; bar = median)")
    fig.suptitle(f"{metric_name}: distribution comparison")
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def age_stratified_fig(epoch, metric, config_start: float, metric_name: str,
                       out_png: str) -> str:
    """ECDFs of a metric within age quartiles of one configuration — reveals a
    stable-vs-full gap that is really an age gradient (collinearity check)."""
    epoch = np.asarray(epoch, float)
    metric = np.asarray(metric, float)
    age = (epoch - config_start) / _DAY
    m = np.isfinite(age) & np.isfinite(metric)
    age, metric = age[m], metric[m]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    if age.size >= 8:
        q = np.quantile(age, [0, .25, .5, .75, 1.0])
        cols = ["#bee3f8", "#63b3ed", "#3182ce", "#2c5282"]
        for i in range(4):
            sel = (age >= q[i]) & (age <= q[i + 1])
            if sel.sum() < 2:
                continue
            xs, ys = _ecdf(metric[sel])
            ax.step(xs, ys, where="post", color=cols[i], lw=1.6,
                    label=f"age quartile {i+1}: {q[i]:.1f}–{q[i+1]:.1f} days")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("cumulative fraction of trials (ECDF)")
    ax.set_title(f"{metric_name} by configuration age\n"
                 "(if the curves separate, 'stability' tracks age)")
    ax.legend(fontsize=8, title="config age")
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


# ---------------- windowed (zoomed) access-resistance views ----------------- #
# Zoom into a [win_lo, win_hi] slice of the record. At this scale calendar time
# (hours / dates) is what a reader needs to answer "when did the seizure occur",
# so the x-axis is wall-clock datetime rather than the elapsed-days axis used by
# the full-record figures. Every seizure inside the window is drawn as a dashed
# red line labelled with its time -- "label when they occur".

def _ts(epochs):
    """Epoch seconds -> python datetimes for a calendar x-axis."""
    return [_dt.datetime.fromtimestamp(float(e)) for e in np.atleast_1d(epochs)]


def _plot_ra_window(ax, ra, intervals, lo, hi):
    """Draw each config's record-electrode Rₐ clipped to [lo, hi] (+2% pad for
    line continuity) on a calendar x-axis. Returns the number of series drawn."""
    pad = 0.02 * (hi - lo)
    drawn = 0
    for k, iv in enumerate(intervals):
        te, r = ra.get(k, (np.empty(0), np.empty(0)))
        te = np.asarray(te, float)
        if not te.size:
            continue
        m = (te >= lo - pad) & (te <= hi + pad)
        if not m.any():
            continue
        ax.plot(_ts(te[m]), np.asarray(r, float)[m], lw=1.1,
                color=_CFG_COLORS[k % 4], marker=".", ms=3,
                label=f"Config {_cfg_letter(k)}, {iv['record_site']} "
                      f"record electrode Rₐ")
        drawn += 1
    return drawn


def _shade_stable_window(ax, run, lo, hi):
    """Shade the part of the detected stable window that falls inside [lo, hi]."""
    if not run.get("found"):
        return []
    s, e = max(float(run["epoch_start"]), lo), min(float(run["epoch_end"]), hi)
    if e <= s:
        return []
    ax.axvspan(_dt.datetime.fromtimestamp(s), _dt.datetime.fromtimestamp(e),
               color="#38a169", alpha=0.18, zorder=0)
    return [Patch(facecolor="#38a169", alpha=0.3, label="detected stable window")]


def _event_lines_window(ax, events, lo, hi):
    """Battery / cage change lines that fall inside [lo, hi]; legend handles."""
    if not events:
        return []
    handles = []
    for key, col, ls in (("battery", _BATTERY_COL, "-"),
                         ("cage", _CAGE_COL, ":")):
        eps = [float(e) for e in (events.get(key) or []) if lo <= e <= hi]
        for ep in eps:
            ax.axvline(_dt.datetime.fromtimestamp(ep), color=col, lw=0.8,
                       ls=ls, alpha=0.5, zorder=1)
        if eps:
            handles.append(Line2D([0], [0], color=col, lw=1.0, ls=ls,
                                  label=f"{key} change ({len(eps)})"))
    return handles


def _seizure_marks_window(ax, seizure_epochs, lo, hi, time_fmt):
    """Dashed red line + time label for every seizure in [lo, hi]. Labels sit
    just inside the top of the axes (blended transform, so never clipped by the
    data range) and alternate between two heights so close neighbours don't
    collide. Returns the legend handle list and the count shown."""
    e = np.asarray(seizure_epochs, float)
    e = np.sort(e[(e >= lo) & (e <= hi)])
    trans = ax.get_xaxis_transform()                   # x = data, y = axes frac
    bbox = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7)
    for i, ep in enumerate(e):
        xn = mdates.date2num(_dt.datetime.fromtimestamp(float(ep)))
        ax.axvline(xn, color="#e53e3e", lw=1.1, ls="--", alpha=0.85, zorder=4)
        ax.text(xn, 0.95 - 0.14 * (i % 3),            # 3 lanes: separate clusters
                _dt.datetime.fromtimestamp(float(ep)).strftime(time_fmt),
                transform=trans, rotation=90, va="top", ha="center",
                fontsize=6, color="#c53030", zorder=5, bbox=bbox, clip_on=True)
    return [Line2D([0], [0], color="#e53e3e", lw=1.1, ls="--",
                   label=f"seizure ({e.size})")], int(e.size)


def _format_time_axis(ax, lo, hi):
    """Pick calendar tick locator / formatter appropriate to the window span."""
    span_h = (hi - lo) / 3600.0
    if span_h <= 30:                                   # ~24 h
        maj, fmt = mdates.HourLocator(interval=3), mdates.DateFormatter("%H:%M")
    elif span_h <= 60:                                 # ~2 days
        maj = mdates.HourLocator(interval=6)
        fmt = mdates.DateFormatter("%b %d\n%H:%M")
    else:                                              # ~1 week+
        maj, fmt = mdates.DayLocator(), mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_locator(maj)
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlim(_dt.datetime.fromtimestamp(lo), _dt.datetime.fromtimestamp(hi))


def ra_over_time_window_fig(ra: dict, intervals: list, run: dict,
                            seizure_epochs, out_png: str, *, win_lo: float,
                            win_hi: float, label: str, animal: str = "BCH111",
                            events: dict | None = None) -> str:
    """Zoomed access-resistance-over-time figure for the [win_lo, win_hi] epoch
    window. Both record electrodes' Rₐ (clipped to the window), the stable-window
    shading, battery / cage lines, and a dashed labelled line for every seizure
    inside the window, on a wall-clock x-axis."""
    assert win_hi > win_lo, "window must be non-empty"
    assert out_png, "out_png required"
    span_h = (win_hi - win_lo) / 3600.0
    time_fmt = "%H:%M" if span_h <= 30 else "%b %d %H:%M"
    fig, ax = plt.subplots(figsize=(12, 5))
    _plot_ra_window(ax, ra, intervals, win_lo, win_hi)
    sh = _shade_stable_window(ax, run, win_lo, win_hi)
    ev = _event_lines_window(ax, events, win_lo, win_hi)
    sz, n_sz = _seizure_marks_window(ax, seizure_epochs, win_lo, win_hi,
                                     time_fmt)
    _format_time_axis(ax, win_lo, win_hi)
    ax.set_xlabel("time of day" if span_h <= 30 else "calendar date / time")
    ax.set_ylabel("access resistance Rₐ (kΩ)")
    d0 = _dt.datetime.fromtimestamp(win_lo).strftime("%b %d %H:%M")
    d1 = _dt.datetime.fromtimestamp(win_hi).strftime("%b %d %H:%M")
    ax.set_title(f"{animal} — access resistance over time · {label} window "
                 f"({d0} – {d1}) · {n_sz} seizure(s) labelled")
    # Legend below the plot so it never covers the top-anchored seizure labels.
    # Drop it lower when the date ticks are two-line (the ~2-day window) so it
    # clears the x-axis label.
    yoff = -0.24 if 30 < span_h <= 60 else -0.14
    ax.legend(handles=ax.get_legend_handles_labels()[0] + sh + ev + sz,
              loc="upper center", bbox_to_anchor=(0.5, yoff),
              fontsize=7.5, framealpha=0.9, ncol=5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png
