"""PNG renderers for the weekly stimulus-stability report.

Three figures per stimulated channel:
  * ``plot_stim_trace`` -- an average stimulus waveform (amplitude vs ms), used
    at the per-file, per-day, and weekly-grand-average levels (Agg matplotlib).
  * ``plot_stim_vs_evoked`` -- stim-magnitude vs evoked-magnitude scatter +
    regression, to see whether they correlate (Agg matplotlib + scipy).
  * ``impedance_png`` -- the access-resistance (Rₐ) drift trend, matching the QC
    Monitor Overview card, rendered from Plotly via kaleido (best-effort).

Matplotlib is a declared dependency; kaleido is best-effort (the impedance PNG is
skipped, not fatal, if it's unavailable) -- mirroring ``preictal/figures.py``.
"""

from __future__ import annotations

import logging
from datetime import datetime

import matplotlib

matplotlib.use("Agg")                     # headless; safe in the daemon thread

import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402

from src.alerting.rules import _drift_stat  # noqa: E402  (pure rolling-median stat)

logger = logging.getLogger("qc_monitor.notifications.stim_figures")

_ACCENT = "#5e7ce2"
_STIM = "#e2a15e"
_FIT = "#ff453a"
_FIGSIZE = (9.0, 4.2)


def _parse_chunk_dt(s: str):
    """'YYYY_MM_DD__HH_MM_SS' -> datetime, or None."""
    try:
        return datetime.strptime(str(s), "%Y_%m_%d__%H_%M_%S")
    except (ValueError, TypeError):
        return None


# Distinct, chronologically-ordered day colours (cool -> warm) -- a qualitative
# palette, NOT a heat-map gradient; the order + a legend make it readable.
_DAY_COLORS = ["#3b4cc0", "#2aa198", "#2ca02c", "#bcbd22", "#ff7f0e",
               "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def _day_label(day: str) -> str:
    try:
        return datetime.strptime(str(day)[:10], "%Y_%m_%d").strftime("%b %d")
    except (ValueError, TypeError):
        return str(day)


def day_color_map(days_sorted) -> dict:
    """{day: distinct colour} assigned in chronological order."""
    return {d: _DAY_COLORS[i % len(_DAY_COLORS)]
            for i, d in enumerate(days_sorted)}


def plot_day_overlay(file_traces, daily_traces, title: str, out_png: str,
                     day_color: dict, ylabel: str = "LFP amplitude") -> str | None:
    """Overlay every file trace (thin, day-coloured) with the per-day MEAN traces
    (bold) on top -- one distinct colour per day + a chronological legend. Each
    of *file_traces* / *daily_traces* is (day, time_ms, y), pre-cropped to the
    window. Returns the path, or None if no daily mean was drawn."""
    drew = False
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for day, tm, y in file_traces:
        if tm and y and len(tm) == len(y) >= 3:
            ax.plot(tm, y, color=day_color.get(day, "#888"), lw=0.5, alpha=0.22)
    for day, tm, y in sorted(daily_traces):
        if not (tm and y and len(tm) == len(y) >= 3):
            continue
        ax.plot(tm, y, color=day_color.get(day, "#888"), lw=2.2,
                label=_day_label(day))
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.axhline(0.0, color="#888", lw=0.6, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, ncol=2, framealpha=0.6, title="day")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_stim_trace(time_ms, y, title: str, out_png: str, *,
                    ylabel: str = "amplitude", color: str = _ACCENT,
                    sem=None) -> str | None:
    """Average waveform (amplitude vs ms) -> PNG, with a shaded mean +/- SEM band
    when *sem* is given. Returns the path, or None when there's nothing
    plottable. Used for the LFP evoked-response figures at each time window."""
    assert out_png, "out_png required"
    if time_ms is None or y is None or len(time_ms) != len(y) or len(y) < 3:
        return None
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if sem is not None and len(sem) == len(y):
        ya = np.asarray(y, dtype=float)
        sa = np.asarray(sem, dtype=float)
        ax.fill_between(time_ms, ya - sa, ya + sa, color=color, alpha=0.22,
                        linewidth=0, label="± SEM")
    ax.plot(time_ms, y, color=color, lw=1.5, label="mean")
    ax.axhline(0.0, color="#888", lw=0.6, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    if sem is not None:
        ax.legend(fontsize=8, framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_stim_vs_evoked(stim_mags, evoked_mags, title: str, out_png: str, *,
                        days=None, day_color=None, stim_win=None,
                        evoked_win=(1.0, 50.0)) -> dict:
    """LFP stim-artifact vs LFP evoked magnitude scatter + regression -> PNG.

    Points are coloured by recording DAY with distinct colours + a chronological
    legend (not a gradient); the axes name the measurement windows. Returns
    ``{r, r2, p, n, slope, png}`` (``png`` None when <3 paired points).
    Regression via ``scipy.stats.linregress``, skipped when every x is identical
    -- mirrors chronic_evoked._add_regression."""
    xs = np.asarray([float(x) for x in stim_mags], dtype=float)
    ys = np.asarray([float(y) for y in evoked_mags], dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    dd = ([d for d, k in zip(days, ok) if k] if days is not None else None)
    xs, ys = xs[ok], ys[ok]
    out = {"r": float("nan"), "r2": float("nan"), "p": float("nan"),
           "n": int(xs.size), "slope": float("nan"), "png": None}
    if xs.size < 3:
        return out
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if dd is not None and day_color:
        for day in sorted(set(dd)):
            idx = [i for i, d in enumerate(dd) if d == day]
            ax.scatter(xs[idx], ys[idx], s=24, alpha=0.8, linewidths=0,
                       color=day_color.get(day, "#888"), label=_day_label(day))
        ax.legend(fontsize=7, ncol=2, framealpha=0.6, title="day")
    else:
        ax.scatter(xs, ys, s=14, c=_ACCENT, alpha=0.6, linewidths=0)
    if np.ptp(xs) > 0:
        from scipy.stats import linregress
        lr = linregress(xs, ys)
        xline = np.array([xs.min(), xs.max()])
        ax.plot(xline, lr.slope * xline + lr.intercept, color=_FIT, lw=2.0)
        out.update(r=float(lr.rvalue), r2=float(lr.rvalue ** 2),
                   p=float(lr.pvalue), slope=float(lr.slope))
        title = (f"{title}  ·  r={lr.rvalue:.2f} r²={lr.rvalue ** 2:.2f} "
                 f"p={lr.pvalue:.1e} n={xs.size}")
    sw = f" ({stim_win[0]:g} to {stim_win[1]:g} ms)" if stim_win else ""
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"LFP stim-artifact magnitude, peak-to-trough{sw}")
    ax.set_ylabel(f"LFP evoked magnitude, peak-to-trough "
                  f"({evoked_win[0]:g}–{evoked_win[1]:g} ms)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    out["png"] = out_png
    return out


def impedance_png(rows: list[dict], out_png: str, *, title: str,
                  pct: float = 40.0, window: int = 10, min_history: int = 4,
                  engine: str = "auto", highlight=None,
                  value_key: str = "access_r_kohm", ylabel: str = "Rₐ (kΩ)",
                  series_name: str = "Rₐ") -> str | None:
    """Per-channel impedance-metric drift trend -> PNG: the metric line +
    rolling-median baseline band + a red-ringed latest point when drifting.
    *highlight* = (start_datetime, end_datetime) shades a date window (used to
    mark 'this week' on the full-history plot).

    *value_key* selects which per-row metric to plot (``access_r_kohm`` for the
    access-resistance Rₐ trend, the default; ``slow_ss_kohm`` for the slow-phase
    steady-state impedance Z_ss trend), with *ylabel* / *series_name* labelling
    the axis + trace accordingly. Both ride the SAME ``impedance_series_by_channel``
    rows, which carry both metrics.

    Tries the Overview's Plotly look via kaleido first (``engine`` 'auto'/'plotly')
    for dashboard fidelity, then falls back to a portable matplotlib render so the
    figure is ALWAYS produced even where kaleido/Chrome is unavailable. Returns
    the path, or None only when there's nothing to plot. *rows* are oldest->newest
    from ``impedance_series_by_channel``."""
    vals = [r.get(value_key) for r in rows if r.get(value_key) is not None]
    if len(vals) < 2:
        return None
    dts = [_parse_chunk_dt(r.get("chunk_datetime")) for r in rows]
    x = dts if all(d is not None for d in dts) else list(range(len(rows)))
    y = [r.get(value_key) for r in rows]
    hl = highlight if (x and not isinstance(x[0], int)) else None  # dates only
    stat = _drift_stat(vals, window, min_history)
    drifting = bool(stat and abs(stat["drift_pct"]) >= pct)
    sub = f"  ·  Δ {stat['drift_pct']:+.0f}% vs baseline" if stat else ""
    if engine in ("auto", "plotly"):
        p = _impedance_plotly(x, y, stat, drifting, pct, title + sub, out_png, hl,
                              ylabel, series_name)
        if p is not None:
            return p
        if engine == "plotly":
            return None
    return _impedance_mpl(x, y, stat, drifting, pct, title + sub, out_png, hl,
                          ylabel, series_name)


def _impedance_plotly(x, y, stat, drifting, pct, title, out_png, highlight,
                      ylabel="Rₐ (kΩ)", series_name="Rₐ"):
    """Overview-fidelity Plotly render -> PNG via kaleido; None if unavailable."""
    try:
        import plotly.graph_objects as go
    except Exception:                                     # noqa: BLE001
        return None
    fig = go.Figure()
    if highlight is not None:
        fig.add_vrect(x0=highlight[0], x1=highlight[1], line_width=0,
                      fillcolor="rgba(255,159,10,0.16)",
                      annotation_text="this week", annotation_position="top left")
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=series_name,
                             line=dict(color=_ACCENT, width=1.8),
                             marker=dict(size=5)))
    if stat is not None:
        base, band = stat["baseline"], stat["baseline"] * pct / 100.0
        fig.add_hrect(y0=base - band, y1=base + band, line_width=0,
                      fillcolor="rgba(94,124,226,0.10)")
        fig.add_hline(y=base, line=dict(color="#888", dash="dash", width=1))
        if drifting:
            fig.add_trace(go.Scatter(
                x=[x[-1]], y=[y[-1]], mode="markers", showlegend=False,
                marker=dict(size=11, color="rgba(0,0,0,0)",
                            line=dict(color=_FIT, width=2))))
    fig.update_layout(title=dict(text=title, font=dict(size=13)),
                      template="plotly_white", height=420, width=900,
                      margin=dict(l=60, r=20, t=48, b=44), showlegend=False)
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text="recording date")
    try:
        fig.write_image(out_png, scale=2)                 # needs kaleido + Chrome
    except Exception as e:                                # noqa: BLE001
        logger.info("kaleido render unavailable, using matplotlib fallback: %s", e)
        return None
    return out_png


def _impedance_mpl(x, y, stat, drifting, pct, title, out_png, highlight,
                   ylabel="Rₐ (kΩ)", series_name="Rₐ"):
    """Portable matplotlib (Agg) render of the same metric trend -- always works."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if highlight is not None:
        ax.axvspan(highlight[0], highlight[1], color="#ff9f0a", alpha=0.16,
                   label="this week")
    ax.plot(x, y, color=_ACCENT, lw=1.8, marker="o", markersize=4,
            label=series_name)
    if stat is not None:
        base, band = stat["baseline"], stat["baseline"] * pct / 100.0
        ax.axhspan(base - band, base + band, color=_ACCENT, alpha=0.10)
        ax.axhline(base, color="#888", ls="--", lw=1.0, label="baseline")
        if drifting:
            ax.scatter([x[-1]], [y[-1]], s=120, facecolors="none",
                       edgecolors=_FIT, linewidths=2.0, zorder=5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("recording date")
    ax.grid(True, alpha=0.2)
    if highlight is not None:
        ax.legend(fontsize=8, framealpha=0.6, loc="best")
    if x and not isinstance(x[0], int):
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
