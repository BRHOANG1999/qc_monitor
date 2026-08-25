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


def plot_stim_trace(time_ms, y, title: str, out_png: str, *,
                    ylabel: str = "amplitude", color: str = _ACCENT) -> str | None:
    """Average waveform (amplitude vs ms) -> PNG. Returns the path, or None when
    there's nothing plottable. Used for the LFP evoked-response figures at each
    time window (stim-artifact and evoked)."""
    assert out_png, "out_png required"
    if time_ms is None or y is None or len(time_ms) != len(y) or len(y) < 3:
        return None
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(time_ms, y, color=color, lw=1.5)
    ax.axhline(0.0, color="#888", lw=0.6, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_stim_vs_evoked(stim_mags, evoked_mags, title: str, out_png: str, *,
                        times=None, stim_win=None,
                        evoked_win=(1.0, 50.0)) -> dict:
    """Stim-magnitude vs evoked-magnitude scatter + regression -> PNG.

    Points are coloured by *times* (per-file epoch seconds) with a date colorbar,
    so the week's time window is visible on the graph; the axes name the
    measurement windows (stim *stim_win*, evoked *evoked_win*, ms). Returns
    ``{r, r2, p, n, slope, png}`` (``png`` None when <3 paired points).
    Regression via ``scipy.stats.linregress``, skipped when every stim magnitude
    is identical -- mirrors chronic_evoked._add_regression."""
    xs = np.asarray([float(x) for x in stim_mags], dtype=float)
    ys = np.asarray([float(y) for y in evoked_mags], dtype=float)
    ts = np.asarray(times, dtype=float) if times is not None else None
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ts is not None:
        ok &= np.isfinite(ts)
    xs, ys = xs[ok], ys[ok]
    ts = ts[ok] if ts is not None else None
    out = {"r": float("nan"), "r2": float("nan"), "p": float("nan"),
           "n": int(xs.size), "slope": float("nan"), "png": None}
    if xs.size < 3:
        return out
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if ts is not None and np.ptp(ts) > 0:
        sc = ax.scatter(xs, ys, s=22, c=ts, cmap="viridis", alpha=0.8,
                        linewidths=0)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("recording time")
        from datetime import datetime
        tk = np.linspace(ts.min(), ts.max(), 4)
        cb.set_ticks(tk)
        cb.set_ticklabels([datetime.fromtimestamp(t).strftime("%m-%d %H:%M")
                           for t in tk])
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
                  engine: str = "auto") -> str | None:
    """Access-resistance (Rₐ) drift trend for one channel -> PNG: Rₐ line +
    rolling-median baseline band + a red-ringed latest point when drifting.

    Tries the Overview's Plotly look via kaleido first (``engine`` 'auto'/'plotly')
    for dashboard fidelity, then falls back to a portable matplotlib render so the
    figure is ALWAYS produced even where kaleido/Chrome is unavailable. Returns
    the path, or None only when there's nothing to plot. *rows* are oldest->newest
    ``{chunk_datetime, access_r_kohm}`` from ``impedance_series_by_channel``."""
    vals = [r.get("access_r_kohm") for r in rows if r.get("access_r_kohm")
            is not None]
    if len(vals) < 2:
        return None
    dts = [_parse_chunk_dt(r.get("chunk_datetime")) for r in rows]
    x = dts if all(d is not None for d in dts) else list(range(len(rows)))
    y = [r.get("access_r_kohm") for r in rows]
    stat = _drift_stat(vals, window, min_history)
    drifting = bool(stat and abs(stat["drift_pct"]) >= pct)
    sub = f"  ·  Δ {stat['drift_pct']:+.0f}% vs baseline" if stat else ""
    if engine in ("auto", "plotly"):
        p = _impedance_plotly(x, y, stat, drifting, pct, title + sub, out_png)
        if p is not None:
            return p
        if engine == "plotly":
            return None
    return _impedance_mpl(x, y, stat, drifting, pct, title + sub, out_png)


def _impedance_plotly(x, y, stat, drifting, pct, title, out_png):
    """Overview-fidelity Plotly render -> PNG via kaleido; None if unavailable."""
    try:
        import plotly.graph_objects as go
    except Exception:                                     # noqa: BLE001
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Rₐ",
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
    fig.update_yaxes(title_text="Rₐ (kΩ)")
    fig.update_xaxes(title_text="recording")
    try:
        fig.write_image(out_png, scale=2)                 # needs kaleido + Chrome
    except Exception as e:                                # noqa: BLE001
        logger.info("kaleido render unavailable, using matplotlib fallback: %s", e)
        return None
    return out_png


def _impedance_mpl(x, y, stat, drifting, pct, title, out_png):
    """Portable matplotlib (Agg) render of the same Rₐ trend -- always works."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(x, y, color=_ACCENT, lw=1.8, marker="o", markersize=4)
    if stat is not None:
        base, band = stat["baseline"], stat["baseline"] * pct / 100.0
        ax.axhspan(base - band, base + band, color=_ACCENT, alpha=0.10)
        ax.axhline(base, color="#888", ls="--", lw=1.0)
        if drifting:
            ax.scatter([x[-1]], [y[-1]], s=120, facecolors="none",
                       edgecolors=_FIT, linewidths=2.0, zorder=5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Rₐ (kΩ)")
    ax.set_xlabel("recording")
    ax.grid(True, alpha=0.2)
    if x and not isinstance(x[0], int):
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
