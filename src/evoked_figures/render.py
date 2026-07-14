"""Headless matplotlib (Agg) renderers: a scatter-only and a percentile-shaded
PNG per (animal, metric), matching the chronic-evoked tab's look (10-90 light
band, 25-75 dark band, rolling-median line)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")               # headless; no display, safe in a worker

from datetime import timedelta       # noqa: E402

import matplotlib.dates as mdates    # noqa: E402
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

# rgba mirroring chronic_evoked._add_ribbon (#5e7ce2 at 0.13 / 0.30).
_BAND_LIGHT = (0.369, 0.486, 0.886, 0.13)
_BAND_DARK = (0.369, 0.486, 0.886, 0.30)
_MEDIAN = "#5e7ce2"
_POINTS = "#8892b0"
_FIGSIZE = (9.0, 4.2)

# A big animal is millions of epochs (BCH040 ~2.3M). Beyond ~150k points the
# scatter is a solid alpha smear -- more points cost minutes and change nothing
# visible. We thin the DRAWN cloud (evenly strided, so the shape is preserved)
# and say so on the figure. The percentile bands are computed from EVERY point
# upstream, so the statistics are exact regardless.
_MAX_SCATTER = 150_000
_MAX_SCATTER_UNDERLAY = 50_000
# Break the bands/median across gaps longer than this (recordings are gappy;
# a count-window rolling percentile would otherwise draw a line through weeks
# of silence). Override via evoked_figures.gap_break_hours.
_DEFAULT_GAP_SEC = 24 * 3600.0


def _thin(dts, vals, cap: int):
    """(dts, vals, n_total, thinned?) evenly strided down to at most *cap*."""
    n = len(vals)
    if n <= cap:
        return dts, vals, n, False
    step = (n // cap) + 1
    return dts[::step], vals[::step], n, True


def _break_gaps(dts, arrays, gap_sec: float):
    """Insert a NaN sample at every time gap longer than *gap_sec* so the bands
    and median LINE BREAK instead of bridging it.

    The rolling percentile uses a COUNT window, not a time window, so two
    consecutive band samples can be weeks apart. Drawn naively, matplotlib joins
    them with a confident straight line through a period where no recording
    exists -- which reads as data. Recordings are gappy by nature (BCH040 has a
    ~6-week hole), so this is not cosmetic: it's the difference between showing
    silence and inventing a trend.
    """
    n = len(dts)
    if n < 2 or gap_sec <= 0:
        return dts, arrays
    ts = np.fromiter((d.timestamp() for d in dts), dtype=float, count=n)
    gi = np.where(np.diff(ts) > gap_sec)[0]        # gap between i and i+1
    if gi.size == 0:
        return dts, arrays
    pos = gi + 1
    new_dts = list(dts)
    for p in sorted(pos.tolist(), reverse=True):   # reverse keeps idx valid
        new_dts.insert(p, dts[p - 1] + timedelta(seconds=1))
    new_arrays = [np.insert(np.asarray(a, dtype=float), pos, np.nan)
                  for a in arrays]
    return new_dts, new_arrays


def _time_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.figure.autofmt_xdate()
    ax.grid(True, alpha=0.2)


def render_scatter(dts, vals, title: str, ylabel: str, out_png: str) -> str:
    """Scatter-only figure: one point per evoked response over time."""
    d, v, n_tot, thinned = _thin(dts, vals, _MAX_SCATTER)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.scatter(d, v, s=6, c=_POINTS, alpha=0.5, linewidths=0)
    if thinned:                                   # never hide the thinning
        title = f"{title}  [showing {len(v):,} of {n_tot:,} points]"
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    _time_axis(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def render_percentile(band_dts, p10, p25, p50, p75, p90, title: str,
                      ylabel: str, out_png: str,
                      pts_dts=None, pts_vals=None,
                      gap_sec: float = _DEFAULT_GAP_SEC) -> str:
    """Percentile-shaded figure: 10-90 + 25-75 bands + rolling median, over a
    faint scatter for context. Bands/median BREAK across time gaps longer than
    *gap_sec* so a stretch with no recordings reads as a gap, not a trend."""
    band_dts, (p10, p25, p50, p75, p90) = _break_gaps(
        band_dts, (p10, p25, p50, p75, p90), gap_sec)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if pts_dts is not None and len(pts_dts):
        pd_, pv_, _n, _t = _thin(pts_dts, pts_vals, _MAX_SCATTER_UNDERLAY)
        ax.scatter(pd_, pv_, s=4, c=_POINTS, alpha=0.18, linewidths=0)
    ax.fill_between(band_dts, p10, p90, color=_BAND_LIGHT, label="10–90%")
    ax.fill_between(band_dts, p25, p75, color=_BAND_DARK, label="25–75%")
    ax.plot(band_dts, p50, color=_MEDIAN, lw=2.0, label="rolling median")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8, framealpha=0.6)
    _time_axis(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
