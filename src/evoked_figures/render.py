"""Headless matplotlib (Agg) renderers: a scatter-only and a percentile-shaded
PNG per (animal, metric), matching the chronic-evoked tab's look (10-90 light
band, 25-75 dark band, rolling-median line)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")               # headless; no display, safe in a worker

import matplotlib.dates as mdates    # noqa: E402
import matplotlib.pyplot as plt      # noqa: E402

# rgba mirroring chronic_evoked._add_ribbon (#5e7ce2 at 0.13 / 0.30).
_BAND_LIGHT = (0.369, 0.486, 0.886, 0.13)
_BAND_DARK = (0.369, 0.486, 0.886, 0.30)
_MEDIAN = "#5e7ce2"
_POINTS = "#8892b0"
_FIGSIZE = (9.0, 4.2)


def _time_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.figure.autofmt_xdate()
    ax.grid(True, alpha=0.2)


def render_scatter(dts, vals, title: str, ylabel: str, out_png: str) -> str:
    """Scatter-only figure: one point per evoked response over time."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.scatter(dts, vals, s=6, c=_POINTS, alpha=0.5, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    _time_axis(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def render_percentile(band_dts, p10, p25, p50, p75, p90, title: str,
                      ylabel: str, out_png: str,
                      pts_dts=None, pts_vals=None) -> str:
    """Percentile-shaded figure: 10-90 + 25-75 bands + rolling median, over a
    faint scatter for context."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if pts_dts is not None and len(pts_dts):
        ax.scatter(pts_dts, pts_vals, s=4, c=_POINTS, alpha=0.18, linewidths=0)
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
