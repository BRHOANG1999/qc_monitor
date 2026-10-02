"""Continuous window-dominant state timeline at 24 h / 2 day / 1 week scales.

Three stacked colour strips (one per time-scale), each a 10-min dominant-state
series with seizure onsets marked and labelled. Reproduces the lost Fig 07
(``07_state_timeline_24h_2d_1wk``) from the committed pipeline.
"""

from __future__ import annotations

import datetime as _dt
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates           # noqa: E402
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.colors import to_rgb        # noqa: E402
from matplotlib.patches import Patch        # noqa: E402
import numpy as np                          # noqa: E402

from . import config as C, trajectory as T, figures as _G  # noqa: E402
_savefig = _G._savefig

_WINDOWS = [("1 week", 7 * 86400.0), ("2 days", 2 * 86400.0),
            ("24 hours", 86400.0)]


def _rgb_row(dom: np.ndarray) -> np.ndarray:
    """[1 x n x 3] RGB image from a dominant-state row (-1 -> no-data colour)."""
    rgb = np.zeros((1, dom.size, 3))
    for i, s in enumerate(dom):
        rgb[0, i] = to_rgb(C.NODATA_COLOR if s < 0 else C.STATE_COLORS[int(s)])
    return rgb


def _strip(ax, tl, lo, hi, onsets, label):
    """Draw one dominant-state strip for [lo, hi] (epoch sec)."""
    sel = (tl["centers"] >= lo) & (tl["centers"] <= hi)
    dom = tl["dominant"][sel]
    if dom.size == 0:
        dom = np.array([-1])
        lo_e, hi_e = lo, hi
    else:
        c = tl["centers"][sel]
        lo_e, hi_e = c[0], c[-1]
    ax.imshow(_rgb_row(dom), aspect="auto", origin="lower",
              extent=[mdates.date2num(_dt.datetime.fromtimestamp(lo_e)),
                      mdates.date2num(_dt.datetime.fromtimestamp(hi_e)), 0, 1],
              interpolation="nearest", zorder=0)
    nsz = 0
    for o in np.sort(onsets):
        if lo <= o <= hi:
            x = mdates.date2num(_dt.datetime.fromtimestamp(o))
            ax.axvline(x, color=C.SEIZURE_COLOR, lw=1.3, zorder=3)
            ax.text(x, 1.02, _dt.datetime.fromtimestamp(o).strftime("%m-%d %H:%M"),
                    rotation=90, va="bottom", ha="center", fontsize=6,
                    color=C.TEXT, transform=ax.get_xaxis_transform())
            nsz += 1
    ax.set_xlim(mdates.date2num(_dt.datetime.fromtimestamp(lo)),
                mdates.date2num(_dt.datetime.fromtimestamp(hi)))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    span_h = (hi - lo) / 3600.0
    loc = mdates.DayLocator() if span_h > 60 else mdates.HourLocator(
        interval=3 if span_h <= 30 else 6)
    fmt = mdates.DateFormatter("%m-%d" if span_h > 60 else "%m-%d %H:%M")
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)
    ax.tick_params(colors=C.MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(C.MUTED)
    ax.set_ylabel(f"{label}\n{nsz} sz", color=C.TEXT, fontsize=9, rotation=0,
                  ha="right", va="center")


def render_timeline(df, onsets, out_png: str, *, bin_sec: float = 600.0,
                    anchor: float | None = None) -> str:
    """Three stacked dominant-state strips (1 wk / 2 d / 24 h) ending at *anchor*
    (default: last epoch). Seizure onsets marked + labelled."""
    tl = T.dominant_state_timeline(df, bin_sec=bin_sec)
    end = anchor if anchor is not None else float(tl["centers"][-1])
    fig, axes = plt.subplots(len(_WINDOWS), 1, figsize=(15, 6.5), facecolor=C.BG)
    for ax, (label, dur) in zip(axes, _WINDOWS):
        ax.set_facecolor(C.PANEL)
        _strip(ax, tl, end - dur, end, np.asarray(onsets, float), label)
    handles = [Patch(color=C.STATE_COLORS[s], label=f"state {s}")
               for s in range(C.K_STATES)]
    handles.append(Patch(color=C.NODATA_COLOR, label="no data"))
    handles.append(plt.Line2D([0], [0], color=C.SEIZURE_COLOR, lw=1.3,
                              label="seizure onset"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                     fontsize=8, framealpha=0.0, bbox_to_anchor=(0.5, -0.02))
    for t in leg.get_texts():
        t.set_color(C.TEXT)
    blab = f"{int(bin_sec//60)}-min" if bin_sec >= 60 else f"{int(bin_sec)}-s"
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · continuous window-dominant state "
                 f"timeline ({blab} dominant state)",
                 color=C.TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png
