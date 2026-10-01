"""Pre-ictal state-trajectory animations in a shared PCA space.

As a time cursor travels toward seizure onset, each evoked-response epoch lights
up in PCA space in time order and earlier ones dim (but never fully) -- a comet
whose trail is the pre-ictal trajectory through state space. The PCA embedding is
fit once over ALL events (states.fit_states); each seizure's trajectory is drawn
within that fixed space. Two builds are produced by the caller: lead seizures
only, and all seizures (including clustered followers).

Writes a GIF always (Pillow) and an MP4 when ffmpeg is available.
"""

from __future__ import annotations

import datetime as _dt
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manim        # noqa: E402
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
from matplotlib.colors import to_rgba       # noqa: E402

from . import config as C                   # noqa: E402


def _seizure_track(df, onset, pre_min, post_min):
    """Epochs of one seizure, sorted by time: (pc1, pc2, state, minutes_to_onset).
    minutes_to_onset is negative before onset, 0 at onset."""
    t = df["t_epoch"].to_numpy(float)
    rel = (t - onset) / 60.0
    m = (rel >= -pre_min) & (rel <= post_min) & np.isfinite(df["pc1"].to_numpy())
    idx = np.where(m)[0]
    idx = idx[np.argsort(rel[idx])]
    return (df["pc1"].to_numpy()[idx], df["pc2"].to_numpy()[idx],
            df["state"].to_numpy()[idx], rel[idx])


def _comet_rgba(states, rel, cursor, trail_min, floor=0.18):
    """Per-point RGBA: points at/under *cursor* visible, newer brighter; older
    fade to *floor* over *trail_min*. Returns (rgba array, size array, visible)."""
    vis = rel <= cursor
    age = cursor - rel
    w = np.clip(1.0 - age / trail_min, floor, 1.0)
    rgba = np.zeros((len(states), 4))
    for i, s in enumerate(states):
        base = C.STATE_COLORS[int(s)] if s >= 0 else C.MUTED
        rgba[i] = to_rgba(base, alpha=w[i] if vis[i] else 0.0)
    size = np.where(vis, 18 + 60 * (w ** 2), 0.0)
    return rgba, size, vis


def animate_set(model, onsets, labels, out_basename, *, pre_min=120.0,
                post_min=10.0, step_min=2.0, trail_min=40.0, fps=12,
                progress=None) -> dict:
    """One animation stepping through each seizure in *onsets* sequentially, each
    in the shared PCA space. Saves <out_basename>.gif (+ .mp4 if ffmpeg)."""
    df = model.df
    allx, ally = df["pc1"].to_numpy(), df["pc2"].to_numpy()
    fin = np.isfinite(allx) & np.isfinite(ally)
    tracks = [_seizure_track(df, o, pre_min, post_min) for o in onsets]
    steps = np.arange(-pre_min, post_min + step_min, step_min)
    # frame list: (seizure_index, cursor_minutes)
    frames = [(si, c) for si in range(len(onsets)) for c in steps]

    fig = plt.figure(figsize=(9.5, 7.2), facecolor=C.BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[6, 1], hspace=0.28)
    axp = fig.add_subplot(gs[0, 0]); axt = fig.add_subplot(gs[1, 0])
    for ax in (axp, axt):
        ax.set_facecolor(C.PANEL)
        for sp in ax.spines.values():
            sp.set_color(C.MUTED)
        ax.tick_params(colors=C.MUTED, labelsize=8)
    axp.scatter(allx[fin], ally[fin], s=4, c=C.NODATA_COLOR, alpha=0.25,
                linewidths=0, zorder=0)                      # shared backdrop
    axp.set_xlabel("PC1", color=C.TEXT); axp.set_ylabel("PC2", color=C.TEXT)
    xpad = np.nanpercentile(allx[fin], [1, 99]); ypad = np.nanpercentile(ally[fin], [1, 99])
    axp.set_xlim(xpad[0] - 2, xpad[1] + 2); axp.set_ylim(ypad[0] - 2, ypad[1] + 2)
    comet = axp.scatter([], [], zorder=5)
    axt.set_xlim(-pre_min, post_min); axt.set_ylim(0, 1); axt.set_yticks([])
    axt.axvline(0, color=C.SEIZURE_COLOR, lw=1.5)
    axt.set_xlabel("minutes to seizure onset", color=C.TEXT)
    cursor = axt.axvline(-pre_min, color=C.ACCENT, lw=2.0)
    title = axp.set_title("", color=C.TEXT, fontsize=11, loc="left")

    def update(fi):
        si, c = frames[fi]
        x, y, st, rel = tracks[si]
        if len(x):
            rgba, size, _ = _comet_rgba(st, rel, c, trail_min)
            comet.set_offsets(np.column_stack([x, y]))
            comet.set_facecolors(rgba); comet.set_edgecolors(rgba); comet.set_sizes(size)
        else:
            comet.set_offsets(np.empty((0, 2)))
        cursor.set_xdata([c, c])
        title.set_text(f"{C.ANIMAL} · {C.CHANNEL} · seizure {si+1}/{len(onsets)} "
                       f"· {labels[si]} · t={c:+.0f} min")
        if progress and fi % 40 == 0:
            progress(fi, len(frames))
        return comet, cursor, title

    ani = manim.FuncAnimation(fig, update, frames=len(frames), blit=False,
                              interval=1000 / fps)
    os.makedirs(os.path.dirname(out_basename), exist_ok=True)
    gif = out_basename + ".gif"
    ani.save(gif, writer=manim.PillowWriter(fps=fps))
    mp4 = None
    if manim.writers.is_available("ffmpeg"):
        mp4 = out_basename + ".mp4"
        ani.save(mp4, writer=manim.FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    return {"gif": gif, "mp4": mp4, "n_frames": len(frames),
            "n_seizures": len(onsets)}


def seizure_labels(onsets) -> list:
    return [_dt.datetime.fromtimestamp(float(o)).strftime("%m-%d %H:%M")
            for o in onsets]
