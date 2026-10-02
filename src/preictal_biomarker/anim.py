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
    """Epochs of one seizure, sorted by time: (pc1, pc2, state, minutes_to_onset,
    t_epoch). minutes_to_onset is negative before onset, 0 at onset."""
    t = df["t_epoch"].to_numpy(float)
    rel = (t - onset) / 60.0
    m = (rel >= -pre_min) & (rel <= post_min) & np.isfinite(df["pc1"].to_numpy())
    idx = np.where(m)[0]
    idx = idx[np.argsort(rel[idx])]
    return (df["pc1"].to_numpy()[idx], df["pc2"].to_numpy()[idx],
            df["state"].to_numpy()[idx], rel[idx], t[idx])


def _wave_idx_for(track_t, wave_t):
    """Nearest waveform-epoch index for each track epoch time (NaN if none ≤5s)."""
    if wave_t is None or wave_t.size == 0:
        return np.full(track_t.size, -1, int)
    order = np.argsort(wave_t); wt = wave_t[order]
    pos = np.clip(np.searchsorted(wt, track_t), 1, wt.size - 1)
    near = np.where(np.abs(wt[pos] - track_t) < np.abs(wt[pos - 1] - track_t),
                    pos, pos - 1)
    out = order[near]
    out[np.abs(wave_t[out] - track_t) > 5.0] = -1
    return out


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


def _export_frames(fig, update, n_frames, frames_dir, fps) -> str:
    """Save every frame as a PNG + a step-through HTML viewer (slider / arrow keys)."""
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        if f.startswith("frame_") and f.endswith(".png"):
            os.remove(os.path.join(frames_dir, f))
    for fi in range(n_frames):
        update(fi)
        fig.savefig(os.path.join(frames_dir, f"frame_{fi:05d}.png"), dpi=110,
                    facecolor=C.BG)
    html = _VIEWER_HTML.replace("__N__", str(n_frames)).replace("__FPS__", str(fps))
    with open(os.path.join(frames_dir, "viewer.html"), "w", encoding="utf-8") as f:
        f.write(html)
    return frames_dir


_VIEWER_HTML = """<!doctype html><html><head><meta charset=utf-8><title>frames</title>
<style>body{background:#15151f;color:#f0f0f5;font-family:sans-serif;text-align:center;margin:0}
img{max-width:100%;height:auto}.row{margin:8px}button{background:#1e1e2f;color:#f0f0f5;border:1px solid #9aa0b4;padding:4px 10px;cursor:pointer}</style></head>
<body><div class=row><button id=prev>&#9664; prev</button> <button id=play>play</button>
<button id=next>next &#9654;</button> <input id=sl type=range min=0 max="__N__" value=0 style="width:55%">
<span id=lbl></span> &nbsp;(&#8592;/&#8594; to step)</div>
<img id=im><script>
const N=__N__,FPS=__FPS__;let i=0,t=null;
const im=document.getElementById('im'),sl=document.getElementById('sl'),lbl=document.getElementById('lbl');
sl.max=N-1;function show(k){i=(k+N)%N;im.src='frame_'+String(i).padStart(5,'0')+'.png';sl.value=i;lbl.textContent=i+' / '+(N-1);}
prev.onclick=()=>show(i-1);next.onclick=()=>show(i+1);sl.oninput=()=>show(+sl.value);
document.onkeydown=e=>{if(e.key==='ArrowRight')show(i+1);if(e.key==='ArrowLeft')show(i-1);};
play.onclick=function(){if(t){clearInterval(t);t=null;this.textContent='play';}else{t=setInterval(()=>show(i+1),1000/FPS);this.textContent='pause';}};
show(0);</script></body></html>"""


def animate_set(model, onsets, labels, out_basename, *, pre_min=120.0,
                post_min=10.0, step_min=2.0, trail_min=40.0, fps=12,
                progress=None, waveforms=None, wave_t=None, wave_time_ms=None,
                lfp_env=None, frames_dir=None) -> dict:
    """Step each seizure through the shared PCA space. Optional panels: per-epoch
    stim-blanked evoked LFP (waveforms), and the CONTINUOUS LFP envelope over the
    full −pre_min..post_min window with the seizure at 0 (lfp_env, one dict per
    onset). Both are time-locked to the cursor. frames_dir also exports per-frame
    PNGs + a step-through HTML viewer. Saves <out_basename>.mp4 (+ .gif)."""
    df = model.df
    allx, ally = df["pc1"].to_numpy(), df["pc2"].to_numpy()
    fin = np.isfinite(allx) & np.isfinite(ally)
    tracks = [_seizure_track(df, o, pre_min, post_min) for o in onsets]
    has_lfp = waveforms is not None and wave_time_ms is not None
    has_env = lfp_env is not None
    widx = [_wave_idx_for(tr[4], wave_t) for tr in tracks] if has_lfp else None
    steps = np.arange(-pre_min, post_min + step_min, step_min)
    frames = [(si, c) for si in range(len(onsets)) for c in steps]
    n_trail = 6

    heights = [6.0]; names = ["p"]
    if has_lfp:
        heights.append(2.2); names.append("l")
    heights.append(2.2 if has_env else 0.7); names.append("c" if has_env else "t")
    fig = plt.figure(figsize=(9.5, 1.15 * sum(heights) + 1), facecolor=C.BG)
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.45)
    ax = {names[i]: fig.add_subplot(gs[i, 0]) for i in range(len(heights))}
    axp = ax["p"]; axl = ax.get("l"); axc = ax.get("c"); axt = ax.get("t")
    for a in ax.values():
        a.set_facecolor(C.PANEL)
        for sp in a.spines.values():
            sp.set_color(C.MUTED)
        a.tick_params(colors=C.MUTED, labelsize=8)
    axp.scatter(allx[fin], ally[fin], s=4, c=C.NODATA_COLOR, alpha=0.25,
                linewidths=0, zorder=0)
    axp.set_xlabel("PC1", color=C.TEXT); axp.set_ylabel("PC2", color=C.TEXT)
    xp = np.nanpercentile(allx[fin], [1, 99]); yp = np.nanpercentile(ally[fin], [1, 99])
    axp.set_xlim(xp[0] - 2, xp[1] + 2); axp.set_ylim(yp[0] - 2, yp[1] + 2)
    comet = axp.scatter([], [], zorder=5)
    lfp_lines = []
    if has_lfp:
        yl = np.nanpercentile(waveforms, [0.3, 99.7])
        axl.set_xlim(wave_time_ms[0], wave_time_ms[-1]); axl.set_ylim(yl[0], yl[1])
        axl.axhline(0, color=C.MUTED, lw=0.5)
        axl.set_xlabel("ms since stim (evoked, stim-blanked)", color=C.TEXT)
        axl.set_ylabel("evoked", color=C.TEXT)
        lfp_lines = [axl.plot([], [], lw=0.8)[0] for _ in range(n_trail)]
    cur_ax = axc if has_env else axt
    cur_ax.set_xlim(-pre_min, post_min)
    cur_ax.axvline(0, color=C.SEIZURE_COLOR, lw=1.5)
    cur_ax.set_xlabel("minutes to seizure onset" + (" (continuous LFP)"
                      if has_env else ""), color=C.TEXT)
    if has_env:
        axc.set_ylabel("LFP", color=C.TEXT)
    else:
        axt.set_ylim(0, 1); axt.set_yticks([])
    cursor = cur_ax.axvline(-pre_min, color=C.ACCENT, lw=2.0)
    title = axp.set_title("", color=C.TEXT, fontsize=11, loc="left")
    env_state = {"si": -1, "fill": None}

    def update(fi):
        si, c = frames[fi]
        x, y, st, rel, _tep = tracks[si]
        if len(x):
            rgba, size, _ = _comet_rgba(st, rel, c, trail_min)
            comet.set_offsets(np.column_stack([x, y]))
            comet.set_facecolors(rgba); comet.set_edgecolors(rgba); comet.set_sizes(size)
        else:
            comet.set_offsets(np.empty((0, 2)))
        if has_lfp:
            recent = np.where(rel <= c)[0][-n_trail:][::-1]
            wi = widx[si]
            for k, ln in enumerate(lfp_lines):
                if k < len(recent) and wi[recent[k]] >= 0:
                    j = recent[k]; s = int(st[j])
                    ln.set_data(wave_time_ms, waveforms[wi[j]])
                    ln.set_color(C.STATE_COLORS[s] if s >= 0 else C.MUTED)
                    ln.set_alpha(1.0 if k == 0 else max(0.12, 1 - 0.22 * k))
                    ln.set_linewidth(2.4 if k == 0 else 0.8)
                else:
                    ln.set_data([], [])
        if has_env and si != env_state["si"]:
            if env_state["fill"] is not None:
                env_state["fill"].remove(); env_state["fill"] = None
            e = lfp_env[si]
            if e and e.get("env_t") is not None and len(e["env_t"]):
                env_state["fill"] = axc.fill_between(
                    e["env_t"], e["env_lo"], e["env_hi"], color="#8aa0c0",
                    alpha=0.8, lw=0)
                lo = np.nanpercentile(e["env_lo"], 0.5)
                hi = np.nanpercentile(e["env_hi"], 99.5)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    axc.set_ylim(lo, hi)
            env_state["si"] = si
        cursor.set_xdata([c, c])
        title.set_text(f"{C.ANIMAL} · {C.CHANNEL} · seizure {si+1}/{len(onsets)} "
                       f"· {labels[si]} · t={c:+.0f} min")
        if progress and fi % 40 == 0:
            progress(fi, len(frames))
        return comet, cursor, title

    os.makedirs(os.path.dirname(out_basename), exist_ok=True)
    out = {"n_frames": len(frames), "n_seizures": len(onsets)}
    if frames_dir:
        out["frames"] = _export_frames(fig, update, len(frames), frames_dir, fps)
    ani = manim.FuncAnimation(fig, update, frames=len(frames), blit=False,
                              interval=1000 / fps)
    if manim.writers.is_available("ffmpeg"):
        mp4 = out_basename + ".mp4"
        ani.save(mp4, writer=manim.FFMpegWriter(fps=fps, bitrate=2400))
        out["mp4"] = mp4
    if not frames_dir:                                     # gif only when not stepping
        gif = out_basename + ".gif"
        ani.save(gif, writer=manim.PillowWriter(fps=fps)); out["gif"] = gif
    plt.close(fig)
    return out


def seizure_labels(onsets) -> list:
    return [_dt.datetime.fromtimestamp(float(o)).strftime("%m-%d %H:%M")
            for o in onsets]
