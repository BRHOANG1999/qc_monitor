"""Animated per-seizure distribution of one metric approaching a seizure.

For a chosen metric, walk a 30-min window (``SLIDING_WIDTH_SEC``) from a lookback
(default 6 h) down to the window right up to onset ([0, 30 min]), sliding by
``STEP_SEC`` (default 15 min). Each frame is the distribution of that metric over
the stimuli whose time-to-onset falls in the window -- histogram + smoothed
density + the individual stim values (rug), on FIXED axes so frames are
comparable, with the far-from-onset baseline drawn faintly behind every frame.
Frames are ordered far -> near, so the animation walks up to the seizure; the
frames are encoded as a looping GIF.

Pure + Dash-free (matplotlib Agg via ``Figure``/``FigureCanvasAgg``, so it is
thread-safe off the request thread) so the lens can call it from a background
worker and it is unit-testable. Operates on ``matrix.build_matrix``'s
``event x metric`` frame (meta columns ``time_to_onset_sec``, ``seizure_idx``,
``phase``); ``sid='group'`` pools every seizure's pre-onset stimuli by tto.
"""

from __future__ import annotations

import io
import re
import zipfile

import numpy as np

from src.periictal import config as _cfg

# Frame-march defaults (seconds). Width matches the preictal window; a half-width
# slide keeps the animation smooth without pooling unrelated windows.
WIDTH_SEC = _cfg.SLIDING_WIDTH_SEC          # 1800 (30 min)
STEP_SEC = WIDTH_SEC / 2.0                  # 900 (15 min)
LOOKBACK_SEC = _cfg.SLIDING_BAND_HI_SEC     # 21600 (6 h)

_MAX_FRAMES = 128                           # NASA Rule 2: bounded frame count
_MAX_SEIZURES = 500
_BINS = 26
_BASELINE_FRAC = 0.25                       # farthest quarter of frames = baseline

# Dark-theme palette (matches the dashboard: surfaces ~#1e1e2f, text #f0f0f5).
_BG = "#1e1e2f"
_PANEL = "#26263a"
_TEXT = "#f0f0f5"
_MUTED = "#9a9ab0"
_ACCENT = "#5e7ce2"
_BASELINE = "#6b6b80"
_ONSET = "#e26e6e"


def _pre_rows(full, feature: str):
    """(tto, values) for the finite pre-onset stimuli of *feature*, as arrays."""
    assert feature in full.columns, f"metric {feature!r} not in matrix"
    assert "time_to_onset_sec" in full.columns and "phase" in full.columns, \
        "matrix missing meta columns"
    pre = full[full["phase"].to_numpy() == "pre"]
    tto = pre["time_to_onset_sec"].to_numpy(dtype=float)
    val = pre[feature].to_numpy(dtype=float)
    ok = np.isfinite(tto) & np.isfinite(val) & (tto > 0)
    return tto[ok], val[ok], pre["seizure_idx"].to_numpy()[ok]


def window_offsets(*, width=WIDTH_SEC, step=STEP_SEC,
                   lookback=LOOKBACK_SEC) -> np.ndarray:
    """Window START offsets (sec before onset) from far -> near: the farthest
    window is [lookback-width, lookback]; the nearest is [0, width] (right up to
    onset). Descending, so frame order walks toward the seizure."""
    assert width > 0 and step > 0 and lookback >= width, "bad march params"
    o_max = lookback - width
    n = int(o_max // step) + 1
    n = max(1, min(n, _MAX_FRAMES))
    offs = o_max - np.arange(n, dtype=float) * step     # far -> near
    return np.clip(offs, 0.0, None)


def window_frames(full, sid, feature: str, *, width=WIDTH_SEC, step=STEP_SEC,
                  lookback=LOOKBACK_SEC) -> list[dict]:
    """Ordered frames (far -> near onset) for *sid* (an int seizure_idx or
    ``'group'`` to pool all seizures). Each frame: ``{lo, hi, center, label,
    values}`` where ``values`` is the metric over that window's stimuli."""
    tto, val, sid_arr = _pre_rows(full, feature)
    if sid != "group":
        m = sid_arr == sid
        tto, val = tto[m], val[m]
    frames: list[dict] = []
    for i, o in enumerate(window_offsets(width=width, step=step,
                                         lookback=lookback)):
        assert i < _MAX_FRAMES, "frame runaway"
        sel = (tto >= o) & (tto <= o + width)
        frames.append({"lo": float(o), "hi": float(o + width),
                       "center": float(o + width / 2.0),
                       "label": _lead_label(o, o + width),
                       "values": val[sel]})
    return frames


def _lead_label(lo: float, hi: float) -> str:
    """'−2.5 to −2.0 h before onset', or '0 to −0.5 h (onset)' for the last one."""
    a, b = hi / 3600.0, lo / 3600.0            # hi is farther back than lo
    if lo <= 1e-6:
        return f"−0.5–0 h  •  right up to onset"
    return f"−{a:.1f} to −{b:.1f} h before onset"


# --------------------------------------------------------------------- #
#  Rendering (matplotlib Agg -> PIL frames -> looping GIF)
# --------------------------------------------------------------------- #
def _kde_curve(values, x_grid):
    """Smoothed density on *x_grid*, or None when too few / degenerate points."""
    v = np.asarray(values, dtype=float)
    if v.size < 8 or np.ptp(v) <= 0:
        return None
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(v)(x_grid)
    except Exception:  # noqa: BLE001 -- singular covariance etc.: skip the curve
        return None


def _prepare(frames: list[dict]) -> dict:
    """Fixed axes + per-frame draw data (histogram density, KDE curve, mean, n)
    and the far-baseline density, so every frame shares one x/y scale."""
    pooled = np.concatenate([f["values"] for f in frames if f["values"].size]) \
        if any(f["values"].size for f in frames) else np.array([])
    assert pooled.size >= 1, "no stimuli in any window for this metric/seizure"
    lo, hi = np.percentile(pooled, [1, 99])
    if hi <= lo:
        lo, hi = float(pooled.min()) - 1.0, float(pooled.max()) + 1.0
    pad = 0.05 * (hi - lo)
    x_lo, x_hi = lo - pad, hi + pad
    bins = np.linspace(x_lo, x_hi, _BINS + 1)
    x_grid = np.linspace(x_lo, x_hi, 200)
    n_base = max(1, int(round(len(frames) * _BASELINE_FRAC)))
    base_vals = np.concatenate([f["values"] for f in frames[:n_base]
                                if f["values"].size] or [np.array([])])
    base_kde = _kde_curve(base_vals, x_grid)
    y_max = float(base_kde.max()) if base_kde is not None else 0.0
    per_frame = []
    for f in frames:
        v = f["values"]
        dens = (np.histogram(v, bins=bins, density=True)[0]
                if v.size else np.zeros(_BINS))
        kde = _kde_curve(v, x_grid)
        y_max = max(y_max, float(dens.max()) if dens.size else 0.0,
                    float(kde.max()) if kde is not None else 0.0)
        per_frame.append({"dens": dens, "kde": kde,
                          "mean": float(np.mean(v)) if v.size else np.nan,
                          "n": int(v.size)})
    return {"x_lo": x_lo, "x_hi": x_hi, "bins": bins, "x_grid": x_grid,
            "y_max": (y_max or 1.0) * 1.18, "base_kde": base_kde,
            "base_mean": float(np.mean(base_vals)) if base_vals.size else np.nan,
            "per_frame": per_frame}


def _draw_frame(frame, fd, ax_data, feature, title, k, total):
    """Render ONE frame to a PIL RGBA image on the shared fixed axes."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image
    fig = Figure(figsize=(6.6, 4.1), dpi=100, facecolor=_BG)
    ax = fig.subplots()
    ax.set_facecolor(_PANEL)
    centres = 0.5 * (ax_data["bins"][:-1] + ax_data["bins"][1:])
    width = ax_data["bins"][1] - ax_data["bins"][0]
    ax.bar(centres, fd["dens"], width=width * 0.95, color=_ACCENT, alpha=0.55,
           edgecolor="none", zorder=2)
    if ax_data["base_kde"] is not None:                 # faint far-baseline
        ax.fill_between(ax_data["x_grid"], ax_data["base_kde"], color=_BASELINE,
                        alpha=0.22, zorder=1)
        ax.plot(ax_data["x_grid"], ax_data["base_kde"], color=_BASELINE,
                lw=1.0, alpha=0.6, zorder=1)
    if fd["kde"] is not None:
        ax.plot(ax_data["x_grid"], fd["kde"], color=_ACCENT, lw=2.0, zorder=3)
    if fd["n"]:                                         # rug of the actual stims
        rug_y = -0.045 * ax_data["y_max"]
        ax.plot(np.clip(frame["values"], ax_data["x_lo"], ax_data["x_hi"]),
                np.full(fd["n"], rug_y), "|", color=_TEXT, alpha=0.5, ms=7,
                zorder=4)
    if np.isfinite(fd["mean"]):
        ax.axvline(fd["mean"], color=_ONSET, lw=1.6, zorder=5)
    if np.isfinite(ax_data["base_mean"]):
        ax.axvline(ax_data["base_mean"], color=_BASELINE, lw=1.0, ls="--",
                   zorder=2)
    _style_axes(ax, ax_data, feature, title, frame["label"], fd["n"], k, total)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = Image.frombuffer("RGBA", canvas.get_width_height(),
                           bytes(canvas.buffer_rgba()), "raw", "RGBA", 0, 1)
    return img.convert("RGB")


def _style_axes(ax, ax_data, feature, title, lead_label, n, k, total):
    """Dark-theme axis cosmetics + the frame's time/onset annotations."""
    ax.set_xlim(ax_data["x_lo"], ax_data["x_hi"])
    ax.set_ylim(-0.07 * ax_data["y_max"], ax_data["y_max"])
    ax.set_xlabel(feature, color=_TEXT, fontsize=10)
    ax.set_ylabel("density", color=_TEXT, fontsize=10)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#3a3a52")
    ax.set_title(title, color=_TEXT, fontsize=11, pad=12, loc="left")
    ax.text(0.99, 1.02, f"{lead_label}    (n={n})", transform=ax.transAxes,
            ha="right", va="bottom", color=_ACCENT, fontsize=9)
    ax.text(0.99, -0.16, f"frame {k + 1}/{total}", transform=ax.transAxes,
            ha="right", va="top", color=_MUTED, fontsize=7)
    ax.figure.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.14)


def render_seizure_gif(full, sid, feature: str, *, label: str = "",
                       width=WIDTH_SEC, step=STEP_SEC, lookback=LOOKBACK_SEC,
                       fps: float = 3.0) -> bytes:
    """Animated GIF (bytes) of *feature*'s distribution marching to onset for
    *sid* (int seizure_idx or ``'group'``). Raises ValueError if there is no
    usable data."""
    assert feature, "feature required"
    frames = window_frames(full, sid, feature, width=width, step=step,
                           lookback=lookback)
    if not any(f["values"].size for f in frames):
        raise ValueError("no pre-onset stimuli for this metric/seizure")
    ax_data = _prepare(frames)
    who = label or ("all seizures" if sid == "group" else f"seizure {sid}")
    title = f"{feature} · {who}"
    imgs = [_draw_frame(f, fd, ax_data, feature, title, k, len(frames))
            for k, (f, fd) in enumerate(zip(frames, ax_data["per_frame"]))]
    return _encode_gif(imgs, fps)


def _encode_gif(imgs, fps: float) -> bytes:
    """Encode RGB frames into a looping, palette-quantised GIF."""
    assert imgs, "no frames to encode"
    dur_ms = int(max(1.0, 1000.0 / max(0.5, float(fps))))
    pal = [im.quantize(colors=128, method=2) for im in imgs]  # 2 = MEDIANCUT
    buf = io.BytesIO()
    pal[0].save(buf, format="GIF", save_all=True, append_images=pal[1:],
                duration=dur_ms, loop=0, disposal=2, optimize=True)
    return buf.getvalue()


def safe_name(text: str) -> str:
    """Filesystem-safe token from a seizure label / feature name."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_") or "x"


def render_all_seizures_zip(full, feature: str, sids, labels: dict, *,
                            width=WIDTH_SEC, step=STEP_SEC,
                            lookback=LOOKBACK_SEC, fps: float = 3.0,
                            progress=None) -> bytes:
    """One GIF per seizure in *sids*, bundled into an in-memory .zip (bytes).
    *labels* maps seizure_idx -> display label; *progress(i, n, label)* is called
    per seizure. Seizures with no usable stimuli are skipped (noted in the zip)."""
    assert feature, "feature required"
    sids = list(sids)[:_MAX_SEIZURES]
    buf = io.BytesIO()
    skipped: list[str] = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, sid in enumerate(sids):
            assert i < _MAX_SEIZURES, "seizure runaway"
            lab = labels.get(sid, f"seizure {sid}")
            if progress:
                progress(i, len(sids), lab)
            try:
                gif = render_seizure_gif(full, sid, feature, label=lab,
                                         width=width, step=step,
                                         lookback=lookback, fps=fps)
            except ValueError:
                skipped.append(str(lab))
                continue
            zf.writestr(f"{safe_name(lab)}__{safe_name(feature)}.gif", gif)
        if skipped:
            zf.writestr("_skipped.txt",
                        "No usable pre-onset stimuli for:\n" + "\n".join(skipped))
    return buf.getvalue()

