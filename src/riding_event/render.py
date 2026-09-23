"""Dark-themed figures for riding-event analysis.

Palette matches the repo's dark notification figures (source of truth:
``notifications/evoked_digest`` + ``dashboard/design.COLOR_ACCENT``): surfaces
``#1e1e2f``/``#26263a``, text ``#f0f0f5``, accent ``#5e7ce2``. House style: the
title states the finding, axes carry domain-term labels with units, every panel
has a legend (with n=), raw examples sit under summaries, and dense overlays use
the memory-safe ``LineDensity`` rasterizer. ``savefig`` bakes the dark background.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from src.periictal.erpimage import decimate_rows  # noqa: E402
from src.notifications.trace_density import LineDensity, robust_ylim  # noqa: E402
from src.utils.decimate import envelope as _decim_envelope  # noqa: E402

logger = logging.getLogger("qc_monitor.riding_event.render")

_BG = "#1e1e2f"          # figure background
_PANEL = "#26263a"       # axes facecolor
_TEXT = "#f0f0f5"        # titles / labels
_MUTED = "#9a9ab0"       # ticks / spines
_SPINE = "#3a3a52"
_ACCENT = "#5e7ce2"      # clean / primary
_EVENT = "#ff7043"       # the riding event (salient)
_MAXROWS = 900           # NASA Rule 3: bound the rasterized image height


def _style(ax) -> None:
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_SPINE)
    ax.grid(True, alpha=0.15, color=_MUTED)


def _lab(ax, *, title=None, xlabel=None, ylabel=None) -> None:
    if title:
        ax.set_title(title, color=_TEXT, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=_TEXT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=_TEXT, fontsize=10)


def _legend(ax, **kw) -> None:
    leg = ax.legend(frameon=False, fontsize=8, **kw)
    if leg:
        for t in leg.get_texts():
            t.set_color(_TEXT)


def _finish(fig, out_png: str) -> str:
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor=_BG)
    plt.close(fig)
    return out_png


def _examples(idx_mask, n: int) -> np.ndarray:
    """Up to *n* evenly-spaced indices where *idx_mask* is True."""
    where = np.flatnonzero(idx_mask)
    if where.size == 0:
        return where
    take = np.linspace(0, where.size - 1, min(n, where.size)).astype(int)
    return where[take]


# ---------------------------------------------------------------- Prong A --- #

def fig_raw_overlay(res: dict, out_png: str, *, n_rows: int = 36) -> str:
    """RIDGELINE of the event-flagged epochs: each event's post-stim waveform,
    peak-normalized and stacked with a vertical offset (earliest at the bottom,
    time-coloured), with the median template on top. Stacking makes each
    individual event legible instead of a flat overlaid smear."""
    tm = np.asarray(res["time_ms"])
    lo, hi = res["flag_win_ms"][0], res["spec_win_ms"][1]
    m = (tm >= lo) & (tm <= hi)
    x = tm[m]
    ev = np.flatnonzero(res["event_mask"])
    if ev.size == 0:
        return _empty(out_png, f"{res['animal']} {res['channel']}: no event epochs")
    if ev.size > n_rows:
        ev = ev[np.linspace(0, ev.size - 1, n_rows).astype(int)]
    tr = res["traces"][np.ix_(ev, np.flatnonzero(m))]
    norm = np.max(np.abs(tr), axis=1, keepdims=True)
    tr = tr / np.where(norm > 0, norm, 1.0)          # peak-normalized shapes
    gap = 1.25
    cmap = plt.get_cmap("turbo")
    fig, ax = plt.subplots(figsize=(9.5, 8.4), facecolor=_BG)
    _style(ax)
    for i, row in enumerate(tr):
        off = i * gap
        c = cmap(0.05 + 0.9 * i / max(1, tr.shape[0] - 1))
        ax.fill_between(x, off, row + off, color=c, alpha=0.18, linewidth=0)
        ax.plot(x, row + off, color=c, lw=0.8)
    tnorm = res["template"][m]
    tnorm = tnorm / (np.max(np.abs(tnorm)) or 1.0)
    top = tr.shape[0] * gap
    ax.plot(x, tnorm + top, color=_TEXT, lw=2.0, label="median template")
    ax.set_yticks([0, top])
    ax.set_yticklabels(["earliest event", "latest / template"])
    _lab(ax, title=f"{res['animal']} {res['channel']} — event-epoch ridgeline "
                   f"({tr.shape[0]} of {res['n_event']} events, peak-normalized)",
         xlabel="time from stim (ms)", ylabel="event (time order →)")
    _legend(ax, loc="upper right")
    return _finish(fig, out_png)


def fig_residual_erpimage(res: dict, out_png: str) -> str:
    """Residual ERP-image: rows = epochs (time order), cols = post-stim ms,
    colour = residual amplitude. The riding events read as bright rows; a right-
    edge stripe marks the event-flagged epochs."""
    tm = np.asarray(res["time_ms"])
    lo, hi = res["flag_win_ms"][0], res["spec_win_ms"][1]
    cols = (tm >= lo) & (tm <= hi)
    z = res["resid"][:, cols]                    # [epochs x samp]
    row_ms = tm[cols]
    # Decimate SAMPLES (columns) for display; keep every epoch as a row (strided
    # only if there are more epochs than the image height).
    zt, col_ms = decimate_rows(z.T, row_ms, max_rows=600)   # -> [samp' x epochs]
    img = zt.T                                              # [epochs x samp']
    if img.shape[0] > _MAXROWS:
        step = int(np.ceil(img.shape[0] / _MAXROWS))
        img = img[::step]
        ev = res["event_mask"][::step]
    else:
        ev = res["event_mask"]
    vmax = float(np.nanpercentile(np.abs(img), 99)) or 1.0
    fig, ax = plt.subplots(figsize=(9.5, 6.0), facecolor=_BG)
    _style(ax)
    im = ax.imshow(img, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=(col_ms[0], col_ms[-1], 0, img.shape[0]))
    for i in np.flatnonzero(ev):                 # event-epoch stripe
        ax.plot([col_ms[-1]], [i + 0.5], marker="s", ms=2.0, color=_EVENT)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("residual (µV)", color=_TEXT, fontsize=9)
    cb.ax.tick_params(colors=_MUTED, labelsize=8)
    _lab(ax, title=f"{res['animal']} {res['channel']} — residual after template "
                   f"subtraction (orange = event epoch)",
         xlabel="time from stim (ms)", ylabel="epoch (time order →)")
    return _finish(fig, out_png)


def fig_density(res: dict, out_png: str) -> str:
    """Full-density residual overlay (every trace) — event vs clean, memory-safe
    via ``LineDensity``."""
    tm = np.asarray(res["time_ms"])
    lo, hi = res["flag_win_ms"][0], res["spec_win_ms"][1]
    cols = (tm >= lo) & (tm <= hi)
    x = tm[cols]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), facecolor=_BG, sharey=True)
    groups = [("event", res["event_mask"], _EVENT),
              ("clean", res["clean_mask"], _ACCENT)]
    lohi = _resid_extrema(res["resid"][:, cols])
    y0, y1 = robust_ylim(lohi)
    for ax, (name, mask, color) in zip(axes, groups):
        _style(ax)
        ld = LineDensity((float(x[0]), float(x[-1])), (y0, y1))
        for j in np.flatnonzero(mask):
            ld.add(x, res["resid"][j, cols])
        ax.imshow(ld.rgba(_hex_rgb(color)), extent=ld.extent(), origin="upper",
                  aspect="auto")
        ax.axhline(0, color=_MUTED, lw=0.5)
        _lab(ax, title=f"{name} residuals (n={int(mask.sum())})",
             xlabel="time from stim (ms)")
    axes[0].set_ylabel("residual (µV)", color=_TEXT, fontsize=10)
    fig.suptitle(f"{res['animal']} {res['channel']} — residual density "
                 f"(every trace)", color=_TEXT, fontsize=11, x=0.02, ha="left")
    return _finish(fig, out_png)


def _resid_extrema(z) -> np.ndarray:
    return np.column_stack([np.nanmin(z, axis=1), np.nanmax(z, axis=1)])


def fig_spectral(res: dict, out_png: str) -> str:
    """The core comparison: (top) epoch-averaged Welch PSDs of event/clean
    residuals and raw + the template; (bottom) event-over-clean excess in dB with
    the excess band shaded and the fundamental + harmonics annotated. Mains
    line-noise (50/60/120/180 Hz) is marked dotted so it isn't read as the
    event's rhythm."""
    f = np.asarray(res["freqs"])
    if f.size == 0:
        return _empty(out_png, f"{res['animal']} {res['channel']}: no spectra")
    fmax = min(3000.0, 0.5 * res["fs"])
    keep = (f > 0) & (f <= fmax)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 8.0), facecolor=_BG,
                                   sharex=True, height_ratios=[3, 2])
    _spectral_top(ax1, res, f, keep, fmax)
    _spectral_bottom(ax2, res, f, keep, fmax)
    return _finish(fig, out_png)


def _spectral_top(ax, res, f, keep, fmax) -> None:
    _style(ax)
    curves = [("event residual", res["psd_event_resid"], _EVENT, 2.0),
              ("clean residual", res["psd_clean_resid"], _ACCENT, 1.3),
              ("event raw", res["psd_event_raw"], "#ffb59e", 1.0),
              ("clean raw", res["psd_clean_raw"], "#9fb2f0", 1.0),
              ("template", res["psd_template"], _TEXT, 1.2)]
    for name, psd, c, lw in curves:
        p = np.asarray(psd)
        if p.size == f.size:
            ax.semilogy(f[keep], np.maximum(p[keep], 1e-20), color=c, lw=lw,
                        label=name)
    for ln in res.get("line_hz", []):
        if ln <= fmax:
            ax.axvline(ln, color=_MUTED, lw=0.5, ls=":", alpha=0.5)
    _lab(ax, title=f"{res['animal']} {res['channel']} — power spectrum: event "
                   f"windows vs clean responses",
         ylabel="PSD (µV²/Hz)")
    _legend(ax, loc="upper right", ncol=2)


def _spectral_bottom(ax, res, f, keep, fmax) -> None:
    _style(ax)
    exc = np.asarray(res["excess_db"])
    if exc.size == f.size:
        ax.plot(f[keep], exc[keep], color=_EVENT, lw=1.6, label="event − clean")
        ax.fill_between(f[keep], 0, exc[keep], where=exc[keep] > 0,
                        color=_EVENT, alpha=0.18)
    ax.axhline(0, color=_MUTED, lw=0.6)
    lo, hi = res.get("excess_band", (np.nan, np.nan))
    if np.isfinite(lo) and np.isfinite(hi):
        ax.axvspan(lo, hi, color=_ACCENT, alpha=0.12,
                   label=f"excess band {lo:.0f}–{hi:.0f} Hz")
    fund = res.get("fundamental", {})
    f0 = fund.get("fundamental_hz", float("nan"))
    if np.isfinite(f0):
        ax.axvline(f0, color=_TEXT, lw=1.0, ls="--",
                   label=f"fundamental {f0:.0f} Hz")
        for h in fund.get("harmonics", []):
            if 1 < h["n"] and h["hz"] <= fmax:
                ax.axvline(h["hz"], color=_TEXT, lw=0.6, ls=":", alpha=0.6)
    for ln in res.get("line_hz", []):
        if ln <= fmax:
            ax.axvline(ln, color=_MUTED, lw=0.5, ls=":", alpha=0.5)
    _lab(ax, xlabel="frequency (Hz)", ylabel="excess power (dB)")
    _legend(ax, loc="upper right")


# ---------------------------------------------------------------- Prong B --- #

def fig_candidates(det: dict, out_png: str, *, animal="", channel="",
                   example_sec: float = 20.0) -> str:
    """A continuous window with the band envelope, detection threshold and
    candidate marks — shows what the detector fired on."""
    fs = float(det["fs"])
    env = np.asarray(det["env"])
    if env.size == 0:
        return _empty(out_png, f"{animal} {channel}: no envelope")
    locs = np.asarray(det["cand_locs"])
    c0 = int(locs[0]) if locs.size else 0        # centre on the first candidate
    half = int(example_sec * fs / 2)
    a = max(0, c0 - half)
    b = min(env.size, c0 + half)
    x, y, _d = _decim_envelope(env[a:b], fs, 6000, t_start=a / fs)
    fig, ax = plt.subplots(figsize=(11.0, 4.2), facecolor=_BG)
    _style(ax)
    ax.plot(x, y, color=_ACCENT, lw=0.7, label="band envelope")
    if np.isfinite(det.get("cand_thr", np.nan)):
        ax.axhline(det["cand_thr"], color=_EVENT, lw=1.0, ls="--",
                   label="detection threshold")
    if np.isfinite(det.get("cand_thr_hi", np.inf)):
        ax.axhline(det["cand_thr_hi"], color="#ffd166", lw=1.0, ls=":",
                   label="noise ceiling")
    inwin = locs[(locs >= a) & (locs < b)]
    if inwin.size:
        ax.plot(inwin / fs, env[inwin], "v", color=_EVENT, ms=6,
                label=f"candidates (n={inwin.size})")
    _lab(ax, title=f"{animal} {channel} — candidate events on continuous LFP "
                   f"({det['band'][0]:.0f}–{det['band'][1]:.0f} Hz envelope)",
         xlabel="time (s)", ylabel="envelope (µV)")
    _legend(ax, loc="upper right")
    return _finish(fig, out_png)


def fig_alignment(tmpl: dict, out_png: str, *, fs: float, animal="", channel="",
                  title_extra="") -> str:
    """Event snippets under the median template + the snippet-to-template
    correlation distribution. The reference frame follows ``tmpl['aligned_by']``:
    'stim' (stim-locked events, x = time from stim, no edge marker) or
    'rising_edge' (spontaneous events, x = time from rising edge at 0)."""
    if not tmpl or tmpl.get("template") is None:
        return _empty(out_png, f"{animal} {channel}: no template")
    aligned = np.asarray(tmpl["aligned"])
    template = np.asarray(tmpl["template"])
    by = tmpl.get("aligned_by", "rising_edge")
    t0 = float(tmpl.get("t0_ms", 0.0))
    t = t0 + (np.arange(template.size) / fs) * 1000.0
    frame = "time from stim (ms)" if by == "stim" else "time from rising edge (ms)"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4), facecolor=_BG,
                                   width_ratios=[3, 2])
    _style(ax1)
    kept = tmpl.get("kept")
    show = np.flatnonzero(kept) if kept is not None and kept.any() \
        else np.arange(aligned.shape[0])
    for j in show[np.linspace(0, show.size - 1, min(200, show.size)).astype(int)]:
        ax1.plot(t, aligned[j], color=_ACCENT, lw=0.4, alpha=0.15)
    ax1.plot(t, template, color=_TEXT, lw=2.2, label=f"template (n={tmpl['n']})")
    if by != "stim":                               # rising-edge marker at t=0
        ax1.axvline(0.0, color=_EVENT, lw=0.8, ls="--", label="rising edge")
    _lab(ax1, title=f"{animal} {channel} — event template {title_extra}",
         xlabel=frame, ylabel="LFP (µV)")
    _legend(ax1, loc="upper right")
    _style(ax2)
    corr = np.asarray(tmpl.get("corr", []))
    corr = corr[np.isfinite(corr)]
    if corr.size:
        ax2.hist(corr, bins=30, color=_ACCENT, alpha=0.8)
    _lab(ax2, title="snippet–template correlation", xlabel="Pearson r",
         ylabel="count")
    return _finish(fig, out_png)


def fig_matched_filter(det: dict, out_png: str, *, animal="", channel="",
                       prox: dict | None = None) -> str:
    """Matched-filter correlation over the recording with the threshold +
    detections, and the detection-score histogram (with near/far-from-onset split
    when validation onsets are supplied)."""
    fs = float(det["fs"])
    r = np.asarray(det.get("r_series", []))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=_BG,
                                   width_ratios=[3, 2])
    _style(ax1)
    if r.size:
        x, y, _d = _decim_envelope(r, fs, 8000, t_start=0.0)
        ax1.plot(x, y, color=_ACCENT, lw=0.6, label="matched-filter r")
        ax1.axhline(det.get("thresh", 0.7), color=_EVENT, lw=1.0, ls="--",
                    label="threshold")
        dl = np.asarray(det.get("det_locs", []))
        if dl.size:
            ax1.plot(dl / fs, np.asarray(det["det_scores"]), "v", color=_EVENT,
                     ms=5, label=f"detections (n={dl.size})")
    _lab(ax1, title=f"{animal} {channel} — matched-filter detection",
         xlabel="time (s)", ylabel="correlation r")
    _legend(ax1, loc="upper right")
    _style(ax2)
    sc = np.asarray(det.get("det_scores", []))
    if sc.size:
        ax2.hist(sc, bins=25, color=_ACCENT, alpha=0.85)
    sub = "" if not prox else \
        f"\n{prox['n_near']} near / {prox['n_far']} far of {prox['n_onsets']} onsets"
    _lab(ax2, title=f"detection scores{sub}", xlabel="r at detection",
         ylabel="count")
    return _finish(fig, out_png)


# ------------------------------------------------------------------ util --- #

def _hex_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _empty(out_png: str, msg: str) -> str:
    fig, ax = plt.subplots(figsize=(8.0, 2.0), facecolor=_BG)
    _style(ax)
    ax.text(0.5, 0.5, msg, color=_TEXT, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    return _finish(fig, out_png)
