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
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from src.periictal.erpimage import decimate_rows  # noqa: E402
from src.notifications.trace_density import LineDensity, robust_ylim  # noqa: E402
from src.riding_event import primitives as _p  # noqa: E402
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
    """Save + close. Writes to a PID-unique temp then atomically replaces the
    target, so a target locked by a viewer / AV / stale handle warns-and-skips
    instead of crashing a long run (a partial PNG is never left behind)."""
    import warnings
    with warnings.catch_warnings():                # gridspec + tight_layout warns
        warnings.simplefilter("ignore")
        fig.tight_layout()
    tmp = f"{out_png}.{os.getpid()}.tmp.png"       # keep .png so format infers
    try:
        fig.savefig(tmp, dpi=140, facecolor=_BG, format="png")
        os.replace(tmp, out_png)
    except OSError as e:                          # locked target / transient FS
        logger.warning("could not write %s: %s", out_png, e)
        try:
            os.remove(tmp)
        except OSError:
            pass
    finally:
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


def fig_spectral_clusters(res: dict, out_png: str, *, k: int = 3,
                          n_ex: int = 5) -> str:
    """Cluster the flagged event epochs by PSD SHAPE and show each cluster's
    spectral footprint (median PSD, peak Hz labelled) beside example traces — so
    the 247 Hz ripple is separated from any other footprint (e.g. a sharp-line
    artifact) that the HF flag also caught."""
    from src.riding_event import spectra as _sp
    ev = np.flatnonzero(res["event_mask"])
    tm = np.asarray(res["time_ms"])
    fs = float(res["fs"])
    win = res["spec_win_ms"]
    f, psds = _sp.epoch_psds(res["resid"][ev], fs, time_ms=tm, win_ms=win) \
        if ev.size else (np.empty(0), np.empty((0, 0)))
    if psds.shape[0] < 4:
        return _empty(out_png, f"{res['animal']} {res['channel']}: too few events")
    labels = _sp.cluster_spectra(psds, f, k=k)
    uniq = sorted(set(labels.tolist()))
    fmax = min(1500.0, 0.5 * fs)
    keep = (f > 0) & (f <= fmax)
    fig, axes = plt.subplots(len(uniq), 2, figsize=(11.5, 2.5 * len(uniq) + 0.7),
                             facecolor=_BG, squeeze=False,
                             gridspec_kw={"width_ratios": [2, 3]})
    cmap = plt.get_cmap("turbo")
    for row, lab in enumerate(uniq):
        idx = np.flatnonzero(labels == lab)
        c = cmap(0.1 + 0.8 * row / max(1, len(uniq) - 1))
        _cluster_psd_panel(axes[row][0], f, psds, idx, keep, lab, c)
        _cluster_trace_panel(axes[row][1], tm, res["traces"], ev[idx], win, c, n_ex)
    fig.suptitle(f"{res['animal']} {res['channel']} — event epochs clustered by "
                 f"spectral footprint ({len(uniq)} groups of {ev.size})",
                 color=_TEXT, fontsize=11, x=0.02, ha="left")
    return _finish(fig, out_png)


def _cluster_psd_panel(ax, f, psds, idx, keep, lab, c) -> None:
    _style(ax)
    for j in idx[:80]:
        ax.semilogy(f[keep], np.maximum(psds[j][keep], 1e-20), color=c, lw=0.3,
                    alpha=0.15)
    med = np.median(psds[idx], axis=0)
    ax.semilogy(f[keep], np.maximum(med[keep], 1e-20), color=c, lw=2.2,
                label=f"cluster {lab} (n={idx.size})")
    for pk_hz, pk_p in _psd_peaks(f[keep], med[keep], n=2):   # distinctive HF peaks
        ax.annotate(f"{pk_hz:.0f} Hz", xy=(pk_hz, pk_p), xytext=(4, 2),
                    textcoords="offset points", color=_TEXT, fontsize=11,
                    fontweight="bold")
    _lab(ax, xlabel="frequency (Hz)", ylabel="PSD (µV²/Hz)")
    _legend(ax, loc="upper right")


def _psd_peaks(fb, mb, *, fmin=100.0, n=2):
    """The *n* most PROMINENT peaks of a PSD above *fmin* Hz (on log power, so
    1/f doesn't win) — the cluster's distinctive footprint frequencies."""
    from scipy.signal import find_peaks
    fb = np.asarray(fb, dtype=np.float64)
    mb = np.asarray(mb, dtype=np.float64)
    band = fb >= fmin
    if band.sum() < 3:
        return []
    lg = np.log10(np.maximum(mb, 1e-20))
    idx, props = find_peaks(np.where(band, lg, -np.inf), prominence=0.3,
                            distance=max(1, int(fb.size / 40)))
    if idx.size == 0:
        return []
    top = idx[np.argsort(-props["prominences"])[:int(n)]]
    return [(float(fb[i]), float(mb[i])) for i in sorted(top)]


def _cluster_trace_panel(ax, tm, traces, epochs, win, c, n_ex) -> None:
    _style(ax)
    take = epochs[np.linspace(0, epochs.size - 1, min(n_ex, epochs.size)).astype(int)]
    off_step = 1.2
    for i, j in enumerate(take):
        tr = traces[j]
        norm = np.max(np.abs(tr[(tm >= win[0]) & (tm <= win[1])])) or 1.0
        ax.plot(tm, tr / norm + i * off_step, color=c, lw=0.7)
    ax.axvspan(win[0], win[1], color=_TEXT, alpha=0.05)
    ax.set_xlim(min(tm[0], -10.0), win[1] + 10.0)
    ax.set_yticks([])
    _lab(ax, xlabel="time from stim (ms)", ylabel="example traces")


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
    fig = plt.figure(figsize=(9.5, 10.2), facecolor=_BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 3, 2], hspace=0.32)
    ax0 = fig.add_subplot(gs[0])                     # traces feeding the PSD
    ax1 = fig.add_subplot(gs[1])                     # PSDs
    ax2 = fig.add_subplot(gs[2], sharex=ax1)         # excess (shares freq axis)
    _spectral_traces(ax0, res)
    _spectral_top(ax1, res, f, keep, fmax)
    _spectral_bottom(ax2, res, f, keep, fmax)
    return _finish(fig, out_png)


def _spectral_traces(ax, res) -> None:
    """Top panel: a few event (orange) and clean (blue) example epochs with the
    PSD analysis window shaded — what actually goes INTO the spectra below."""
    _style(ax)
    tm = np.asarray(res["time_ms"])
    lo, hi = res["spec_win_ms"]
    for j in _examples(res["clean_mask"], 4):
        ax.plot(tm, res["traces"][j], color=_ACCENT, lw=0.5, alpha=0.45)
    for j in _examples(res["event_mask"], 4):
        ax.plot(tm, res["traces"][j], color=_EVENT, lw=0.7, alpha=0.8)
    ax.axvspan(lo, hi, color=_TEXT, alpha=0.07)
    ax.plot([], [], color=_EVENT, lw=1.4, label="event epochs")
    ax.plot([], [], color=_ACCENT, lw=1.4, label="clean epochs")
    ax.set_xlim(min(tm[0], -10.0), hi + 20.0)
    _lab(ax, title=f"{res['animal']} {res['channel']} — example traces feeding the "
                   f"PSD (shaded = {lo:.0f}-{hi:.0f} ms analysis window)",
         xlabel="time from stim (ms)", ylabel="LFP (µV)")
    _legend(ax, loc="upper right")


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
    for ln in res.get("line_hz", []):
        if ln <= fmax:
            ax.axvline(ln, color=_MUTED, lw=0.5, ls=":", alpha=0.5)
    # Label the (up to 2) most prominent excess-power peaks with their Hz, big.
    for pk_hz, pk_db in _top_excess_peaks(f, exc, res, fmax, n=2):
        ax.axvline(pk_hz, color=_TEXT, lw=1.2, ls="--")
        ax.annotate(f"{pk_hz:.0f} Hz", xy=(pk_hz, pk_db), xytext=(6, 4),
                    textcoords="offset points", color=_TEXT, fontsize=12,
                    fontweight="bold")
    _lab(ax, xlabel="frequency (Hz)", ylabel="excess power (dB)")
    _legend(ax, loc="upper right")


def _top_excess_peaks(f, exc, res, fmax, n=2):
    """The *n* most prominent excess-power peaks (Hz, dB) in the excess band,
    excluding mains-line bins — the event's rhythm(s) to label big."""
    from scipy.signal import find_peaks
    if exc.size != f.size:
        return []
    lo, hi = res.get("excess_band", (4.0, fmax))
    lo = lo if np.isfinite(lo) else 4.0
    hi = hi if np.isfinite(hi) else fmax
    band = (f >= lo) & (f <= min(hi, fmax)) & np.isfinite(exc)
    line = np.zeros(f.size, dtype=bool)
    for ln in res.get("line_hz", []):
        line |= np.abs(f - ln) <= 3.0
    cand = band & ~line
    if not cand.any():
        return []
    d = np.where(cand, exc, -np.inf)
    idx, props = find_peaks(d, height=2.0, distance=max(1, int(f.size / 60)))
    if idx.size == 0:
        return []
    top = idx[np.argsort(-props["peak_heights"])[:int(n)]]
    return [(float(f[i]), float(exc[i])) for i in sorted(top)]


# ---------------------------------------------------------------- Prong B --- #

def fig_candidates(det: dict, out_png: str, *, animal="", channel="",
                   example_sec: float = 20.0) -> str:
    """A continuous window with the band envelope, detection threshold and
    candidate marks — shows what the detector fired on."""
    fs = float(det["fs"])
    env = np.asarray(det["env"])
    if env.size == 0:
        return _empty(out_png, f"{animal} {channel}: no envelope")
    # Centre on a DETECTION (a real event) in a window that has no noise-ceiling
    # spike (the giant transient the user objected to), so the example is legible.
    half = int(example_sec * fs / 2)
    dl = np.asarray(det.get("det_locs", []), dtype=np.int64)
    c0 = _clean_window_center(env, dl, det.get("cand_thr_hi", np.inf), half)
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
    st = det.get("stim_times_sec")                   # mark stims: the periodic
    if st is not None and len(st):                   # peaks ARE the stim artifacts
        sv = np.asarray(st, dtype=np.float64)
        sv = sv[(sv >= a / fs) & (sv <= b / fs)]
        for i, s in enumerate(sv):
            ax.axvline(s, color="#ff3b3b", lw=0.6, ls=":", alpha=0.5,
                       label="stim (artifact)" if i == 0 else None)
    inwin = dl[(dl >= a) & (dl < b)]
    if inwin.size:
        ax.plot(inwin / fs, env[np.clip(inwin, 0, env.size - 1)], "v",
                color=_EVENT, ms=7, label=f"detections in window (n={inwin.size})")
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


def fig_detected_events(signal, det: dict, out_png: str, *, fs: float, animal="",
                        channel="", n: int = 12, pre_ms: float = 20.0,
                        post_ms: float = 100.0, onset_sec=None,
                        frac=None, prom=None,
                        phfo_frac: float = _p.DEFAULT_PHFO_FRAC,
                        phfo_prom: float = _p.DEFAULT_PHFO_PROM) -> str:
    """The actual LFP traces at matched-filter detections (a grid, one per
    detection, aligned to the detection at t=0). Shows what the detector fired on
    so it can be judged by eye -- raw LFP (blue) + band-passed HF trace (orange).
    With *onset_sec* (a seizure recording), shows the detections nearest the
    onset; else the strongest.

    When ``frac``/``prom`` (per-detection pHFO metrics, aligned to
    ``det['det_locs']``) are given, each panel is framed and labelled by the pHFO
    GATE: green = pHFO (frac>=*phfo_frac* & prom>=*phfo_prom*), red = rejected LFD
    (low-frequency deflection). This is what separates the real HF packets from the
    slow DC-shift false positives."""
    x = np.asarray(signal, dtype=np.float64)
    dl = np.asarray(det.get("det_locs", []), dtype=np.int64)
    sc = np.asarray(det.get("det_scores", []), dtype=np.float64)
    if dl.size == 0:
        return _empty(out_png, f"{animal} {channel}: no detections")
    if onset_sec is not None:                       # detections nearest the seizure
        order = np.argsort(np.abs(dl / fs - float(onset_sec)))[:int(n)]
    else:
        order = np.argsort(-sc)[:int(n)]            # strongest first
    sel = dl[order]
    fr = np.asarray(frac, float)[order] if frac is not None else None
    pr = np.asarray(prom, float)[order] if prom is not None else None
    gate_sel = _phfo_gate_mask(fr, pr, phfo_frac, phfo_prom)  # over SHOWN
    n_keep = int(gate_sel.sum()) if gate_sel is not None else -1
    pre = max(1, int(round(pre_ms * 1e-3 * fs)))
    post = max(1, int(round(post_ms * 1e-3 * fs)))
    t = (np.arange(-pre, post) / fs) * 1000.0
    band = det.get("band", (120.0, 990.0))
    nrow = int(np.ceil(sel.size / 3))
    fig, axes = plt.subplots(nrow, 3, figsize=(11.5, 2.15 * nrow + 0.6),
                             facecolor=_BG, squeeze=False)
    for ax in axes.flat:
        _style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
    for i, (ax, loc, sco) in enumerate(zip(axes.flat, sel, sc[order])):
        lo, hi = loc - pre, loc + post
        if lo < 0 or hi > x.size:
            continue
        seg = x[lo:hi]
        hf = _ef_bandpass(seg, fs, band)
        ax.plot(t, seg, color=_ACCENT, lw=0.6)
        ax.plot(t, hf, color=_EVENT, lw=0.7, alpha=0.8)
        ax.axvline(0, color=_MUTED, lw=0.6, ls="--")
        _annotate_gate(ax, loc, fs, sco, fr, pr, i, phfo_frac, phfo_prom)
    which = (f"nearest the seizure onset ({onset_sec:.0f}s)"
             if onset_sec is not None else "strongest")
    gate_txt = (f" — pHFO-GATED: {n_keep} of {sel.size} pHFO (green), "
                f"{sel.size - n_keep} LFD rejected (red)" if n_keep >= 0 else "")
    fig.suptitle(f"{animal} {channel} — {sel.size} detections {which}{gate_txt}\n"
                 f"(blue = LFP, orange = {band[0]:.0f}-{band[1]:.0f} Hz; t=0 = "
                 f"detection)", color=_TEXT, fontsize=10.5, x=0.02, ha="left")
    return _finish(fig, out_png)


def _phfo_gate_mask(frac, prom, phfo_frac, phfo_prom):
    """Boolean 'is a real pHFO' mask over detections, or None when metrics were
    not supplied."""
    if frac is None or prom is None:
        return None
    fr = np.asarray(frac, float)
    pr = np.asarray(prom, float)
    return np.isfinite(fr) & (fr >= float(phfo_frac)) & (pr >= float(phfo_prom))


def _annotate_gate(ax, loc, fs, sco, fr, pr, i, phfo_frac, phfo_prom) -> None:
    """Per-panel title + coloured frame from the pHFO gate (green pHFO / red LFD);
    plain title when no metrics were supplied."""
    if fr is None or pr is None or i >= fr.size:
        ax.set_title(f"{loc/fs:.1f}s  r={sco:.2f}", color=_TEXT, fontsize=8,
                     loc="left")
        return
    keep = np.isfinite(fr[i]) and fr[i] >= phfo_frac and pr[i] >= phfo_prom
    col = "#3ddc84" if keep else "#ff5d5d"
    tag = "pHFO" if keep else "LFD"
    ax.set_title(f"{loc/fs:.1f}s  r={sco:.2f}  {tag}  f={fr[i]:.2f} p={pr[i]:.0f}",
                 color=col, fontsize=8, loc="left")
    for sp in ax.spines.values():
        sp.set_color(col)
        sp.set_linewidth(1.8)


def _ef_bandpass(seg, fs, band):
    from src.utils import evoked_features as _ef
    lo = max(1.0, float(band[0]))
    hi = min(float(band[1]), 0.49 * float(fs))
    if hi <= lo or seg.size < 20:
        return np.zeros_like(seg)
    return _ef._bandpass(seg[None, :], float(fs), lo, hi)[0]


def fig_matched_filter(det: dict, out_png: str, *, animal="", channel="",
                       prox: dict | None = None, onset_sec=None) -> str:
    """Matched-filter correlation over the recording with the threshold +
    detections, and the detection-score histogram (with near/far-from-onset split
    when validation onsets are supplied)."""
    fs = float(det["fs"])
    r = _blank_near_stims(np.asarray(det.get("r_series", []), dtype=np.float64),
                          fs, det.get("stim_times_sec"), det.get("excl_pad_sec"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=_BG,
                                   width_ratios=[3, 2])
    _style(ax1)
    if r.size:
        xb, ymax = _bin_max(r, fs, 3000)             # upper envelope: peak r/bin
        ax1.plot(xb, ymax, color=_ACCENT, lw=0.5,
                 label="matched-filter r (peak/bin, stim windows blanked)")
        ax1.axhline(det.get("thresh", 0.7), color=_EVENT, lw=1.0, ls="--",
                    label="threshold")
        dl = np.asarray(det.get("det_locs", []))
        if dl.size:
            ax1.plot(dl / fs, np.asarray(det["det_scores"]), "v", color=_EVENT,
                     ms=5, label=f"detections (n={dl.size})")
        ax1.set_ylim(det.get("thresh", 0.7) - 0.25, 1.02)
    if onset_sec is not None:
        ax1.axvline(float(onset_sec), color="#ff3b3b", lw=1.5,
                    label="seizure onset")
    _lab(ax1, title=f"{animal} {channel} — spontaneous-event detections "
                    f"(between stims)",
         xlabel="time (s)", ylabel="correlation r")
    _legend(ax1, loc="upper right")
    _style(ax2)
    sc = np.asarray(det.get("det_scores", []))
    if sc.size:
        ax2.hist(sc, bins=25, color=_ACCENT, alpha=0.85)
    sub = ""
    if prox and "enrichment" in prox:                # seizure-recording format
        sub = (f"\n{prox['n_within_60s']} within ±60 s of onset "
               f"({prox['enrichment']:.1f}× the overall rate)")
    elif prox and "n_onsets" in prox:                # scored-onsets proximity
        sub = f"\n{prox['n_near']} near / {prox['n_far']} far of {prox['n_onsets']} onsets"
    _lab(ax2, title=f"detection scores{sub}", xlabel="r at detection",
         ylabel="count")
    return _finish(fig, out_png)


# ------------------------------------------------------------------ util --- #

def _blank_near_stims(r, fs, stim_times_sec, pad_sec):
    """NaN out the matched-filter r within ``pad_sec`` of each stim so the plot
    shows only the INTER-STIM correlation (the stim-evoked responses, which the
    template matches by construction, are excluded from detection anyway)."""
    r = np.asarray(r, dtype=np.float64)
    if r.size == 0 or stim_times_sec is None or not len(stim_times_sec) \
            or not pad_sec:
        return r
    out = r.copy()
    pad = max(1, int(round(float(pad_sec) * float(fs))))
    n = out.size
    ts = np.asarray(stim_times_sec, dtype=np.float64)
    assert ts.size < 5_000_000, "stim count exceeds bound"   # NASA Rule 2
    for s in ts:
        i = int(round(float(s) * float(fs)))
        lo, hi = max(0, i - pad), min(n, i + pad)
        if hi > lo:
            out[lo:hi] = np.nan
    return out


def _clean_window_center(env, det_locs, thr_hi, half: int) -> int:
    """A detection sample whose +/-half window contains no envelope point above the
    noise ceiling (so the example window has no giant transient). Falls back to a
    middle detection, else the signal centre."""
    n = int(np.asarray(env).size)
    dl = np.asarray(det_locs, dtype=np.int64) if det_locs is not None \
        else np.empty(0, dtype=np.int64)
    if dl.size == 0:
        return n // 2
    e = np.asarray(env, dtype=np.float64)
    for c in dl:                                     # bounded by detection count
        a, b = max(0, int(c) - half), min(n, int(c) + half)
        if not np.isfinite(thr_hi) or bool(np.all(e[a:b] < thr_hi)):
            return int(c)
    return int(dl[dl.size // 2])


def _bin_max(r, fs, nbins: int):
    """Per-bin MAX of the matched-filter r (the upper envelope), so the plot is a
    clean line tracing where correlation peaks occur -- not a solid min/max fill
    (the r spans +/-1 in every bin, so the fill hides everything). NaN (blanked
    stim) bins render as gaps."""
    r = np.asarray(r, dtype=np.float64)
    n = r.size
    assert n > 0 and fs > 0, "need samples and fs"
    nb = int(min(max(1, nbins), n))
    edges = np.linspace(0, n, nb + 1).astype(int)
    xb = (edges[:-1] + edges[1:]) / (2.0 * float(fs))
    ymax = np.full(nb, np.nan)
    for i in range(nb):                              # bounded: nb <= nbins
        seg = r[edges[i]:edges[i + 1]]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            ymax[i] = float(seg.max())
    return xb, ymax


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
