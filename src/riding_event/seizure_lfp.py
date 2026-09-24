"""Descriptive per-seizure LFP viewer for the riding-event subsystem.

For each scored seizure onset, render the continuous LFP on the pHFO detection
channel:

  * a WIDE decimated (min/max) envelope over ``[onset - pre_h, onset + post_h]``
    with the pHFO event RATE overlaid, so you can see where events/min rises
    relative to onset;
  * a finer ZOOM envelope around onset (``+/- zoom_min``) with the individual
    pHFO detections ticked, so the peri-onset waveform is visible.

DESCRIPTIVE ONLY -- this is not an inferential test. With the Sept-14 boundary
there are too few scored seizures (3, one night) to support a dose-response /
calibration curve; this just shows what those seizures look like on the channel
the detector runs on.

Reads continuous recordings (multi-GB) via the shared chunk cache; the pHFO
detections are the ones already cached by the periictal pipeline (no re-detect).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from src.riding_event import detect as _det                  # noqa: E402
from src.riding_event import periictal as _pe                # noqa: E402
from src.riding_event import render as _r                    # noqa: E402
from src.utils import chunk_cache as _cc                     # noqa: E402

_SEC_H = 3600.0


def _decimate_minmax(seg, fs, t0_min, *, px_per_min):
    """Min/max envelope of ``seg`` in fixed pixel bins. Returns
    (t_center_min, lo, hi). Bounded number of bins (NASA Rule 2/3)."""
    n = seg.size
    dur_min = n / fs / 60.0
    nb = int(max(1, min(20000, round(dur_min * px_per_min))))
    edges = np.linspace(0, n, nb + 1).astype(np.int64)
    lo = np.full(nb, np.nan)
    hi = np.full(nb, np.nan)
    tc = np.empty(nb)
    for k in range(nb):                                # bounded by nb <= 20000
        a, b = edges[k], edges[k + 1]
        if b > a:
            s = seg[a:b]
            lo[k] = float(np.min(s))
            hi[k] = float(np.max(s))
        tc[k] = t0_min + (0.5 * (a + b) / fs / 60.0)
    return tc, lo, hi


def gather_seizure_lfp(store, animal, onset_epoch, tmpl, *, pre_h=2.0, post_h=1.0,
                       zoom_min=2.0, thresh=0.7, wide_ppm=6.0, zoom_ppm=3000.0,
                       progress=None):
    """Continuous LFP + pHFO detections around one seizure onset.

    Returns a dict with the wide envelope (``env_t`` minutes rel onset, ``env_lo``,
    ``env_hi``), the zoom envelope (``z_t`` seconds rel onset, ``z_lo``, ``z_hi``),
    pHFO detection times (``phfo_min`` rel onset), and a coverage-aware pHFO rate
    curve (``rate_c`` bin centers min, ``rate`` events/min)."""
    prog = progress or (lambda *_a: None)
    assert onset_epoch > 0 and pre_h > 0 and post_h > 0, "bad window"
    t_lo = onset_epoch - pre_h * _SEC_H
    t_hi = onset_epoch + post_h * _SEC_H
    recs = [r for r in _pe.animal_recordings(store, animal)
            if r["start_epoch"] < t_hi and r["start_epoch"] + r["duration"] > t_lo]
    assert len(recs) < _pe._MAX_RECS, "too many recordings in the window"
    env_t, env_lo, env_hi = [], [], []
    z_t, z_lo, z_hi = [], [], []
    phfo_abs = []
    ch_name = None
    z_lo_s, z_hi_s = onset_epoch - zoom_min * 60.0, onset_epoch + zoom_min * 60.0
    for i, r in enumerate(recs):
        prog(f"sz@{onset_epoch:.0f}: rec {i + 1}/{len(recs)} "
             f"{os.path.basename(r['file_path'])}")
        d = _pe.detect_recording(store, animal, r, tmpl, thresh=thresh)
        if d["mf"].size:
            keep = (np.isfinite(d["frac"]) & (d["frac"] >= _pe._PHFO_FRAC)
                    & (d["prom"] >= _pe._PHFO_PROM))
            ph = d["mf"][keep]
            phfo_abs.append(ph[(ph >= t_lo) & (ph <= t_hi)])
        loaded = _det.load_channel(r["file_path"], animal)
        if loaded is None:
            continue
        signal, fs, ch = loaded
        ch_name = ch_name or ch
        try:
            ts = r["start_epoch"]
            n = signal.size
            i0 = max(0, int((t_lo - ts) * fs))
            i1 = min(n, int((t_hi - ts) * fs))
            if i1 > i0:
                seg = signal[i0:i1]
                t0m = (ts + i0 / fs - onset_epoch) / 60.0
                tc, lo, hi = _decimate_minmax(seg, fs, t0m, px_per_min=wide_ppm)
                env_t.append(tc); env_lo.append(lo); env_hi.append(hi)
            j0 = max(0, int((z_lo_s - ts) * fs))
            j1 = min(n, int((z_hi_s - ts) * fs))
            if j1 > j0:
                zseg = signal[j0:j1]
                t0s = (ts + j0 / fs - onset_epoch)
                ztc, zlo, zhi = _decimate_minmax(zseg, fs, t0s / 60.0,
                                                 px_per_min=zoom_ppm)
                z_t.append(ztc * 60.0); z_lo.append(zlo); z_hi.append(zhi)
        finally:
            del signal
            _cc.clear()

    def _cat(parts):
        return np.concatenate(parts) if parts else np.empty(0)
    et, elo, ehi = _cat(env_t), _cat(env_lo), _cat(env_hi)
    o = np.argsort(et) if et.size else np.empty(0, int)
    zt, zlo, zhi = _cat(z_t), _cat(z_lo), _cat(z_hi)
    zo = np.argsort(zt) if zt.size else np.empty(0, int)
    phfo = _cat(phfo_abs)
    phfo_min = (phfo - onset_epoch) / 60.0 if phfo.size else np.empty(0)
    # coverage-aware pHFO rate (events/min) in 5-min bins
    bin_min = 5.0
    edges = np.arange(-pre_h * 60.0, post_h * 60.0 + bin_min, bin_min)
    cov = _pe._coverage_seconds(recs, onset_epoch, edges)
    ct, _ = np.histogram(phfo_min, bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(cov > 0, ct / (cov / 60.0), np.nan)
    return {"onset_epoch": onset_epoch, "channel": ch_name,
            "env_t": et[o], "env_lo": elo[o], "env_hi": ehi[o],
            "z_t": zt[zo], "z_lo": zlo[zo], "z_hi": zhi[zo],
            "phfo_min": phfo_min, "rate_c": 0.5 * (edges[:-1] + edges[1:]),
            "rate": rate, "pre_h": pre_h, "post_h": post_h, "zoom_min": zoom_min}


def _sz_label(onset_epoch, racine=None):
    s = _dt.datetime.fromtimestamp(float(onset_epoch)).strftime("%b %d %H:%M")
    return f"{s}" + (f" R{racine}" if racine is not None else "")


def fig_seizure_lfp(gathers, animal, out, *, labels=None):
    """Montage: one row per seizure -- wide envelope + pHFO rate (left), onset
    zoom + pHFO ticks (right)."""
    n = len(gathers)
    assert n >= 1, "need at least one seizure"
    labels = labels or [_sz_label(g["onset_epoch"]) for g in gathers]
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.1 * n), facecolor=_r._BG,
                             squeeze=False, gridspec_kw={"width_ratios": [2.3, 1]})
    for i, g in enumerate(gathers):
        axw, axz = axes[i][0], axes[i][1]
        for a in (axw, axz):
            a.set_facecolor("#26263a")
            a.tick_params(colors=_r._MUTED, labelsize=8)
            for sp in a.spines.values():
                sp.set_color("#3a3a52")
        # wide envelope
        if g["env_t"].size:
            axw.fill_between(g["env_t"], g["env_lo"], g["env_hi"],
                             color=_r._ACCENT, lw=0, alpha=0.85)
        axw.axvline(0, color="#ff6b6b", lw=1.6)
        axw.set_xlim(-g["pre_h"] * 60.0, g["post_h"] * 60.0)
        axw.set_ylabel("LFP (a.u.)", color=_r._TEXT, fontsize=9)
        ch = g.get("channel") or "?"
        axw.set_title(f"{labels[i]}  —  {ch}   (0 = onset)", color=_r._TEXT,
                      fontsize=10, loc="left")
        # pHFO rate on twin axis
        axr = axw.twinx()
        axr.tick_params(colors="#ffd479", labelsize=8)
        ok = np.isfinite(g["rate"])
        axr.plot(g["rate_c"][ok], g["rate"][ok], color="#ffd479", lw=1.8,
                 marker="o", ms=2.5)
        axr.set_ylabel("pHFO / min", color="#ffd479", fontsize=9)
        axr.set_ylim(bottom=0)
        # event rug at the bottom of the wide panel
        if g["phfo_min"].size:
            yb = g["env_lo"][np.isfinite(g["env_lo"])]
            y0 = float(np.min(yb)) if yb.size else 0.0
            axw.plot(g["phfo_min"], np.full(g["phfo_min"].size, y0),
                     "|", color="#ffd479", ms=5, alpha=0.5)
        if i == n - 1:
            axw.set_xlabel("minutes to onset", color=_r._TEXT, fontsize=9)
        # zoom
        if g["z_t"].size:
            axz.fill_between(g["z_t"], g["z_lo"], g["z_hi"],
                             color=_r._ACCENT, lw=0, alpha=0.9)
        axz.axvline(0, color="#ff6b6b", lw=1.6)
        zm = g["zoom_min"] * 60.0
        axz.set_xlim(-zm, zm)
        pin = g["phfo_min"] * 60.0
        pin = pin[(pin >= -zm) & (pin <= zm)]
        if pin.size and g["z_hi"].size:
            yt = float(np.nanmax(g["z_hi"]))
            axz.plot(pin, np.full(pin.size, yt), "v", color="#ffd479", ms=4,
                     alpha=0.8)
        axz.set_title(f"onset ±{g['zoom_min']:.0f} min", color=_r._TEXT,
                      fontsize=10, loc="left")
        if i == n - 1:
            axz.set_xlabel("seconds to onset", color=_r._TEXT, fontsize=9)
    fig.suptitle(f"{animal} · per-seizure LFP on the pHFO channel "
                 f"(descriptive; n={n} scored seizures ≥ Sep 14)",
                 color=_r._TEXT, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=125, facecolor=_r._BG)
    plt.close(fig)
    return out


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", default="BCH111")
    ap.add_argument("--since", default="2026-09-14",
                    help="only seizures with onset on/after this date (hard boundary)")
    ap.add_argument("--lead-gap-h", type=float, default=6.0,
                    help="cluster-leader filter: drop any seizure with a prior "
                         "seizure within this many hours (0 = keep all)")
    ap.add_argument("--pre-h", type=float, default=2.0)
    ap.add_argument("--post-h", type=float, default=1.0)
    ap.add_argument("--zoom-min", type=float, default=2.0)
    ap.add_argument("--thresh", type=float, default=0.7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import yaml
    from src.db.store import Store
    from src.preictal.isi import scored_seizures, leading_seizures
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    store = Store(cfg["database"]["path"])
    cut = _dt.datetime.fromisoformat(a.since).timestamp()
    szs = sorted((s for s in scored_seizures(store, a.animal)
                  if s.onset_epoch >= cut), key=lambda s: s.onset_epoch)
    print(f"{len(szs)} scored seizures >= {a.since}", flush=True)
    if a.lead_gap_h > 0:
        kept = leading_seizures(szs, a.lead_gap_h * _SEC_H)
        dropped = [s for s in szs if s not in kept]
        for s in dropped:
            print(f"  dropped follower (within {a.lead_gap_h:g}h of a prior "
                  f"seizure): {_sz_label(s.onset_epoch, s.racine)}", flush=True)
        szs = kept
        print(f"{len(szs)} cluster-leading seizures kept", flush=True)
    if not szs:
        print("no seizures in window"); return 1
    print("building pHFO template…", flush=True)
    tmpl = _pe.ensure_template(store, a.animal)
    prog = lambda m: print("  ", m, flush=True)             # noqa: E731
    gathers, labels = [], []
    for k, s in enumerate(szs):
        print(f"[{k + 1}/{len(szs)}] {_sz_label(s.onset_epoch, s.racine)}", flush=True)
        g = gather_seizure_lfp(store, a.animal, s.onset_epoch, tmpl, pre_h=a.pre_h,
                               post_h=a.post_h, zoom_min=a.zoom_min, thresh=a.thresh,
                               progress=prog)
        gathers.append(g)
        labels.append(_sz_label(s.onset_epoch, s.racine))
    out = a.out or os.path.join(_pe._root(a.animal),
                                f"{a.animal}_seizure_lfp_since_{a.since}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig_seizure_lfp(gathers, a.animal, out, labels=labels)
    print("SAVED:", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
