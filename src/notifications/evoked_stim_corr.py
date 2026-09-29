"""Stimulus-delivery vs evoked-response correlation (BCH111).

Two *stimulus-side* quantities describe each recording's delivered pulse:

  * **stim P2P**   -- the recorded stim-artifact peak-to-peak amplitude
                      (``stimulusPeakAmplitudes - stimulusTroughAmplitudes``),
                      gain-corrected (input-referred).
  * **Z_ss**       -- the steady-state impedance of the stimulus response
                      (``channel_impedance.slow_ss_kohm``; already gain-corrected).

This module correlates BOTH of them against the **evoked-response metrics**
(peak-to-trough, RMS, line-length, ... ; amplitude metrics gain-corrected), one
point per recording, over a span. Because the recorded stim P2P barely moves at a
fixed commanded charge, the P2P axis only spreads across charge eras (5/10/20 nC)
-- which are partly confounded with time -- so the scatter colours points BY DATE
to expose that confound; Z_ss drifts within any multi-week window.

Outputs (dark theme): a Spearman-rho heatmap (metrics x {P2P, Z_ss}) and a
per-predictor scatter grid (z-scored metric vs z-scored predictor, date-coloured).

CLI: ``python -m src.notifications.evoked_stim_corr --animal BCH111
      --config config/config.yaml [--days N] [--out DIR]``
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import matplotlib.dates as mdates          # noqa: E402
import numpy as np                         # noqa: E402

from src.evoked_figures import data as _d                        # noqa: E402
from src.notifications import evoked_digest as _ed               # noqa: E402
from src.notifications import evoked_package as _pkg             # noqa: E402
from src.notifications.evoked_windowed import (                  # noqa: E402
    _file_gain, _pretty, _raw_base, _robust_z, _spearman)
from src.utils.amplifier_records import (                        # noqa: E402
    load_amplifier_gains, _norm_chan, _norm_chan_free)
from src.utils.evoked_output import read_feature_sidecar         # noqa: E402

# amplitude-scaled metrics (divide by gain); shape/timing metrics are gain-invariant.
_AMPL_METRICS = ["peak_to_trough", "trough_amplitude", "rms_amplitude",
                 "line_length", "max_slope"]
_INV_METRICS = ["peak_latency_ms", "early_late_ratio"]
_METRICS = _AMPL_METRICS + _INV_METRICS

_DT_RE = re.compile(r"__(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})")


def _file_dt(fp):
    """Recording datetime parsed from the ``__YYYY_MM_DD__HH_MM_SS`` stamp."""
    m = _DT_RE.search(os.path.basename(str(fp)))
    return datetime(*[int(x) for x in m.groups()]) if m else None


# --------------------------------------------------------------------- #
#  data sources
# --------------------------------------------------------------------- #
def _zss_map(db_path: str, animal: str) -> dict:
    """``{normalized raw-base: Z_ss kOhm}`` from channel_impedance for *animal*."""
    assert db_path, "db_path required"
    out: dict = {}
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT pf.file_path fp, ci.slow_ss_kohm z
               FROM channel_impedance ci JOIN processed_files pf ON pf.id = ci.file_id
               WHERE ci.channel_name LIKE ? AND ci.slow_ss_kohm IS NOT NULL""",
            (f"%{animal}%",)).fetchall()
    finally:
        conn.close()
    for r in rows:
        b = os.path.basename(str(r["fp"]))
        if b.lower().endswith(".mat"):
            b = b[:-4]
        out[_norm_chan_free(b)] = float(r["z"])
    return out


def _match_key(keys, channel: str):
    t = _norm_chan(channel)
    for k in keys:
        if _norm_chan(k) == t:
            return k
    return None


def _light_stim_p2p(path: str, channel: str) -> float:
    """Median recorded stim-artifact P2P (|peak-trough|) via a LIGHT h5py read of
    only the two small amplitude datasets (never the traces). NaN when unreadable."""
    import h5py
    try:
        with h5py.File(path, "r") as g:
            grp = g.get("allAnimalResults")
            if grp is None:
                return np.nan
            node = grp.get(channel) or grp.get(_match_key(list(grp.keys()), channel))
            if node is None or "stimulusPeakAmplitudes" not in node:
                return np.nan
            sp = np.asarray(node["stimulusPeakAmplitudes"][()], float).ravel()
            st = np.asarray(node["stimulusTroughAmplitudes"][()], float).ravel()
    except (OSError, KeyError, ValueError, TypeError):
        return np.nan
    n = min(sp.size, st.size)
    if n == 0:
        return np.nan
    p2p = np.abs(sp[:n] - st[:n])
    p2p = p2p[np.isfinite(p2p)]
    return float(np.median(p2p)) if p2p.size else np.nan


def _recording_row(fp, animal, channel, gains, zmap) -> dict | None:
    """One recording's row: date, gain, stim P2P, Z_ss and the median (gain-corrected)
    evoked metrics. None when the file has no rows for *channel*."""
    rows = [r for r in (read_feature_sidecar(fp, animal) or [])
            if r.get("channel") == channel]
    dt = _file_dt(fp)
    if not rows or dt is None:
        return None
    g = _file_gain(gains, fp, animal, channel)
    gd = g if (g and g > 0) else np.nan
    out = {"dt": dt, "gain": gd}
    for m in _METRICS:
        vals = np.asarray([_d._nan(r.get(m)) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        med = float(np.median(vals)) if vals.size else np.nan
        if m in _AMPL_METRICS and np.isfinite(gd):
            med = med / gd
        out[m] = med
    p2p = _light_stim_p2p(fp, channel)
    out["p2p"] = p2p / gd if (np.isfinite(p2p) and np.isfinite(gd)) else p2p
    out["zss"] = zmap.get(_norm_chan_free(_raw_base(fp)), np.nan)
    return out


def _list_files(animal, evoked_dir, days) -> list:
    """Sorted (by time) evoked .mat files for *animal*, optionally the last *days*."""
    import glob
    files = [f for f in glob.glob(os.path.join(evoked_dir, f"*{animal}*_evoked.mat"))]
    dated = [(("_file_dt", _file_dt(f)), f) for f in files]
    dated = [(dt, f) for (_t, dt), f in dated if dt is not None]
    dated.sort(key=lambda x: x[0])
    if days and dated:
        cutoff = dated[-1][0] - timedelta(days=float(days))
        dated = [(dt, f) for dt, f in dated if dt >= cutoff]
    return [f for _dt, f in dated]


def _pick_channel(files, animal, override):
    """Primary channel: *override*, else the most common sidecar channel that carries
    the animal id (prefer an 'SR' electrode)."""
    if override:
        return override
    from collections import Counter
    cnt: Counter = Counter()
    for fp in files[:40]:
        for r in (read_feature_sidecar(fp, animal) or []):
            ch = r.get("channel")
            if ch and animal.lower() in ch.lower():
                cnt[ch] += 1
    if not cnt:
        return None
    return max(cnt, key=lambda c: (("sr" in c.lower()), cnt[c]))


# --------------------------------------------------------------------- #
#  correlate
# --------------------------------------------------------------------- #
def correlate(rows, metrics) -> dict:
    """Spearman rho of each *metric* vs stim P2P and vs Z_ss, across recordings."""
    p2p = np.asarray([r["p2p"] for r in rows], dtype=float)
    zss = np.asarray([r["zss"] for r in rows], dtype=float)
    out: dict = {}
    for m in metrics:
        y = np.asarray([r[m] for r in rows], dtype=float)
        out[m] = {"p2p": _spearman(p2p, y), "zss": _spearman(zss, y)}
    return out


# --------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------- #
_PREDS = [("p2p", "stim P2P"), ("zss", "Z_ss (steady-state)")]


def _fig_heatmap(corr, metrics, meta, out) -> str:
    """Heatmap: metric x {stim P2P, Z_ss} Spearman rho (rho + n annotated)."""
    order = sorted(metrics, key=lambda m: -np.nanmax(
        [abs(corr[m][k][0]) for k, _l in _PREDS] or [0]))
    Z = np.array([[corr[m][k][0] for k, _l in _PREDS] for m in order])
    fig, ax = plt.subplots(figsize=(5.8, 0.55 * len(order) + 2.2), facecolor=_ed._BG)
    ax.set_facecolor(_ed._PANEL)
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(_PREDS)))
    ax.set_xticklabels([l for _k, l in _PREDS], fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_pretty(m) for m in order], fontsize=9)
    ax.tick_params(colors=_ed._MUTED)
    for r, m in enumerate(order):
        for c, (k, _l) in enumerate(_PREDS):
            rho, n = corr[m][k]
            txt = f"{rho:+.2f}\nn={n}" if np.isfinite(rho) else "n/a"
            # saturated (|rho|>0.5) cells are dark -> light text; pale cells -> dark
            ax.text(c, r, txt, ha="center", va="center", fontsize=7,
                    color="#f0f0f5" if (np.isfinite(rho) and abs(rho) > 0.5)
                    else "#111")
    cb = fig.colorbar(im, ax=ax, fraction=0.06)
    cb.set_label("Spearman ρ", color=_ed._MUTED)
    cb.ax.tick_params(colors=_ed._MUTED)
    ax.set_title(f"{meta['animal']} · {meta['channel']} · stim delivery ↔ evoked "
                 f"(Spearman ρ)\n{meta['span']}", color=_ed._TEXT, fontsize=10,
                 loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=_ed._BG)
    plt.close(fig)
    return out


def _scatter_panel(ax, x, y, dnum, metric, pred_label):
    """One metric panel: z(metric) vs z(predictor), date-coloured, rho + fit."""
    xz, yz = _robust_z(x), _robust_z(y)
    ok = np.isfinite(xz) & np.isfinite(yz)
    ax.set_facecolor(_ed._PANEL)
    sc = ax.scatter(xz[ok], yz[ok], c=dnum[ok], s=10, cmap="viridis",
                    alpha=0.7, linewidths=0)
    rho, n = _spearman(x, y)
    if ok.sum() >= 2 and np.ptp(xz[ok]) > 0:            # robust-ish linear guide
        b = np.polyfit(xz[ok], yz[ok], 1)
        xs = np.array([np.nanmin(xz[ok]), np.nanmax(xz[ok])])
        ax.plot(xs, np.polyval(b, xs), color="#f0f0f5", lw=1.2)
    ax.axhline(0, color=_ed._MUTED, lw=0.4, alpha=0.4)
    ax.axvline(0, color=_ed._MUTED, lw=0.4, alpha=0.4)
    ax.set_title(f"{_pretty(metric)}  ρ={rho:+.2f}" if np.isfinite(rho)
                 else f"{_pretty(metric)}  ρ=n/a", color=_ed._TEXT, fontsize=9,
                 loc="left")
    ax.set_xlabel(f"z({pred_label})", color=_ed._MUTED, fontsize=8)
    ax.tick_params(colors=_ed._MUTED, labelsize=7)
    return sc


def _fig_scatter(rows, metrics, pred_key, pred_label, meta, out) -> str:
    """Scatter grid: each metric (z) vs the predictor (z), points coloured by date."""
    x = np.asarray([r[pred_key] for r in rows], dtype=float)
    dnum = np.asarray([mdates.date2num(r["dt"]) for r in rows], dtype=float)
    ncol = 4
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.0 * nrow),
                             facecolor=_ed._BG, squeeze=False)
    sc = None
    for i, m in enumerate(metrics):
        y = np.asarray([r[m] for r in rows], dtype=float)
        sc = _scatter_panel(axes[i // ncol][i % ncol], x, y, dnum, m, pred_label)
    for j in range(len(metrics), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{meta['animal']} · {meta['channel']} · evoked metric (z) vs "
                 f"{pred_label} (z) · {meta['span']}", color=_ed._TEXT, fontsize=12)
    # own layout (avoids the x-label / next-row-title collision) + dedicated colourbar.
    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.08,
                        hspace=0.55, wspace=0.26)
    if sc is not None:                                   # shared date colourbar
        cb = fig.colorbar(sc, cax=fig.add_axes([0.925, 0.15, 0.012, 0.7]))
        cb.set_label("recording date", color=_ed._MUTED, fontsize=8)
        cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        cb.ax.tick_params(colors=_ed._MUTED, labelsize=7)
    fig.savefig(out, dpi=125, facecolor=_ed._BG)
    plt.close(fig)
    return out


# --------------------------------------------------------------------- #
#  build
# --------------------------------------------------------------------- #
def build_stim_corr(animal, evoked_dir, db_path, out_dir, *, days=None,
                    channel_override=None, config=None, gains=None,
                    progress=None) -> dict:
    """Assemble per-recording (stim P2P, Z_ss, evoked metrics) and render the
    correlation heatmap + per-predictor scatter grids. Returns a manifest dict."""
    assert animal and evoked_dir, "animal and evoked_dir required"
    if gains is None and config is not None:
        if progress:
            progress("loading amplifier gains…")
        gains = load_amplifier_gains(config)
    zmap = _zss_map(db_path, animal)
    files = _list_files(animal, evoked_dir, days)
    if not files:
        return {"animal": animal, "empty": True, "reason": "no recordings"}
    channel = _pick_channel(files, animal, channel_override)
    if channel is None:
        return {"animal": animal, "empty": True, "reason": "no channel"}
    if progress:
        progress(f"channel {channel}; scanning {len(files)} recordings…")
    rows = []
    for i, fp in enumerate(files):
        if progress and i % 25 == 0:
            progress(f"recordings… ({i}/{len(files)})")
        r = _recording_row(fp, animal, channel, gains, zmap)
        if r is not None:
            rows.append(r)
    n_z = int(np.sum([np.isfinite(r["zss"]) for r in rows]))
    n_p = int(np.sum([np.isfinite(r["p2p"]) for r in rows]))
    if progress:
        progress(f"{len(rows)} recordings ({n_p} with P2P, {n_z} with Z_ss); "
                 f"rendering…")
    span = (f"{rows[0]['dt']:%Y-%m-%d} … {rows[-1]['dt']:%Y-%m-%d} "
            f"({len(rows)} recordings)")
    meta = {"animal": animal, "channel": channel, "span": span}
    os.makedirs(out_dir, exist_ok=True)
    corr = correlate(rows, _METRICS)
    heat = _fig_heatmap(corr, _METRICS, meta, os.path.join(out_dir, "corr_heatmap.png"))
    scatters = {}
    for k, label in _PREDS:
        p = os.path.join(out_dir, f"scatter_{k}.png")
        scatters[k] = _fig_scatter(rows, _METRICS, k, label, meta, p)
    return {"animal": animal, "channel": channel, "n_recordings": len(rows),
            "n_p2p": n_p, "n_zss": n_z, "span": span, "heatmap": heat,
            "scatters": scatters, "corr": corr, "out_dir": out_dir}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", default="BCH111")
    ap.add_argument("--days", type=float, default=None, help="last N days (default all)")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import yaml
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    evoked_dir = cfg["chronic_evoked"]["evoked_output_dir"]
    db_path = (cfg.get("database", {}) or {}).get("path") or "data/monitor.db"
    out = a.out or os.path.join("data", "evoked_stim_corr", a.animal)
    res = build_stim_corr(a.animal, evoked_dir, db_path, out, days=a.days,
                          channel_override=a.channel, config=cfg,
                          progress=lambda m: print("  ", m, flush=True))
    print("RESULT:", {k: v for k, v in res.items() if k != "corr"})
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
