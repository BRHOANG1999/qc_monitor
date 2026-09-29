"""Stimulus-artifact (LFP) vs evoked-response correlation (BCH111).

Two *stimulus-side* quantities describe each recording's LFP stim artifact, both
from the impedance analysis of the recording (animal/LFP) channel -- the stimulus
COPY channel is used only for pulse alignment, never for the amplitude itself:

  * **Ra**   -- access resistance (fast-phase voltage step / current, kOhm), the
                artifact ONSET measure.  ``channel_impedance.access_r_kohm``.
  * **Z_ss** -- steady-state impedance (slow-phase plateau / current, kOhm).
                ``channel_impedance.slow_ss_kohm``.

Both are measured on the artifact DURING current delivery (at onset), so they do
NOT overlap the evoked-response window.  (A raw +/-1 ms LFP P2P is not used: the
artifact rails at +/-10.9 V there, so its raw P2P is the clip level, not a real
amplitude -- the impedance module reads the non-railed edge/plateau instead.)

They are correlated against the **evoked-response metrics** RE-MEASURED over a
separate, non-overlapping window (default 2-50 ms post-stim; amplitude metrics
gain-corrected).  One point PER EPOCH: each epoch's evoked metric vs its
recording's Ra / Z_ss (broadcast).  Z_ss/Ra vary per recording (electrode/tissue
drift), not per epoch.

Outputs (dark theme): a Spearman-rho heatmap (metrics x {Ra, Z_ss}); per-predictor
per-epoch density (hexbin); and a **mean + overlaid example traces grouped by the
predictor** figure -- recordings binned by Ra (or Z_ss), each bin's evoked traces
overlaid with its mean, turbo low->high, so a real coupling is visible in the raw
waveforms.

CLI: ``python -m src.notifications.evoked_stim_corr --animal BCH111
      --config config/config.yaml [--since YYYY-MM-DD] [--window 2-50]``
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

from src.notifications import evoked_digest as _ed               # noqa: E402
from src.notifications.evoked_windowed import (                  # noqa: E402
    _file_gain, _pretty, _raw_base, _robust_z, _spearman, _win_cfg, _win_tag)
from src.utils import evoked_features as _ef                     # noqa: E402
from src.utils.amplifier_records import (                        # noqa: E402
    load_amplifier_gains, _norm_chan, _norm_chan_free)
from src.utils.evoked_output import read_feature_sidecar         # noqa: E402

# amplitude-scaled metrics (÷ gain); shape/timing metrics are gain-invariant.
_AMPL_METRICS = ["peak_to_trough", "trough_amplitude", "rms_amplitude",
                 "line_length", "max_slope"]
_INV_METRICS = ["peak_latency_ms", "early_late_ratio"]
_METRICS = _AMPL_METRICS + _INV_METRICS

# stimulus-side predictors (both LFP-artifact impedances, per recording).
_PREDS = [("ra", "Ra (access resistance, kΩ)"),
          ("zss", "Z_ss (steady-state, kΩ)")]

_FS = 20000.0                              # native evoked sampling
DEFAULT_WINDOW = (2.0, 50.0)               # evoked window (non-overlapping w/ artifact)
_DT_RE = re.compile(r"__(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})")


def _file_dt(fp):
    """Recording datetime from the ``__YYYY_MM_DD__HH_MM_SS`` stamp; None if absent."""
    m = _DT_RE.search(os.path.basename(str(fp)))
    return datetime(*[int(x) for x in m.groups()]) if m else None


def _match_key(keys, channel: str):
    t = _norm_chan(channel)
    for k in keys:
        if _norm_chan(k) == t:
            return k
    return None


# --------------------------------------------------------------------- #
#  data sources
# --------------------------------------------------------------------- #
def _imp_map(db_path: str, animal: str) -> dict:
    """``{normalized raw-base: (Ra_kOhm, Z_ss_kOhm)}`` from channel_impedance."""
    assert db_path, "db_path required"
    out: dict = {}
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT pf.file_path fp, ci.access_r_kohm ra, ci.slow_ss_kohm z
               FROM channel_impedance ci JOIN processed_files pf ON pf.id = ci.file_id
               WHERE ci.channel_name LIKE ?
                 AND (ci.access_r_kohm IS NOT NULL OR ci.slow_ss_kohm IS NOT NULL)""",
            (f"%{animal}%",)).fetchall()
    finally:
        conn.close()
    for r in rows:
        b = os.path.basename(str(r["fp"]))
        if b.lower().endswith(".mat"):
            b = b[:-4]
        out[_norm_chan_free(b)] = (
            float(r["ra"]) if r["ra"] is not None else np.nan,
            float(r["z"]) if r["z"] is not None else np.nan)
    return out


def _win_traces(path, channel, win):
    """(traces[:, win], time_ms[win]) via a windowed h5py read of ONLY the analysis
    window columns (no baseline needed, so pre-stim samples are not read)."""
    import h5py
    try:
        with h5py.File(path, "r") as g:
            grp = g.get("allAnimalResults")
            if grp is None:
                return None
            node = grp.get(channel) or grp.get(_match_key(list(grp.keys()), channel))
            if node is None or "evokedData" not in node or "timeAxis" not in node:
                return None
            t = np.asarray(node["timeAxis"][()], float).ravel()
            c = np.where((t >= win[0]) & (t <= win[1]))[0]
            if c.size < 2:
                return None
            c0, c1 = int(c[0]), int(c[-1]) + 1
            tr = np.asarray(node["evokedData"][:, c0:c1], float)
    except (OSError, KeyError, ValueError, TypeError):
        return None
    return tr, t[c0:c1]


def _collect_file(fp, animal, channel, gains, imp, metrics, window):
    """(epoch_arrays, recording_row) for one file. Metrics RE-MEASURED over *window*
    per epoch (gain-corrected amplitudes); each epoch carries its recording's Ra +
    Z_ss (broadcast). The recording row also holds the gain-corrected MEAN evoked
    trace over the window (for the grouped-trace overlay)."""
    dt = _file_dt(fp)
    if dt is None:
        return None, None
    wt = _win_traces(fp, channel, window)
    if wt is None:
        return None, None
    tr, tw = wt
    n = tr.shape[0]
    if n < 2:
        return None, None
    g = _file_gain(gains, fp, animal, channel)
    gd = g if (g and g > 0) else np.nan
    ra, zss = imp.get(_norm_chan_free(_raw_base(fp)), (np.nan, np.nan))
    feats = _ef.compute_all(_ef.trial_moving_average(tr), tw, _FS, cfg=_win_cfg(window))
    ep = {m: (np.asarray(feats.get(m), float) / gd
              if (m in _AMPL_METRICS and np.isfinite(gd))
              else np.asarray(feats.get(m), float)) for m in metrics}
    good = np.ones(n, bool)
    for m in metrics:
        good = good & np.isfinite(ep[m])
    ep = {m: v[good] for m, v in ep.items()}
    if not good.any():
        return None, None
    ep["ra"] = np.full(int(good.sum()), ra)
    ep["zss"] = np.full(int(good.sum()), zss)
    mean_tr = np.nanmean(tr, axis=0)
    rec = {"dt": dt, "ra": ra, "zss": zss, "tw": tw,
           "mean_trace": (mean_tr / gd if np.isfinite(gd) else mean_tr)}
    for m in metrics:
        rec[m] = float(np.median(ep[m])) if ep[m].size else np.nan
    return ep, rec


def _list_files(animal, evoked_dir, days, since=None) -> list:
    """Sorted evoked .mat files for *animal*; optionally last *days* and/or >= *since*."""
    import glob
    files = glob.glob(os.path.join(evoked_dir, f"*{animal}*_evoked.mat"))
    dated = [(dt, f) for dt, f in ((_file_dt(f), f) for f in files) if dt is not None]
    dated.sort(key=lambda x: x[0])
    if days and dated:
        cutoff = dated[-1][0] - timedelta(days=float(days))
        dated = [(dt, f) for dt, f in dated if dt >= cutoff]
    if since is not None:
        dated = [(dt, f) for dt, f in dated if dt >= since]
    return [f for _dt, f in dated]


def _pick_channel(files, animal, override):
    """Primary channel: *override*, else the most common sidecar channel carrying
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
def correlate_epochs(epochs, metrics) -> dict:
    """Spearman rho of each metric vs Ra and vs Z_ss, per epoch (predictor broadcast)."""
    out: dict = {}
    for pk, _lab in _PREDS:
        x = epochs.get(pk, np.array([]))
        out[pk] = {m: _spearman(x, epochs.get(m, np.array([]))) for m in metrics}
    return out


# --------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------- #
def _fig_heatmap(corr, metrics, meta, out) -> str:
    """Heatmap: metric x {Ra, Z_ss} Spearman rho (rho + n annotated)."""
    cols = _PREDS
    order = sorted(metrics, key=lambda m: -np.nanmax(
        [abs(corr[k][m][0]) for k, _l in cols] or [0]))
    Z = np.array([[corr[k][m][0] for k, _l in cols] for m in order])
    fig, ax = plt.subplots(figsize=(5.8, 0.55 * len(order) + 2.2), facecolor=_ed._BG)
    ax.set_facecolor(_ed._PANEL)
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([l for _k, l in cols], fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_pretty(m) for m in order], fontsize=9)
    ax.tick_params(colors=_ed._MUTED)
    for r, m in enumerate(order):
        for cc, (k, _l) in enumerate(cols):
            rho, n = corr[k][m]
            txt = f"{rho:+.2f}\nn={n}" if np.isfinite(rho) else "n/a"
            ax.text(cc, r, txt, ha="center", va="center", fontsize=7,
                    color="#f0f0f5" if (np.isfinite(rho) and abs(rho) > 0.5) else "#111")
    cb = fig.colorbar(im, ax=ax, fraction=0.06)
    cb.set_label("Spearman ρ", color=_ed._MUTED)
    cb.ax.tick_params(colors=_ed._MUTED)
    ax.set_title(f"{meta['animal']} · {meta['channel']} · {meta['window']} evoked ↔ "
                 f"LFP-artifact impedance (Spearman ρ)\n{meta['span']}",
                 color=_ed._TEXT, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=_ed._BG)
    plt.close(fig)
    return out


def _fig_pred_hexbin(epochs, metrics, pred_key, pred_label, meta, out) -> str:
    """Per-epoch density: each evoked metric (z) vs the predictor (raw kΩ)."""
    x = epochs.get(pred_key, np.array([]))
    ncol = 4
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.0 * nrow),
                             facecolor=_ed._BG, squeeze=False)
    for i, m in enumerate(metrics):
        ax = axes[i // ncol][i % ncol]
        ax.set_facecolor(_ed._PANEL)
        y = _robust_z(epochs.get(m, np.array([])))
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum():
            ax.hexbin(x[ok], y[ok], gridsize=45, cmap="viridis", mincnt=1, bins="log")
        rho, n = _spearman(x, epochs.get(m, np.array([])))
        ax.set_title(f"{_pretty(m)}  ρ={rho:+.2f}" if np.isfinite(rho)
                     else f"{_pretty(m)}  ρ=n/a", color=_ed._TEXT, fontsize=9, loc="left")
        ax.set_xlabel(pred_label, color=_ed._MUTED, fontsize=8)
        ax.set_ylabel("z(metric)", color=_ed._MUTED, fontsize=8)
        ax.tick_params(colors=_ed._MUTED, labelsize=7)
    for j in range(len(metrics), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{meta['animal']} · {meta['channel']} · {meta['window']} evoked "
                 f"metric (z) vs {pred_label} · {meta['span']} · {x.size} epochs",
                 color=_ed._TEXT, fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.09,
                        hspace=0.55, wspace=0.32)
    fig.savefig(out, dpi=125, facecolor=_ed._BG)
    plt.close(fig)
    return out


def _fig_trace_by_pred(recs, pred_key, pred_label, meta, out, n_bins=6) -> str | None:
    """Mean + overlaid example evoked traces GROUPED by the predictor: recordings
    binned into *n_bins* quantiles of Ra (or Z_ss); each bin's per-recording mean
    traces drawn faint with the bin mean bold, turbo low->high. Makes a real coupling
    (bigger/shifted evoked response at higher Ra / Z_ss) visible in the raw waveforms."""
    rr = [r for r in recs if np.isfinite(r.get(pred_key, np.nan))
          and r.get("mean_trace") is not None]
    if len(rr) < n_bins:
        return None
    tw = rr[0]["tw"]
    vals = np.array([r[pred_key] for r in rr])
    edges = np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1))
    which = np.clip(np.digitize(vals, edges[1:-1]), 0, n_bins - 1)
    colors = plt.cm.turbo(np.linspace(0.12, 0.92, n_bins))
    fig, ax = plt.subplots(figsize=(9.5, 5.6), facecolor=_ed._BG)
    ax.set_facecolor(_ed._PANEL)
    for b in range(n_bins):
        grp = [rr[i]["mean_trace"] for i in range(len(rr)) if which[i] == b]
        if not grp:
            continue
        for tr in grp:                                # faint per-recording examples
            ax.plot(tw, tr, color=colors[b], lw=0.4, alpha=0.14)
        ax.plot(tw, np.nanmean(grp, axis=0), color=colors[b], lw=2.4,  # bold bin mean
                label=f"{edges[b]:.2f}–{edges[b + 1]:.2f} (n={len(grp)})")
    ax.set_xlim(tw[0], tw[-1])
    ax.set_xlabel("ms since stim", color=_ed._TEXT, fontsize=10)
    ax.set_ylabel("evoked response (gain-corrected)", color=_ed._TEXT, fontsize=10)
    ax.tick_params(colors=_ed._MUTED, labelsize=8)
    leg = ax.legend(title=pred_label, fontsize=8, frameon=False, loc="upper right")
    leg.get_title().set_color(_ed._TEXT)
    for t in leg.get_texts():
        t.set_color(_ed._TEXT)
    ax.set_title(f"{meta['animal']} · {meta['channel']} · {meta['window']} evoked "
                 f"response grouped by {pred_label} · {meta['span']}",
                 color=_ed._TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=_ed._BG)
    plt.close(fig)
    return out


# --------------------------------------------------------------------- #
#  build
# --------------------------------------------------------------------- #
def _accumulate(files, animal, channel, gains, imp, window, progress) -> tuple:
    """One pass: per-EPOCH arrays (metrics + broadcast Ra/Z_ss) and per-recording rows
    (Ra, Z_ss, mean trace). Returns (epochs concat arrays, recs)."""
    keys = ("ra", "zss", *_METRICS)
    ep_all = {k: [] for k in keys}
    recs = []
    for i, fp in enumerate(files):
        if progress and i % 25 == 0:
            progress(f"recordings… ({i}/{len(files)})")
        ep, rec = _collect_file(fp, animal, channel, gains, imp, _METRICS, window)
        if rec is not None:
            recs.append(rec)
        if ep is not None:
            for k in keys:
                ep_all[k].append(ep[k])
    epochs = {k: (np.concatenate(v) if v else np.array([])) for k, v in ep_all.items()}
    return epochs, recs


def build_stim_corr(animal, evoked_dir, db_path, out_dir, *, days=None, since=None,
                    window=DEFAULT_WINDOW, channel_override=None, config=None,
                    gains=None, progress=None) -> dict:
    """Correlate the LFP-artifact impedances (Ra, Z_ss) against the evoked metrics
    RE-MEASURED over *window* (default 2-50 ms), per epoch, then render the heatmap,
    per-predictor density, and grouped-trace overlays. *since* restricts by date."""
    assert animal and evoked_dir, "animal and evoked_dir required"
    if gains is None and config is not None:
        if progress:
            progress("loading amplifier gains…")
        gains = load_amplifier_gains(config)
    imp = _imp_map(db_path, animal)
    files = _list_files(animal, evoked_dir, days, since=since)
    if not files:
        return {"animal": animal, "empty": True, "reason": "no recordings"}
    channel = _pick_channel(files, animal, channel_override)
    if channel is None:
        return {"animal": animal, "empty": True, "reason": "no channel"}
    wtag = _win_tag(window)
    if progress:
        progress(f"channel {channel}; {wtag} window; scanning {len(files)} "
                 f"recordings (per-epoch, re-measured)…")
    epochs, recs = _accumulate(files, animal, channel, gains, imp, window, progress)
    if not recs:
        return {"animal": animal, "empty": True, "reason": "no matched responses"}
    n_ep = int(epochs["ra"].size)
    n_ra = int(np.sum([np.isfinite(r["ra"]) for r in recs]))
    n_z = int(np.sum([np.isfinite(r["zss"]) for r in recs]))
    if progress:
        progress(f"{len(recs)} recordings, {n_ep} epochs ({n_ra} with Ra, {n_z} with "
                 f"Z_ss); rendering…")
    span = f"{recs[0]['dt']:%Y-%m-%d} … {recs[-1]['dt']:%Y-%m-%d}"
    meta = {"animal": animal, "channel": channel, "span": span, "window": wtag}
    os.makedirs(out_dir, exist_ok=True)
    corr = correlate_epochs(epochs, _METRICS)
    heat = _fig_heatmap(corr, _METRICS, meta, os.path.join(out_dir, "corr_heatmap.png"))
    scatters, overlays = {}, {}
    for k, label in _PREDS:
        scatters[k] = _fig_pred_hexbin(epochs, _METRICS, k, label, meta,
                                       os.path.join(out_dir, f"scatter_{k}.png"))
        overlays[k] = _fig_trace_by_pred(recs, k, label, meta,
                                         os.path.join(out_dir, f"traces_by_{k}.png"))
    return {"animal": animal, "channel": channel, "window": wtag,
            "n_recordings": len(recs), "n_epochs": n_ep, "n_ra": n_ra, "n_zss": n_z,
            "span": span, "heatmap": heat, "scatters": scatters, "overlays": overlays,
            "corr": corr, "out_dir": out_dir}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", default="BCH111")
    ap.add_argument("--days", type=float, default=None, help="last N days (default all)")
    ap.add_argument("--since", default=None, help="only on/after YYYY-MM-DD")
    ap.add_argument("--window", default="2-50", help="evoked window ms, e.g. 2-50")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import yaml
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    evoked_dir = cfg["chronic_evoked"]["evoked_output_dir"]
    db_path = (cfg.get("database", {}) or {}).get("path") or "data/monitor.db"
    out = a.out or os.path.join("data", "evoked_stim_corr", a.animal)
    since = datetime.fromisoformat(a.since) if a.since else None
    window = tuple(float(x) for x in a.window.split("-"))
    res = build_stim_corr(a.animal, evoked_dir, db_path, out, days=a.days, since=since,
                          window=window, channel_override=a.channel, config=cfg,
                          progress=lambda m: print("  ", m, flush=True))
    print("RESULT:", {k: v for k, v in res.items() if k != "corr"})
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
