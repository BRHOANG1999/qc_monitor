"""Peri-ictal pHFO-RATE trend: does the pHFO event rate rise in the hours
before a seizure (a pre-ictal biomarker)?

For a seizure, gather the continuous recordings over a long pre-onset horizon
(hours), detect pHFO events in each, and bin them into a RATE vs time-to-onset.
Two rate metrics per bin (the user wanted both): the HF-envelope candidate rate
(cheaper, HF-band-specific) and the matched-filter detection rate (the
template-correlated events). Both EXCLUDE +/-250 ms around each stim, so
stim-evoked responses never inflate the rate; bins with no recording coverage are
NaN (a gap is not a zero).

Heavy: reads many multi-GB recordings. Detections are cached per recording (so
overlapping pre-onset windows across seizures are detected once), and the chunk
cache is cleared after each recording so a serial sweep never accumulates the
multi-GB signal matrices. Run one seizure per worker (``_main``); aggregate the
per-seizure trajectory CSVs into individual + averaged figures with ``aggregate``.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from src.preictal.isi import parse_chunk_datetime
from src.riding_event import detect as _det
from src.riding_event import primitives as _prim
from src.utils import chunk_cache as _cc

logger = logging.getLogger("qc_monitor.riding_event.periictal")

_MAX_RECS = 200          # NASA Rule 2: bound the per-seizure recording scan.
_SEC_H = 3600.0
_PHFO_FRAC = _prim.DEFAULT_PHFO_FRAC     # pHFO gate: min HF-band energy fraction
_PHFO_PROM = _prim.DEFAULT_PHFO_PROM     # pHFO gate: min HF-envelope prominence

# Bright qualitative palette (reads on the #26263a panel); one colour per seizure.
_SZ_COLORS = ("#5e7ce2", "#2dd4bf", "#f4d35e", "#c084fc", "#4ade80",
              "#f472b6", "#38bdf8", "#fb923c", "#a3e635", "#e5484d")


def _sz_label(traj: dict) -> str:
    """Legend label for a seizure: local onset time + Racine grade."""
    import datetime
    dt = datetime.datetime.fromtimestamp(float(traj["onset_epoch"]))
    rac = traj.get("racine")
    tag = ""
    if rac not in (None, ""):
        try:
            tag = f"  ·  R{int(rac)}"
        except (TypeError, ValueError):
            tag = f"  ·  R{rac}"
    return dt.strftime("%b %d  %H:%M") + tag


def _root(animal: str) -> str:
    return os.path.join("data", "derivatives", "riding_event", animal)


def ensure_template(store, animal: str, *, recompute: bool = False) -> dict:
    """Build + cache the pHFO template (stim-locked, from a chronicStim
    recording via Prong A). Returns ``{template, fs, band}``. Cached to
    ``<root>/ripple_template.npz`` (legacy filename) so every worker loads it,
    not rebuilds it."""
    from src.riding_event import residual as _res
    from src.riding_event import run as _run
    path = os.path.join(_root(animal), "ripple_template.npz")
    if os.path.exists(path) and not recompute:
        d = np.load(path)
        return {"template": d["template"], "fs": float(d["fs"]),
                "band": (float(d["band"][0]), float(d["band"][1]))}
    recs = _run.discover(store, animal)
    ranked = _run.autopick(store, recs, animal, scan_files=12)
    top = ranked[0] if ranked else None
    assert top and top.get("evoked_path"), "no recording to build a template from"
    res = _res.analyze_recording(top["evoked_path"], animal)
    assert res is not None, "template recording has no usable channel"
    tmpl = _det.harvest_from_prongA(res)
    assert tmpl and tmpl.get("template") is not None, "could not build a template"
    band = _run._detect_band(None, res)
    os.makedirs(_root(animal), exist_ok=True)
    np.savez(path, template=tmpl["template"], fs=res["fs"],
             band=np.asarray(band, dtype=float))
    return {"template": tmpl["template"], "fs": float(res["fs"]), "band": band}


def animal_recordings(store, animal: str) -> list[dict]:
    """[{file_id, file_path, start_epoch, duration}] for every recording that
    carries an *animal* evoked channel, chronological. duration from
    processed_files (falls back to the nominal file span)."""
    conn = store._connect()
    try:
        rows = conn.execute(
            """SELECT DISTINCT pf.id AS id, pf.file_path AS fp,
                      pf.chunk_datetime AS dt, pf.duration_sec AS dur
               FROM evoked_waveforms ew JOIN processed_files pf
                 ON pf.id = ew.file_id
               WHERE ew.channel_name LIKE ? AND pf.chunk_datetime IS NOT NULL""",
            (f"{animal}%",)).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        dt = parse_chunk_datetime(r["dt"])         # from processed_files, NOT the
        if dt is None:                             # raw path (no _evoked.mat suffix)
            continue
        dur = float(r["dur"]) if r["dur"] else 3600.0
        out.append({"file_id": int(r["id"]), "file_path": r["fp"],
                    "start_epoch": dt.timestamp(), "duration": dur})
    out.sort(key=lambda d: d["start_epoch"])
    return out


def _det_cache_paths(animal: str, file_id: int) -> dict:
    d = os.path.join(_root(animal), "det_cache")
    return {"cand": os.path.join(d, f"{file_id}_cand.npy"),
            "mf": os.path.join(d, f"{file_id}_mf.npy"),
            "frac": os.path.join(d, f"{file_id}_frac.npy"),
            "prom": os.path.join(d, f"{file_id}_prom.npy")}


def detect_recording(store, animal: str, rec: dict, tmpl: dict, *,
                     thresh: float = 0.7) -> dict:
    """One recording's detections as absolute epochs: HF-envelope ``candidates``,
    matched-filter ``mf`` detections, and per-mf-detection pHFO metrics (``frac``,
    ``prom`` from ``primitives.phfo_metrics``) so a pHFO gate can reject LFDs
    (low-frequency deflections) later WITHOUT re-reading. Cached per file_id (all
    four arrays); the chunk cache is cleared after the read so memory doesn't
    accumulate across a serial sweep."""
    p = _det_cache_paths(animal, rec["file_id"])
    if all(os.path.exists(p[k]) for k in p):
        return {"cand": np.load(p["cand"]), "mf": np.load(p["mf"]),
                "frac": np.load(p["frac"]), "prom": np.load(p["prom"])}
    stim_times = _recording_stim_times(store, animal, rec["file_id"])
    loaded = _det.load_channel(rec["file_path"], animal)
    if loaded is None:
        return _save_det(p, np.empty(0), np.empty(0), np.empty(0), np.empty(0))
    signal, fs, _ch = loaded
    try:
        d = _det.run_detector(signal, fs, band=tmpl["band"],
                              template_override=tmpl["template"],
                              exclude_times_sec=stim_times, thresh=thresh)
        det_locs = np.asarray(d["det_locs"], dtype=np.int64)
        from src.riding_event import primitives as _p
        frac, prom = _p.phfo_metrics(signal, det_locs, fs, tmpl["band"])
    finally:
        del signal
        _cc.clear()                                # free the multi-GB chunk
    s0 = rec["start_epoch"]
    cand = s0 + np.asarray(d["cand_locs"], dtype=np.float64) / fs
    mf = s0 + det_locs.astype(np.float64) / fs
    return _save_det(p, cand, mf, frac, prom)


def _save_det(p: dict, cand, mf, frac, prom) -> dict:
    os.makedirs(os.path.dirname(p["cand"]), exist_ok=True)
    np.save(p["cand"], cand)
    np.save(p["mf"], mf)
    np.save(p["frac"], frac)
    np.save(p["prom"], prom)
    return {"cand": cand, "mf": mf, "frac": frac, "prom": prom}


def _recording_stim_times(store, animal: str, file_id: int):
    """The recording's own stim times (seconds into it) from its evoked .mat, for
    blanking. None when unavailable."""
    from src.utils.animal import is_animal_channel, split_animal_electrode
    from src.utils.evoked_output import read_file_evoked
    ep = store.evoked_output_path_for_file(int(file_id))
    if not (ep and os.path.exists(ep)):
        return None
    try:
        chans = read_file_evoked(ep, only_animals=[animal])
    except Exception:                              # noqa: BLE001
        return None
    for ch, rec in chans.items():
        if split_animal_electrode(ch)[0] == animal and is_animal_channel(ch):
            return rec.get("times")
    return None


def seizure_trajectory(store, animal: str, onset_epoch: float, tmpl: dict, *,
                       horizon_h: float = 6.0, post_h: float = 3.0,
                       bin_min: float = 10.0, thresh: float = 0.7,
                       phfo_frac: float = _PHFO_FRAC, phfo_prom: float = _PHFO_PROM,
                       progress=None) -> dict:
    """pHFO-rate trajectory for one seizure: bin the candidate, matched-filter, and
    pHFO-CONFIRMED matched-filter detections over ``[onset - horizon_h,
    onset + post_h]`` by time-to-onset. A detection is a pHFO when its HF fraction
    >= *phfo_frac* AND HF-envelope prominence >= *phfo_prom* (rejects LFDs /
    low-frequency deflections). Per-bin counts + covered seconds + rates
    (events/min), NaN where uncovered. Recordings cached (thresholds re-tunable
    without re-reading)."""
    prog = progress or (lambda *_a: None)
    t_lo = onset_epoch - horizon_h * _SEC_H
    t_hi = onset_epoch + post_h * _SEC_H
    recs = [r for r in animal_recordings(store, animal)
            if r["start_epoch"] < t_hi and r["start_epoch"] + r["duration"] > t_lo]
    assert len(recs) < _MAX_RECS, "too many recordings in the horizon"
    edges = np.arange(-horizon_h * 60.0, post_h * 60.0 + bin_min, bin_min)
    cand_all, mf_all, phfo_all = [], [], []
    for i, r in enumerate(recs):
        prog(f"seizure@{onset_epoch:.0f}: rec {i+1}/{len(recs)} "
             f"{os.path.basename(r['file_path'])}")
        d = detect_recording(store, animal, r, tmpl, thresh=thresh)
        cand_all.append(d["cand"])
        mf_all.append(d["mf"])
        keep = np.isfinite(d["frac"]) & (d["frac"] >= phfo_frac) \
            & (d["prom"] >= phfo_prom)
        phfo_all.append(d["mf"][keep] if d["mf"].size else np.empty(0))
    cand = np.concatenate(cand_all) if cand_all else np.empty(0)
    mf = np.concatenate(mf_all) if mf_all else np.empty(0)
    phfo = np.concatenate(phfo_all) if phfo_all else np.empty(0)
    cov = _coverage_seconds(recs, onset_epoch, edges)
    cand_ct, _ = np.histogram((cand - onset_epoch) / 60.0, bins=edges)
    mf_ct, _ = np.histogram((mf - onset_epoch) / 60.0, bins=edges)
    phfo_ct, _ = np.histogram((phfo - onset_epoch) / 60.0, bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_of = lambda ct: np.where(cov > 0, ct / (cov / 60.0), np.nan)
        cand_rate, mf_rate, phfo_rate = r_of(cand_ct), r_of(mf_ct), r_of(phfo_ct)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"onset_epoch": onset_epoch, "centers_min": centers,
            "cand_count": cand_ct, "mf_count": mf_ct, "phfo_count": phfo_ct,
            "cover_sec": cov, "cand_rate": cand_rate, "mf_rate": mf_rate,
            "phfo_rate": phfo_rate, "n_recs": len(recs), "horizon_h": horizon_h,
            "bin_min": bin_min}


def _coverage_seconds(recs, onset_epoch, edges) -> np.ndarray:
    """Recording-covered seconds in each time-to-onset bin (minutes edges)."""
    cov = np.zeros(edges.size - 1, dtype=np.float64)
    for r in recs:                                 # bounded by _MAX_RECS
        a = (r["start_epoch"] - onset_epoch) / 60.0
        b = a + r["duration"] / 60.0
        lo = np.maximum(edges[:-1], a)
        hi = np.minimum(edges[1:], b)
        cov += np.maximum(0.0, hi - lo) * 60.0
    return cov


def write_trajectory_csv(traj: dict, path: str, racine=None) -> str:
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["onset_epoch", "racine", "center_min", "cand_count",
                    "mf_count", "phfo_count", "cover_sec", "cand_rate_per_min",
                    "mf_rate_per_min", "phfo_rate_per_min"])
        for i in range(traj["centers_min"].size):
            w.writerow([f"{traj['onset_epoch']:.0f}",
                        "" if racine is None else int(racine),
                        f"{traj['centers_min'][i]:.1f}",
                        int(traj["cand_count"][i]), int(traj["mf_count"][i]),
                        int(traj["phfo_count"][i]), f"{traj['cover_sec'][i]:.1f}",
                        f"{traj['cand_rate'][i]:.5g}", f"{traj['mf_rate'][i]:.5g}",
                        f"{traj['phfo_rate'][i]:.5g}"])
    return path


def load_trajectories(animal: str) -> list[dict]:
    """Every per-seizure trajectory CSV under ``<root>/periictal/``, as dicts with
    ``centers_min`` + ``cand_rate``/``mf_rate``/``phfo_rate`` (NaN where uncovered)
    + ``racine``."""
    import csv
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(_root(animal), "periictal",
                                           "traj_*.csv"))):
        cen, cr, mr, rr, onset, rac = [], [], [], [], None, None
        cc, mc, rc, cov = [], [], [], []           # counts + coverage (for rebin)
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                onset = float(row["onset_epoch"])
                rac = row.get("racine") or None
                cen.append(float(row["center_min"]))
                cr.append(float(row["cand_rate_per_min"] or "nan"))
                mr.append(float(row["mf_rate_per_min"] or "nan"))
                rr.append(float(row.get("phfo_rate_per_min") or "nan"))
                cc.append(float(row.get("cand_count") or 0))
                mc.append(float(row.get("mf_count") or 0))
                rc.append(float(row.get("phfo_count") or 0))
                cov.append(float(row.get("cover_sec") or 0))
        if cen:
            out.append({"onset_epoch": onset, "racine": rac,
                        "centers_min": np.asarray(cen), "cand_rate": np.asarray(cr),
                        "mf_rate": np.asarray(mr), "phfo_rate": np.asarray(rr),
                        "cand_count": np.asarray(cc), "mf_count": np.asarray(mc),
                        "phfo_count": np.asarray(rc), "cover_sec": np.asarray(cov),
                        "path": p})
    return out


def _rebin(traj: dict, coarse_min: float) -> dict:
    """Re-bin a fine-binned trajectory into *coarse_min* windows EXACTLY: counts
    and covered-seconds are additive, so a coarse rate = summed counts / summed
    covered minutes (NaN where no sub-bin had coverage). This is coverage-weighted
    -- averaging the per-bin RATES would mis-weight low-coverage bins. Returns a
    traj-shaped dict on the coarse grid; passes through when already coarse."""
    fc = np.asarray(traj["centers_min"], dtype=np.float64)
    if fc.size < 2 or "cover_sec" not in traj:
        return traj
    w = float(np.median(np.diff(fc)))
    edges = np.concatenate([fc - w / 2.0, [fc[-1] + w / 2.0]])
    cedges = np.arange(float(edges[0]), float(edges[-1]) + coarse_min, coarse_min)
    if cedges.size < 2:
        return traj
    idx = np.clip(np.searchsorted(cedges, fc, side="right") - 1, 0, cedges.size - 2)
    nb = cedges.size - 1
    out = {"onset_epoch": traj["onset_epoch"], "racine": traj.get("racine"),
           "path": traj.get("path"),
           "centers_min": 0.5 * (cedges[:-1] + cedges[1:])}
    cov = np.zeros(nb)
    np.add.at(cov, idx, np.nan_to_num(traj["cover_sec"]))
    out["cover_sec"] = cov
    for ck, rk in (("cand_count", "cand_rate"), ("mf_count", "mf_rate"),
                   ("phfo_count", "phfo_rate")):
        ct = np.zeros(nb)
        np.add.at(ct, idx, np.nan_to_num(traj[ck]))
        out[ck] = ct
        with np.errstate(divide="ignore", invalid="ignore"):
            out[rk] = np.where(cov > 0, ct / (cov / 60.0), np.nan)
    return out


def periictal_stats(trajs: list, key: str = "phfo_rate", *,
                    baseline_h=(6.0, 4.0), last_h: float = 1.0) -> dict:
    """Per-seizure baseline (``baseline_h`` hours before onset) vs last-hour rate
    of *key*, a Wilcoxon paired test that the last hour exceeds baseline, and the
    median fold-change. baseline_h=(6,4) = the -6..-4 h window."""
    ref = trajs[0]["centers_min"] / 60.0
    base_m = (ref <= -baseline_h[1]) & (ref >= -baseline_h[0])
    last_m = ref >= -last_h
    base, last, rac = [], [], []
    for t in trajs:
        y = t.get(key)
        if y is None or y.size != ref.size:
            continue
        b = _safe_mean(y, base_m)
        l = _safe_mean(y, last_m)
        if np.isfinite(b) and np.isfinite(l):
            base.append(b)
            last.append(l)
            rac.append(t.get("racine"))
    base, last = np.asarray(base), np.asarray(last)
    out = {"n": int(base.size), "baseline_mean": float(np.mean(base)) if base.size else np.nan,
           "lasthour_mean": float(np.mean(last)) if last.size else np.nan,
           "median_fold": float(np.median(last / np.where(base > 0, base, np.nan)))
           if base.size else np.nan, "p_wilcoxon": np.nan, "per_racine": {}}
    if base.size >= 5:
        try:
            from scipy.stats import wilcoxon
            out["p_wilcoxon"] = float(wilcoxon(last, base, alternative="greater").pvalue)
        except Exception:                            # noqa: BLE001
            pass
    for g in sorted(set(r for r in rac if r)):
        m = np.array([r == g for r in rac])
        if m.sum():
            out["per_racine"][g] = {"n": int(m.sum()),
                                    "median_fold": float(np.median(
                                        last[m] / np.where(base[m] > 0, base[m], np.nan)))}
    return out


def aggregate(animal: str, out_png: str | None = None, *,
              coarse_min: float = 30.0, min_gap_h: float | None = None,
              trajs_override: list | None = None) -> str | None:
    """Individual + averaged peri-ictal pHFO-rate figure (dark-themed): three
    panels (pHFO rate, the same baseline-normalized, HF-envelope candidates for
    contrast). Each seizure is a distinctly COLOURED trajectory (legend below) so
    you can see which seizures drive the trend, over the across-seizure mean +/-
    SEM (bold white). Bins are averaged into *coarse_min* windows (default 30 min);
    the horizon runs from pre-onset through the post-onset tail. Returns the figure
    path (None when no trajectories)."""
    trajs_raw = (trajs_override if trajs_override is not None
                 else load_trajectories(animal))
    if not trajs_raw:
        return None
    if min_gap_h and min_gap_h > 0:                    # cluster-leaders only
        from src.preictal.isi import leading_mask
        mask = leading_mask([t["onset_epoch"] for t in trajs_raw],
                            min_gap_h * _SEC_H)
        kept = [t for t, k in zip(trajs_raw, mask) if k]
        for t, k in zip(trajs_raw, mask):
            if not k:
                logger.info("aggregate: dropped follower seizure @%.0f (within %.1fh "
                            "of a prior seizure)", t["onset_epoch"], min_gap_h)
        trajs_raw = kept or trajs_raw
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.riding_event import render as _r
    trajs = [_rebin(t, coarse_min) for t in trajs_raw]
    ref = trajs[0]["centers_min"]
    xh = ref / 60.0
    hh = abs(xh[0]) + coarse_min / 120.0           # first bin's LEADING edge, in h
    hp = xh[-1] + coarse_min / 120.0               # last bin's trailing edge, in h
    out_png = out_png or os.path.join(_root(animal), "periictal",
                                      f"{animal}_periictal_phfo_rate.png")
    stats = periictal_stats(trajs, "phfo_rate")
    _write_stats(animal, trajs)
    _print_per_seizure_folds(trajs, "phfo_rate")
    colors = _SZ_COLORS
    fig, ax4 = plt.subplots(4, 1, figsize=(11.0, 12.8), facecolor=_r._BG,
                            gridspec_kw={"height_ratios": [1, 1, 1, 0.34]})
    axes, lax = ax4[:3], ax4[3]
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    lax.axis("off")
    handles = _agg_panel(axes[0], trajs, "phfo_rate", xh, "pHFO matched-filter "
                         "(LFDs rejected)", _r, "events / min", colors=colors)
    st = (f"last hour vs -6..-4 h baseline:  {stats['median_fold']:.2f}× "
          f"(p={stats['p_wilcoxon']:.3g}, n={stats['n']})")
    axes[0].text(0.02, 0.84, st, transform=axes[0].transAxes, color="#ffd166",
                 fontsize=10, fontweight="bold", va="top")
    _agg_panel(axes[1], trajs, "phfo_rate", xh, "pHFO rate, baseline-normalized "
               "(each ÷ its -6..-4 h mean)", _r, "× baseline",
               normalize=True, colors=colors)
    axes[1].axhline(1.0, color=_r._MUTED, lw=0.8, ls=":")
    _agg_panel(axes[2], trajs, "cand_rate", xh, "HF-envelope candidates "
               "(non-specific, for contrast)", _r, "events / min", colors=colors)
    axes[2].set_xlabel("hours to seizure onset (0 = onset)", color=_r._TEXT,
                       fontsize=10)
    for a in (axes[0], axes[1]):
        plt.setp(a.get_xticklabels(), visible=False)
    valid = [(h, _sz_label(t)) for h, t in zip(handles, trajs) if h is not None]
    leg = lax.legend([h for h, _ in valid], [l for _, l in valid], loc="center",
                     ncol=min(4, max(1, len(valid))), frameon=False, fontsize=8.5,
                     title=f"seizure onset time  ·  Racine  ({len(valid)} seizures, "
                     f"{int(coarse_min)}-min bins)", handlelength=1.8,
                     columnspacing=1.6)
    if leg:
        leg.get_title().set_color(_r._TEXT)
        for tx in leg.get_texts():
            tx.set_color(_r._TEXT)
    fig.suptitle(f"{animal} — peri-ictal pHFO rate, {hh:.0f} h before to "
                 f"{hp:.0f} h after onset ({len(trajs)} seizures, "
                 f"{int(coarse_min)}-min bins)",
                 color=_r._TEXT, fontsize=12, x=0.02, ha="left")
    return _r._finish(fig, out_png)


def _safe_mean(y, mask) -> float:
    """np.nanmean over *mask* without the all-NaN 'empty slice' warning."""
    v = np.asarray(y)[mask]
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan


def _sz_label_ascii(traj: dict) -> str:
    """ASCII-only seizure label for the console (no mojibake on a cp1252 shell)."""
    import datetime
    dt = datetime.datetime.fromtimestamp(float(traj["onset_epoch"]))
    rac = traj.get("racine")
    tag = f" R{rac}" if rac not in (None, "") else ""
    return dt.strftime("%b %d %H:%M") + tag


def _print_per_seizure_folds(trajs: list, key: str = "phfo_rate") -> None:
    """Progress readout: each seizure's last-hour / (-6..-4 h) baseline fold, so a
    'not significant, but a few seizures fire pHFOs often' pattern is visible."""
    ref = trajs[0]["centers_min"] / 60.0
    base_m = (ref <= -4.0) & (ref >= -6.0)
    last_m = ref >= -1.0
    rows = []
    for t in trajs:
        y = t.get(key)
        if y is None or y.size != ref.size:
            continue
        b = _safe_mean(y, base_m)
        l = _safe_mean(y, last_m)
        fold = l / b if (np.isfinite(b) and b > 0) else np.nan
        rows.append((_sz_label_ascii(t), b, l, fold))
    rows.sort(key=lambda r: (-r[3] if np.isfinite(r[3]) else 1e9))
    print(f"per-seizure last-hour vs baseline fold ({key}):", flush=True)
    for lab, b, l, fold in rows:
        fs = f"{fold:5.2f}x" if np.isfinite(fold) else "  n/a"
        print(f"  {lab:>18}   base={b:6.2f}  last={l:6.2f}  fold={fs}", flush=True)


def _baseline(y, xh) -> float:
    m = (xh <= -4.0) & (xh >= -6.0)
    b = _safe_mean(y, m) if m.any() else np.nan
    return b if (np.isfinite(b) and b > 0) else np.nan


def _agg_panel(ax, trajs, key, xh, title, _r, ylabel, *, normalize=False,
               colors=None) -> list:
    """Draw one panel: each seizure a distinctly coloured line (+ dot markers),
    the across-seizure mean +/- SEM in bold white on top. Returns the per-seizure
    line handles (index-aligned to *trajs*, None where a seizure had no data on
    this grid) so the caller can build one shared legend."""
    import warnings
    from matplotlib import patheffects as pe
    ax.set_facecolor(_r._PANEL)
    ax.tick_params(colors=_r._MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_r._SPINE)
    ax.grid(True, alpha=0.15, color=_r._MUTED)
    mat = np.full((len(trajs), xh.size), np.nan)
    handles = []
    for i, t in enumerate(trajs):
        y = t.get(key)
        if y is None or y.size != xh.size:
            handles.append(None)
            continue
        if normalize:
            b = _baseline(y, xh)
            y = y / b if np.isfinite(b) else np.full_like(y, np.nan)
        c = colors[i % len(colors)] if colors else _r._ACCENT
        ln, = ax.plot(xh, y, color=c, lw=1.3, alpha=0.9, marker="o",
                      markersize=3.0, solid_capstyle="round")
        handles.append(ln)
        mat[i] = y
    n = np.sum(np.isfinite(mat), axis=0)
    with warnings.catch_warnings():                # all-NaN (gap) columns warn
        warnings.simplefilter("ignore")
        mean = np.where(n >= 1, np.nanmean(mat, axis=0), np.nan)
        sd = np.where(n >= 2, np.nanstd(mat, axis=0), np.nan)
    sem = np.where(n >= 2, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    ok = n >= 2
    ax.fill_between(xh, np.where(ok, mean - sem, np.nan),
                    np.where(ok, mean + sem, np.nan), color=_r._MUTED, alpha=0.22)
    ax.plot(xh, np.where(n >= 1, mean, np.nan), color="#ffffff", lw=2.8, zorder=6,
            path_effects=[pe.Stroke(linewidth=4.6, foreground=_r._BG), pe.Normal()],
            label=f"mean ± SEM (n≤{len(trajs)})")
    ax.axvline(0.0, color="#ff3b3b", lw=1.4, zorder=5, label="seizure onset")
    ax.set_title(title, color=_r._TEXT, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=_r._TEXT, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for tx in leg.get_texts():
        tx.set_color(_r._TEXT)
    return handles


def _write_stats(animal: str, trajs: list) -> str:
    """Per-metric baseline-vs-last-hour stats to a CSV next to the figure."""
    import csv
    path = os.path.join(_root(animal), "periictal", f"{animal}_periictal_stats.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "baseline_mean_per_min", "lasthour_mean_per_min",
                    "median_fold", "p_wilcoxon_greater", "per_racine"])
        for key in ("phfo_rate", "mf_rate", "cand_rate"):
            s = periictal_stats(trajs, key)
            w.writerow([key, s["n"], f"{s['baseline_mean']:.4g}",
                        f"{s['lasthour_mean']:.4g}", f"{s['median_fold']:.3g}",
                        f"{s['p_wilcoxon']:.3g}", s["per_racine"]])
    return path


# ---------------------------------------------------------------------------
# Camp 2: pHFOs in the 0-100 ms post-stim window (from the evoked .mat), scored by
# the SAME ABSOLUTE gate as camp 1 (frac + prominence per epoch), NOT a percentile.
# The complement to camp 1 (all pHFO events, LFDs rejected, from the continuous
# LFP). Cached per file_id so the union of horizons is read once.
# ---------------------------------------------------------------------------
_EVK_WIN_MS = (2.0, 100.0)      # post-artifact .. 100 ms post-stim (the riding event)


def _evk_cache_paths(animal: str, file_id: int) -> dict:
    d = os.path.join(_root(animal), "evk_cache")     # "p" suffix = pHFO-gated (the
    return {"flag": os.path.join(d, f"{file_id}_pflag.npy"),   # absolute frac+prom
            "stim": os.path.join(d, f"{file_id}_pstim.npy")}   # gate, not percentile)


def evoked_riding_recording(store, animal: str, rec: dict, *,
                            win_ms=_EVK_WIN_MS, band=None) -> tuple:
    """Absolute-epoch times of (flagged, all) stimuli for ONE recording: a stimulus
    is 'flagged' when its 0-100 ms post-stim window carries a pHFO by the SAME
    ABSOLUTE gate the continuous detector uses (``primitives.phfo_metrics_epochs``:
    frac >= DEFAULT_PHFO_FRAC & prom >= DEFAULT_PHFO_PROM) -- NOT Prong-A's
    per-recording 95th-percentile flag (which pins the flagged fraction ~5% by
    construction and cannot show an absolute trend). Cached per file_id."""
    p = _evk_cache_paths(animal, rec["file_id"])
    if all(os.path.exists(p[k]) for k in p):
        return np.load(p["flag"]), np.load(p["stim"])
    from src.riding_event import residual as _res
    from src.utils.evoked_output import read_file_evoked
    ep = store.evoked_output_path_for_file(int(rec["file_id"]))
    if not (ep and os.path.exists(ep)):
        return _save_evk(p, np.empty(0), np.empty(0))
    if band is None:
        band = ensure_template(store, animal)["band"]
    try:
        chans = read_file_evoked(ep, only_animals=[animal])
        ch = _res.pick_channel(chans, animal)
    except Exception:                              # noqa: BLE001 -- bad file, skip
        ch = None
    if ch is None:
        return _save_evk(p, np.empty(0), np.empty(0))
    r = chans[ch]
    traces = np.asarray(r["traces"], dtype=np.float64)
    time_ms = np.asarray(r["time_ms"], dtype=np.float64)
    times = np.asarray(r.get("times") or [], dtype=np.float64)
    if traces.ndim != 2 or times.size == 0:
        return _save_evk(p, np.empty(0), np.empty(0))
    fs = 1000.0 / float(np.mean(np.diff(time_ms)))
    traces = _prim.blank_artifact_epochs(traces, time_ms,
                                         pre_ms=_prim.DEFAULT_ARTIFACT_MS[0],
                                         post_ms=_prim.DEFAULT_ARTIFACT_MS[1])
    m = (time_ms >= float(win_ms[0])) & (time_ms <= float(win_ms[1]))
    frac, prom = _prim.phfo_metrics_epochs(traces[:, m], fs, band)
    keep = (np.isfinite(frac) & (frac >= _PHFO_FRAC) & (prom >= _PHFO_PROM))
    n = int(min(times.size, keep.size))
    s0 = float(rec["start_epoch"])
    stim = s0 + times[:n]
    flag = s0 + times[:n][keep[:n]]
    return _save_evk(p, flag, stim)


def _save_evk(p: dict, flag, stim) -> tuple:
    os.makedirs(os.path.dirname(p["flag"]), exist_ok=True)
    np.save(p["flag"], flag)
    np.save(p["stim"], stim)
    return flag, stim


def seizure_evoked_trajectory(store, animal: str, onset_epoch: float, *,
                              horizon_h: float = 6.0, post_h: float = 0.5,
                              bin_min: float = 10.0, win_ms=_EVK_WIN_MS,
                              progress=None) -> dict:
    """Camp-2 trajectory for one seizure: bin the riding-event stimuli over the
    pre-onset horizon into a RATE (events/min) and a FRACTION (flagged / all
    stimuli). Recordings cached per file_id (evoked .mat), NaN where uncovered."""
    prog = progress or (lambda *_a: None)
    t_lo = onset_epoch - horizon_h * _SEC_H
    t_hi = onset_epoch + post_h * _SEC_H
    recs = [r for r in animal_recordings(store, animal)
            if r["start_epoch"] < t_hi and r["start_epoch"] + r["duration"] > t_lo]
    assert len(recs) < _MAX_RECS, "too many recordings in the horizon"
    edges = np.arange(-horizon_h * 60.0, post_h * 60.0 + bin_min, bin_min)
    flag_all, stim_all = [], []
    for i, r in enumerate(recs):
        prog(f"evoked seizure@{onset_epoch:.0f}: rec {i+1}/{len(recs)} "
             f"{os.path.basename(r['file_path'])}")
        flag, stim = evoked_riding_recording(store, animal, r, win_ms=win_ms)
        flag_all.append(flag)
        stim_all.append(stim)
    flag = np.concatenate(flag_all) if flag_all else np.empty(0)
    stim = np.concatenate(stim_all) if stim_all else np.empty(0)
    cov = _coverage_seconds(recs, onset_epoch, edges)
    flag_ct, _ = np.histogram((flag - onset_epoch) / 60.0, bins=edges)
    stim_ct, _ = np.histogram((stim - onset_epoch) / 60.0, bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(cov > 0, flag_ct / (cov / 60.0), np.nan)
        frac = np.where(stim_ct > 0, flag_ct / stim_ct, np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"onset_epoch": onset_epoch, "centers_min": centers,
            "flag_count": flag_ct, "stim_count": stim_ct, "cover_sec": cov,
            "ride_rate": rate, "ride_frac": frac, "n_recs": len(recs)}


def write_evk_csv(traj: dict, path: str, racine=None) -> str:
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["onset_epoch", "racine", "center_min", "flag_count",
                    "stim_count", "cover_sec", "ride_rate_per_min", "ride_frac"])
        for i in range(traj["centers_min"].size):
            w.writerow([f"{traj['onset_epoch']:.0f}",
                        "" if racine is None else int(racine),
                        f"{traj['centers_min'][i]:.1f}",
                        int(traj["flag_count"][i]), int(traj["stim_count"][i]),
                        f"{traj['cover_sec'][i]:.1f}",
                        f"{traj['ride_rate'][i]:.5g}", f"{traj['ride_frac'][i]:.5g}"])
    return path


def load_evk_trajectories(animal: str) -> list[dict]:
    """Every per-seizure camp-2 CSV under ``<root>/periictal/evk_*.csv``."""
    import csv
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(_root(animal), "periictal",
                                           "evk_*.csv"))):
        cen, rr, rf, onset, rac = [], [], [], None, None
        fc, sc, cov = [], [], []
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                onset = float(row["onset_epoch"])
                rac = row.get("racine") or None
                cen.append(float(row["center_min"]))
                rr.append(float(row["ride_rate_per_min"] or "nan"))
                rf.append(float(row["ride_frac"] or "nan"))
                fc.append(float(row.get("flag_count") or 0))
                sc.append(float(row.get("stim_count") or 0))
                cov.append(float(row.get("cover_sec") or 0))
        if cen:
            out.append({"onset_epoch": onset, "racine": rac,
                        "centers_min": np.asarray(cen), "ride_rate": np.asarray(rr),
                        "ride_frac": np.asarray(rf), "flag_count": np.asarray(fc),
                        "stim_count": np.asarray(sc), "cover_sec": np.asarray(cov),
                        "path": p})
    return out


def _rebin_evk(traj: dict, coarse_min: float) -> dict:
    """Re-bin a camp-2 trajectory into coarse windows: flag/stim counts and
    coverage are additive; rate = flags / covered-min, frac = flags / stimuli."""
    fc = np.asarray(traj["centers_min"], dtype=np.float64)
    if fc.size < 2:
        return traj
    w = float(np.median(np.diff(fc)))
    edges = np.concatenate([fc - w / 2.0, [fc[-1] + w / 2.0]])
    cedges = np.arange(float(edges[0]), float(edges[-1]) + coarse_min, coarse_min)
    if cedges.size < 2:
        return traj
    idx = np.clip(np.searchsorted(cedges, fc, side="right") - 1, 0, cedges.size - 2)
    nb = cedges.size - 1
    flag, stim, cov = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    np.add.at(flag, idx, np.nan_to_num(traj["flag_count"]))
    np.add.at(stim, idx, np.nan_to_num(traj["stim_count"]))
    np.add.at(cov, idx, np.nan_to_num(traj["cover_sec"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(cov > 0, flag / (cov / 60.0), np.nan)
        frac = np.where(stim > 0, flag / stim, np.nan)
    return {"onset_epoch": traj["onset_epoch"], "racine": traj.get("racine"),
            "path": traj.get("path"),
            "centers_min": 0.5 * (cedges[:-1] + cedges[1:]),
            "flag_count": flag, "stim_count": stim, "cover_sec": cov,
            "ride_rate": rate, "ride_frac": frac}


def _one_evk(args) -> tuple:
    """Pool worker: compute + cache one recording's camp-2 detections. Opens its
    own Store so it runs in a separate process."""
    file_id, file_path, start_epoch, animal, config_path, win_ms = args
    import yaml

    from src.db.store import Store
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    store = Store(cfg["database"]["path"])
    rec = {"file_id": file_id, "file_path": file_path, "start_epoch": start_epoch}
    flag, stim = evoked_riding_recording(store, animal, rec, win_ms=win_ms)
    return (file_id, int(np.asarray(flag).size), int(np.asarray(stim).size))


def compute_evoked_camp(animal: str, config_path: str, *, jobs: int = 4,
                        horizon_h: float = 6.0, post_h: float = 0.5,
                        bin_min: float = 10.0, win_ms=_EVK_WIN_MS) -> str | None:
    """Camp 2 for every seizure that already has a camp-1 trajectory: read each
    UNIQUE evoked .mat across the horizons once (bounded process pool, cached per
    file_id), then bin per seizure into evk_*.csv and build the comparison figure."""
    import yaml
    from concurrent.futures import ProcessPoolExecutor

    from src.db.store import Store
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    store = Store(cfg["database"]["path"])
    onsets = [(t["onset_epoch"], t.get("racine")) for t in load_trajectories(animal)]
    assert onsets, "no camp-1 trajectories to mirror -- run camp 1 first"
    recs = animal_recordings(store, animal)
    uniq = {}
    for on, _r in onsets:
        lo, hi = on - horizon_h * _SEC_H, on + post_h * _SEC_H
        for r in recs:
            if r["start_epoch"] < hi and r["start_epoch"] + r["duration"] > lo:
                uniq[r["file_id"]] = r
    todo = [r for r in uniq.values() if not all(
        os.path.exists(x) for x in _evk_cache_paths(animal, r["file_id"]).values())]
    print(f"camp2: {len(uniq)} unique recordings across {len(onsets)} seizures, "
          f"{len(todo)} to read on {min(jobs, max(1, len(todo)))} workers...",
          flush=True)
    tasks = [(r["file_id"], r["file_path"], r["start_epoch"], animal, config_path,
              win_ms) for r in todo]
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, min(int(jobs), len(tasks)))) as ex:
            for i, res in enumerate(ex.map(_one_evk, tasks)):
                print(f"  [{i+1}/{len(tasks)}] file {res[0]}: {res[1]} riding / "
                      f"{res[2]} stimuli", flush=True)
    for on, rac in onsets:                          # bin per seizure (cache: instant)
        traj = seizure_evoked_trajectory(store, animal, on, horizon_h=horizon_h,
                                         post_h=post_h, bin_min=bin_min, win_ms=win_ms)
        out = os.path.join(_root(animal), "periictal", f"evk_{int(on)}.csv")
        write_evk_csv(traj, out, racine=rac)
        print(f"  seizure {int(on)}: {traj['n_recs']} recs -> {out}", flush=True)
    fig = aggregate_camps(animal)
    print(f"camps comparison figure -> {fig}", flush=True)
    return fig


def aggregate_camps(animal: str, out_png: str | None = None, *,
                    coarse_min: float = 30.0) -> str | None:
    """Two-camp comparison figure: camp 1 (all pHFO events, LFDs rejected,
    continuous) vs camp 2 (riding events in the 0-100 ms post-stim window). Panels:
    camp-1 per-seizure rate, camp-2 per-seizure rate, and the two camps' baseline-
    normalized MEANS overlaid with each camp's last-hour-vs-baseline stats."""
    t1, t2 = load_trajectories(animal), load_evk_trajectories(animal)
    if not t1 or not t2:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.riding_event import render as _r
    c1 = [_rebin(t, coarse_min) for t in t1]
    c2 = [_rebin_evk(t, coarse_min) for t in t2]
    xh, xh2 = c1[0]["centers_min"] / 60.0, c2[0]["centers_min"] / 60.0
    out_png = out_png or os.path.join(_root(animal), "periictal",
                                      f"{animal}_periictal_camps.png")
    colors = _SZ_COLORS
    fig, ax4 = plt.subplots(4, 1, figsize=(11.0, 12.8), facecolor=_r._BG,
                            gridspec_kw={"height_ratios": [1, 1, 1, 0.34]})
    axes, lax = ax4[:3], ax4[3]
    lax.axis("off")
    h1 = _agg_panel(axes[0], c1, "phfo_rate", xh, "CAMP 1: all pHFO events "
                    "(continuous LFP, LFDs rejected)", _r, "events / min",
                    colors=colors)
    _agg_panel(axes[1], c2, "ride_rate", xh2, "CAMP 2: riding events in the "
               "0-100 ms window after each stim", _r, "events / min", colors=colors)
    _camp_compare(axes[2], c1, c2, xh, xh2, _r)
    axes[2].set_xlabel("hours to seizure onset (0 = onset)", color=_r._TEXT,
                       fontsize=10)
    for a in (axes[0], axes[1]):
        plt.setp(a.get_xticklabels(), visible=False)
    valid = [(h, _sz_label(t)) for h, t in zip(h1, c1) if h is not None]
    leg = lax.legend([h for h, _ in valid], [l for _, l in valid], loc="center",
                     ncol=min(4, max(1, len(valid))), frameon=False, fontsize=8.5,
                     title=f"seizure onset time  ·  Racine  ({len(valid)} seizures, "
                     f"{int(coarse_min)}-min bins)")
    if leg:
        leg.get_title().set_color(_r._TEXT)
        for tx in leg.get_texts():
            tx.set_color(_r._TEXT)
    fig.suptitle(f"{animal} — two-camp peri-ictal comparison ({len(c1)} seizures, "
                 f"{int(coarse_min)}-min bins)", color=_r._TEXT, fontsize=12,
                 x=0.02, ha="left")
    return _r._finish(fig, out_png)


def _camp_compare(ax, c1, c2, xh, xh2, _r) -> None:
    """Panel 3: the two camps' baseline-normalized across-seizure means overlaid,
    each with its last-hour-vs-baseline fold + Wilcoxon p (which camp trends
    harder / is more significant)."""
    import warnings
    ax.set_facecolor(_r._PANEL)
    ax.tick_params(colors=_r._MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_r._SPINE)
    ax.grid(True, alpha=0.15, color=_r._MUTED)
    ax.axhline(1.0, color=_r._MUTED, lw=0.8, ls=":")
    specs = [("CAMP 1 (all pHFO)", c1, "phfo_rate", xh, _r._ACCENT),
             ("CAMP 2 (0-100 ms post-stim)", c2, "ride_rate", xh2, _r._EVENT)]
    for name, trajs, key, x, col in specs:
        mat = np.full((len(trajs), x.size), np.nan)
        for i, t in enumerate(trajs):
            y = t.get(key)
            if y is None or y.size != x.size:
                continue
            b = _baseline(y, x)
            mat[i] = y / b if np.isfinite(b) else np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n = np.sum(np.isfinite(mat), axis=0)
            mean = np.where(n >= 1, np.nanmean(mat, axis=0), np.nan)
        st = periictal_stats(trajs, key)
        ax.plot(x, mean, color=col, lw=2.6, marker="o", markersize=3.0,
                label=f"{name}: {st['median_fold']:.2f}x  p={st['p_wilcoxon']:.3g}"
                f"  n={st['n']}")
    ax.axvline(0.0, color="#ff3b3b", lw=1.4, label="seizure onset")
    ax.set_title("baseline-normalized comparison (each camp ÷ its -6..-4 h mean)",
                 color=_r._TEXT, fontsize=11, loc="left")
    ax.set_ylabel("× baseline", color=_r._TEXT, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for tx in leg.get_texts():
        tx.set_color(_r._TEXT)


def _one_seizure(onset: float, animal: str, config_path: str, *,
                 horizon_h: float, post_h: float, bin_min: float,
                 thresh: float, phfo_frac: float = _PHFO_FRAC,
                 phfo_prom: float = _PHFO_PROM) -> tuple:
    """Compute + write ONE seizure's trajectory. Opens its OWN Store so it can run
    in a separate process (the pool worker); prints progress with the onset tag."""
    import yaml

    from src.db.store import Store
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    store = Store(cfg["database"]["path"])
    tmpl = ensure_template(store, animal)          # loads the cached template
    traj = seizure_trajectory(store, animal, onset, tmpl, horizon_h=horizon_h,
                              post_h=post_h, bin_min=bin_min, thresh=thresh,
                              phfo_frac=phfo_frac, phfo_prom=phfo_prom,
                              progress=lambda m: print("  ", m, flush=True))
    out = os.path.join(_root(animal), "periictal", f"traj_{int(onset)}.csv")
    write_trajectory_csv(traj, out, racine=_racine_for_onset(store, animal, onset))
    print(f"seizure {int(onset)}: {traj['n_recs']} recordings -> {out}", flush=True)
    return (onset, traj["n_recs"], out)


def _racine_for_onset(store, animal: str, onset: float):
    """Racine grade of the scored seizure at *onset* (nearest within 2 s)."""
    from src.preictal.isi import scored_seizures
    best, bd = None, 2.0
    for z in scored_seizures(store, animal):
        d = abs(z.onset_epoch - onset)
        if d < bd:
            best, bd = z.racine, d
    return best


def run_batch(animal: str, onsets, config_path: str, *, jobs: int = 3,
              horizon_h: float = 6.0, post_h: float = 3.0, bin_min: float = 10.0,
              thresh: float = 0.7, phfo_frac: float = _PHFO_FRAC,
              phfo_prom: float = _PHFO_PROM) -> str | None:
    """Compute several seizures' trajectories in a bounded PROCESS pool (true
    parallelism across seizures; each worker clears the chunk cache after every
    recording so peak memory ~= jobs recordings), then aggregate. Returns the
    aggregate figure path."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(float(o), animal, config_path) for o in onsets]
    n = len(tasks)
    print(f"running {n} seizures on {min(jobs, n)} workers "
          f"(horizon -{horizon_h} h .. +{post_h} h)...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(int(jobs), n))) as ex:
        futs = [ex.submit(_one_seizure, o, a, c, horizon_h=horizon_h,
                          post_h=post_h, bin_min=bin_min, thresh=thresh,
                          phfo_frac=phfo_frac, phfo_prom=phfo_prom)
                for (o, a, c) in tasks]
        for i, fu in enumerate(futs):
            try:
                r = fu.result()
                print(f"[{i+1}/{n}] done: seizure {int(r[0])} ({r[1]} recs)",
                      flush=True)
            except Exception as e:                 # noqa: BLE001 -- one bad seizure
                print(f"[{i+1}/{n}] FAILED: {e}", flush=True)
    fig = aggregate(animal)
    print(f"aggregate figure -> {fig}", flush=True)
    return fig


def _main(argv=None) -> int:
    import argparse

    import yaml

    from src.db.store import Store

    ap = argparse.ArgumentParser(description="Peri-ictal pHFO-rate")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", default="BCH111")
    ap.add_argument("--onset-epoch", type=float, default=None,
                    help="one seizure onset (absolute epoch)")
    ap.add_argument("--onsets", default=None,
                    help="comma-separated onsets -> parallel batch + aggregate")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="just rebuild the aggregate figure(s) from existing CSVs")
    ap.add_argument("--camp2", action="store_true",
                    help="compute camp 2 (pHFO in 0-100 ms post-stim, from the "
                    "evoked .mat) for every seizure + the two-camp comparison")
    ap.add_argument("--horizon-h", type=float, default=6.0)
    ap.add_argument("--post-h", type=float, default=3.0)
    ap.add_argument("--bin-min", type=float, default=10.0)
    ap.add_argument("--thresh", type=float, default=0.7)
    ap.add_argument("--phfo-frac", type=float, default=_PHFO_FRAC,
                    help="pHFO gate: min HF-band energy fraction")
    ap.add_argument("--phfo-prom", type=float, default=_PHFO_PROM,
                    help="pHFO gate: min HF-envelope prominence (reject LFDs)")
    args = ap.parse_args(argv)
    if args.aggregate_only:
        print("aggregate ->", aggregate(args.animal))
        print("camps ->", aggregate_camps(args.animal))
        return 0
    if args.camp2:
        compute_evoked_camp(args.animal, args.config, jobs=args.jobs,
                            horizon_h=args.horizon_h, post_h=args.post_h,
                            bin_min=args.bin_min)
        return 0
    if args.onsets:
        onsets = [float(x) for x in args.onsets.split(",") if x.strip()]
        run_batch(args.animal, onsets, args.config, jobs=args.jobs,
                  horizon_h=args.horizon_h, post_h=args.post_h,
                  bin_min=args.bin_min, thresh=args.thresh,
                  phfo_frac=args.phfo_frac, phfo_prom=args.phfo_prom)
        return 0
    assert args.onset_epoch is not None, "pass --onset-epoch or --onsets"
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    store = Store(cfg["database"]["path"])
    tmpl = ensure_template(store, args.animal)
    traj = seizure_trajectory(store, args.animal, args.onset_epoch, tmpl,
                              horizon_h=args.horizon_h, post_h=args.post_h,
                              bin_min=args.bin_min, thresh=args.thresh,
                              phfo_frac=args.phfo_frac,
                              phfo_prom=args.phfo_prom,
                              progress=lambda m: print("  ", m, flush=True))
    out = os.path.join(_root(args.animal), "periictal",
                       f"traj_{int(args.onset_epoch)}.csv")
    write_trajectory_csv(traj, out,
                         racine=_racine_for_onset(store, args.animal,
                                                  args.onset_epoch))
    print(f"seizure {int(args.onset_epoch)}: {traj['n_recs']} recordings -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
