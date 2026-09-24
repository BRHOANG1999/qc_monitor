"""Peri-ictal ripple-RATE trend: does the ripple event rate rise in the hours
before a seizure (a pre-ictal biomarker)?

For a seizure, gather the continuous recordings over a long pre-onset horizon
(hours), detect ripple events in each, and bin them into a RATE vs time-to-onset.
Two rate metrics per bin (the user wanted both): the HF-envelope candidate rate
(cheaper, ripple-band-specific) and the matched-filter detection rate (the
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
from src.utils import chunk_cache as _cc

logger = logging.getLogger("qc_monitor.riding_event.periictal")

_MAX_RECS = 200          # NASA Rule 2: bound the per-seizure recording scan.
_SEC_H = 3600.0


def _root(animal: str) -> str:
    return os.path.join("data", "derivatives", "riding_event", animal)


def ensure_template(store, animal: str, *, recompute: bool = False) -> dict:
    """Build + cache the ripple template (stim-locked, from a chronicStim
    recording via Prong A). Returns ``{template, fs, band}``. Cached to
    ``<root>/ripple_template.npz`` so every worker loads it, not rebuilds it."""
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
            "sust": os.path.join(d, f"{file_id}_sust.npy")}


def detect_recording(store, animal: str, rec: dict, tmpl: dict, *,
                     thresh: float = 0.7) -> dict:
    """One recording's detections as absolute epochs: HF-envelope ripple
    ``candidates``, matched-filter ``mf`` detections, and per-mf-detection ripple
    metrics (``frac``, ``sust`` from ``primitives.ripple_metrics``) so a ripple
    gate can reject slow waves later WITHOUT re-reading. Cached per file_id (all
    four arrays); the chunk cache is cleared after the read so memory doesn't
    accumulate across a serial sweep."""
    p = _det_cache_paths(animal, rec["file_id"])
    if all(os.path.exists(p[k]) for k in p):
        return {"cand": np.load(p["cand"]), "mf": np.load(p["mf"]),
                "frac": np.load(p["frac"]), "sust": np.load(p["sust"])}
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
        w = int(np.asarray(tmpl["template"]).size)
        from src.riding_event import primitives as _p
        frac, sust = _p.ripple_metrics(signal, det_locs, fs, tmpl["band"], w)
    finally:
        del signal
        _cc.clear()                                # free the multi-GB chunk
    s0 = rec["start_epoch"]
    cand = s0 + np.asarray(d["cand_locs"], dtype=np.float64) / fs
    mf = s0 + det_locs.astype(np.float64) / fs
    return _save_det(p, cand, mf, frac, sust)


def _save_det(p: dict, cand, mf, frac, sust) -> dict:
    os.makedirs(os.path.dirname(p["cand"]), exist_ok=True)
    np.save(p["cand"], cand)
    np.save(p["mf"], mf)
    np.save(p["frac"], frac)
    np.save(p["sust"], sust)
    return {"cand": cand, "mf": mf, "frac": frac, "sust": sust}


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
                       horizon_h: float = 6.0, post_h: float = 0.5,
                       bin_min: float = 10.0, thresh: float = 0.7,
                       ripple_frac: float = 0.5, ripple_sust: float = 0.30,
                       progress=None) -> dict:
    """Ripple-rate trajectory for one seizure: bin the candidate, matched-filter,
    and RIPPLE-CONFIRMED matched-filter detections over ``[onset - horizon_h,
    onset + post_h]`` by time-to-onset. A detection is ripple-confirmed when its
    HF fraction >= *ripple_frac* AND sustained fraction >= *ripple_sust* (rejects
    slow waves / lone transients). Per-bin counts + covered seconds + rates
    (events/min), NaN where uncovered. Recordings cached (thresholds re-tunable
    without re-reading)."""
    prog = progress or (lambda *_a: None)
    t_lo = onset_epoch - horizon_h * _SEC_H
    t_hi = onset_epoch + post_h * _SEC_H
    recs = [r for r in animal_recordings(store, animal)
            if r["start_epoch"] < t_hi and r["start_epoch"] + r["duration"] > t_lo]
    assert len(recs) < _MAX_RECS, "too many recordings in the horizon"
    edges = np.arange(-horizon_h * 60.0, post_h * 60.0 + bin_min, bin_min)
    cand_all, mf_all, rip_all = [], [], []
    for i, r in enumerate(recs):
        prog(f"seizure@{onset_epoch:.0f}: rec {i+1}/{len(recs)} "
             f"{os.path.basename(r['file_path'])}")
        d = detect_recording(store, animal, r, tmpl, thresh=thresh)
        cand_all.append(d["cand"])
        mf_all.append(d["mf"])
        rip = np.isfinite(d["frac"]) & (d["frac"] >= ripple_frac) \
            & (d["sust"] >= ripple_sust)
        rip_all.append(d["mf"][rip] if d["mf"].size else np.empty(0))
    cand = np.concatenate(cand_all) if cand_all else np.empty(0)
    mf = np.concatenate(mf_all) if mf_all else np.empty(0)
    ripe = np.concatenate(rip_all) if rip_all else np.empty(0)
    cov = _coverage_seconds(recs, onset_epoch, edges)
    cand_ct, _ = np.histogram((cand - onset_epoch) / 60.0, bins=edges)
    mf_ct, _ = np.histogram((mf - onset_epoch) / 60.0, bins=edges)
    rip_ct, _ = np.histogram((ripe - onset_epoch) / 60.0, bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_of = lambda ct: np.where(cov > 0, ct / (cov / 60.0), np.nan)
        cand_rate, mf_rate, rip_rate = r_of(cand_ct), r_of(mf_ct), r_of(rip_ct)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"onset_epoch": onset_epoch, "centers_min": centers,
            "cand_count": cand_ct, "mf_count": mf_ct, "rip_count": rip_ct,
            "cover_sec": cov, "cand_rate": cand_rate, "mf_rate": mf_rate,
            "rip_rate": rip_rate, "n_recs": len(recs), "horizon_h": horizon_h,
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
                    "mf_count", "rip_count", "cover_sec", "cand_rate_per_min",
                    "mf_rate_per_min", "rip_rate_per_min"])
        for i in range(traj["centers_min"].size):
            w.writerow([f"{traj['onset_epoch']:.0f}",
                        "" if racine is None else int(racine),
                        f"{traj['centers_min'][i]:.1f}",
                        int(traj["cand_count"][i]), int(traj["mf_count"][i]),
                        int(traj["rip_count"][i]), f"{traj['cover_sec'][i]:.1f}",
                        f"{traj['cand_rate'][i]:.5g}", f"{traj['mf_rate'][i]:.5g}",
                        f"{traj['rip_rate'][i]:.5g}"])
    return path


def load_trajectories(animal: str) -> list[dict]:
    """Every per-seizure trajectory CSV under ``<root>/periictal/``, as dicts with
    ``centers_min`` + ``cand_rate``/``mf_rate``/``rip_rate`` (NaN where uncovered)
    + ``racine``."""
    import csv
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(_root(animal), "periictal",
                                           "traj_*.csv"))):
        cen, cr, mr, rr, onset, rac = [], [], [], [], None, None
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                onset = float(row["onset_epoch"])
                rac = row.get("racine") or None
                cen.append(float(row["center_min"]))
                cr.append(float(row["cand_rate_per_min"] or "nan"))
                mr.append(float(row["mf_rate_per_min"] or "nan"))
                rr.append(float(row.get("rip_rate_per_min") or "nan"))
        if cen:
            out.append({"onset_epoch": onset, "racine": rac,
                        "centers_min": np.asarray(cen), "cand_rate": np.asarray(cr),
                        "mf_rate": np.asarray(mr), "rip_rate": np.asarray(rr),
                        "path": p})
    return out


def periictal_stats(trajs: list, key: str = "rip_rate", *,
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
        b = np.nanmean(y[base_m])
        l = np.nanmean(y[last_m])
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


def aggregate(animal: str, out_png: str | None = None) -> str | None:
    """Individual + averaged peri-ictal ripple-rate figure (dark-themed): two
    panels (HF candidate rate, matched-filter rate), each with every seizure's
    trajectory (faint) + the across-seizure mean +/- SEM (bold), x = hours to
    onset. Returns the figure path (None when no trajectories)."""
    trajs = load_trajectories(animal)
    if not trajs:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.riding_event import render as _r
    ref = trajs[0]["centers_min"]
    xh = ref / 60.0
    out_png = out_png or os.path.join(_root(animal), "periictal",
                                      f"{animal}_periictal_ripplerate.png")
    stats = periictal_stats(trajs, "rip_rate")
    _write_stats(animal, trajs)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.4), facecolor=_r._BG,
                             sharex=True)
    _agg_panel(axes[0], trajs, "rip_rate", xh, "ripple-confirmed matched-filter "
               "(slow waves rejected)", _r, "events / min")
    st = (f"last hour vs -6..-4 h baseline:  {stats['median_fold']:.2f}× "
          f"(p={stats['p_wilcoxon']:.3g}, n={stats['n']})")
    axes[0].text(0.02, 0.94, st, transform=axes[0].transAxes, color="#ffd166",
                 fontsize=10, fontweight="bold", va="top")
    _agg_panel(axes[1], trajs, "rip_rate", xh, "ripple-confirmed, baseline-"
               "normalized (each ÷ its -6..-4 h mean)", _r, "× baseline",
               normalize=True)
    axes[1].axhline(1.0, color=_r._MUTED, lw=0.8, ls=":")
    _agg_panel(axes[2], trajs, "cand_rate", xh, "HF-envelope candidates "
               "(non-specific, for contrast)", _r, "events / min")
    axes[2].set_xlabel("hours to seizure onset (0 = onset)", color=_r._TEXT,
                       fontsize=10)
    fig.suptitle(f"{animal} — peri-ictal ripple rate, {abs(xh[0]):.0f} h before "
                 f"onset ({len(trajs)} seizures)", color=_r._TEXT, fontsize=12,
                 x=0.02, ha="left")
    return _r._finish(fig, out_png)


def _baseline(y, xh) -> float:
    m = (xh <= -4.0) & (xh >= -6.0)
    b = np.nanmean(y[m]) if m.any() else np.nan
    return b if (np.isfinite(b) and b > 0) else np.nan


def _agg_panel(ax, trajs, key, xh, title, _r, ylabel, *, normalize=False) -> None:
    import warnings
    ax.set_facecolor(_r._PANEL)
    ax.tick_params(colors=_r._MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_r._SPINE)
    ax.grid(True, alpha=0.15, color=_r._MUTED)
    mat = np.full((len(trajs), xh.size), np.nan)
    for i, t in enumerate(trajs):
        y = t.get(key)
        if y is None or y.size != xh.size:
            continue
        if normalize:
            b = _baseline(y, xh)
            y = y / b if np.isfinite(b) else np.full_like(y, np.nan)
        ax.plot(xh, y, color=_r._ACCENT, lw=0.7, alpha=0.28)
        mat[i] = y
    n = np.sum(np.isfinite(mat), axis=0)
    with warnings.catch_warnings():                # all-NaN (gap) columns warn
        warnings.simplefilter("ignore")
        mean = np.where(n >= 1, np.nanmean(mat, axis=0), np.nan)
        sd = np.where(n >= 2, np.nanstd(mat, axis=0), np.nan)
    sem = np.where(n >= 2, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    ok = n >= 2
    ax.fill_between(xh, np.where(ok, mean - sem, np.nan),
                    np.where(ok, mean + sem, np.nan), color=_r._EVENT, alpha=0.2)
    ax.plot(xh, np.where(n >= 1, mean, np.nan), color=_r._EVENT, lw=2.4,
            label=f"mean ± SEM (n≤{len(trajs)})")
    ax.axvline(0.0, color="#ff3b3b", lw=1.4, label="seizure onset")
    ax.set_title(title, color=_r._TEXT, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=_r._TEXT, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for tx in leg.get_texts():
        tx.set_color(_r._TEXT)


def _write_stats(animal: str, trajs: list) -> str:
    """Per-metric baseline-vs-last-hour stats to a CSV next to the figure."""
    import csv
    path = os.path.join(_root(animal), "periictal", f"{animal}_periictal_stats.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "baseline_mean_per_min", "lasthour_mean_per_min",
                    "median_fold", "p_wilcoxon_greater", "per_racine"])
        for key in ("rip_rate", "mf_rate", "cand_rate"):
            s = periictal_stats(trajs, key)
            w.writerow([key, s["n"], f"{s['baseline_mean']:.4g}",
                        f"{s['lasthour_mean']:.4g}", f"{s['median_fold']:.3g}",
                        f"{s['p_wilcoxon']:.3g}", s["per_racine"]])
    return path


def _one_seizure(onset: float, animal: str, config_path: str, *,
                 horizon_h: float, post_h: float, bin_min: float,
                 thresh: float) -> tuple:
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
              horizon_h: float = 6.0, post_h: float = 0.5, bin_min: float = 10.0,
              thresh: float = 0.7) -> str | None:
    """Compute several seizures' trajectories in a bounded PROCESS pool (true
    parallelism across seizures; each worker clears the chunk cache after every
    recording so peak memory ~= jobs recordings), then aggregate. Returns the
    aggregate figure path."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(float(o), animal, config_path) for o in onsets]
    n = len(tasks)
    print(f"running {n} seizures on {min(jobs, n)} workers "
          f"(horizon {horizon_h} h)...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(int(jobs), n))) as ex:
        futs = [ex.submit(_one_seizure, o, a, c, horizon_h=horizon_h,
                          post_h=post_h, bin_min=bin_min, thresh=thresh)
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

    ap = argparse.ArgumentParser(description="Peri-ictal ripple-rate")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", default="BCH111")
    ap.add_argument("--onset-epoch", type=float, default=None,
                    help="one seizure onset (absolute epoch)")
    ap.add_argument("--onsets", default=None,
                    help="comma-separated onsets -> parallel batch + aggregate")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="just rebuild the aggregate figure from existing CSVs")
    ap.add_argument("--horizon-h", type=float, default=6.0)
    ap.add_argument("--post-h", type=float, default=0.5)
    ap.add_argument("--bin-min", type=float, default=10.0)
    ap.add_argument("--thresh", type=float, default=0.7)
    args = ap.parse_args(argv)
    if args.aggregate_only:
        print("aggregate ->", aggregate(args.animal))
        return 0
    if args.onsets:
        onsets = [float(x) for x in args.onsets.split(",") if x.strip()]
        run_batch(args.animal, onsets, args.config, jobs=args.jobs,
                  horizon_h=args.horizon_h, post_h=args.post_h,
                  bin_min=args.bin_min, thresh=args.thresh)
        return 0
    assert args.onset_epoch is not None, "pass --onset-epoch or --onsets"
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    store = Store(cfg["database"]["path"])
    tmpl = ensure_template(store, args.animal)
    traj = seizure_trajectory(store, args.animal, args.onset_epoch, tmpl,
                              horizon_h=args.horizon_h, post_h=args.post_h,
                              bin_min=args.bin_min, thresh=args.thresh,
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
