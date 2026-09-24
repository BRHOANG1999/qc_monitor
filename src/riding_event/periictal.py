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

from src.riding_event import detect as _det
from src.utils import chunk_cache as _cc
from src.utils.evoked_output import parse_recording_dt

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
        dt = parse_recording_dt(r["fp"]) or None
        if dt is None:
            continue
        dur = float(r["dur"]) if r["dur"] else 3600.0
        out.append({"file_id": int(r["id"]), "file_path": r["fp"],
                    "start_epoch": dt.timestamp(), "duration": dur})
    out.sort(key=lambda d: d["start_epoch"])
    return out


def _det_cache_paths(animal: str, file_id: int) -> tuple:
    d = os.path.join(_root(animal), "det_cache")
    return (os.path.join(d, f"{file_id}_cand.npy"),
            os.path.join(d, f"{file_id}_mf.npy"))


def detect_recording(store, animal: str, rec: dict, tmpl: dict, *,
                     thresh: float = 0.7) -> tuple:
    """(candidate_epochs, mf_epochs) — absolute epochs of the HF-envelope ripple
    candidates and the matched-filter detections in one recording. Cached to disk
    per file_id; the chunk cache is cleared after the read so memory doesn't
    accumulate across a serial sweep."""
    cand_p, mf_p = _det_cache_paths(animal, rec["file_id"])
    if os.path.exists(cand_p) and os.path.exists(mf_p):
        return np.load(cand_p), np.load(mf_p)
    stim_times = _recording_stim_times(store, animal, rec["file_id"])
    loaded = _det.load_channel(rec["file_path"], animal)
    if loaded is None:
        _save_empty(cand_p, mf_p)
        return np.empty(0), np.empty(0)
    signal, fs, _ch = loaded
    try:
        d = _det.run_detector(signal, fs, band=tmpl["band"],
                              template_override=tmpl["template"],
                              exclude_times_sec=stim_times, thresh=thresh)
    finally:
        del signal
        _cc.clear()                                # free the multi-GB chunk
    s0 = rec["start_epoch"]
    cand = s0 + np.asarray(d["cand_locs"], dtype=np.float64) / fs
    mf = s0 + np.asarray(d["det_locs"], dtype=np.float64) / fs
    os.makedirs(os.path.dirname(cand_p), exist_ok=True)
    np.save(cand_p, cand)
    np.save(mf_p, mf)
    return cand, mf


def _save_empty(cand_p, mf_p) -> None:
    os.makedirs(os.path.dirname(cand_p), exist_ok=True)
    np.save(cand_p, np.empty(0))
    np.save(mf_p, np.empty(0))


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
                       progress=None) -> dict:
    """Ripple-rate trajectory for one seizure: bin the candidate + matched-filter
    detections over ``[onset - horizon_h, onset + post_h]`` by time-to-onset.
    Returns per-bin counts, per-bin covered seconds, and rates (events/min), with
    NaN where no recording covers the bin. Detects each overlapping recording
    (cached)."""
    prog = progress or (lambda *_a: None)
    t_lo = onset_epoch - horizon_h * _SEC_H
    t_hi = onset_epoch + post_h * _SEC_H
    recs = [r for r in animal_recordings(store, animal)
            if r["start_epoch"] < t_hi and r["start_epoch"] + r["duration"] > t_lo]
    assert len(recs) < _MAX_RECS, "too many recordings in the horizon"
    edges = np.arange(-horizon_h * 60.0, post_h * 60.0 + bin_min, bin_min)
    cand_all, mf_all = [], []
    for i, r in enumerate(recs):
        prog(f"seizure@{onset_epoch:.0f}: rec {i+1}/{len(recs)} "
             f"{os.path.basename(r['file_path'])}")
        c, m = detect_recording(store, animal, r, tmpl, thresh=thresh)
        cand_all.append(c)
        mf_all.append(m)
    cand = (np.concatenate(cand_all) if cand_all else np.empty(0))
    mf = (np.concatenate(mf_all) if mf_all else np.empty(0))
    cov = _coverage_seconds(recs, onset_epoch, edges)
    cand_ct, _ = np.histogram((cand - onset_epoch) / 60.0, bins=edges)
    mf_ct, _ = np.histogram((mf - onset_epoch) / 60.0, bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        cand_rate = np.where(cov > 0, cand_ct / (cov / 60.0), np.nan)
        mf_rate = np.where(cov > 0, mf_ct / (cov / 60.0), np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"onset_epoch": onset_epoch, "centers_min": centers,
            "cand_count": cand_ct, "mf_count": mf_ct, "cover_sec": cov,
            "cand_rate": cand_rate, "mf_rate": mf_rate, "n_recs": len(recs),
            "horizon_h": horizon_h, "bin_min": bin_min}


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


def write_trajectory_csv(traj: dict, path: str) -> str:
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["onset_epoch", "center_min", "cand_count", "mf_count",
                    "cover_sec", "cand_rate_per_min", "mf_rate_per_min"])
        for i in range(traj["centers_min"].size):
            w.writerow([f"{traj['onset_epoch']:.0f}",
                        f"{traj['centers_min'][i]:.1f}",
                        int(traj["cand_count"][i]), int(traj["mf_count"][i]),
                        f"{traj['cover_sec'][i]:.1f}",
                        f"{traj['cand_rate'][i]:.5g}", f"{traj['mf_rate'][i]:.5g}"])
    return path


def load_trajectories(animal: str) -> list[dict]:
    """Every per-seizure trajectory CSV under ``<root>/periictal/``, as dicts with
    ``centers_min``, ``cand_rate``, ``mf_rate`` (NaN where uncovered)."""
    import csv
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(_root(animal), "periictal",
                                           "traj_*.csv"))):
        cen, cr, mr, onset = [], [], [], None
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                onset = float(row["onset_epoch"])
                cen.append(float(row["center_min"]))
                cr.append(float(row["cand_rate_per_min"] or "nan"))
                mr.append(float(row["mf_rate_per_min"] or "nan"))
        if cen:
            out.append({"onset_epoch": onset, "centers_min": np.asarray(cen),
                        "cand_rate": np.asarray(cr), "mf_rate": np.asarray(mr),
                        "path": p})
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
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.6), facecolor=_r._BG,
                             sharex=True)
    for ax, key, title in [(axes[0], "cand_rate",
                            "HF-envelope ripple candidates (ripple-band, stim-blanked)"),
                           (axes[1], "mf_rate",
                            "matched-filter detections (template-correlated)")]:
        _agg_panel(ax, trajs, key, xh, title, _r, plt)
    axes[1].set_xlabel("hours to seizure onset (0 = onset)", color=_r._TEXT,
                       fontsize=10)
    fig.suptitle(f"{animal} — peri-ictal ripple rate over "
                 f"{abs(xh[0]):.0f} h before onset ({len(trajs)} seizures)",
                 color=_r._TEXT, fontsize=12, x=0.02, ha="left")
    return _r._finish(fig, out_png)


def _agg_panel(ax, trajs, key, xh, title, _r, plt) -> None:
    ax.set_facecolor(_r._PANEL)
    ax.tick_params(colors=_r._MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_r._SPINE)
    ax.grid(True, alpha=0.15, color=_r._MUTED)
    mat = np.full((len(trajs), xh.size), np.nan)
    for i, t in enumerate(trajs):
        y = t[key]
        if y.size == xh.size:
            ax.plot(xh, y, color=_r._ACCENT, lw=0.7, alpha=0.28)
            mat[i] = y
    import warnings
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
    ax.set_ylabel("events / min", color=_r._TEXT, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for tx in leg.get_texts():
        tx.set_color(_r._TEXT)


def _main(argv=None) -> int:
    import argparse

    import yaml

    from src.db.store import Store

    ap = argparse.ArgumentParser(description="Peri-ictal ripple-rate for one seizure")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", default="BCH111")
    ap.add_argument("--onset-epoch", type=float, required=True)
    ap.add_argument("--horizon-h", type=float, default=6.0)
    ap.add_argument("--post-h", type=float, default=0.5)
    ap.add_argument("--bin-min", type=float, default=10.0)
    ap.add_argument("--thresh", type=float, default=0.7)
    args = ap.parse_args(argv)
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
    write_trajectory_csv(traj, out)
    print(f"seizure {int(args.onset_epoch)}: {traj['n_recs']} recordings -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
