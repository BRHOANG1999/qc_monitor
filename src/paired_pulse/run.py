"""Build the paired-pulse matrix + render the four figures + CSV + manifest."""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess

import numpy as np

from . import config as C, data as D, figures as G

_GROUPS = {
    "s1": ["s1_peak_to_trough", "s1_rms_amplitude", "s1_line_length",
           "s1_max_slope", "s1_phfo_present"],
    "s2": ["s2_peak_to_trough", "s2_rms_amplitude", "s2_line_length",
           "s2_max_slope", "s2_phfo_present"],
    "ppr": ["ppr_peak_to_trough", "ppr_rms_amplitude", "ppr_line_length",
            "ppr_max_slope"],
}
_GROUP_NAME = {"s1": "pulse 1 (S1)", "s2": "pulse 2 (S2)", "ppr": "PPR (S2/S1)"}
_HORIZONS = [600.0, 1800.0, 3600.0, 7200.0]


def _preictal_filter(mat, onsets, buffer=None):
    """Keep only PRE-ictal epochs: next onset ahead AND > buffer since the previous
    onset (so on/post-onset data is excluded from pre-ictal binning)."""
    buffer = C.POSTICTAL_BUFFER_SEC if buffer is None else buffer
    t = mat["t_epoch"].to_numpy(float)
    ons = np.sort(np.asarray(onsets, float))
    if not ons.size:
        return mat
    nxt_i = np.searchsorted(ons, t, side="left")
    nxt = np.where(nxt_i < ons.size, ons[np.clip(nxt_i, 0, ons.size - 1)], np.inf)
    pidx = np.searchsorted(ons, t, side="right") - 1
    prev = np.where(pidx >= 0, ons[np.clip(pidx, 0, ons.size - 1)], -np.inf)
    keep = (nxt - t > 0) & (t - prev > buffer)
    return mat[keep].reset_index(drop=True)


def _label(col):
    return (col.replace("peak_to_trough", "p2p").replace("rms_amplitude", "rms")
            .replace("line_length", "line-len").replace("max_slope", "slope")
            .replace("phfo_present", "pHFO").replace("_", " "))


def _ctx():
    from src.db.store import Store
    from src.dashboard.data_helpers import load_config
    cfg = load_config()
    return Store(cfg["database"]["path"]), cfg["chronic_evoked"]["evoked_output_dir"]


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=C._ROOT).decode().strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def run(*, since=None, force=False, apply=True) -> dict:
    """Build the matrix (always) and, when apply, render figures + CSV + manifest."""
    from src.preictal import isi as _isi
    store, ed = _ctx()
    mat = D.build_pp_matrix(store, ed, since=since, force=force)
    med = {f: float(mat[f"ppr_{f}"].median()) for f in C.FEATURES
           if f"ppr_{f}" in mat.columns}
    print(f"[paired_pulse] {len(mat)} pairs from {mat['file'].nunique()} files; "
          f"PPR medians {({k: round(v, 3) for k, v in med.items()})}", flush=True)
    if not apply:
        print("[paired_pulse] dry run (pass --apply to write figures/CSV)", flush=True)
        return {"mat": mat, "medians": med}
    onsets = np.sort(np.array([s.onset_epoch for s in _isi.leading_seizures(
        _isi.scored_seizures(store, C.ANIMAL), 6 * 3600.0)], float))
    p = lambda n: os.path.join(C.OUT_DIR, n)
    out = {"over_time": G.ppr_over_time_fig(mat, p("01_ppr_over_time.png"),
                                            onsets=onsets),
           "overlay": G.s1_s2_overlay_fig(D.mean_waveforms(ed, since=since),
                                          p("02_s1_s2_overlay.png")),
           "vs_seizure": G.ppr_vs_seizure_fig(mat, p("03_ppr_vs_seizure.png")),
           "distribution": G.ppr_distribution_fig(mat, p("04_ppr_distribution.png"))}
    csv = p("ppr_epochs.csv.gz")
    mat.to_csv(csv, index=False, compression="gzip"); out["csv"] = csv
    man = {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
           "git": _git_sha(), "n_pairs": int(len(mat)),
           "n_files": int(mat["file"].nunique()), "isi_ms": C.ISI_MS,
           "win_ms": list(C.WIN), "features": C.FEATURES, "ppr_median": med,
           "n_lead_in_window": int(mat["n_lead_in_window"].iloc[0]) if len(mat) else 0}
    with open(p("manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    for k, v in out.items():
        print(f"[paired_pulse] {k} -> {v}", flush=True)
    return {"mat": mat, "out": out, "manifest": man}


def run_periictal(*, since=None, force=False) -> dict:
    """Pre-ictal trend + P(event within H | metric) for S1 / S2 / PPR, each for LEAD
    and ALL seizures, plus the -2 h..+1 h peri-ictal PPR trajectory (line per seizure
    + mean). Figures land in data/BCH111_paired_pulse/periictal/."""
    from src.preictal import isi as _isi
    store, ed = _ctx()
    mat = D.build_pp_matrix(store, ed, since=since, force=force)
    labels = {c: _label(c) for g in _GROUPS.values() for c in g}
    od = os.path.join(C.OUT_DIR, "periictal")
    p = lambda n: os.path.join(od, n)
    sz = _isi.scored_seizures(store, C.ANIMAL)
    t = mat["t_epoch"].to_numpy(float); lo, hi = t.min(), t.max()
    allon = np.sort(np.array([s.onset_epoch for s in sz], float))
    leadon = np.sort(np.array([s.onset_epoch for s in
                               _isi.leading_seizures(sz, 6 * 3600.0)], float))
    onset_col = {"lead": "time_to_onset_sec", "all": "tto_any_sec"}
    onset_arr = {"lead": leadon, "all": allon}
    out = {}
    for g, cols in _GROUPS.items():
        cols = [c for c in cols if c in mat.columns]
        for oset, ocol in onset_col.items():
            mf = _preictal_filter(mat, onset_arr[oset])     # exclude on/post-onset
            tag = f"{_GROUP_NAME[g]} · {oset} seizures"
            out[f"{g}_{oset}_trend"] = G.metric_trend_fig(
                mf, cols, ocol, p(f"{g}_{oset}_trend.png"), labels=labels,
                title=f"{C.ANIMAL} · {C.CHANNEL} · pre-ictal trend — {tag}",
                ref1=(g == "ppr"))
            out[f"{g}_{oset}_prob"] = G.seizure_prob_vs_metric_fig(
                mf, cols, ocol, p(f"{g}_{oset}_prob.png"), horizons=_HORIZONS,
                labels=labels, title=f"{C.ANIMAL} · {C.CHANNEL} · "
                f"P(event within H | metric) — {tag}")
    for oset, ons in (("lead", leadon), ("all", allon)):
        ons = ons[(ons >= lo) & (ons <= hi)]
        out[f"periictal_ppr_{oset}"] = G.periictal_trajectory_fig(
            mat, ons, p(f"periictal_ppr_{oset}.png"),
            title=f"{C.ANIMAL} · {C.CHANNEL} · peri-ictal PPR (p2p) — "
            f"{oset} seizures (n={ons.size}; 1 = equal)")
    for oset, ons in (("lead", leadon), ("all", allon)):
        out[f"ppr_dist_prox_{oset}"] = G.ppr_distribution_proximity_fig(
            mat, ons, p(f"ppr_distribution_proximity_{oset}.png"),
            title=f"{C.ANIMAL} · {C.CHANNEL} · PPR distribution by seizure proximity "
            f"— {oset} seizures")
    for k, v in out.items():
        print(f"[paired_pulse] {k} -> {v}", flush=True)
    return {"mat": mat, "out": out}


def run_null(*, since=None, force=False, n_surr=300) -> dict:
    """Circular-shift null for the pre-ictal trends + the peri-ictal PPR trajectory
    (bins 1/5/10 min), plus an example-evoked-responses figure. periictal/null/."""
    from src.preictal import isi as _isi
    store, ed = _ctx()
    mat = D.build_pp_matrix(store, ed, since=since, force=force)
    labels = {c: _label(c) for g in _GROUPS.values() for c in g}
    od = os.path.join(C.OUT_DIR, "periictal", "null")
    p = lambda n: os.path.join(od, n)
    sz = _isi.scored_seizures(store, C.ANIMAL)
    t = mat["t_epoch"].to_numpy(float); lo, hi = t.min(), t.max()
    allon = np.sort(np.array([s.onset_epoch for s in sz], float))
    leadon = np.sort(np.array([s.onset_epoch for s in
                               _isi.leading_seizures(sz, 6 * 3600.0)], float))
    onsets = {"lead": leadon[(leadon >= lo) & (leadon <= hi)],
              "all": allon[(allon >= lo) & (allon <= hi)]}
    out = {}
    out["examples"] = G.example_pairs_fig(
        D.example_pairs(ed, since=since), p("examples.png"),
        title=f"{C.ANIMAL} · {C.CHANNEL} · example paired-pulse evoked responses")
    for g, cols in _GROUPS.items():
        cols = [c for c in cols if c in mat.columns]
        for oset, ons in onsets.items():
            res = {c: D.trend_shift_null(mat, ons, feature=c, n_surr=n_surr)
                   for c in cols}
            out[f"{g}_{oset}_trend_null"] = G.trend_null_fig(
                res, p(f"{g}_{oset}_trend_null.png"), labels=labels, ref1=(g == "ppr"),
                title=f"{C.ANIMAL} · {C.CHANNEL} · pre-ictal trend vs shift null — "
                f"{_GROUP_NAME[g]} · {oset} seizures")
    for oset, ons in onsets.items():
        for bm in (1.0, 5.0, 10.0):
            res = D.trajectory_shift_null(mat, ons, feature="ppr_peak_to_trough",
                                          bin_min=bm, n_surr=n_surr)
            out[f"traj_null_{oset}_{int(bm)}min"] = G.trajectory_null_fig(
                res, p(f"periictal_ppr_null_{oset}_{int(bm)}min.png"),
                title=f"{C.ANIMAL} · {C.CHANNEL} · peri-ictal PPR vs shift null — "
                f"{oset} seizures · {int(bm)}-min bins")
    wb = D.waveform_by_leadtime(store, ed, since=since)      # S2/residual per bin
    for oset in ("lead", "all"):
        out[f"s2_wave_{oset}"] = G.waveform_by_bin_fig(
            wb, p(f"s2_waveform_by_bin_{oset}.png"), onset_set=oset, which="s2",
            title=f"{C.ANIMAL} · {C.CHANNEL} · averaged S2 response per pre-ictal "
            f"lead-time bin — {oset} seizures")
        out[f"resid_wave_{oset}"] = G.waveform_by_bin_fig(
            wb, p(f"residual_waveform_by_bin_{oset}.png"), onset_set=oset,
            which="resid", title=f"{C.ANIMAL} · {C.CHANNEL} · S2 − S1 residual per "
            f"pre-ictal lead-time bin — {oset} seizures")
    for k, v in out.items():
        print(f"[paired_pulse] {k} -> {v}", flush=True)
    return {"mat": mat, "out": out}


def run_continuous(*, since=None, force=False) -> dict:
    """Recreate the per-seizure trajectory heatmap + continuous timeline, but coloured
    by a CONTINUOUS measure (PPR, and separately S2 p2p) instead of discrete k-means
    state. periictal/continuous/."""
    from src.preictal import isi as _isi
    store, ed = _ctx()
    mat = D.build_pp_matrix(store, ed, since=since, force=force)
    od = os.path.join(C.OUT_DIR, "periictal", "continuous")
    p = lambda n: os.path.join(od, n)
    sz = _isi.scored_seizures(store, C.ANIMAL)
    t = mat["t_epoch"].to_numpy(float); lo, hi = t.min(), t.max()
    allon = np.sort(np.array([s.onset_epoch for s in sz], float))
    leadon = np.sort(np.array([s.onset_epoch for s in
                               _isi.leading_seizures(sz, 6 * 3600.0)], float))
    meas = [("ppr", "ppr_peak_to_trough", True, "PPR (S2/S1) p2p"),
            ("s2", "s2_peak_to_trough", False, "S2 p2p")]
    out = {}
    for oset, ons in (("lead", leadon), ("all", allon)):
        ons = ons[(ons >= lo) & (ons <= hi)]
        for tag, feat, div, name in meas:
            out[f"{tag}_heatmap_{oset}"] = G.continuous_trajectory_heatmap(
                mat, ons, p(f"{tag}_traj_heatmap_{oset}.png"), feature=feat,
                diverging=div, title=f"{C.ANIMAL} · {C.CHANNEL} · per-seizure "
                f"peri-ictal {name} — {oset} seizures")
    for tag, feat, div, name in meas:
        out[f"{tag}_timeline"] = G.continuous_timeline_fig(
            mat, allon, p(f"{tag}_timeline.png"), feature=feat, diverging=div,
            title=f"{C.ANIMAL} · {C.CHANNEL} · continuous {name} timeline")
    for k, v in out.items():
        print(f"[paired_pulse] {k} -> {v}", flush=True)
    return {"mat": mat, "out": out}
