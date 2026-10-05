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
    onset_col = {"lead": "time_to_onset_sec", "all": "tto_any_sec"}
    out = {}
    for g, cols in _GROUPS.items():
        cols = [c for c in cols if c in mat.columns]
        for oset, ocol in onset_col.items():
            tag = f"{_GROUP_NAME[g]} · {oset} seizures"
            out[f"{g}_{oset}_trend"] = G.metric_trend_fig(
                mat, cols, ocol, p(f"{g}_{oset}_trend.png"), labels=labels,
                title=f"{C.ANIMAL} · {C.CHANNEL} · pre-ictal trend — {tag}",
                ref1=(g == "ppr"))
            out[f"{g}_{oset}_prob"] = G.seizure_prob_vs_metric_fig(
                mat, cols, ocol, p(f"{g}_{oset}_prob.png"), horizons=_HORIZONS,
                labels=labels, title=f"{C.ANIMAL} · {C.CHANNEL} · "
                f"P(event within H | metric) — {tag}")
    sz = _isi.scored_seizures(store, C.ANIMAL)
    t = mat["t_epoch"].to_numpy(float); lo, hi = t.min(), t.max()
    allon = np.sort(np.array([s.onset_epoch for s in sz], float))
    leadon = np.sort(np.array([s.onset_epoch for s in
                               _isi.leading_seizures(sz, 6 * 3600.0)], float))
    for oset, ons in (("lead", leadon), ("all", allon)):
        ons = ons[(ons >= lo) & (ons <= hi)]
        out[f"periictal_ppr_{oset}"] = G.periictal_trajectory_fig(
            mat, ons, p(f"periictal_ppr_{oset}.png"),
            title=f"{C.ANIMAL} · {C.CHANNEL} · peri-ictal PPR (p2p) — "
            f"{oset} seizures (n={ons.size}; 1 = equal)")
    for k, v in out.items():
        print(f"[paired_pulse] {k} -> {v}", flush=True)
    return {"mat": mat, "out": out}
