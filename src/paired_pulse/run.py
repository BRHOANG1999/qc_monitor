"""Build the paired-pulse matrix + render the four figures + CSV + manifest."""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess

from . import config as C, data as D, figures as G


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
    from src.preictal_biomarker import features as F
    store, ed = _ctx()
    mat = D.build_pp_matrix(store, ed, since=since, force=force)
    med = {f: float(mat[f"ppr_{f}"].median()) for f in C.FEATURES
           if f"ppr_{f}" in mat.columns}
    print(f"[paired_pulse] {len(mat)} pairs from {mat['file'].nunique()} files; "
          f"PPR medians {({k: round(v, 3) for k, v in med.items()})}", flush=True)
    if not apply:
        print("[paired_pulse] dry run (pass --apply to write figures/CSV)", flush=True)
        return {"mat": mat, "medians": med}
    onsets = F.lead_onsets(store)
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
