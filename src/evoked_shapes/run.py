"""Orchestrate Stages 1 to 4 for one animal: assemble -> preprocess -> structure ->
cluster -> choose k -> render figures + report + provenance manifest.

``analyze`` is pure compute (returns every result object, no writes) so it is unit-
testable on synthetic data; ``build_animal`` adds the figures, report, and manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from src.evoked_shapes import cluster as _cl
from src.evoked_shapes import config as _cfg
from src.evoked_shapes import data as _data
from src.evoked_shapes import preprocess as _pp
from src.evoked_shapes import render as _render
from src.evoked_shapes import report as _report
from src.evoked_shapes import select_k as _sk
from src.evoked_shapes import structure as _st

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, cwd=_REPO_ROOT)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def analyze(X: np.ndarray, meta, *, config: dict | None = None,
            n_boot: int | None = None, log: Callable[[str], None] | None = None
            ) -> dict:
    """Run Stages 1(normalize) to 4 on an in-memory trace matrix. Returns a dict with
    every result object (pre, pca, structure, cluster sweep, select-k readouts, and
    the chosen range). Deterministic given the config seed."""
    cfg = config or {}
    seed = _cfg.seed(cfg)
    k_range = _cfg.k_range(cfg)
    nb = int(n_boot if n_boot is not None else _cfg.n_boot(cfg))
    pre = _pp.preprocess(X, mode=_cfg.norm_mode(cfg),
                         qc_min_smoothness=_cfg.qc_min_smoothness(cfg), log=log)
    Xn = pre["Xn"]
    meta_kept = meta.loc[pre["keep"]].reset_index(drop=True) if hasattr(meta, "loc") else meta
    assert Xn.shape[0] >= 4, "too few trials survived QC for shape analysis"

    if log is not None:
        log("Stage 2: PCA + dip + diffusion map...")
    pca = _st.shape_pca(Xn, seed=seed)
    Z, explained = pca["Z"], pca["explained"]
    corr_vals = _st.pairwise_correlations(Xn, seed=seed)
    diff = _st.diffusion_map(Xn, seed=seed)
    pc_dips = _st.dip_scan(Z, n_axes=4, n_boot=max(nb, 200), seed=seed)
    diff_dip = _st.dip_test(diff["coords"][:, 0], n_boot=max(nb, 200), seed=seed)
    struct = _st.summarize(pc_dips=pc_dips, diff_dip=diff_dip, corr_vals=corr_vals,
                           explained=explained)

    if log is not None:
        log("Stage 3: correlation k-means sweep...")
    sweep = _cl.cluster_sweep(Xn, k_range=k_range, seed=seed)
    templates_by_k = {k: r["templates"] for k, r in sweep.items()}
    maxcorr = {k: _cl.max_offdiag_corr(T) for k, T in templates_by_k.items()}

    if log is not None:
        log(f"Stage 4: bounding k over {k_range} ({nb} bootstraps)...")
    sessions = (meta_kept["session"].to_numpy() if hasattr(meta_kept, "columns")
                else np.zeros(Xn.shape[0]))
    bic = _sk.gmm_bic_curve(Z, k_range, seed=seed)
    heldout = _sk.gmm_heldout_ll(Z, sessions, k_range, seed=seed)
    stability = _sk.stability_curve(Xn, k_range, n_boot=nb, seed=seed)
    sel = _sk.select_range(bic=bic, heldout=heldout, stability=stability,
                           templates_by_k=templates_by_k)
    return {"pre": pre, "Xn": Xn, "meta": meta_kept, "pca": pca, "Z": Z,
            "explained": explained, "corr_vals": corr_vals, "diff": diff,
            "structure": struct, "sweep": sweep, "templates_by_k": templates_by_k,
            "maxcorr": maxcorr, "bic": bic, "heldout": heldout,
            "stability": stability, "select": sel, "n_boot": nb}


def _write_manifest(animal_dir: str, animal: str, ds: dict, res: dict) -> None:
    inp = sorted(ds.get("stim_coverage", {}).items())
    sig = hashlib.sha256(json.dumps(inp).encode("utf-8")).hexdigest()[:16]
    manifest = {
        "animal": animal, "channel": ds.get("channel"),
        "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_gathered": ds.get("n_gathered", 0),
        "n_kept": res["pre"].get("n_kept", 0),
        "verdict": res["structure"].get("verdict"),
        "k_lo": res["select"]["k_lo"], "k_hi": res["select"]["k_hi"],
        "coverage_signature": sig, "stim_coverage": dict(inp),
    }
    tmp = os.path.join(animal_dir, "manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, os.path.join(animal_dir, "manifest.json"))


def build_animal(animal: str, config: dict, *, apply: bool = True,
                 cache: bool = True,
                 log: Callable[[str], None] | None = None) -> dict:
    """Full pipeline for *animal*: assemble the dataset, analyze, and (when apply)
    write the Stage 2 to 4 figures, report.md, and manifest.json under
    ``data/derivatives/evoked_shapes/<animal>/``. ``apply=False`` is a dry-run."""
    assert animal, "animal required"
    ds = _data.build_dataset(config, animal, cache=cache, log=log)
    if ds["n_gathered"] < 4:
        return {"animal": animal, "channel": ds.get("channel"),
                "n_gathered": ds["n_gathered"], "note": "too few trials"}
    res = analyze(ds["X"], ds["meta"], config=config, log=log)
    sel = res["select"]
    k_star = sel["k_lo"]
    summary = {"animal": animal, "channel": ds.get("channel"),
               "n_gathered": ds["n_gathered"], "n_kept": res["pre"]["n_kept"],
               "verdict": res["structure"]["verdict"],
               "k_lo": sel["k_lo"], "k_hi": sel["k_hi"]}
    if not apply:
        return summary

    animal_dir = os.path.join(_cfg.out_root(config), animal)
    os.makedirs(animal_dir, exist_ok=True)
    figs = {
        "structure": _render.fig_structure(
            res["structure"], res["Z"], res["corr_vals"],
            res["diff"]["coords"], os.path.join(animal_dir, "stage2_structure.png")),
        "templates": _render.fig_templates(
            res["templates_by_k"][k_star], res["sweep"][k_star]["labels"],
            ds["time_ms"], res["Xn"], os.path.join(animal_dir, "stage3_templates.png")),
        "select_k": _render.fig_select_k(
            res["bic"], res["heldout"], res["stability"], res["maxcorr"],
            sel["k_lo"], sel["k_hi"], os.path.join(animal_dir, "stage4_select_k.png")),
    }
    n_pre = int(np.isfinite(res["meta"]["time_to_seizure"].to_numpy()).sum()) \
        if hasattr(res["meta"], "columns") else 0
    text = _report.build_report(
        animal=animal, channel=ds.get("channel"), ds=ds, pre=res["pre"],
        struct=res["structure"], sel=sel, figures=figs, n_boot=res["n_boot"],
        config_notes={"window_ms": _cfg.window_ms(config), "norm": _cfg.norm_mode(config),
                      "n_preictal": n_pre})
    _report.write_report(os.path.join(animal_dir, "report.md"), text)
    _write_manifest(animal_dir, animal, ds, res)
    if log is not None:
        log(f"wrote figures + report.md + manifest.json to {animal_dir}")
    summary["out_dir"] = animal_dir
    return summary
