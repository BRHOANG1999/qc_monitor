"""Orchestration for the pre-ictal biomarker rebuild.

Thin, import-light glue so both the CLI and the notebook call the same steps:
load/build the feature matrix, fit states, compute occupancy + null, render the
figures. Each function returns plain objects so the notebook can inspect them.
"""

from __future__ import annotations

import os

from src.db.store import Store
from src.dashboard.data_helpers import load_config
from . import config as C, features as F, states as S, occupancy as O, figures as G


def _ctx():
    cfg = load_config()
    return (Store(cfg["database"]["path"]),
            cfg["chronic_evoked"]["evoked_output_dir"],
            cfg["database"]["path"])


def load_states(*, force_features: bool = False):
    """Feature matrix -> fitted StateModel."""
    store, evoked_dir, db = _ctx()
    df = F.build_feature_matrix(store, evoked_dir, db, force=force_features)
    return F, S.fit_states(df), store


def run_phase0(*, force_features: bool = False) -> dict:
    """Build/fit everything and render the Phase-0 anchor figure."""
    store, evoked_dir, db = _ctx()
    df = F.build_feature_matrix(store, evoked_dir, db, force=force_features)
    model = S.fit_states(df)
    pre = S.preictal_mask(model.df)
    base = S.baseline_mask(model.df)
    occ = O.observed(model.df, pre, base)
    out = os.path.join(C.OUT_DIR, "rebuild_00_anchor_states.png")
    G.anchor_figure(model, occ, pre, out)
    ev = model.explained_var
    print(f"[pipeline] rows={len(model.df)} stated={int((model.df.state>=0).sum())}"
          f" | PC var%={[round(100*x,1) for x in ev]}"
          f" cum={round(100*ev.sum(),1)}")
    print(f"[pipeline] state counts: "
          f"{model.df['state'].value_counts().sort_index().to_dict()}")
    print(f"[pipeline] labels: {model.labels}")
    print(f"[pipeline] pre-ictal occ %: {occ['preictal'].round(1)} "
          f"(n={occ['n_pre']}) | baseline: {occ['baseline'].round(1)} "
          f"(n={occ['n_base']})")
    print(f"[pipeline] anchor figure -> {out}")
    return {"model": model, "occ": occ, "pre": pre, "base": base, "out": out}


def run_full(*, force_features: bool = False, n_surr: int = 2000) -> dict:
    """Render the full static figure set (anchor, occupancy + null comparison,
    null-method intermediaries, per-seizure trajectories, timeline)."""
    from . import trajectory as TR, timeline as TL
    store, evoked_dir, db = _ctx()
    df = F.build_feature_matrix(store, evoked_dir, db, force=force_features)
    model = S.fit_states(df)
    lead = F.lead_onsets(store)
    allon = F.scoped_onsets(store)
    print(f"[pipeline] lead seizures={lead.size} | all in-scope={allon.size}")
    pre = S.preictal_mask(model.df); base = S.baseline_mask(model.df)
    occ = O.observed(model.df, pre, base)
    p = lambda n: os.path.join(C.OUT_DIR, n)
    out = {}
    out["anchor"] = G.anchor_figure(model, occ, pre, p("rebuild_00_anchor_states.png"))
    null = O.circular_shift_null(model.df, lead, n_surr=n_surr, seed=0)
    out["null_compare"] = G.occupancy_null_compare(
        null, p("rebuild_05_occupancy_null_compare.png"), baseline=occ["baseline"])
    out["null_worked"] = G.null_worked_example(null, p("rebuild_05_null_worked_example.png"))
    out["null_schematic"] = G.null_schematic(model.df, lead, p("rebuild_04_null_schematic.png"))
    out["traj_lead"] = G.per_seizure_trajectory_fig(
        TR.per_seizure_trajectory(model.df, lead), p("rebuild_06_traj_lead.png"),
        title_suffix=" — lead seizures")
    out["traj_all"] = G.per_seizure_trajectory_fig(
        TR.per_seizure_trajectory(model.df, allon), p("rebuild_06_traj_all.png"),
        title_suffix=" — all seizures")
    out["timeline"] = TL.render_timeline(model.df, lead, p("rebuild_07_state_timeline.png"))
    for j in range(C.K_STATES):
        print(f"  state {j} [{model.labels[j]}]: obs={null['observed'][j]:.1f}% "
              f"null med={null['median'][j]:.1f}% p={null['p'][j]:.2f}")
    for k, v in out.items():
        print(f"[pipeline] {k} -> {v}")
    return {"model": model, "occ": occ, "null": null, "out": out}


def run_anim(*, step_lead: float = 2.0, step_all: float = 3.0, fps: int = 12,
             smoke: bool = False) -> dict:
    """Render the shared-PCA pre-ictal trajectory animations: lead + all."""
    from . import anim as A
    store, evoked_dir, db = _ctx()
    model = S.fit_states(F.build_feature_matrix(store, evoked_dir, db))
    lead = F.lead_onsets(store); allon = F.scoped_onsets(store)
    if smoke:                                       # 1 seizure, coarse — validate
        r = A.animate_set(model, lead[:1], A.seizure_labels(lead[:1]),
                          os.path.join(C.OUT_DIR, "anim", "smoke"),
                          step_min=20.0, fps=6, progress=lambda i, n: None)
        print("[pipeline] smoke:", r); return {"smoke": r}
    pr = lambda i, n: print(f"  frame {i}/{n}", flush=True) if i % 60 == 0 else None
    rl = A.animate_set(model, lead, A.seizure_labels(lead),
                       os.path.join(C.OUT_DIR, "anim", "preictal_traj_lead"),
                       step_min=step_lead, fps=fps, progress=pr)
    print("[pipeline] lead anim:", rl)
    ra = A.animate_set(model, allon, A.seizure_labels(allon),
                       os.path.join(C.OUT_DIR, "anim", "preictal_traj_all"),
                       step_min=step_all, fps=fps, progress=pr)
    print("[pipeline] all anim:", ra)
    return {"lead": rl, "all": ra}


if __name__ == "__main__":
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    if "--anim" in sys.argv:
        run_anim(smoke="--smoke" in sys.argv)
    elif "--full" in sys.argv:
        run_full(force_features="--force" in sys.argv)
    else:
        run_phase0(force_features="--force" in sys.argv)
