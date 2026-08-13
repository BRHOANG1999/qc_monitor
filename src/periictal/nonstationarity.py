"""Nonstationarity control: is a sliding-window preictal AUC a real pre-ictal
signal, or just feature drift over hours?

Runs the SAME ``sliding_window_auc`` machinery on four conditions and differences
them:

  * evoked @ real seizures  (the current sliding-AUC output)
  * passive @ real seizures (the mirror negative window: evoked [+g,+post] ->
    passive [-post,-g])
  * evoked @ NULL onsets    (random deep-interictal fake onsets)
  * passive @ NULL onsets

The null is K independent MATCHED draws (each N = seizure count, via
``null_onsets``), pooled into a per-(feature, window) null band + an
observed-vs-null empirical p (``preictal.validation.null_stats``). The headline is
Δ = AUC(seizure) − AUC(null): Δ > 0 means the separation survives the control.

Pure (no Dash). Heavy I/O (passive sidecars + the null epoch read) so the caller
runs it on a background thread with a ``progress`` callback.
"""

from __future__ import annotations

import warnings

import numpy as np

from src.periictal import config as _cfg
from src.periictal import passive as _passive
from src.periictal.matrix import (build_null_matrix, epoch_columns_for,
                                  near_seizure_filter)
from src.periictal.null_onsets import draw_null_set
from src.periictal.persist import build_matrix_cached
from src.periictal.sliding_auc import sliding_window_auc
from src.preictal.validation import null_stats


def _group_auc(result: dict) -> tuple[np.ndarray, list]:
    """Mean-across-seizures AUC matrix (feature x window) + the feature order, from
    a ``sliding_window_auc`` result. NaN where no unit scored a cell."""
    feats = list(result.get("features", []))
    n_win = int(result["offsets"].size) if result.get("offsets") is not None else 0
    ps = result.get("per_seizure", {})
    if not ps or n_win == 0:
        return np.full((len(feats), n_win), np.nan), feats
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        return np.nanmean(np.stack([p["auc"] for p in ps.values()]), axis=0), feats


def _seizure_stack(result: dict) -> np.ndarray:
    """Per-unit AUC cube (n_units x feature x window) for the overlay spread."""
    ps = result.get("per_seizure", {})
    if not ps:
        feats = list(result.get("features", []))
        n_win = int(result["offsets"].size) if result.get("offsets") is not None else 0
        return np.empty((0, len(feats), n_win))
    return np.stack([p["auc"] for p in ps.values()])


def _score_null_variant(store, animal, evoked_dir, *, draws, feats, sidecar_variant,
                        feature_cfg, window_sec, nwin, band_lo, band_hi,
                        progress, label) -> dict:
    """Read the epoch columns ONCE (union prefilter over all fake onsets) then
    score every draw. Returns per-draw group-AUC cube + per-draw feature means."""
    all_fakes = (np.concatenate([d.onsets for d in draws if d.onsets.size])
                 if any(d.onsets.size for d in draws) else np.array([]))
    progress(f"reading {label} epochs for the null…")
    cols = epoch_columns_for(store, animal, evoked_dir, metrics=feats,
                             sidecar_variant=sidecar_variant, feature_cfg=feature_cfg,
                             extra_onsets=all_fakes, window_sec=window_sec)
    gaucs, groups = [], []
    for k, d in enumerate(draws):
        if d.onsets.size == 0:
            continue
        progress(f"scoring {label} null draw {k + 1}/{len(draws)}…")
        nm = build_null_matrix(d.onsets, cols, feats, window_sec=window_sec)
        r = sliding_window_auc(nm, feats, n_windows=nwin, band_lo=band_lo,
                               band_hi=band_hi)
        g, _ = _group_auc(r)
        gaucs.append(g)
        groups.append(dict(r.get("group", {})))
    return {"draw_gaucs": gaucs, "draw_groups": groups}


def _aggregate_variant(sz_res: dict, null: dict) -> dict:
    """Combine one variant's seizure result + its K null draws into the figure-
    ready block: group AUCs, null median/band, and Δ + per-feature p."""
    sz_gauc, feats = _group_auc(sz_res)
    sz_group = dict(sz_res.get("group", {}))
    n_win = sz_gauc.shape[1] if sz_gauc.ndim == 2 else 0
    gaucs = null["draw_gaucs"]
    if gaucs:
        stack = np.stack(gaucs)                                  # K x M x W
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            null_median = np.nanmedian(stack, axis=0)
            band = np.nanpercentile(stack, [2.5, 97.5], axis=0)
    else:
        stack = np.empty((0, len(feats), n_win))
        null_median = np.full((len(feats), n_win), np.nan)
        band = np.stack([null_median, null_median])
    # per-feature seizure-unit mean AUC across draws
    null_group_draws = {f: [g.get(f, np.nan) for g in null["draw_groups"]]
                        for f in feats}
    null_group_median = {f: (float(np.nanmedian(v)) if any(np.isfinite(v)) else np.nan)
                         for f, v in null_group_draws.items()}
    delta_feat = {f: (sz_group.get(f, np.nan) - null_group_median.get(f, np.nan))
                  for f in feats}
    p = {f: null_stats(sz_group.get(f, np.nan), null_group_draws.get(f, []))["p"]
         for f in feats}
    order = sorted(feats, key=lambda f: (-sz_group[f] if np.isfinite(sz_group.get(f, np.nan))
                                         else np.inf))
    return {
        "feats": feats, "sz_gauc": sz_gauc, "sz_stack": _seizure_stack(sz_res),
        "sz_group": sz_group, "null_median": null_median, "null_band": band,
        "null_stack": stack, "null_group_median": null_group_median,
        "null_group_draws": null_group_draws, "delta": sz_gauc - null_median,
        "delta_feat": delta_feat, "p": p, "ranked": order,
        "n_units": int(sz_res.get("n_seizures_used", 0)),
    }


def run_control(store, animal: str, evoked_dir: str, cache_dir: str, *,
                base_full_evoked, feats_evoked, seed: int = 0, k_draws: int = 20,
                buffer_sec: float | None = None, nwin: int = _cfg.SLIDING_N_WINDOWS,
                band_lo_sec: float = _cfg.SLIDING_BAND_LO_SEC,
                band_hi_sec: float = _cfg.SLIDING_BAND_HI_SEC,
                evoked_sidecar_variant: str = "evoked", evoked_feature_cfg=None,
                evoked_from_ms: float = 1.0, evoked_to_ms: float = 200.0,
                protocol: str | None = None, progress=None) -> dict:
    """The full control. *base_full_evoked* / *feats_evoked* come from the already-
    built scope-bar evoked matrix (its ``full`` + ``metrics``). Returns a figure-
    ready dict keyed by variant ('evoked' | 'passive')."""
    prog = progress or (lambda *_a: None)
    window_sec = float(band_hi_sec)
    # Passive = the negative mirror of the evoked window ([+g,+post] -> [-post,-g]).
    passive_cfg = _passive.window_config(-float(evoked_to_ms), -float(evoked_from_ms))
    passive_feats = _cfg.metrics_for_variant("passive")

    # Warm the mirror-window passive sidecars first (build_matrix_cached only
    # READS them), exactly as the explorer's _worker does. Near-seizure prefilter
    # keeps this to the files that can contribute a row (~15 s/file, cached after).
    prog("warming the passive (matched pre-stim) window…")
    _passive.warm_variant(
        animal, evoked_dir, "passive", passive_cfg, protocol=protocol,
        file_filter=near_seizure_filter(store, animal, window_sec),
        progress=lambda d, n, _fp: prog(f"windowing passive traces… ({d}/{n})"))

    prog("building the passive (matched pre-stim) matrix…")
    passive_full = build_matrix_cached(
        store, animal, evoked_dir, cache_dir, protocol=protocol,
        window_sec=window_sec, variant="passive", sidecar_variant="passive",
        feature_cfg=passive_cfg)

    prog("scoring the seizure conditions…")
    ev = base_full_evoked
    ev_pre = ev[ev["phase"].to_numpy() == "pre"] if len(ev) else ev
    pa_pre = (passive_full[passive_full["phase"].to_numpy() == "pre"]
              if len(passive_full) else passive_full)
    sz_evoked = sliding_window_auc(ev_pre, feats_evoked, n_windows=nwin,
                                   band_lo=band_lo_sec, band_hi=band_hi_sec)
    sz_passive = sliding_window_auc(pa_pre, passive_feats, n_windows=nwin,
                                    band_lo=band_lo_sec, band_hi=band_hi_sec)

    prog("drawing null onsets…")
    draws = draw_null_set(store, animal, evoked_dir,
                          seeds=list(range(int(seed), int(seed) + int(k_draws))),
                          buffer_sec=buffer_sec, window_sec=window_sec,
                          band_lo_sec=band_lo_sec, band_hi_sec=band_hi_sec)
    n_placed = int(np.median([d.n_placed for d in draws])) if draws else 0
    reason = draws[0].reason if draws else "no draws"

    null_evoked = _score_null_variant(
        store, animal, evoked_dir, draws=draws, feats=feats_evoked,
        sidecar_variant=evoked_sidecar_variant, feature_cfg=evoked_feature_cfg,
        window_sec=window_sec, nwin=nwin, band_lo=band_lo_sec, band_hi=band_hi_sec,
        progress=prog, label="evoked")
    null_passive = _score_null_variant(
        store, animal, evoked_dir, draws=draws, feats=passive_feats,
        sidecar_variant="passive", feature_cfg=passive_cfg, window_sec=window_sec,
        nwin=nwin, band_lo=band_lo_sec, band_hi=band_hi_sec, progress=prog,
        label="passive")

    prog("aggregating…")
    return {
        "empty": sz_evoked.get("n_seizures_used", 0) == 0,
        "offsets": sz_evoked.get("offsets"),
        "win_labels": sz_evoked.get("win_labels"),
        "n_seizures": int(sz_evoked.get("n_seizures_used", 0)),
        "n_null_placed": n_placed,
        "n_null_requested": (draws[0].n_requested if draws else 0),
        "k_draws": int(k_draws), "null_reason": reason,
        "evoked_window_ms": (evoked_from_ms, evoked_to_ms),
        "passive_window_ms": (-evoked_to_ms, -evoked_from_ms),
        "variants": {
            "evoked": _aggregate_variant(sz_evoked, null_evoked),
            "passive": _aggregate_variant(sz_passive, null_passive),
        },
    }
