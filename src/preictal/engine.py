"""Pre-ictal sweep orchestrator (Stage 1).

For every animal in scope: enumerate confirmed behavioral seizures, build each
one's decimated pre-ictal feature trajectory (bounded by its ISI ceiling), run
the Morlet CWT on a COMMON scale grid, and aggregate per-scale coefficient
summaries across seizures -> ``preictal_scale_summary`` + a BIDS pocket. Stages
2-3 add the bin-vs-bin AUC matrix, LOSO CV, the surrogate null, and the
collapse-to-a-vs-scale deliverable.
"""

from __future__ import annotations

import logging

import numpy as np
import pywt

from src.preictal import cwt as _cwt
from src.preictal import pocket as _pocket
from src.preictal import scoring as _scoring
from src.preictal import validation as _validation
from src.preictal.isi import (inter_seizure_intervals, leadtime_bins,
                               lookback_ceilings, scored_seizures)
from src.preictal.registry import resolve_features
from src.preictal.trajectory import (feature_trajectory, gather_leadup_signal,
                                      robust_z)

logger = logging.getLogger(__name__)


class Cancelled(Exception):
    """Raised to abort a sweep when the job is cancelled."""


def _resolve_channel_index(store, seizure) -> int:
    """The animal's EEG channel index in its session; falls back to the first
    EEG channel, else 0 (synthetic sessions with no config)."""
    try:
        from src.utils.animal import is_animal_channel, split_animal_electrode
        from src.utils.session_config import discover_from_session_dir
        sc = discover_from_session_dir(seizure.session_dir)
        for i in sc.eeg_channels:
            name = sc.channel_names[i] if i < len(sc.channel_names) else ""
            if isinstance(name, str) and is_animal_channel(name):
                a, _ = split_animal_electrode(name)
                if a == seizure.animal_id:
                    return int(i)
        return int(sc.eeg_channels[0]) if sc.eeg_channels else 0
    except Exception:  # noqa: BLE001 -- fallback keeps the sweep alive
        return 0


def _period_bounds(period_start: str | None, period_end: str | None):
    """Epoch [lo, hi) for the rolling window from ISO dates (period_end is
    inclusive -> +1 day). None on either side = unbounded (adhoc = all-time)."""
    from datetime import datetime, timedelta
    lo = hi = None
    if period_start:
        try:
            lo = datetime.fromisoformat(period_start).timestamp()
        except ValueError:
            pass
    if period_end:
        try:
            hi = (datetime.fromisoformat(period_end)
                  + timedelta(days=1)).timestamp()
        except ValueError:
            pass
    return lo, hi


def _enumerate_seizures(store, animals: list[str], buffer_sec: float,
                         min_lead: float, lo: float | None = None,
                         hi: float | None = None,
                         max_lookback: float | None = None):
    """(usable_pairs, in_scope_seizure_rows). ROLLING scope: analyze only
    seizures whose ONSET falls in [lo, hi) -- but ISI + lookback ceiling still
    use each animal's FULL seizure history (the predecessor that sets a
    window's ceiling may be from before the period). A pair is (Seizure,
    ceiling_sec) with a real pre-ictal window."""
    pairs, rows = [], []
    for a in (animals or []):
        szs = scored_seizures(store, a)                     # full history
        isis = inter_seizure_intervals(szs)
        ceils = lookback_ceilings(szs, buffer_sec, max_lookback)
        for sz, isi_v, ceil in zip(szs, isis, ceils):
            if lo is not None and sz.onset_epoch < lo:
                continue
            if hi is not None and sz.onset_epoch >= hi:
                continue
            if ceil and ceil > min_lead:
                pairs.append((sz, float(ceil)))
            rows.append({"file_id": sz.file_id, "animal_id": sz.animal_id,
                         "chunk_datetime": sz.chunk_datetime, "eo_sec": sz.eo_sec,
                         "bb_sec": sz.bb_sec, "racine": sz.racine,
                         "seizure_type": sz.seizure_type,
                         "onset_epoch": sz.onset_epoch, "isi_sec": isi_v,
                         "ceiling_sec": ceil, "used": 0})
    return pairs, rows


def _f(x):
    """Float, with NaN -> None so unscoreable scales store as SQL NULL."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return None if xf != xf else xf


def _score_feature(store, feat, pairs, scales, freqs, bin_edges, step_sec,
                    target_fs, wavelet, channel_role, n_surrogates, rng,
                    cancel_fn):
    """Per-scale deliverable rows for ONE feature. Builds each seizure's
    lead-time-binned CWT coefficients, then per scale: the bin-vs-bin AUC matrix
    -> collapse gradient, leave-one-seizure-out CI, surrogate-null percentile/p,
    and the nearest-vs-farthest forecasting ROC/PR. Returns (rows, used_files).
    bin_idx is scale-independent (per-sample lead-time), computed once."""
    n_scales = int(len(scales))
    n_bins = int(len(bin_edges) - 1)
    seizure_mags: list = []        # each |W| array [n_scales, L] (float32)
    seizure_binidx: list = []      # each [L] lead-time bin index (nearest=0)
    used_files: set = set()
    for sz, ceil in pairs:
        if cancel_fn and cancel_fn():
            raise Cancelled()
        ch = _resolve_channel_index(store, sz)
        sig, fs = gather_leadup_signal(store, sz, ceil, ch)
        if sig is None:
            continue
        t = feature_trajectory(sig, fs, feat, step_sec, target_fs)
        if t.size < 4:
            continue
        coeffs, _ = pywt.cwt(robust_z(t), scales, wavelet,
                             sampling_period=step_sec)
        seizure_mags.append(np.abs(coeffs).astype(np.float32))
        # Sample k covers time onset-ceiling + k*step; lead-time (before onset)
        # = ceiling - (k+0.5)*step. Assign to common bins (edges ascending ->
        # nearest-onset bin = index 0).
        lead = float(ceil) - (np.arange(t.size) + 0.5) * step_sec
        bidx = np.clip(np.searchsorted(bin_edges, lead, side="right") - 1,
                       0, n_bins - 1)
        seizure_binidx.append(bidx)
        used_files.add(sz.file_id)

    n_used = len(seizure_mags)
    rows: list = []
    if n_used == 0:
        return rows, used_files
    for s in range(n_scales):
        seizure_values = [m[s] for m in seizure_mags]
        per_seizure_bins = [[v[bi == b] for b in range(n_bins)]
                            for v, bi in zip(seizure_values, seizure_binidx)]
        pooled = _validation._pool(per_seizure_bins, range(n_used), n_bins)
        collapse = _scoring.collapse_gradient(_scoring.auc_matrix(pooled))
        loso = _validation.loso_collapse(per_seizure_bins)
        surr = _validation.circular_shift_null(seizure_values, seizure_binidx,
                                               n_bins, n_surrogates, rng)
        nstat = _validation.null_stats(collapse, surr)
        near = pooled[0]
        far = next((pooled[b] for b in range(n_bins - 1, -1, -1)
                    if pooled[b].size), np.array([], dtype=float))
        roc, pr = _scoring.forecasting_scores(near, far)
        allmag = np.concatenate(seizure_values)
        rows.append({
            "feature": feat.name, "channel_role": channel_role,
            "scale_index": s, "scale": float(scales[s]),
            "pseudo_freq_hz": float(freqs[s]),
            "coeff_mean": float(allmag.mean()), "coeff_std": float(allmag.std()),
            "coeff_max": float(allmag.max()), "n_seizures": n_used,
            "collapse_stat": _f(collapse),
            "collapse_loso_mean": _f(loso["mean"]),
            "collapse_loso_ci_lo": _f(loso["ci_lo"]),
            "collapse_loso_ci_hi": _f(loso["ci_hi"]),
            "null_mean": _f(nstat["mean"]), "null_std": _f(nstat["std"]),
            "null_percentile": _f(nstat["percentile"]), "null_p": _f(nstat["p"]),
            "forecast_roc_auc": _f(roc), "forecast_pr_auc": _f(pr)})
    return rows, used_files


def run_sweep(store, config: dict, scope: str = "adhoc",
               period_start: str | None = None, period_end: str | None = None,
               animals: list[str] | None = None, cancel_fn=None) -> int:
    """Execute one sweep; returns the run_id. Persists a 'done' or 'failed'
    run either way. *cancel_fn* () -> bool aborts cooperatively between
    seizures."""
    pcfg = (config or {}).get("preictal", {}) or {}
    tcfg = pcfg.get("trajectory", {}) or {}
    ccfg = pcfg.get("cwt", {}) or {}
    ecfg = pcfg.get("events", {}) or {}
    wavelet = ccfg.get("wavelet", "cmor1.5-1.0")
    step_sec = float(tcfg.get("step_sec", 1.0))
    target_fs = float(tcfg.get("target_fs", 500.0))
    min_lead = float(ccfg.get("min_leadtime_sec", 1.0))
    max_lookback = float(ccfg.get("max_leadtime_sec", 21600.0))   # cap: 6 h
    spo = int(ccfg.get("scales_per_octave", 4))
    buffer = float(ecfg.get("post_ictal_buffer_sec", 300.0))
    channel_role = "eeg"

    if not animals:
        try:
            animals = store.list_all_animals()
        except Exception:  # noqa: BLE001
            animals = []
    version_id = None
    try:
        version_id = store.create_settings_version(pcfg, "preictal")
    except Exception:  # noqa: BLE001
        pass
    root = pcfg.get("derivatives_root")
    run_id = store.create_preictal_run(
        scope, period_start, period_end, animals or [],
        event_source=ecfg.get("event_source", "behavioral"), version_id=version_id)
    try:
        lo, hi = _period_bounds(period_start, period_end)
        pairs, seizure_rows = _enumerate_seizures(store, animals, buffer,
                                                  min_lead, lo, hi, max_lookback)
        ceilings = [c for _, c in pairs]
        scale_rows: list[dict] = []
        n_scales = 0
        if pairs:
            scales = _cwt.scales_for_band(step_sec, 1.0 / max(ceilings),
                                          1.0 / min_lead, wavelet, spo)
            freqs = _cwt.pseudo_freqs(scales, step_sec, wavelet)
            n_scales = int(len(scales))
            # Common lead-time bins (nearest-onset first) up to the LONGEST
            # ceiling; short-ISI seizures only reach the nearer bins.
            bin_edges = np.asarray(leadtime_bins(max(ceilings), min_lead).edges_sec,
                                   dtype=float)
            vcfg = (pcfg.get("validation", {}) or {})
            n_surr = int((vcfg.get("surrogate_null", {}) or {})
                         .get("n_surrogates", 200))
            rng = np.random.default_rng(int(vcfg.get("seed", 0)))
            used_files: set = set()
            for feat in resolve_features(pcfg.get("features", ["line_length"])):
                if cancel_fn and cancel_fn():
                    raise Cancelled()
                frows, ufiles = _score_feature(
                    store, feat, pairs, scales, freqs, bin_edges, step_sec,
                    target_fs, wavelet, channel_role, n_surr, rng, cancel_fn)
                scale_rows.extend(frows)
                used_files |= ufiles
            for r in seizure_rows:
                if r["file_id"] in used_files:
                    r["used"] = 1
        store.insert_preictal_scale_summaries(run_id, scale_rows, version_id)
        cstats = {}
        if ceilings:
            arr = np.asarray(ceilings, dtype=float)
            cstats = {"min": float(arr.min()),
                      "median": float(np.median(arr)),
                      "max": float(arr.max())}
        store.finish_preictal_run(run_id, "done", n_seizures=len(seizure_rows),
                                  n_scales=n_scales, ceiling_stats=cstats)
        if root:
            try:
                _pocket.write_run_pocket(
                    root, run_id, scope, period_start, period_end,
                    {"animals": animals or [], "n_seizures": len(seizure_rows),
                     "ceiling_min": cstats.get("min"),
                     "ceiling_median": cstats.get("median"),
                     "ceiling_max": cstats.get("max"), "config": pcfg},
                    scale_rows, seizure_rows)
            except Exception as e:  # noqa: BLE001 -- pocket is best-effort
                logger.warning("preictal pocket write failed: %s", e)
        return run_id
    except Cancelled:
        store.finish_preictal_run(run_id, "failed", error="cancelled")
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("preictal run %s failed", run_id)
        store.finish_preictal_run(run_id, "failed", error=str(e))
        raise
