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
from src.preictal.isi import (behavioral_seizures, inter_seizure_intervals,
                               lookback_ceilings)
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


def _enumerate_seizures(store, animals: list[str], buffer_sec: float,
                         min_lead: float):
    """(usable_pairs, all_seizure_rows). A pair is (Seizure, ceiling_sec) with a
    real pre-ictal window; rows mirror every seizure for the pocket."""
    pairs, rows = [], []
    for a in (animals or []):
        szs = behavioral_seizures(store, a)
        isis = inter_seizure_intervals(szs)
        ceils = lookback_ceilings(szs, buffer_sec)
        for sz, isi_v, ceil in zip(szs, isis, ceils):
            if ceil and ceil > min_lead:
                pairs.append((sz, float(ceil)))
            rows.append({"file_id": sz.file_id, "animal_id": sz.animal_id,
                         "chunk_datetime": sz.chunk_datetime, "eo_sec": sz.eo_sec,
                         "bb_sec": sz.bb_sec, "racine": sz.racine,
                         "seizure_type": sz.seizure_type,
                         "onset_epoch": sz.onset_epoch, "isi_sec": isi_v,
                         "ceiling_sec": ceil, "used": 0})
    return pairs, rows


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
        pairs, seizure_rows = _enumerate_seizures(store, animals, buffer,
                                                  min_lead)
        ceilings = [c for _, c in pairs]
        scale_rows: list[dict] = []
        n_scales = 0
        if pairs:
            scales = _cwt.scales_for_band(step_sec, 1.0 / max(ceilings),
                                          1.0 / min_lead, wavelet, spo)
            freqs = _cwt.pseudo_freqs(scales, step_sec, wavelet)
            n_scales = int(len(scales))
            used_files: set = set()
            for feat in resolve_features(pcfg.get("features", ["line_length"])):
                acc: list[list] = [[] for _ in range(n_scales)]
                used = 0
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
                    mag = np.abs(coeffs)
                    for s in range(n_scales):
                        acc[s].append(mag[s])
                    used += 1
                    used_files.add(sz.file_id)
                for s in range(n_scales):
                    if not acc[s]:
                        continue
                    m = np.concatenate(acc[s])
                    scale_rows.append({
                        "feature": feat.name, "channel_role": channel_role,
                        "scale_index": s, "scale": float(scales[s]),
                        "pseudo_freq_hz": float(freqs[s]),
                        "coeff_mean": float(m.mean()), "coeff_std": float(m.std()),
                        "coeff_max": float(m.max()), "n_seizures": used})
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
