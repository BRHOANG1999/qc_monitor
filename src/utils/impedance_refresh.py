"""Populate channel_impedance from evoked waveforms + gain + stim params.

Runs decoupled from the MATLAB pipeline: whenever a file has per-channel
mean evoked traces (``evoked_waveforms``) but no computed transfer
impedance yet, resolve its recording-amplifier gain (File_Records sheet)
and commanded stim current, then compute + store both phase impedances.

Also re-tries files that were skipped only because the gain was unknown,
so a later File_Records edit gets picked up without a manual rescan.
"""

from __future__ import annotations

import json
import logging
import os
import time

from src.utils.amplifier_records import load_amplifier_gains, resolve_gain
from src.utils.animal import is_animal_channel, split_animal_electrode
from src.utils.impedance import (
    access_resistance, phase_currents, normalize_ratio, slow_steady_state,
)

logger = logging.getLogger("qc_monitor.utils.impedance_refresh")


def stimulated_indices(eeg_channels, stim_copy_channels) -> set:
    """Channel indices that are actually STIMULATED, not just recorded.

    Lab convention (validated across recordings): a `stimCopy` channel
    immediately precedes each stimulated animal channel; record-only
    animals are appended with no preceding stimCopy. So an eeg channel is
    stimulated iff the index before it is a stim-copy channel. Record-only
    channels (e.g. BCH111SR) must NOT get a transfer impedance.
    """
    sc = set(stim_copy_channels or [])
    return {int(i) for i in (eeg_channels or []) if (int(i) - 1) in sc}


def _session_stim_indices(store, session_dir: str) -> set:
    """Stimulated channel index set for a session (empty if unknown /
    non-stim)."""
    if not session_dir:
        return set()
    with store.connection() as conn:
        row = conn.execute(
            """SELECT eeg_channels, stim_copy_channels
               FROM session_config WHERE session_dir = ?""",
            (session_dir,),
        ).fetchone()
    if not row:
        return set()
    try:
        eeg = json.loads(row["eeg_channels"] or "[]")
        sc = json.loads(row["stim_copy_channels"] or "[]")
    except (json.JSONDecodeError, TypeError):
        return set()
    return stimulated_indices(eeg, sc)


def _stim_params(store, file_id: int, session_dir: str) -> dict:
    """Commanded stim (charge_nC, pulse_width_us, ratio) for a file.

    Prefer the per-file STIM_REPORT values (stim_qc); fall back to the
    per-session config. Returns a dict with possibly-None values.
    """
    with store.connection() as conn:
        row = conn.execute(
            """SELECT charge_nC, pulse_width_us, neg_pulse_ratio
               FROM stim_qc WHERE file_id = ?
               ORDER BY (charge_nC > 0) DESC, id DESC LIMIT 1""",
            (file_id,),
        ).fetchone()
        if row is None and session_dir:
            row = conn.execute(
                """SELECT stim_charge_nC AS charge_nC,
                          stim_pulse_width_us AS pulse_width_us,
                          stim_neg_ratio AS neg_pulse_ratio
                   FROM session_config WHERE session_dir = ?""",
                (session_dir,),
            ).fetchone()
    if row is None:
        return {"charge": None, "pw": None, "ratio": None}
    return {"charge": row["charge_nC"], "pw": row["pulse_width_us"],
            "ratio": row["neg_pulse_ratio"]}


def _file_meta(store, file_id: int) -> dict | None:
    with store.connection() as conn:
        row = conn.execute(
            """SELECT session_name, session_dir, file_path
               FROM processed_files WHERE id = ?""",
            (file_id,),
        ).fetchone()
    return dict(row) if row else None


def _process_one_file(store, file_id: int, gains) -> int:
    """Compute + upsert impedance for every animal channel of one file.

    Returns the number of channel rows written (0 if the file has no
    animal channels or waveforms).
    """
    meta = _file_meta(store, file_id)
    if meta is None:
        return 0
    # Check for a stimulated channel FIRST (cheap session_config read) and bail
    # BEFORE the expensive per-file blob read. A non-stim recording never gets a
    # channel_impedance row, so it stays in files_missing_impedance forever and
    # was re-read on EVERY backfill run (boot + every 30 min) -- ~939 files of
    # multi-MB evoked_waveform blobs = GBs of disk reads that saturated the disk
    # and made the dashboard's Overview builds collide/oscillate (1s <-> minutes).
    session_dir = meta.get("session_dir") or ""
    stim_idx = _session_stim_indices(store, session_dir)
    if not stim_idx:
        return 0                       # no stimulated channel -> nothing to do
    waveforms = store.get_evoked_waveform_by_file(file_id)
    if not waveforms:
        return 0
    stim = _stim_params(store, file_id, session_dir)
    file_base = os.path.basename(meta.get("file_path") or "")
    session_name = meta.get("session_name") or ""
    # Accumulate every channel's row, then write them in ONE batched, locked
    # transaction (store.upsert_channel_impedance_batch) -- not one tiny
    # connect->commit per channel, which was the main 'database is locked'
    # source when a backfill swept thousands of files.
    batch = []
    for wf in waveforms:
        ch_name = wf.get("channel_name") or ""
        if not ch_name or not is_animal_channel(ch_name):
            continue
        # Only channels actually stimulated (preceded by a stimCopy).
        if int(wf.get("channel") or -1) not in stim_idx:
            continue
        animal, elec = split_animal_electrode(ch_name)
        gain = resolve_gain(gains, session_name, file_base, ch_name,
                            animal or "", elec or "")
        rec = _access_r_record(wf, gain, stim)
        batch.append((file_id, int(wf.get("channel") or 0), ch_name,
                      animal or "", elec or "", rec))
    store.upsert_channel_impedance_batch(batch)
    return len(batch)


def _access_r_record(wf: dict, gain, stim: dict) -> dict:
    """Build the channel_impedance record: the ohmic-step ACCESS RESISTANCE
    (the metric) plus the inputs needed to reproduce it. The legacy peak/mean
    impedance fields are intentionally left unset (None)."""
    i_pos, i_neg = phase_currents(stim["charge"], stim["pw"], stim["ratio"])
    ar = access_resistance(wf.get("mean_trace"), wf.get("time_axis_ms"),
                           wf.get("stim_mean_trace"), gain, i_pos, i_neg)
    # Slow-phase steady-state impedance only when Rₐ passed its QC gates
    # (same saturation/agreement checks exclude bad recordings).
    zss = {"slow_ss_raw": None, "slow_ss_kohm": None}
    if ar["r_access_kohm"] is not None:
        zss = slow_steady_state(wf.get("mean_trace"), wf.get("time_axis_ms"),
                                wf.get("stim_mean_trace"), gain, i_neg)
    return {
        "gain": (float(gain) if gain not in (None, "") else None),
        "charge_nc": _as_float(stim["charge"]),
        "pulse_width_us": _as_float(stim["pw"]),
        "neg_ratio": normalize_ratio(stim["ratio"]),
        "i_pos_ua": i_pos, "i_neg_ua": i_neg,
        "v_pos_raw": None, "v_neg_raw": None,        # legacy peak/mean: dropped
        "impedance_pos_kohm": None, "impedance_neg_kohm": None,
        "access_r_kohm": ar["r_access_kohm"],
        "access_r_reversal_kohm": ar["r_reversal_kohm"],
        "access_r_offset_kohm": ar["r_offset_kohm"],
        "slow_ss_raw": zss["slow_ss_raw"],
        "slow_ss_kohm": zss["slow_ss_kohm"],
    }


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def refresh_impedance(store, config: dict, *, limit: int | None = None,
                      log_every: int = 200, throttle_sec: float = 0.0) -> int:
    """Backfill/refresh channel_impedance. Returns channel rows written.

    Processes files with no impedance yet plus gain-was-missing retries,
    newest first. Emits progress logs so a long backfill isn't silent.

    *throttle_sec*: sleep this long between files so a big boot-time backfill
    YIELDS the disk + WAL write lock to the dashboard instead of starving it
    (the sweep read large per-channel waveform arrays flat-out). 0 = no throttle
    (interactive callers that want it fast).
    """
    assert store is not None, "store required"
    gains = load_amplifier_gains(config or {})
    todo = list(store.files_missing_impedance(limit=limit))
    seen = set(todo)
    for fid in store.files_needing_access_r(limit=limit):
        if fid not in seen:
            todo.append(fid)
            seen.add(fid)
    total = len(todo)
    if total == 0:
        return 0
    logger.info("Impedance refresh: %d file(s) to compute", total)
    rows = 0
    max_iter = total + 1
    for idx, fid in enumerate(todo):
        assert idx < max_iter, "impedance refresh runaway"
        try:
            rows += _process_one_file(store, fid, gains)
        except Exception:
            logger.exception("impedance compute failed for file_id=%s", fid)
        if (idx + 1) % log_every == 0:
            logger.info("  impedance %d/%d files (%d channel rows)",
                        idx + 1, total, rows)
        if throttle_sec:
            time.sleep(throttle_sec)
    logger.info("Impedance refresh done: %d file(s), %d channel rows",
                total, rows)
    return rows
