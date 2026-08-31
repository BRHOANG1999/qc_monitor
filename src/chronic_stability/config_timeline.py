"""Configuration timeline for a chronic-stim subject.

A CONFIGURATION is the ordered pair ``(stim site, record site)``. Per the
operator's wiring convention, in channel ``BCH111<AREA>`` the NAMED ``<AREA>``
is the RECORD site and the UNNAMED site is the STIM target. We locate the named
electrode with the codebase's positional rule (``stim_map._stimulated_electrode
_index`` -- the electrode preceded by a stimCopy), then apply the operator's
convention to label ``(stim, record)``. A swap is a change of that named
electrode between consecutive sessions in time; a change of stim CHARGE at the
same site is reported separately as a within-configuration event (it is NOT a
new configuration under the ordered-pair definition).

Session start time: ``session_config`` has no datetime column, so we use
``MIN(processed_files.chunk_datetime)`` for the session dir.
"""

from __future__ import annotations

import json
import os

import numpy as np

from src.periictal.stim_map import (_stimulated_electrode_index,
                                    resolve_fingerprint)
from src.preictal.isi import parse_chunk_datetime
from src.utils.animal import split_animal_electrode

_MAX_SESS = 100_000         # NASA Rule 2: explicit scan bound.


def _channel_names(cfg: dict) -> list:
    raw = cfg.get("channel_names")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    return list(raw or [])


def _session_start_epoch(store, session_dir: str) -> float | None:
    """Earliest ``processed_files.chunk_datetime`` for *session_dir*, as an
    absolute epoch. None when the session has no dated files."""
    with store.connection() as conn:
        row = conn.execute(
            "SELECT MIN(chunk_datetime) mn FROM processed_files "
            "WHERE session_dir = ?", (session_dir,)).fetchone()
    mn = row["mn"] if row else None
    dt = parse_chunk_datetime(mn) if mn else None
    return dt.timestamp() if dt else None


def _animal_sites(store, animal: str) -> set:
    """The distinct electrode sites *animal* is seen at across all sessions."""
    sites: set = set()
    for i, cfg in enumerate(store.get_all_session_configs()):
        assert i < _MAX_SESS, "session scan runaway"
        for n in _channel_names(cfg):
            if isinstance(n, str) and split_animal_electrode(n)[0] == animal:
                sites.add(split_animal_electrode(n)[1])
    return {s for s in sites if s}


def bch111_sessions(store, animal: str = "BCH111") -> list[dict]:
    """Every session in which *animal* is a stim-adjacent (named-after-stimCopy)
    channel, with its record/stim sites, stim fingerprint and start epoch.
    Sorted oldest->newest by start. Record-only sessions are skipped (no
    ``(stim, record)`` configuration is defined there) but their count is
    recoverable via :func:`record_only_sessions`."""
    assert animal, "animal required"
    sites = _animal_sites(store, animal)
    out: list[dict] = []
    for i, cfg in enumerate(store.get_all_session_configs()):
        assert i < _MAX_SESS, "session scan runaway"
        names = _channel_names(cfg)
        if not any(split_animal_electrode(str(n))[0] == animal for n in names):
            continue
        idx = _stimulated_electrode_index(names, animal)
        if idx is None:
            continue
        rec_site = split_animal_electrode(names[idx])[1]
        others = sorted(sites - {rec_site})
        stim_site = others[0] if len(others) == 1 else "?"
        sd = cfg.get("session_dir") or ""
        start = _session_start_epoch(store, sd)
        if start is None:
            continue
        fp = resolve_fingerprint(store, sd, animal)
        out.append(_session_record(sd, rec_site, stim_site, start, fp, names, idx))
    out.sort(key=lambda d: d["start_epoch"])
    return out


def _session_record(sd, rec_site, stim_site, start, fp, names, idx) -> dict:
    """One session's descriptor row."""
    return {
        "session_dir": sd,
        "token": os.path.basename(sd).split("__", 1)[0],
        "record_site": rec_site, "stim_site": stim_site,
        "record_channel": names[idx],
        "start_epoch": float(start),
        "charge_nC": fp.charge_nC, "pulse_width_us": fp.pulse_width_us,
        "neg_pulse_ratio": fp.neg_pulse_ratio, "frequency_hz": fp.frequency_hz,
        "gain": fp.gain, "fp_status": fp.status,
        "report_channel": fp.report_channel,
    }


def record_only_sessions(store, animal: str = "BCH111") -> list[str]:
    """Session dirs where *animal* is present but NOT stim-adjacent (no
    configuration defined). Surfaced for the mapping, not analysed."""
    out = []
    for i, cfg in enumerate(store.get_all_session_configs()):
        assert i < _MAX_SESS, "session scan runaway"
        names = _channel_names(cfg)
        present = any(split_animal_electrode(str(n))[0] == animal for n in names)
        if present and _stimulated_electrode_index(names, animal) is None:
            out.append(cfg.get("session_dir") or "")
    return out


def config_intervals(sessions: list[dict]) -> list[dict]:
    """Merge consecutive same-record-site sessions into half-open
    ``[start_epoch, end_epoch)`` configuration intervals. The last interval is
    open-ended (``end_epoch = inf``). Each carries the (stim, record) sites and
    the modal stim charge across its sessions."""
    assert isinstance(sessions, list), "sessions must be a list"
    if not sessions:
        return []
    intervals: list[dict] = []
    i = 0
    n = len(sessions)
    while i < n:
        assert i < _MAX_SESS, "interval scan runaway"
        j = i
        site = sessions[i]["record_site"]
        while j + 1 < n and sessions[j + 1]["record_site"] == site:
            j += 1
        grp = sessions[i:j + 1]
        end = sessions[j + 1]["start_epoch"] if j + 1 < n else float("inf")
        intervals.append(_interval_record(grp, end))
        i = j + 1
    return intervals


def _interval_record(grp: list[dict], end_epoch: float) -> dict:
    """Assemble one configuration interval from its member sessions."""
    charges = [g["charge_nC"] for g in grp if g["charge_nC"] is not None]
    modal = _mode(charges) if charges else None
    s = grp[0]
    return {
        "stim_site": s["stim_site"], "record_site": s["record_site"],
        "record_channel": s["record_channel"],
        "start_epoch": s["start_epoch"], "end_epoch": end_epoch,
        "charges_seen": sorted(set(charges)), "modal_charge_nC": modal,
        "n_sessions": len(grp),
        "config_label": f"(stim={s['stim_site']}, rec={s['record_site']})",
    }


def _mode(vals: list):
    """Most common value (ties -> smallest)."""
    u, c = np.unique(np.asarray(vals, float), return_counts=True)
    return float(u[int(np.argmax(c))])


def detect_config_events(sessions: list[dict]) -> list[dict]:
    """Timeline events between consecutive sessions: ``site_swap`` (record-site
    change = new configuration) and ``charge_change`` (same site, different
    modal charge = within-config event). Each carries full provenance."""
    assert isinstance(sessions, list), "sessions must be a list"
    events: list[dict] = []
    for i in range(1, len(sessions)):
        assert i < _MAX_SESS, "event scan runaway"
        a, b = sessions[i - 1], sessions[i]
        if a["record_site"] != b["record_site"]:
            events.append(_event(a, b, "site_swap"))
        elif _charge_changed(a, b):
            events.append(_event(a, b, "charge_change"))
    return events


def _charge_changed(a: dict, b: dict) -> bool:
    ca, cb = a["charge_nC"], b["charge_nC"]
    return ca is not None and cb is not None and float(ca) != float(cb)


def _event(a: dict, b: dict, kind: str) -> dict:
    """A swap / charge-change event with provenance."""
    return {
        "event_type": kind, "at_epoch": b["start_epoch"],
        "from_config": f"(stim={a['stim_site']}, rec={a['record_site']}"
                       f", {a['charge_nC']}nC)",
        "to_config": f"(stim={b['stim_site']}, rec={b['record_site']}"
                     f", {b['charge_nC']}nC)",
        "provenance": {
            "table": "session_config", "field": "channel_names",
            "session_dir_before": a["session_dir"],
            "session_dir_after": b["session_dir"],
            "report_channel_after": b["report_channel"],
            "fp_status_after": b["fp_status"],
        },
    }


def assign_epochs_to_configs(epochs, intervals: list[dict]) -> np.ndarray:
    """Index (into *intervals*) of the configuration each trial epoch falls in,
    or -1 when before the first / in no interval. Half-open ``[start, end)``."""
    epochs = np.asarray(epochs, float)
    assert epochs.ndim == 1, "epochs must be 1-D"
    out = np.full(epochs.size, -1, dtype=int)
    for k, iv in enumerate(intervals):
        assert k < _MAX_SESS, "assign scan runaway"
        m = (epochs >= iv["start_epoch"]) & (epochs < iv["end_epoch"])
        out[m] = k
    return out
