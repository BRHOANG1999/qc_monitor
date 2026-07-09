"""Behavioral-seizure enumeration + inter-seizure-interval (ISI) accounting.

The pre-ictal lookback for seizure N is bounded by its ISI: a window reaching
past ``onset_{N-1}`` would ingest the previous seizure + its post-ictal tail and
corrupt the pre-ictal label. So each seizure gets a LOOKBACK CEILING

    ceiling_N = onset_N - onset_{N-1} - post_ictal_buffer

and lead-time bins are defined only up to that ceiling (bins beyond it are
dropped, never padded). The first seizure of an animal has no predecessor and
is excluded (no defined pre-ictal window).
"""

from __future__ import annotations

import json
from datetime import datetime

from src.preictal.types import LeadTimeBins, Seizure

# review_state statuses that count as a reviewer-CONFIRMED behavioral seizure.
CONFIRMED_STATUSES = ("pi_approved", "has_events")


def parse_chunk_datetime(s: str | None) -> datetime | None:
    """processed_files.chunk_datetime is stored in TWO formats in the wild --
    ISO 8601 (``2025-02-09T13:59:16``) and the recorder's underscore stamp
    (``2026_03_02__17_18_51``). Parse either; None on anything else."""
    if not s:
        return None
    s = s.strip()
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    try:
        # "YYYY_MM_DD__HH_MM_SS" is 20 chars (note the DOUBLE underscore).
        return datetime.strptime(s[:20], "%Y_%m_%d__%H_%M_%S")
    except (ValueError, IndexError):
        return None


def _event_is_seizure(ev: dict) -> bool:
    """A scored behavioral seizure carries an EEG onset AND a Racine score."""
    return (isinstance(ev, dict)
            and ev.get("EO_sec") not in (None, "")
            and ev.get("racine") not in (None, ""))


def behavioral_seizures(store, animal_id: str,
                         statuses: tuple = CONFIRMED_STATUSES) -> list[Seizure]:
    """Every confirmed behavioral seizure for *animal_id*, chronological by
    ABSOLUTE onset. Reads the LATEST review per (file, animal); each qualifying
    marker (EO + Racine) becomes one Seizure with its absolute onset epoch."""
    assert animal_id, "animal_id required"
    ph = ",".join("?" for _ in statuses)
    latest = store._sql_latest_row_correlated("pf.id")
    with store.connection() as conn:
        rows = conn.execute(
            f"""SELECT rs.file_id, rs.animal_id, rs.markers_json,
                       pf.file_path, pf.session_dir, pf.chunk_datetime
                FROM review_state rs
                JOIN processed_files pf ON pf.id = rs.file_id
                WHERE rs.animal_id = ?
                  AND rs.status IN ({ph})
                  AND rs.markers_json IS NOT NULL
                  AND pf.chunk_datetime IS NOT NULL
                  AND {latest}
                ORDER BY pf.chunk_datetime ASC""",
            (animal_id, *statuses)).fetchall()
    out: list[Seizure] = []
    for i, r in enumerate(rows):
        assert i < 1_000_000, "seizure scan runaway"
        dt = parse_chunk_datetime(r["chunk_datetime"])
        if dt is None:
            continue
        try:
            markers = json.loads(r["markers_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        base = dt.timestamp()
        for ev in markers:
            if not _event_is_seizure(ev):
                continue
            try:
                eo = float(ev["EO_sec"])
            except (TypeError, ValueError):
                continue
            bb = ev.get("BB_sec")
            try:
                bb = float(bb) if bb not in (None, "") else None
            except (TypeError, ValueError):
                bb = None
            try:
                rac = int(ev["racine"])
            except (TypeError, ValueError):
                rac = None
            out.append(Seizure(
                animal_id=animal_id, file_id=int(r["file_id"]),
                file_path=r["file_path"], session_dir=r["session_dir"],
                chunk_datetime=r["chunk_datetime"], eo_sec=eo, bb_sec=bb,
                racine=rac, seizure_type=ev.get("type"),
                onset_epoch=base + eo))
    out.sort(key=lambda s: s.onset_epoch)
    return out


def _load_markers(markers_json) -> list:
    try:
        return json.loads(markers_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return []


def _mk_seizure(animal_id, file_id, file_path, session_dir, chunk_datetime,
                 ev) -> Seizure | None:
    """Build a Seizure from one qualifying (EO+Racine) marker + its file's
    absolute start, or None if the timestamp/onset can't be resolved."""
    dt = parse_chunk_datetime(chunk_datetime)
    if dt is None or not _event_is_seizure(ev):
        return None
    try:
        eo = float(ev["EO_sec"])
    except (TypeError, ValueError, KeyError):
        return None
    bb = ev.get("BB_sec")
    try:
        bb = float(bb) if bb not in (None, "") else None
    except (TypeError, ValueError):
        bb = None
    try:
        rac = int(ev["racine"])
    except (TypeError, ValueError, KeyError):
        rac = None
    return Seizure(animal_id=animal_id, file_id=int(file_id),
                   file_path=file_path, session_dir=session_dir,
                   chunk_datetime=chunk_datetime, eo_sec=eo, bb_sec=bb,
                   racine=rac, seizure_type=ev.get("type"),
                   onset_epoch=dt.timestamp() + eo)


def scored_seizures(store, animal_id: str) -> list[Seizure]:
    """Every scored seizure (EO + Racine) for an animal on the TIMELINE, from
    the live review (ANY status -- how a seizure was scored doesn't change WHEN
    it happened) UNION the historical imported examples, deduped by file (a live
    review supersedes its imported copy). Chronological by absolute onset. This
    is the temporal series the rolling daily/weekly/monthly sweep scopes by
    date."""
    assert animal_id, "animal_id required"
    latest = store._sql_latest_row_correlated("pf.id")
    out: list[Seizure] = []
    seen_files: set = set()
    with store.connection() as conn:
        live = conn.execute(
            f"""SELECT rs.file_id, rs.markers_json, pf.file_path,
                       pf.session_dir, pf.chunk_datetime
                FROM review_state rs
                JOIN processed_files pf ON pf.id = rs.file_id
                WHERE rs.animal_id = ?
                  AND rs.markers_json IS NOT NULL
                  AND pf.chunk_datetime IS NOT NULL
                  AND {latest}""", (animal_id,)).fetchall()
        for r in live:
            seen_files.add(int(r["file_id"]))
            for ev in _load_markers(r["markers_json"]):
                s = _mk_seizure(animal_id, r["file_id"], r["file_path"],
                                r["session_dir"], r["chunk_datetime"], ev)
                if s is not None:
                    out.append(s)
        ext = conn.execute(
            """SELECT e.file_id, e.markers_json, pf.file_path,
                      pf.session_dir, pf.chunk_datetime
               FROM training_external_example e
               JOIN processed_files pf ON pf.id = e.file_id
               WHERE e.animal = ? AND e.has_seizure = 1
                 AND pf.chunk_datetime IS NOT NULL""", (animal_id,)).fetchall()
        for r in ext:
            if int(r["file_id"]) in seen_files:
                continue
            for ev in _load_markers(r["markers_json"]):
                s = _mk_seizure(animal_id, r["file_id"], r["file_path"],
                                r["session_dir"], r["chunk_datetime"], ev)
                if s is not None:
                    out.append(s)
    out.sort(key=lambda s: s.onset_epoch)
    return out


def inter_seizure_intervals(seizures: list[Seizure]) -> list[float | None]:
    """Seconds from the previous seizure's onset to each seizure's onset.
    Element 0 is None (no predecessor). Same length as *seizures*."""
    isi: list[float | None] = []
    prev = None
    for s in seizures:
        isi.append(None if prev is None else (s.onset_epoch - prev))
        prev = s.onset_epoch
    return isi


def lookback_ceilings(seizures: list[Seizure], post_ictal_buffer_sec: float,
                       max_lookback_sec: float | None = None
                       ) -> list[float | None]:
    """Per-seizure lookback ceiling = ISI - post_ictal_buffer, CAPPED at
    *max_lookback_sec*. None where the seizure has no predecessor OR the buffer
    eats the whole interval.

    The cap matters: a long ISI (e.g. a 41-day gap) would otherwise make the
    "pre-ictal" window span days -- not pre-ictal at all (it's the whole
    interictal span), and a multi-million-sample trajectory that explodes CWT
    compute. Fast pre-ictal change lives in a bounded window; long-timescale
    rhythms belong to the evoked_rhythms module, not per-seizure lookback."""
    assert post_ictal_buffer_sec >= 0, "buffer must be >= 0"
    out: list[float | None] = []
    for isi in inter_seizure_intervals(seizures):
        if isi is None:
            out.append(None)
            continue
        ceiling = isi - post_ictal_buffer_sec
        if max_lookback_sec is not None and ceiling > max_lookback_sec:
            ceiling = float(max_lookback_sec)
        out.append(ceiling if ceiling > 0 else None)
    return out


def leadtime_bins(ceiling_sec: float, min_leadtime_sec: float,
                   factor: float = 2.0, max_bins: int = 64) -> LeadTimeBins:
    """Dyadic (log) lead-time bins from *min_leadtime_sec* up to *ceiling_sec*:
    edges min, min*factor, min*factor^2, ... capped at the ceiling. Bin k spans
    ``[edges[k], edges[k+1]]`` seconds-before-onset; centers are geometric
    means. Empty when the ceiling is below the minimum lead-time (ISI too
    short)."""
    assert factor > 1.0, "factor must be > 1"
    assert min_leadtime_sec > 0, "min_leadtime_sec must be > 0"
    if not ceiling_sec or ceiling_sec <= min_leadtime_sec:
        return LeadTimeBins((), (), float(ceiling_sec or 0.0))
    edges = [float(min_leadtime_sec)]
    it = 0
    while edges[-1] < ceiling_sec and it < max_bins:
        it += 1
        nxt = edges[-1] * factor
        if nxt >= ceiling_sec:
            break
        edges.append(nxt)
    edges.append(float(ceiling_sec))          # final edge = the ceiling
    centers = tuple((edges[k] * edges[k + 1]) ** 0.5
                    for k in range(len(edges) - 1))
    return LeadTimeBins(tuple(edges), centers, float(ceiling_sec))
