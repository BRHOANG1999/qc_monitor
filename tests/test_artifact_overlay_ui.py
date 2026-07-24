"""Overview stim-artifact overlay: selection helpers + per-figure builder.

The overlay lets a reviewer pick any animal + stim location(s), and gives EACH
location its own recording date range (they're usually from different periods).
These cover the non-Dash logic behind that: default selection, the per-location
date-range seed, the single-figure builder (ranged vs recent-N fallback + empty
state), and the multi-location body.

Run: pytest tests/test_artifact_overlay_ui.py -q
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                       # noqa: E402
import src.dashboard.tabs.overview as ov             # noqa: E402

_T = [0.0, 0.5, 1.0]


def _rec(store, fid, chunk_dt, animal, ch_name, channel, *, charge=5.0):
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO processed_files (id, file_path, session_dir, "
            "chunk_datetime, has_video) VALUES (?,?,?,?,1)",
            (fid, f"/f{fid}.mat", f"/s{chunk_dt[:10]}", chunk_dt))
        conn.execute(
            "INSERT INTO channel_impedance (file_id, channel, channel_name, "
            "animal_id, electrode, charge_nc, access_r_kohm, slow_ss_kohm, "
            "computed_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (fid, channel, ch_name, animal, ch_name[-2:], charge, 100.0, 50.0,
             chunk_dt))
        conn.execute(
            "INSERT INTO evoked_waveforms (file_id, channel, channel_name, "
            "time_axis_ms, mean_trace) VALUES (?,?,?,?,?)",
            (fid, channel, ch_name, json.dumps(_T),
             json.dumps([float(fid)] * len(_T))))
        conn.commit()


def _seed(store):
    # BCH062: two locations across June; BCH061: a single retired location.
    _rec(store, 1, "2026_06_01__10_00_00", "BCH062", "BCH062SR", 1)
    _rec(store, 2, "2026_06_15__10_00_00", "BCH062", "BCH062SR", 1)
    _rec(store, 3, "2026_06_20__10_00_00", "BCH062", "BCH062SLM", 2)
    _rec(store, 4, "2026_05_01__10_00_00", "BCH061", "BCH061SR", 1)


def _store(tmp_path):
    s = Store(str(tmp_path / "data" / "m.db"))
    _seed(s)
    return s


def _text(component) -> str:
    """Flatten a Dash component tree to its text for substring assertions."""
    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(_text(c) for c in component)
    ch = getattr(component, "children", None)
    return _text(ch) if ch is not None else ""


def _find_ids(component, out=None):
    """Collect every component `id` (dict or str) in a Dash tree."""
    out = [] if out is None else out
    cid = getattr(component, "id", None)
    if cid is not None:
        out.append(cid)
    ch = getattr(component, "children", None)
    if isinstance(ch, (list, tuple)):
        for c in ch:
            _find_ids(c, out)
    elif ch is not None:
        _find_ids(ch, out)
    return out


# ----------------------------------------------------- default selection --- #

def test_default_selection_lists_all_animals_and_locations(tmp_path):
    s = _store(tmp_path)
    animals, animal0, locs0 = ov._artifact_default_selection(s, {})
    assert set(animals) == {"BCH061", "BCH062"}
    assert animal0 in animals
    assert locs0 == ov._impedance_animals_and_electrodes(s, {}).get(animal0)


def test_range_for_is_last_30_days_of_history(tmp_path):
    s = _store(tmp_path)
    # BCH062 SR history spans 06-01 .. 06-15; end anchors on the latest,
    # start is 30 d back but clamped to the earliest available.
    start, end = ov._artifact_range_for(s, "BCH062", ["BCH062SR"])
    assert end == "2026-06-15"
    assert start == "2026-06-01"       # clamped to earliest (< 30 d span)


def test_range_none_for_unknown_channel(tmp_path):
    s = _store(tmp_path)
    assert ov._artifact_range_for(s, "BCH062", ["nope"]) == (None, None)


# --------------------------------------------------- single-figure builder --- #

def test_single_fig_ranged_includes_only_in_range(tmp_path):
    s = _store(tmp_path)
    fig, sub = ov._single_location_fig(
        s, "BCH062", "BCH062SR", "2026-06-10", "2026-06-30")
    assert len(fig.data) == 1                 # only file 2 (06-15) in range
    assert "1 traces" in sub
    assert "2026-06-10" in sub and "2026-06-30" in sub


def test_single_fig_no_recordings_in_range(tmp_path):
    s = _store(tmp_path)
    fig, sub = ov._single_location_fig(
        s, "BCH062", "BCH062SR", "2026-01-01", "2026-01-31")
    assert len(fig.data) == 0
    assert "No recordings in this range" in sub


def test_single_fig_recent_fallback_without_range(tmp_path):
    """No start/end -> recent-N path (preserves the old default behaviour)."""
    s = _store(tmp_path)
    fig, sub = ov._single_location_fig(s, "BCH062", "BCH062SR", None, None)
    assert len(fig.data) == 2                 # both SR recordings
    assert "recent 24" in sub and "2026-" not in sub    # no date span


# ------------------------------------------------------- per-figure body --- #

def test_body_one_dated_block_per_location(tmp_path):
    s = _store(tmp_path)
    body = ov._artifact_overlay_body(s, "BCH062", ["BCH062SR", "BCH062SLM"])
    blocks = body.children
    assert len(blocks) == 2
    ids = _find_ids(body)
    # every location gets its OWN date-range picker + figure, keyed by loc.
    ranges = [i for i in ids if isinstance(i, dict)
              and i.get("type") == "artifact-loc-range"]
    figs = [i for i in ids if isinstance(i, dict)
            and i.get("type") == "artifact-loc-fig"]
    assert {r["loc"] for r in ranges} == {"BCH062SR", "BCH062SLM"}
    assert len(figs) == 2
    assert all(r["animal"] == "BCH062" for r in ranges)


def test_body_empty_locations_message(tmp_path):
    s = _store(tmp_path)
    assert "Pick at least one stim location" in _text(
        ov._artifact_overlay_body(s, "BCH062", []))


def test_body_no_animal_falls_back_to_active_channels(tmp_path):
    """No animal chosen -> the active (newest-session) channels, each as its
    own dated block (preserves the load-and-see default)."""
    s = _store(tmp_path)
    active = {c for _a, c in s.active_impedance_channel_keys()}
    assert active                                # newest session has a channel
    out = ov._artifact_overlay_body(s, None, [])
    ranges = [i for i in _find_ids(out) if isinstance(i, dict)
              and i.get("type") == "artifact-loc-range"]
    assert {r["loc"] for r in ranges} == active


def test_body_empty_store_prompts_for_selection(tmp_path):
    s = Store(str(tmp_path / "data" / "empty.db"))   # nothing seeded
    assert "Pick an animal" in _text(ov._artifact_overlay_body(s, None, []))
