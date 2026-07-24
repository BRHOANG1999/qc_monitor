"""Overview stim-artifact overlay: selection helpers + figure builder.

The overlay lets a reviewer pick any animal + stim location(s) + a recording
date range instead of the active-channels / most-recent-N default. These cover
the non-Dash logic behind that: default selection, the date-range seed, and the
generalised builder (ranged vs recent-N fallback + empty state).

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


# ----------------------------------------------------- default selection --- #

def test_default_selection_lists_all_animals_and_seeds_range(tmp_path):
    s = _store(tmp_path)
    animals, animal0, locs0, (start, end) = ov._artifact_default_selection(s, {})
    assert set(animals) == {"BCH061", "BCH062"}
    assert animal0 in animals
    assert locs0 == ov._impedance_animals_and_electrodes(s, {}).get(animal0)
    assert start and end and start <= end


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


# ------------------------------------------------------------- builder --- #

def test_builder_ranged_includes_only_in_range(tmp_path):
    s = _store(tmp_path)
    out = ov._impedance_artifact_overlay(
        s, [("BCH062", "BCH062SR")], start="2026-06-10", end="2026-06-30")
    txt = _text(out)
    assert "1 traces" in txt                 # only file 2 (06-15) in range
    assert "2026-06-10" in txt and "2026-06-30" in txt


def test_builder_empty_keys_message(tmp_path):
    s = _store(tmp_path)
    out = ov._impedance_artifact_overlay(s, [], start="2026-06-01",
                                         end="2026-06-30")
    assert "Pick an animal" in _text(out)


def test_builder_no_recordings_in_range_message(tmp_path):
    s = _store(tmp_path)
    out = ov._impedance_artifact_overlay(
        s, [("BCH062", "BCH062SR")], start="2026-01-01", end="2026-01-31")
    assert "No recordings for this selection" in _text(out)


def test_builder_recent_fallback_without_range(tmp_path):
    """No start/end -> recent-N path (preserves the old default behaviour)."""
    s = _store(tmp_path)
    out = ov._impedance_artifact_overlay(s, [("BCH062", "BCH062SR")])
    txt = _text(out)
    assert "2 traces" in txt                 # both SR recordings, no date span
    assert "→" not in txt.split("dominant")[-1]     # no range suffix
