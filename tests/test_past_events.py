"""Past scored-event parsing + EEG file location.

Faithful-port checks for src/utils/past_events.py (the BHZCSVParser +
BHZFileDiscovery reimplementation). Run: pytest tests/test_past_events.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import past_events as pe  # noqa: E402

# A representative BHZ filename: date_experiment__<channels>___<datetime>.mat
_FN = ("stimTest-40nC__stimCopy_BCH061SLM_stimCopy_BCH062SLM_BCH060SR_"
       "BCH060SLM___2026_06_02__10_11_41.mat")


# ---- filename parsing ------------------------------------------------ #

def test_extract_channel_names():
    names = pe.extract_channel_names(_FN)
    assert names == ["stimCopy", "BCH061SLM", "stimCopy", "BCH062SLM",
                     "BCH060SR", "BCH060SLM"]


def test_animal_from_name_excludes_label_tokens():
    assert pe.animal_from_name("BCH060SR") == "BCH060"
    assert pe.animal_from_name("stimCopy") == ""
    # A bare "CH3"/"SR" style token must not be read as an animal.
    assert pe.animal_from_name("SR12") == ""


def test_animal_for_event_uses_channel_index():
    # Channel is 1-based into the channel list; index 5 -> BCH060SR.
    assert pe.animal_for_event(_FN, 5) == "BCH060"
    assert pe.animal_for_event(_FN, 2) == "BCH061"
    # Out-of-range channel -> whole-filename fallback (first real animal).
    assert pe.animal_for_event(_FN, 99) == "BCH061"


def test_file_datetime():
    dt = pe.file_datetime(_FN)
    assert dt is not None
    assert (dt.year, dt.month, dt.day, dt.hour) == (2026, 6, 2, 10)
    assert pe.file_datetime("no_timestamp.mat") is None


# ---- row parsing ----------------------------------------------------- #

def _norm(**kw):
    return {k.lower(): v for k, v in kw.items()}


def test_parse_row_real_seizure_new_format():
    row = _norm(folder="X:\\s\\", filename=_FN, fs="20000", Channel="5",
                Peak_Index="123", Score="2", Onset="Hyp (Hypersynchronous)",
                EventEO="40000", EventBB="60000")
    ev = pe.parse_row(row, "x.csv")
    assert ev is not None
    assert ev["type"] == "HYP" and ev["racine"] == 2
    assert ev["animal"] == "BCH060"
    assert ev["EO_idx"] == 40000.0 and ev["BB_idx"] == 60000.0


def test_parse_row_drops_no_event_and_zero_score():
    # Peak_Index NaN -> not a seizure.
    assert pe.parse_row(_norm(filename=_FN, Peak_Index="NaN",
                              Score="3"), "x.csv") is None
    # Score <= 0 -> filtered (matches MATLAB Score>0).
    assert pe.parse_row(_norm(filename=_FN, Peak_Index="5",
                              Score="-1"), "x.csv") is None


def test_parse_row_legacy_pipe_comment():
    # Legacy 16-col: type + EEG-ONSET live in the pipe-delimited Comment.
    row = _norm(folder="Z:\\b\\", filename=_FN, fs="5000", Channel="2",
                Peak_Index="14773455", Score="3",
                Comment="LVF | EEG-ONSET | no video")
    ev = pe.parse_row(row, "leg.csv")
    assert ev["type"] == "LVF"
    assert ev["eeg_onset"] is True


def test_seizure_type_maps_lvhf_to_lvf():
    # The lab writes low-voltage onsets as "LVHF" (High-Frequency); "LVF" is
    # NOT a substring of "LVHF", so it used to parse to None -> "?" and
    # mis-grade a correct LVF. It must map to LVF.
    assert pe._seizure_type("", ["LVHF"]) == "LVF"
    assert pe._seizure_type("LVHF (Low Voltage High Frequency)", []) == "LVF"
    assert pe._seizure_type("", ["HYP"]) == "HYP"
    assert pe._seizure_type("", ["LVF"]) == "LVF"
    # A real CSV comment lead token.
    row = _norm(folder="Z:\\b\\", filename=_FN, fs="5000", Channel="3",
                Peak_Index="10472908", Score="3",
                Comment="LVHF | EEG-ONSET | Gradual, abrupt buildup.")
    assert pe.parse_row(row, "leg.csv")["type"] == "LVF"


def test_markers_for_event_seconds():
    ev = pe.parse_row(_norm(filename=_FN, fs="20000", Channel="5",
                            Peak_Index="1", Score="4", Onset="LVF",
                            EventEO="40000", EventLAS="50000"), "x.csv")
    m = pe.markers_for_event(ev)
    assert m["type"] == "LVF" and m["racine"] == 4
    assert m["EO_sec"] == 2.0 and m["LAS_sec"] == 2.5
    assert m["BO_sec"] is None


def test_markers_eo_falls_back_to_peak_index():
    # Behavioral-only score: no EventEO, but Peak_Index marks the peak.
    ev = pe.parse_row(_norm(filename=_FN, fs="20000", Channel="5",
                            Peak_Index="60000", Score="3", Onset="HYP"),
                      "x.csv")
    m = pe.markers_for_event(ev)
    assert m["EO_sec"] == 3.0          # 60000 / 20000, from Peak_Index


# ---- dedup ----------------------------------------------------------- #

def _ev(animal="BCH060", fn=_FN, stamp=0.0, eeg=False, racine=2):
    return {"animal": animal, "filename": fn, "peak_stamp": stamp,
            "eeg_onset": eeg, "racine": racine, "type": "HYP", "fs": 20000.0}


def test_dedup_prefers_eeg_onset_within_window():
    near = 5.0 / (24 * 60)  # 5 min in datenum days
    out = pe.dedup_events([
        _ev(stamp=0.0, eeg=False),
        _ev(stamp=near, eeg=True),    # same seizure, EEG-ONSET -> winner
    ])
    assert len(out) == 1 and out[0]["eeg_onset"] is True


def test_dedup_keeps_distant_events():
    far = 30.0 / (24 * 60)  # 30 min apart -> two seizures
    out = pe.dedup_events([_ev(stamp=0.0), _ev(stamp=far)])
    assert len(out) == 2


def test_dedup_separates_animals():
    out = pe.dedup_events([_ev(animal="BCH060", stamp=0.0),
                           _ev(animal="BCH061", stamp=0.0)])
    assert len(out) == 2


# ---- load_scored_events ---------------------------------------------- #

_HEADER = ("folder,filename,fs,Cutoff,Channel,Peak_Index,Peak_Stamp,"
           "Peak_Date,Peak_Time,Peak_Start,Peak_Stop,Duration,Score,"
           "Onset,Comment,EventEO")


def test_load_scored_events(tmp_path):
    csv = tmp_path / "20260602_BCH060.csv"
    csv.write_text(
        _HEADER + "\n"
        # a real seizure row
        f"X:\\s\\,{_FN},20000,0.05,5,123,1.0,2026-06-02,10:11:41,0,0,0,"
        "2,LVF (Low Voltage Fast Onset),,40000\n"
        # a no-events row (Peak_Index NaN) -> dropped
        f"X:\\s\\,{_FN},20000,0.05,5,NaN,NaN,2026-06-02,10:11:41,,,,"
        "NaN,,No events,NaN\n",
        encoding="utf-8")
    evs = pe.load_scored_events(str(tmp_path))
    assert len(evs) == 1
    assert evs[0]["type"] == "LVF" and evs[0]["animal"] == "BCH060"


def test_load_scored_events_missing_dir():
    assert pe.load_scored_events("Z:\\does\\not\\exist") == []


# ---- no-event recordings (stage-1 negatives) ------------------------ #

def test_parse_no_event_row():
    # Peak_Index unset -> a no-event recording (the lab's negative rows).
    m = pe.parse_no_event_row(_norm(folder="X:\\s\\", filename=_FN,
                                     fs="20000", Channel="5",
                                     Peak_Index="NaN",
                                     Comment="No events in file"))
    assert m is not None and m["filename"] == _FN
    assert m["animal"] == "BCH060"
    # A row WITH a peak index is a scored event, not a negative.
    assert pe.parse_no_event_row(_norm(filename=_FN, Peak_Index="5")) is None


def test_load_no_event_recordings_excludes_seizure_files(tmp_path):
    other = ("base__stimCopy_BCH039SR___2026_06_03__01_02_03.mat")
    csv = tmp_path / "20260602_BCH060.csv"
    csv.write_text(
        _HEADER + "\n"
        # seizure row for _FN
        f"X:\\s\\,{_FN},20000,0.05,5,123,1.0,2026-06-02,10:11:41,0,0,0,"
        "2,LVF,,40000\n"
        # no-event row for _FN (same file -> excluded as a negative)
        f"X:\\s\\,{_FN},20000,0.05,5,NaN,NaN,2026-06-02,10:11:41,,,,"
        "NaN,,No events,NaN\n"
        # no-event row for a DIFFERENT file -> a real negative
        f"X:\\s\\,{other},20000,0.05,1,NaN,NaN,2026-06-03,01:02:03,,,,"
        "NaN,,No events,NaN\n",
        encoding="utf-8")
    seizures = pe.dedup_events(pe.load_scored_events(str(tmp_path)))
    keys = {(e["folder"], e["filename"]) for e in seizures}
    negs = pe.load_no_event_recordings(str(tmp_path), keys)
    assert len(negs) == 1 and negs[0]["filename"] == other


def test_catalog_no_event_examples():
    metas = [{"folder": "X:\\s\\", "filename": _FN, "fs": 20000.0,
              "channel": 5, "animal": "BCH060"}]
    out = pe.catalog_no_event_examples(metas)
    assert len(out) == 1
    assert out[0]["has_seizure"] is False and out[0]["markers"] == []
    assert out[0]["animal"] == "BCH060"


# ---- channel selection: the electrode after stimCopy --------------- #

def test_recording_channel_after_stimcopy():
    from src.utils.animal import recording_channel_index, stim_copy_indices
    names = ["stimCopy", "BCH061SLM", "stimCopy", "BCH062SLM", "BCH060SR"]
    assert recording_channel_index(names) == 1          # after first stimCopy
    assert sorted(stim_copy_indices(names)) == [0, 2]
    # No stimCopy -> first animal channel.
    assert recording_channel_index(["saline", "BCH040SR"]) == 1
    # stimCopy with a non-animal after it -> fall back to first animal.
    assert recording_channel_index(["stimCopy", "ref", "BCH040SR"]) == 2
    assert recording_channel_index([]) == 0


# ---- EEGLocator ------------------------------------------------------ #

def test_locator_direct_hit(tmp_path):
    d = tmp_path / "sess"
    d.mkdir()
    (d / _FN).write_bytes(b"x")
    loc = pe.EEGLocator(auto_drives=False)
    assert loc.resolve(str(d), _FN) == str(d / _FN)


def test_locator_adds_mat_extension(tmp_path):
    d = tmp_path / "sess"
    d.mkdir()
    (d / (_FN)).write_bytes(b"x")
    stem = _FN[:-4]  # without .mat
    loc = pe.EEGLocator(auto_drives=False)
    assert loc.resolve(str(d), stem) == str(d / _FN)


def test_locator_recursive_search_and_cache(tmp_path):
    # File lives deep under a search root; original folder is wrong.
    root = tmp_path / "drive"
    deep = root / "MONTH" / "session"
    deep.mkdir(parents=True)
    (deep / _FN).write_bytes(b"x")
    store: dict = {}
    loc = pe.EEGLocator(
        [str(root)], auto_drives=False,
        cache_get=store.get,
        cache_put=lambda n, p: store.__setitem__(n, p))
    got = loc.resolve("Z:\\offline\\wrong", _FN)
    assert got == str(deep / _FN)
    assert store[_FN] == got              # positive cache populated

    # Second resolve hits the cache (even with no roots configured).
    loc2 = pe.EEGLocator(auto_drives=False, cache_get=store.get)
    assert loc2.resolve("Z:\\offline\\wrong", _FN) == got


def test_locator_skips_system_dirs(tmp_path):
    root = tmp_path / "drive"
    win = root / "Windows" / "deep"
    win.mkdir(parents=True)
    (win / _FN).write_bytes(b"x")
    loc = pe.EEGLocator([str(root)], auto_drives=False)
    # The only copy is under a skipped dir -> not found.
    assert loc.resolve("", _FN) is None


def test_locator_negative_cache(tmp_path):
    store: dict = {}
    loc = pe.EEGLocator([str(tmp_path)], auto_drives=False,
                        cache_get=store.get,
                        cache_put=lambda n, p: store.__setitem__(n, p))
    assert loc.resolve("", "missing.mat") is None
    assert "missing.mat" in store and store["missing.mat"] is None


# ---- example assembly ------------------------------------------------ #

def test_catalog_examples_no_disk_access():
    # Cataloging is pure metadata -- no locator, no filesystem.
    evs = [
        pe.parse_row(_norm(folder="X:\\sess\\", filename=_FN, fs="20000",
                           Channel="5", Peak_Index="1", Score="2",
                           Onset="HYP", EventEO="40000"), "x.csv"),
        pe.parse_row(_norm(folder="X:\\sess\\", filename=_FN, fs="20000",
                           Channel="5", Peak_Index="2", Score="5",
                           Onset="LVF", EventEO="80000",
                           Peak_Stamp="999"), "x.csv"),
    ]
    seen = []
    out = pe.catalog_examples(evs, progress=lambda i, n, f: seen.append((i, n)))
    assert len(out) == 1
    ex = out[0]
    assert ex["filename"] == _FN and ex["folder"] == "X:\\sess\\"
    assert ex["recorded_path"].endswith(_FN)
    assert ex["animal"] == "BCH060"
    assert len(ex["markers"]) == 2
    assert ex["rep_racine"] == 5 and ex["rep_type"] == "LVF"  # highest racine
    assert ex["channel_names"][4] == "BCH060SR"
    assert seen and seen[-1] == (1, 1)   # one file group; progress reached total


def test_recorded_path_adds_mat_and_strips_trailing_sep():
    rec = pe._recorded_path("X:\\sess\\", "rec_no_ext")
    assert rec.endswith("rec_no_ext.mat")
    assert pe._recorded_path("", "x.mat") == "x.mat"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
