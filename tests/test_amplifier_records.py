"""Tests for File_Records amplifier-gain parsing + resolution."""

import pandas as pd
import pytest

from src.utils.amplifier_records import _parse_frame, resolve_gain


def _frame(rows):
    return pd.DataFrame(rows, columns=[
        "EEG File", "Animal ID & Channel", "Amplifier Gain", "Date"])


def test_pipe_split_and_exact_file_match():
    df = _frame([
        {"EEG File": "stimStability__stimCopy_BCH110SLM_stimCopy_BCH062SR___2026_07_02__15_47_09",
         "Animal ID & Channel": "stimCopy | BCH110SLM | stimCopy | BCH062SR",
         "Amplifier Gain": "1 | 300 | 1 | 150",
         "Date": "2026-07-02"},
    ])
    g = _parse_frame(df)
    # Exact file (basename w/o ext) + channel -> aligned gain.
    fb = "stimStability__stimCopy_BCH110SLM_stimCopy_BCH062SR___2026_07_02__15_47_09.mat"
    assert resolve_gain(g, "", fb, "BCH110SLM") == 300.0
    assert resolve_gain(g, "", fb, "BCH062SR") == 150.0


def test_session_match_when_chunk_timestamp_differs():
    # Sheet logs the session-start chunk; a later chunk in the same session
    # has a different timestamp -> exact file misses, session match hits.
    df = _frame([
        {"EEG File": "sess__stimCopy_BCH110SLM___2026_07_02__12_00_00",
         "Animal ID & Channel": "stimCopy | BCH110SLM",
         "Amplifier Gain": "1 | 300",
         "Date": "2026-07-02"},
    ])
    g = _parse_frame(df)
    session_name = "sess__stimCopy_BCH110SLM_"
    other_chunk = "sess__stimCopy_BCH110SLM___2026_07_02__15_47_09.mat"
    assert resolve_gain(g, session_name, other_chunk, "BCH110SLM") == 300.0


def test_latest_by_channel_fallback():
    df = _frame([
        {"EEG File": "old___2026_01_01__00_00_00",
         "Animal ID & Channel": "BCH110SLM",
         "Amplifier Gain": "200", "Date": "2026-01-01"},
        {"EEG File": "new___2026_06_01__00_00_00",
         "Animal ID & Channel": "BCH110SLM",
         "Amplifier Gain": "300", "Date": "2026-06-01"},
    ])
    g = _parse_frame(df)
    # Unknown file/session -> latest-known gain for the channel wins.
    assert resolve_gain(g, "unknown", "unknown.mat", "BCH110SLM") == 300.0


def test_animal_electrode_fallback():
    df = _frame([
        {"EEG File": "f___2026_06_01__00_00_00",
         "Animal ID & Channel": "BCH062SR",
         "Amplifier Gain": "150", "Date": "2026-06-01"},
    ])
    g = _parse_frame(df)
    # Channel name spelled differently but same animal+electrode.
    assert resolve_gain(g, "x", "y.mat", "BCH062 SR",
                        animal_id="BCH062", electrode="SR") == 150.0


def test_na_gain_skipped():
    df = _frame([
        {"EEG File": "f___2026_06_01__00_00_00",
         "Animal ID & Channel": "BCH110SLM | saline",
         "Amplifier Gain": "300 | N/A", "Date": "2026-06-01"},
    ])
    g = _parse_frame(df)
    fb = "f___2026_06_01__00_00_00.mat"
    assert resolve_gain(g, "", fb, "BCH110SLM") == 300.0
    assert resolve_gain(g, "", fb, "saline") is None


def test_unknown_channel_returns_none():
    df = _frame([
        {"EEG File": "f___2026_06_01__00_00_00",
         "Animal ID & Channel": "BCH110SLM",
         "Amplifier Gain": "300", "Date": "2026-06-01"},
    ])
    g = _parse_frame(df)
    assert resolve_gain(g, "", "f.mat", "BCH999XX") is None
    assert resolve_gain(None, "", "f.mat", "BCH110SLM") is None
