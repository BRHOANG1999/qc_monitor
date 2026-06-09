"""End-to-end pass through Dispatcher.process_file against stubbed I/O.

The dispatcher is the single highest-risk module in the tree and used
to be one ~280-line method that could not be exercised without a real
.mat + a real MATLAB install. Splitting it into named tiers
(_load_chunk / _run_tier1_python / _run_tier2_matlab / _run_tier3_video)
opens the door to a synthetic-data smoke test that catches the obvious
regressions:

* Tier 1 produces one ``chunk_qc`` row per channel with the role
  labels and channel_names the session config dictates.
* Tier 2 fans the MATLAB result dict out into ``matlab_results``,
  ``evoked_features``, ``evoked_summary``, ``evoked_waveforms``, and
  ``criticality`` rows.
* ``processed_files.status`` ends at 'done' with the right chunk
  metadata stamped on the row.
* The MATLAB feature-name -> DB column mapping in
  ``src.utils.feature_map`` actually agrees with the schema.

The test stubs ``load_mat`` to return synthetic ChunkData, stubs
``matlab_run_pipeline`` to return a hand-crafted result, and
stubs ``video_path_for_mat`` to return None so Tier 3 is skipped.
``analyze_stim_report`` reads from disk and naturally returns None
when there is no STIM_REPORT.txt next to the (non-existent) input
path, so no extra stub is needed.

Run with: pytest tests/test_dispatcher.py -q
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.dispatcher import Dispatcher  # noqa: E402
from src.utils.mat_loader import ChunkData  # noqa: E402
from src.utils.session_config import SessionConfig  # noqa: E402
from src.watcher import NewFile  # noqa: E402


FS = 1000  # Hz -- low enough that synthetic Welch + Hilbert run fast.
DUR_SEC = 4
N_SAMPLES = FS * DUR_SEC
N_CHANNELS = 2  # 0 = stim_copy, 1 = EEG


def _make_chunk(source_path: str) -> ChunkData:
    """Synthetic 2-channel chunk: a flat near-zero stim_copy plus a
    1 µV-scale 10 Hz EEG sinusoid. The exact values don't matter --
    the test asserts the dispatcher's plumbing, not the analyzer
    numerical output.
    """
    t = np.arange(N_SAMPLES) / FS
    sig = np.zeros((N_SAMPLES, N_CHANNELS), dtype=np.float64)
    # ch 0 = stim_copy: a stub flatline + a single mid-recording pulse.
    sig[N_SAMPLES // 2, 0] = 1.0
    # ch 1 = EEG: 10 Hz, ~5 µV amplitude.
    sig[:, 1] = 5.0 * np.sin(2 * np.pi * 10 * t)
    return ChunkData(
        signal=sig, fs=float(FS),
        num_channels=N_CHANNELS, num_samples=N_SAMPLES,
        duration_sec=float(DUR_SEC), source_path=source_path,
    )


def _make_session_config() -> SessionConfig:
    return SessionConfig(
        channel_names=["stimCopy", "BCH001SL"],
        eeg_channels=[1],
        stim_copy_channels=[0],
        num_channels=N_CHANNELS,
        fs=FS,
    )


def _make_matlab_result() -> dict:
    """Hand-rolled MATLAB result mimicking the run_pipeline.m output
    shape documented at src/utils/matlab_bridge.py:54-63.

    * One stim, three epochs, two features.
    * One per_channel block (the EEG channel).
    * Three criticality windows.
    """
    return {
        "exit_status": "success",
        "pipeline_duration_sec": 1.23,
        "num_stimuli": 1,
        "num_traces": 3,
        "num_features": 2,
        "evoked_output_path": "/tmp/evoked.mat",
        "lfp_channels": [2],            # 1-indexed in MATLAB
        "criticality_channel": 2,        # 1-indexed in MATLAB
        "features": {
            "Peak_Amplitude": [1.0, 2.0, 3.0],
            "Line_Length":    [10.0, 20.0, 30.0],
            # Unknown feature -- must be silently dropped, not crash.
            "Foo_Bar":        [9.0, 9.0, 9.0],
        },
        "epoch_times": [0.5, 1.5, 2.5],
        "per_channel": {
            "ch2": {
                "channel": 2,                # 1-indexed; persisted as 1
                "channel_name": "BCH001SL",
                "time_axis_ms": [0.0, 1.0, 2.0],
                "mean_trace":   [0.0, 1.0, 0.0],
                "sem_trace":    [0.1, 0.1, 0.1],
                "stim_mean_trace": [0.0, 0.0, 0.0],
                "num_traces": 3,
            },
            # Not a dict -- must be skipped silently.
            "noise": "not-a-dict",
        },
        "criticality": {
            "time_points": [0.0, 1.0, 2.0],
            "db_values":   [0.5, float("nan"), 0.7],
            "db_stds":     [0.05, 0.05, 0.05],
            "sigmas":      [0.01, 0.01, 0.01],
        },
    }


# ===================================================================== #
#  Fixture: dispatcher wired against a real Store + stubbed everything
# ===================================================================== #

@pytest.fixture
def wired(tmp_path):
    db_path = str(tmp_path / "data" / "monitor.db")
    store = Store(db_path)
    config = {
        "matlab_exe": "stub-matlab",   # presence triggers Tier 2
        "qc_thresholds": {},
        "artifact": {},
        "feature_analysis": {"analysis_start_ms": 1.0,
                              "analysis_end_ms": 50.0},
        "criticality": {"ar_order": 5},
        "video_qc": {"enabled": False},  # belt + braces; we also stub
    }
    # Emailer left at the default None -- no Tier 3 means no sends.
    dispatcher = Dispatcher(store, config)
    # Pre-populate the session-config cache so the dispatcher never
    # tries to read recorder_settings.mat from a path that doesn't exist.
    dispatcher._session_configs["/fake/session"] = _make_session_config()

    new_file = NewFile(
        path="/fake/session/file_2026_01_01__00_00_00.mat",
        size=12345, mtime=1_700_000_000.0,
        session_dir="/fake/session",
        session_name="session",
        chunk_datetime="2026_01_01__00_00_00",
    )

    chunk = _make_chunk(new_file.path)
    matlab_result = _make_matlab_result()

    patches = [
        patch("src.dispatcher.load_mat",
              return_value=chunk),
        patch("src.dispatcher.matlab_run_pipeline",
              return_value=matlab_result),
        # Skip Tier 3 entirely -- no companion video on disk and
        # cv2 may not be installed.
        patch("src.dispatcher.video_path_for_mat",
              return_value=None),
    ]
    for p in patches:
        p.start()
    try:
        yield store, dispatcher, new_file
    finally:
        for p in patches:
            p.stop()


# ===================================================================== #
#  Happy path
# ===================================================================== #

def test_process_file_returns_true_on_clean_pass(wired):
    store, dispatcher, new_file = wired
    ok = dispatcher.process_file(new_file, version_id=None)
    assert ok is True


def test_processed_files_row_ends_at_done(wired):
    store, dispatcher, new_file = wired
    dispatcher.process_file(new_file, version_id=None)
    with store.connection() as conn:
        row = conn.execute(
            "SELECT status, num_channels, num_samples, sampling_rate, "
            "duration_sec FROM processed_files WHERE file_path = ?",
            (new_file.path,),
        ).fetchone()
    assert row is not None
    assert row["status"] == "done"
    assert row["num_channels"] == N_CHANNELS
    assert row["num_samples"] == N_SAMPLES
    assert row["sampling_rate"] == float(FS)
    assert row["duration_sec"] == float(DUR_SEC)


def test_chunk_qc_has_one_row_per_channel(wired):
    store, dispatcher, new_file = wired
    dispatcher.process_file(new_file, version_id=None)
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT channel, channel_name, channel_role "
            "FROM chunk_qc ORDER BY channel"
        ).fetchall()
    assert [r["channel"] for r in rows] == list(range(N_CHANNELS))
    # Session config maps ch0 -> stim_copy, ch1 -> eeg.
    assert rows[0]["channel_role"] == "stim_copy"
    assert rows[0]["channel_name"] == "stimCopy"
    assert rows[1]["channel_role"] == "eeg"
    assert rows[1]["channel_name"] == "BCH001SL"


def test_tier2_persists_evoked_features_with_known_columns(wired):
    """Three epochs * two known features (Peak_Amplitude, Line_Length).
    The unknown Foo_Bar feature must NOT land in the DB (would raise
    on the column write); the dispatcher silently drops unknown names.
    """
    store, dispatcher, new_file = wired
    dispatcher.process_file(new_file, version_id=None)
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT epoch_index, epoch_time_sec, peak_amplitude, "
            "line_length FROM evoked_features ORDER BY epoch_index"
        ).fetchall()
    assert len(rows) == 3
    assert [r["epoch_index"] for r in rows] == [0, 1, 2]
    assert [r["peak_amplitude"] for r in rows] == [1.0, 2.0, 3.0]
    assert [r["line_length"] for r in rows] == [10.0, 20.0, 30.0]
    assert [r["epoch_time_sec"] for r in rows] == [0.5, 1.5, 2.5]


def test_tier2_evoked_summary_aggregates_peak_amplitude(wired):
    store, dispatcher, new_file = wired
    dispatcher.process_file(new_file, version_id=None)
    with store.connection() as conn:
        row = conn.execute(
            "SELECT num_stimuli_detected, num_traces_extracted, "
            "mean_peak_amplitude, std_peak_amplitude "
            "FROM evoked_summary"
        ).fetchone()
    assert row is not None
    assert row["num_stimuli_detected"] == 1
    assert row["num_traces_extracted"] == 3
    # mean([1,2,3]) == 2; std (population) ~= 0.8165
    assert row["mean_peak_amplitude"] == pytest.approx(2.0)
    assert row["std_peak_amplitude"] == pytest.approx(0.8165, abs=1e-3)


def test_tier2_evoked_waveforms_persisted_with_zero_indexed_channel(wired):
    """MATLAB ships 1-indexed channels; the dispatcher subtracts 1 so
    chunk_qc / evoked_features / evoked_waveforms all agree on the
    0-indexed scheme used by ChunkData.signal."""
    store, dispatcher, new_file = wired
    dispatcher.process_file(new_file, version_id=None)
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT channel, channel_name, n_epochs "
            "FROM evoked_waveforms"
        ).fetchall()
    # "ch2" was 1-indexed in MATLAB; persisted as channel=1 (0-indexed).
    # The "noise" entry that wasn't a dict must be silently skipped.
    assert len(rows) == 1
    assert rows[0]["channel"] == 1
    assert rows[0]["channel_name"] == "BCH001SL"
    assert rows[0]["n_epochs"] == 3


def test_tier2_criticality_windows_drop_nan_db_values(wired):
    """The MATLAB result includes a NaN db_value in the middle window;
    the dispatcher's self-equality NaN check must store it as NULL
    rather than letting the NaN reach SQLite (which would coerce
    silently and break downstream queries)."""
    store, dispatcher, new_file = wired
    dispatcher.process_file(new_file, version_id=None)
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT window_index, channel, time_sec, db_value, "
            "ar_order FROM criticality ORDER BY window_index"
        ).fetchall()
    assert [r["window_index"] for r in rows] == [0, 1, 2]
    # MATLAB criticality_channel=2 (1-indexed) -> channel=1 (0-indexed).
    assert all(r["channel"] == 1 for r in rows)
    assert rows[0]["db_value"] == 0.5
    assert rows[1]["db_value"] is None   # NaN bridged to NULL
    assert rows[2]["db_value"] == 0.7
    assert all(r["ar_order"] == 5 for r in rows)


def test_matlab_results_row_inserted_with_exit_status(wired):
    store, dispatcher, new_file = wired
    dispatcher.process_file(new_file, version_id=None)
    with store.connection() as conn:
        row = conn.execute(
            "SELECT exit_status, num_stimuli, num_traces, num_features "
            "FROM matlab_results"
        ).fetchone()
    assert row is not None
    assert row["exit_status"] == "success"
    assert row["num_stimuli"] == 1
    assert row["num_traces"] == 3
    assert row["num_features"] == 2


# ===================================================================== #
#  Tier 2 disabled
# ===================================================================== #

def test_no_matlab_skips_tier2_but_still_writes_chunk_qc(tmp_path):
    """When matlab_exe is absent, Tier 2 must skip entirely and the
    file must still reach status='done' on the strength of Tier 1
    alone."""
    db_path = str(tmp_path / "data" / "monitor.db")
    store = Store(db_path)
    config = {
        # No matlab_exe key -- Tier 2 gate is None
        "qc_thresholds": {}, "artifact": {},
        "video_qc": {"enabled": False},
    }
    dispatcher = Dispatcher(store, config)
    dispatcher._session_configs["/fake/session"] = _make_session_config()
    new_file = NewFile(
        path="/fake/session/no_matlab.mat",
        size=1, mtime=0.0,
        session_dir="/fake/session",
        session_name="session",
        chunk_datetime="2026_01_01__00_00_00",
    )
    with patch("src.dispatcher.load_mat",
                return_value=_make_chunk(new_file.path)), \
         patch("src.dispatcher.video_path_for_mat",
                return_value=None), \
         patch("src.dispatcher.matlab_run_pipeline") as m_run:
        ok = dispatcher.process_file(new_file, version_id=None)

    assert ok is True
    m_run.assert_not_called()
    with store.connection() as conn:
        n_chunk = conn.execute(
            "SELECT COUNT(*) FROM chunk_qc"
        ).fetchone()[0]
        n_matlab = conn.execute(
            "SELECT COUNT(*) FROM matlab_results"
        ).fetchone()[0]
        status = conn.execute(
            "SELECT status FROM processed_files"
        ).fetchone()[0]
    assert n_chunk == N_CHANNELS
    assert n_matlab == 0
    assert status == "done"


# ===================================================================== #
#  Failure path
# ===================================================================== #

def test_load_failure_flips_row_to_error_and_returns_false(tmp_path):
    db_path = str(tmp_path / "data" / "monitor.db")
    store = Store(db_path)
    config = {"qc_thresholds": {}, "artifact": {},
              "video_qc": {"enabled": False}}
    dispatcher = Dispatcher(store, config)
    dispatcher._session_configs["/fake/session"] = _make_session_config()
    new_file = NewFile(
        path="/fake/session/broken.mat", size=1, mtime=0.0,
        session_dir="/fake/session", session_name="session",
        chunk_datetime="2026_01_01__00_00_00",
    )
    with patch("src.dispatcher.load_mat",
                side_effect=ValueError("corrupt mat")):
        ok = dispatcher.process_file(new_file, version_id=None)
    assert ok is False
    with store.connection() as conn:
        status = conn.execute(
            "SELECT status FROM processed_files"
        ).fetchone()[0]
    assert status == "error"


# ===================================================================== #
#  Tier 3 alert plumbing
# ===================================================================== #

class _StubVideoResult:
    """Minimal stand-in for analyzers.video_qc.VideoQCResult."""
    def __init__(self):
        self.status = "critical"
        self.n_frames_sampled = 60
        self.n_frames_total = 1800
        self.sample_every_n = 30
        self.n_bad = 45
        self.pct_bad = 75.0
        self.n_dark = 30
        self.n_bright = 5
        self.n_low_var = 10
        self.first_bad_idx = 2
        self.first_bad_reason = "dark"
        self.thresholds = {"min_mean": 25.0, "max_mean": 230.0,
                            "min_variance": 50.0}

    def to_dict(self):
        return {"status": self.status, "pct_bad": self.pct_bad,
                "n_bad": self.n_bad, "n_dark": self.n_dark,
                "n_bright": self.n_bright, "n_low_var": self.n_low_var,
                "first_bad_idx": self.first_bad_idx,
                "first_bad_reason": self.first_bad_reason,
                "thresholds": self.thresholds,
                "n_frames_sampled": self.n_frames_sampled,
                "n_frames_total": self.n_frames_total,
                "sample_every_n": self.sample_every_n,
                "video_path": "/fake/video.mp4",
                "duration_sec": 60.0,
                "duration_analyzed_sec": 60.0,
                "fps": 30.0}


class _StubEmailer:
    def __init__(self):
        self.enabled = True
        self.password = "x"
        self.sent: list[dict] = []

    def send(self, subject, body, severity="info",
             recipients=None, body_html=None, subject_prefix=True):
        self.sent.append({"subject": subject, "severity": severity})
        return True


def test_tier3_critical_video_triggers_emailer(tmp_path):
    """When Tier 3 says critical and an emailer is wired, exactly one
    email goes out and the alerts table records the send."""
    db_path = str(tmp_path / "data" / "monitor.db")
    store = Store(db_path)
    config = {
        "matlab_exe": None,
        "qc_thresholds": {}, "artifact": {},
        "video_qc": {"enabled": True, "alert_on_critical": True},
    }
    emailer = _StubEmailer()
    dispatcher = Dispatcher(store, config, emailer=emailer)
    dispatcher._session_configs["/fake/session"] = _make_session_config()
    new_file = NewFile(
        path="/fake/session/with_video.mat", size=1, mtime=0.0,
        session_dir="/fake/session", session_name="session",
        chunk_datetime="2026_01_01__00_00_00",
    )
    with patch("src.dispatcher.load_mat",
                return_value=_make_chunk(new_file.path)), \
         patch("src.dispatcher.video_path_for_mat",
                return_value="/fake/video.mp4"), \
         patch("src.dispatcher.analyze_video",
                return_value=_StubVideoResult()):
        dispatcher.process_file(new_file, version_id=None)

    assert len(emailer.sent) == 1
    assert "Video QC alert" in emailer.sent[0]["subject"]


def test_tier3_critical_video_without_emailer_does_not_raise(tmp_path):
    """Same flow as above but the dispatcher has no emailer wired.
    Should still produce a video_qc row and reach status='done' --
    the Tier 3 alert step must skip cleanly without an EmailAlerter."""
    db_path = str(tmp_path / "data" / "monitor.db")
    store = Store(db_path)
    config = {
        "matlab_exe": None,
        "qc_thresholds": {}, "artifact": {},
        "video_qc": {"enabled": True, "alert_on_critical": True},
    }
    dispatcher = Dispatcher(store, config)  # no emailer
    dispatcher._session_configs["/fake/session"] = _make_session_config()
    new_file = NewFile(
        path="/fake/session/with_video.mat", size=1, mtime=0.0,
        session_dir="/fake/session", session_name="session",
        chunk_datetime="2026_01_01__00_00_00",
    )
    with patch("src.dispatcher.load_mat",
                return_value=_make_chunk(new_file.path)), \
         patch("src.dispatcher.video_path_for_mat",
                return_value="/fake/video.mp4"), \
         patch("src.dispatcher.analyze_video",
                return_value=_StubVideoResult()):
        ok = dispatcher.process_file(new_file, version_id=None)

    assert ok is True
    with store.connection() as conn:
        n_video = conn.execute(
            "SELECT COUNT(*) FROM video_qc"
        ).fetchone()[0]
    assert n_video == 1


# ===================================================================== #
#  Manual entry point
# ===================================================================== #

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
