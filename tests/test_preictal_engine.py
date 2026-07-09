"""Pre-ictal engine end-to-end (Stage 1): a synthetic store with a planted
pre-ictal oscillation at a known time-scale must run through events -> trajectory
-> CWT -> per-scale summaries + a BIDS pocket, and surface at that scale.

Run with: pytest tests/test_preictal_engine.py -q
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.preictal import engine  # noqa: E402
from src.preictal.registry import _FEATURES, register_feature  # noqa: E402
from src.utils.mat_loader import ChunkData  # noqa: E402

# A controlled test feature: the window mean, so the trajectory == the planted
# per-window signal. (Registered once, guarded against re-import.)
if "_win_mean" not in _FEATURES:
    @register_feature("_win_mean")
    def _win_mean(window, fs):        # noqa: ANN001
        return float(np.mean(window))

_FS = 100
_PERIOD = 40.0                        # trajectory oscillation period (seconds)


def _planted_chunk(_path):
    """A 700 s single-channel chunk whose 100 s..600 s span carries a 40 s
    oscillation in its per-1 s-window mean; elsewhere zero."""
    n = 700 * _FS
    sig = np.zeros((n, 1), dtype=float)
    for i in range(500):              # windows over [100 s, 600 s]
        v = np.sin(2 * np.pi * i / _PERIOD)
        sig[10000 + i * 100:10000 + (i + 1) * 100, 0] = v
    return ChunkData(signal=sig, fs=float(_FS), num_channels=1, num_samples=n,
                     duration_sec=700.0, source_path=_path)


def _config(root):
    return {"preictal": {
        "derivatives_root": root,
        "features": ["_win_mean"],
        "trajectory": {"step_sec": 1.0, "target_fs": float(_FS)},
        "cwt": {"wavelet": "cmor1.5-1.0", "min_leadtime_sec": 1.0,
                "scales_per_octave": 6},
        "events": {"event_source": "behavioral", "post_ictal_buffer_sec": 0.0},
    }}


def test_engine_end_to_end_planted_scale(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "m.db"))
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO processed_files (id, file_path, session_dir, "
            "chunk_datetime, duration_sec, sampling_rate) VALUES "
            "(1,'/sess/1.mat','/sess','2026_01_01__00_00_00',700.0,100.0)")
        # Two seizures in one file: onsets 100 s and 600 s -> ISI 500 s.
        markers = [{"EO_sec": 100.0, "racine": 3, "type": "LVF"},
                   {"EO_sec": 600.0, "racine": 4, "type": "LVF"}]
        conn.execute(
            "INSERT INTO review_state (file_id, user_email, status, animal_id, "
            "markers_json, created_at, updated_at) VALUES "
            "(1,'pi','pi_approved','BCH040',?, 't','t')", (json.dumps(markers),))
        conn.commit()
    monkeypatch.setattr("src.utils.chunk_cache.get_chunk", _planted_chunk)

    root = str(tmp_path / "derivatives" / "preictal_cwt")
    run_id = engine.run_sweep(store, _config(root), scope="adhoc",
                              animals=["BCH040"])

    # Run completed; the first-of-animal seizure is dropped (no ISI), one used.
    run = store.latest_preictal_run("adhoc")
    assert run["id"] == run_id and run["status"] == "done"
    assert run["n_seizures"] == 2
    assert run["ceiling_median_sec"] == 500.0

    rows = store.preictal_scale_summary_for_run(run_id)
    assert rows and all(r["feature"] == "_win_mean" for r in rows)
    assert all(r["n_seizures"] == 1 for r in rows)
    # The planted 40 s oscillation must be the dominant scale.
    peak = max(rows, key=lambda r: r["coeff_mean"])
    assert abs(peak["pseudo_freq_hz"] - 1.0 / _PERIOD) < 0.3 / _PERIOD

    # Pocket written + queryable.
    runs_dir = os.path.join(root, "runs")
    pockets = os.listdir(runs_dir)
    assert len(pockets) == 1
    assert os.path.exists(os.path.join(runs_dir, pockets[0],
                                       "scale_summary.csv"))


def test_period_scope_filters_by_onset_but_keeps_full_isi(tmp_path):
    # Rolling window = one day. A seizure IN the window is analyzed with its
    # ceiling derived from a predecessor BEFORE the window; seizures outside the
    # window are excluded from analysis (but still seed ISI).
    store = Store(str(tmp_path / "data" / "m.db"))
    days = [("2026_07_06__00_00_00", 1),   # before window
            ("2026_07_07__00_00_00", 2),   # in window
            ("2026_07_08__00_00_00", 3)]   # after window
    with store.connection() as conn:
        for i, (cd, _r) in enumerate(days, start=1):
            conn.execute("INSERT INTO processed_files (id, file_path, "
                         "session_dir, chunk_datetime, duration_sec) VALUES "
                         "(?,?,?,?,3600.0)", (i, f"/s/{i}.mat", "/s", cd))
            conn.execute("INSERT INTO review_state (file_id, user_email, status, "
                         "animal_id, markers_json, created_at, updated_at) "
                         "VALUES (?,?,?,?,?,?,?)",
                         (i, "p", "pi_approved", "BCH040",
                          json.dumps([{"EO_sec": 0.0, "racine": 3}]),
                          "t", "t"))
        conn.commit()
    from src.preictal.engine import _enumerate_seizures, _period_bounds
    lo, hi = _period_bounds("2026-07-07", "2026-07-07")
    pairs, rows = _enumerate_seizures(store, ["BCH040"], buffer_sec=300.0,
                                      min_lead=1.0, lo=lo, hi=hi)
    # Only the 2026-07-07 seizure is in scope.
    assert len(rows) == 1 and rows[0]["chunk_datetime"] == "2026_07_07__00_00_00"
    # Its ceiling comes from the 2026-07-06 predecessor (ISI 1 day - buffer).
    assert len(pairs) == 1
    assert abs(pairs[0][1] - (86400.0 - 300.0)) < 1e-6


def test_engine_no_seizures_is_clean_done(tmp_path):
    """No usable seizures -> a clean 'done' run with 0 scales (no crash)."""
    store = Store(str(tmp_path / "data" / "m.db"))
    root = str(tmp_path / "deriv")
    run_id = engine.run_sweep(store, _config(root), scope="daily",
                              animals=["BCH999"])
    run = store.latest_preictal_run("daily")
    assert run["id"] == run_id and run["status"] == "done"
    assert run["n_scales"] == 0
    assert store.preictal_scale_summary_for_run(run_id) == []
