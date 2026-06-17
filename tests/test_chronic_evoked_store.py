"""Chronic Evoked Analyzer store query: per-animal evoked_summary across
all sessions, ordered, with the loose-LIKE guard.

Run with: pytest tests/test_chronic_evoked_store.py -q
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402


def _session(conn, sd, channels):
    conn.execute(
        "INSERT INTO session_config (session_dir, channel_names, "
        "eeg_channels, discovered_at) VALUES (?, ?, ?, '2026-01-01')",
        (sd, json.dumps(channels), json.dumps([1, 2])))


def _file(conn, fid, sd, dt, mean_peak):
    conn.execute(
        "INSERT INTO processed_files (id, file_path, session_dir, "
        "chunk_datetime) VALUES (?, ?, ?, ?)",
        (fid, f"/f/{fid}.mat", sd, dt))
    conn.execute(
        "INSERT INTO evoked_summary (file_id, mean_peak_amplitude, "
        "num_stimuli_detected) VALUES (?, ?, ?)",
        (fid, mean_peak, 10))


def test_per_animal_summary_filters_orders_and_guards(tmp_path):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    sa = "/srv/db/sessA_BCH062SR_"
    sb = "/srv/db/sessB_BCH070SR_"
    sc = "/srv/db/sessC_BCH0620SR_"        # decoy: matches the LIKE
    with store.connection() as conn:
        _session(conn, sa, ["stimCopy", "BCH062SR"])
        _session(conn, sb, ["stimCopy", "BCH070SR"])
        _session(conn, sc, ["stimCopy", "BCH0620SR"])
        # Out of order in the DB; the query must return ASC by time.
        _file(conn, 2, sa, "2026-05-10T06:00:00", 2.0)
        _file(conn, 1, sa, "2026-05-01T06:00:00", 1.0)
        _file(conn, 3, sb, "2026-05-05T06:00:00", 9.0)
        _file(conn, 4, sc, "2026-05-06T06:00:00", 5.0)
        conn.commit()

    rows = store.query_evoked_summary_for_animal("BCH062")
    assert [r["file_id"] for r in rows] == [1, 2]   # only BCH062, ASC
    r0 = rows[0]
    assert {"chunk_datetime", "session_name",
            "mean_peak_amplitude"} <= set(r0)
    assert r0["session_name"] == "sessA_BCH062SR_"

    # The decoy BCH0620 file matches the loose LIKE but the parser must
    # reject it -- the highest-value assertion.
    assert 4 not in [r["file_id"] for r in rows]

    # Other animal isolated; bogus / empty -> [].
    assert [r["file_id"]
            for r in store.query_evoked_summary_for_animal("BCH070")] == [3]
    assert store.query_evoked_summary_for_animal("BCH999") == []
    assert store.query_evoked_summary_for_animal("") == []


def test_hours_cutoff_drops_old_files(tmp_path):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    sd = "/srv/db/sessA_BCH062SR_"
    recent = datetime.now().isoformat()
    old = (datetime.now() - timedelta(days=40)).isoformat()
    with store.connection() as conn:
        _session(conn, sd, ["stimCopy", "BCH062SR"])
        _file(conn, 1, sd, old, 1.0)
        _file(conn, 2, sd, recent, 2.0)
        conn.commit()
    all_rows = store.query_evoked_summary_for_animal("BCH062")
    assert {r["file_id"] for r in all_rows} == {1, 2}
    last24 = store.query_evoked_summary_for_animal("BCH062", hours=24)
    assert [r["file_id"] for r in last24] == [2]   # old one dropped


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
