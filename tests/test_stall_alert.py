"""AlertRuleEngine.check_processing_stalled -- keyed on the recording's
embedded datetime (backup-immune), not file mtime.

Run: pytest tests/test_stall_alert.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store            # noqa: E402
from src.alerting.email_alert import EmailAlerter  # noqa: E402
from src.alerting.rules import AlertRuleEngine     # noqa: E402

_CFG = {"alerting": {"enabled": False, "rules": {"no_data_hours": 2}}}


def _engine(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    return store, AlertRuleEngine(store, EmailAlerter(_CFG), _CFG)


def _seed_done(store, chunk_dt: str):
    with store.connection() as c:
        c.execute(
            "INSERT INTO processed_files (file_path, status, chunk_datetime, "
            "processed_at) VALUES (?, 'done', ?, ?)",
            (f"/{chunk_dt}.mat", chunk_dt, datetime.now().isoformat()))
        c.commit()


def _fired(store, atype="processing_stalled") -> bool:
    return any(a["alert_type"] == atype
               for a in store.get_recent_alerts(hours=1))


def test_genuine_stall_fires(tmp_path):
    store, eng = _engine(tmp_path)
    _seed_done(store, "2020_01_01__04_00_00")     # newest processed
    # A recording from 2020 (>> 2h ago) newer than that, still unprocessed.
    eng.check_processing_stalled({"newest_chunk_dt": "2020_01_01__06_00_00"})
    assert _fired(store)


def test_backup_recopy_does_not_fire(tmp_path):
    """The daily backup re-copies already-processed files with fresh mtimes;
    the newest recording ON the share == newest processed, so no stall."""
    store, eng = _engine(tmp_path)
    _seed_done(store, "2020_01_01__06_00_00")
    eng.check_processing_stalled({"newest_chunk_dt": "2020_01_01__06_00_00"})
    assert not _fired(store)


def test_fresh_recording_does_not_fire(tmp_path):
    """A brand-new recording (newer than processed, but < no_data_hours old)
    is still being worked -- don't alarm prematurely."""
    store, eng = _engine(tmp_path)
    _seed_done(store, "2020_01_01__06_00_00")      # old processed
    fresh = (datetime.now() - timedelta(minutes=10)).strftime(
        "%Y_%m_%d__%H_%M_%S")
    eng.check_processing_stalled({"newest_chunk_dt": fresh})
    assert not _fired(store)


def test_no_recordings_on_share_does_not_fire(tmp_path):
    store, eng = _engine(tmp_path)
    _seed_done(store, "2020_01_01__06_00_00")
    eng.check_processing_stalled({"newest_chunk_dt": ""})
    assert not _fired(store)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
