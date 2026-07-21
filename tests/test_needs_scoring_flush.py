"""Periodic flush of 'Needs more onsets' events into the BHZ day CSVs.

Onsets reach needs_scoring by two routes and only the Submit path exported
inline; the 4-second autosave (upsert_scoring_draft) bypassed it, so those
onsets stayed in the DB only. These pin the worker's gating, idempotence and
crash-safety.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import needs_scoring_flush as NF  # noqa: E402


def _cfg(**over):
    b = {"enabled": True, "auto_flush_enabled": True,
         "auto_flush_interval_sec": 600}
    b.update(over)
    return {"bhz_csv": b}


def test_enabled_by_default():
    enabled, interval = NF._settings(_cfg())
    assert enabled is True and interval == 600.0


def test_disabled_when_auto_flush_off():
    assert NF._settings(_cfg(auto_flush_enabled=False))[0] is False


def test_disabled_when_csv_export_off():
    """If BHZ CSV export itself is off, the flush must not run."""
    assert NF._settings(_cfg(enabled=False))[0] is False
    assert NF._settings({})[0] is False


def test_interval_is_floored():
    """A tiny interval would hammer the share; clamp it."""
    assert NF._settings(_cfg(auto_flush_interval_sec=5))[1] == NF._MIN_INTERVAL_SEC
    assert NF._settings(_cfg(auto_flush_interval_sec="junk"))[1] == \
        NF.DEFAULT_INTERVAL_SEC


def test_start_worker_respects_disabled():
    assert NF.start_worker(object(), _cfg(auto_flush_enabled=False)) is False


def test_flush_once_never_raises(monkeypatch):
    """A failing export must not kill the worker loop."""
    import src.dashboard.tabs.video as V

    def boom(store, config):
        raise RuntimeError("share down")

    monkeypatch.setattr(V, "_export_all_needs_scoring", boom)
    out = NF.flush_once(object(), _cfg())
    assert out["errors"] == 1 and out["rows"] == 0


def test_flush_once_returns_summary(monkeypatch):
    import src.dashboard.tabs.video as V
    monkeypatch.setattr(V, "_export_all_needs_scoring",
                        lambda s, c: {"animals": 2, "files": 5, "rows": 7,
                                      "skipped": 1, "errors": 0})
    out = NF.flush_once(object(), _cfg())
    assert out["rows"] == 7 and out["files"] == 5
