"""Draft re-hydration from review_state.

Regression cover for the incident where BCH062 onsets looked ERASED after
``src/maintenance/revert_stranded_drafts.py`` moved stranded drafts back to the
Flag pool. That script sets status to 'abandoned' and PRESERVES markers_json by
contract -- but the Video Review editor only re-hydrated markers when the status
was exactly 'needs_scoring', so 55 recordings opened blank with their onsets
still sitting in the DB.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.tabs.video import restorable_drafts  # noqa: E402


def _row(status, events):
    return {"status": status, "markers_json": json.dumps(events)}


_ONSET = [{"type": "LVF", "EO_sec": 1627.901, "racine": None, "draft": True}]


# ------------------------------------------------- statuses that restore --- #

def test_needs_scoring_draft_restores():
    assert restorable_drafts(_row("needs_scoring", _ONSET)) == _ONSET


def test_abandoned_draft_restores():
    """The actual incident: reverted stranded draft must not read as empty."""
    assert restorable_drafts(_row("abandoned", _ONSET)) == _ONSET


def test_abandoned_row_preserves_the_exact_onset():
    got = restorable_drafts(_row("abandoned", _ONSET))
    assert got[0]["EO_sec"] == 1627.901
    assert got[0]["type"] == "LVF"


# --------------------------------------------- statuses that must NOT ----- #

def test_finalised_work_is_never_resurrected():
    """Submitted/approved rows are done -- re-hydrating them into the editor
    would present finalised scoring as unsaved draft work."""
    for st in ("pending_pi_review", "pi_approved", "has_events", "no_events"):
        assert restorable_drafts(_row(st, _ONSET)) == [], st


def test_reattributed_row_has_nothing_to_restore():
    """Store.reattribute_review writes the abandoned row with markers=[], so
    onsets moved to another animal never bleed back into the old one."""
    assert restorable_drafts(_row("abandoned", [])) == []


# ------------------------------------------------------------ robustness --- #

def test_missing_and_malformed_rows_are_safe():
    assert restorable_drafts(None) == []
    assert restorable_drafts({}) == []
    assert restorable_drafts({"status": "abandoned"}) == []
    assert restorable_drafts({"status": "abandoned",
                              "markers_json": "{not json"}) == []
    assert restorable_drafts({"status": "abandoned",
                              "markers_json": '{"a": 1}'}) == []
