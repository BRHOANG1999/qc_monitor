"""Daily pre-ictal digest: content + the disabled / no-recipient / no-run guards.

Run with: pytest tests/test_preictal_digests.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import date

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.notifications.preictal_daily import send_preictal_daily  # noqa: E402


class _Emailer:
    def __init__(self):
        self.calls = []

    def send(self, subject, body, body_html=None, recipients=None,
             subject_prefix=True, severity="info", attachments=None,
             file_attachments=None):
        self.calls.append({"subject": subject, "body": body,
                           "html": body_html, "recipients": recipients,
                           "attachments": attachments,
                           "file_attachments": file_attachments})
        return True


_ENABLED = {"notifications": {"preictal_daily": {"enabled": True}},
            "alerting": {"smtp": {"recipients": ["pi@lab"]}}}


def _seed_run(store, *, surviving=True):
    rid = store.create_preictal_run("daily", "2026-07-08", "2026-07-08",
                                    ["BCH040"], derivatives_path="/p/run1")
    row = {"feature": "line_length", "channel_role": "eeg", "scale_index": 3,
           "scale": 8.0, "pseudo_freq_hz": 1.0 / 40.0,
           "coeff_mean": 1.2, "coeff_std": 0.3, "coeff_max": 2.0,
           "n_seizures": 12, "collapse_stat": 0.32 if surviving else 0.01,
           "collapse_loso_mean": 0.30, "collapse_loso_ci_lo": 0.2,
           "collapse_loso_ci_hi": 0.4, "null_p": 0.004 if surviving else 0.6,
           "null_percentile": 99.6, "forecast_roc_auc": 0.71,
           "forecast_pr_auc": 0.4}
    store.insert_preictal_scale_summaries(rid, [row])
    store.finish_preictal_run(rid, "done", n_seizures=12, n_scales=1,
                              ceiling_stats={"min": 100.0, "median": 900.0,
                                             "max": 3600.0})
    return rid


def test_disabled_and_no_recipients(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    assert send_preictal_daily(date(2026, 7, 9),
                               {"notifications": {"preictal_daily":
                                {"enabled": False}}}, store, _Emailer()
                               )["reason"] == "disabled"
    assert send_preictal_daily(
        date(2026, 7, 9),
        {"notifications": {"preictal_daily": {"enabled": True}}},
        store, _Emailer())["reason"] == "no recipients"


def test_no_completed_run(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    out = send_preictal_daily(date(2026, 7, 9), _ENABLED, store, _Emailer())
    assert out["sent"] is False and "no completed" in out["reason"]


def test_sends_and_names_surviving_scale(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    _seed_run(store, surviving=True)
    em = _Emailer()
    out = send_preictal_daily(date(2026, 7, 9), _ENABLED, store, em)
    assert out["sent"] is True and out["survivors"] == 1
    assert len(em.calls) == 1
    body = em.calls[0]["body"]
    assert "line_length" in body and "40s" in body        # named the scale
    assert em.calls[0]["html"]                             # html built
    assert "1 scale" in em.calls[0]["subject"]


def test_reports_when_nothing_survives(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    _seed_run(store, surviving=False)
    em = _Emailer()
    out = send_preictal_daily(date(2026, 7, 9), _ENABLED, store, em)
    assert out["sent"] is True and out["survivors"] == 0
    assert "no scale survives" in em.calls[0]["body"].lower() \
        or "survived the surrogate null" in em.calls[0]["body"].lower()
