"""Peri-ictal stim-fingerprint mapping: the verified positional rule (k-th
stimCopy -> STIM_REPORT Channel k+1, stimulating the NEXT channel), the
record-only path (animal not preceded by a stimCopy), and the fail-loud unknown
path (no report / count mismatch).

Run with: pytest tests/test_periictal_stim_map.py -q
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store            # noqa: E402
from src.periictal import stim_map as sm  # noqa: E402

# A real two-channel STIM_REPORT: Channel 1 = 5 nC, Channel 2 = 2 nC.
_REPORT = """\
--- CHANNEL PARAMETERS ---
Channel 1: ACTIVE
  Charge per phase:     5.0 nC
  Pulse width:          150 us
  Neg pulse ratio:      3.0 (width: 450 us)
  Frequency:            0.50 Hz
  Gain:                 1.00
Channel 2: ACTIVE
  Charge per phase:     2.0 nC
  Pulse width:          150 us
  Neg pulse ratio:      3.0 (width: 450 us)
  Frequency:            0.50 Hz
  Gain:                 1.00
--- CONVERSION CHAIN ---
"""


# ------------------------------------------------------------------ #
#  Pure rule: k-th stimCopy -> Channel k+1; record-only when no stimCopy before
# ------------------------------------------------------------------ #

def test_stim_ordinal_kth_stimcopy():
    # [stimCopy, A, stimCopy, B]: A follows stimCopy#0 -> Channel 1 (k=0);
    # B follows stimCopy#1 -> Channel 2 (k=1).
    names = ["stimCopy", "BCH110SR", "stimCopy", "BCH111SR"]
    assert sm._stim_ordinal(names, 1) == 0
    assert sm._stim_ordinal(names, 3) == 1


def test_stim_ordinal_record_only_when_not_preceded():
    # [stimCopy, A, B, C]: only A is stimulated; B, C are record-only.
    names = ["stimCopy", "BCH110SR", "BCH061SLM", "BCH111SR"]
    assert sm._stim_ordinal(names, 1) == 0
    assert sm._stim_ordinal(names, 2) is None     # B not preceded by stimCopy
    assert sm._stim_ordinal(names, 3) is None     # C not preceded by stimCopy


def test_fingerprint_key_groups_by_hardware():
    a = sm.StimFingerprint("mapped", 2, 2.0, 150.0, 3.0, 0.5, 1.0, True)
    b = sm.StimFingerprint("mapped", 1, 2.0, 150.0, 3.0, 0.5, 1.0, True)
    # Same hardware settings -> same key even though the report channel differs.
    assert a.key() == b.key() == "2nC/150us/x3/0.5Hz/g1"
    assert sm.StimFingerprint("record_only").key() == "record_only"
    inactive = sm.StimFingerprint("mapped", 2, 2.0, 150.0, 3.0, 0.5, 1.0, False)
    assert "inactive" in inactive.key()


# ------------------------------------------------------------------ #
#  Integration: resolve against a session_config row + an on-disk report
# ------------------------------------------------------------------ #

def _seed_session(store, session_dir, names):
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO session_config (session_dir, channel_names, "
            "discovered_at) VALUES (?,?,?)",
            (session_dir, json.dumps(names), "2026-07-13"))
        conn.commit()


def test_resolve_mapped_second_channel(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    sess = str(tmp_path / "chronicStim-5nC-2nC__stimCopy_BCH110SR_stimCopy_BCH111SR_")
    os.makedirs(sess, exist_ok=True)
    with open(os.path.join(sess, "x_STIM_REPORT.txt"), "w") as f:
        f.write(_REPORT)
    _seed_session(store, sess,
                  ["stimCopy", "BCH110SR", "stimCopy", "BCH111SR"])
    fp = sm.resolve_fingerprint(store, sess, "BCH111")
    assert fp.status == "mapped" and fp.report_channel == 2
    assert fp.charge_nC == 2.0 and fp.active is True
    assert fp.key() == "2nC/150us/x3/0.5Hz/g1"


def test_resolve_record_only_when_not_stimulated(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    sess = str(tmp_path / "stimStability__stimCopy_BCH110SR_BCH061SLM_BCH111SR_")
    os.makedirs(sess, exist_ok=True)
    with open(os.path.join(sess, "x_STIM_REPORT.txt"), "w") as f:
        f.write(_REPORT)
    # BCH111SR is at index 3, preceded by BCH061SLM (not a stimCopy).
    _seed_session(store, sess,
                  ["stimCopy", "BCH110SR", "BCH061SLM", "BCH111SR"])
    fp = sm.resolve_fingerprint(store, sess, "BCH111")
    assert fp.status == "record_only"


def test_resolve_unknown_when_no_report(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    sess = str(tmp_path / "chronicStim__stimCopy_BCH111SR_")   # no report file
    os.makedirs(sess, exist_ok=True)
    _seed_session(store, sess, ["stimCopy", "BCH111SR"])
    fp = sm.resolve_fingerprint(store, sess, "BCH111")
    assert fp.status == "unknown" and fp.report_channel == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
