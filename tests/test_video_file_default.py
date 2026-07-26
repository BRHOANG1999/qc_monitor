"""File-dropdown default resolution for the Video Review tab.

Regression cover for the "Now viewing (scoring)" banner showing the wrong
recording: a CROSS-SESSION queue/flag click set (session, file) in one shot, but
the session change fired _update_files, whose State read of the just-set file
value raced and fell through to the session's OLDEST chunk ("hour 1 of N")
instead of the clicked recording. The fix routes an explicit (session, file)
request through a store that _pick_file_default honours deterministically,
independent of that State read.

Run: pytest tests/test_video_file_default.py -q
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dashboard.tabs.video import _pick_file_default, _req_file  # noqa: E402

_SESS = "/data/chronicStim"
_OPTS = [10, 11, 12, 13]        # option file_ids; 10 = oldest (options[0])


def _req(fid, session=_SESS):
    return {"session_dir": session, "file_id": fid}


# ---------------------------------------------- the incident it fixes --- #

def test_request_wins_over_stale_current_value_cross_session():
    """THE bug: clicked file 13, but current_value is a PREVIOUS session's file
    not in options. Without the request we fell to options[0] (oldest = hour 1);
    the request pins the clicked file."""
    got = _pick_file_default(_OPTS, _SESS, bridge=None, requested=_req(13),
                             current_value=999)
    assert got == 13


def test_without_request_the_old_race_falls_to_oldest():
    """Documents the pre-fix behaviour the request had to override."""
    got = _pick_file_default(_OPTS, _SESS, bridge=None, requested=None,
                             current_value=999)
    assert got == 10        # options[0] = oldest = "hour 1"


# ------------------------------------------------------------ priority --- #

def test_bridge_beats_request():
    """A fresh LFP-Browser hand-off outranks a (possibly older) queue request."""
    got = _pick_file_default(_OPTS, _SESS, bridge=_req(11), requested=_req(13),
                             current_value=12)
    assert got == 11


def test_request_beats_current_value():
    got = _pick_file_default(_OPTS, _SESS, bridge=None, requested=_req(13),
                             current_value=12)
    assert got == 13


def test_current_value_preserved_when_no_request():
    got = _pick_file_default(_OPTS, _SESS, bridge=None, requested=None,
                             current_value=12)
    assert got == 12


# --------------------------------------------- stale-guard by session --- #

def test_request_for_other_session_is_ignored():
    """A request naming a DIFFERENT session must not hijack this load -- else a
    stale queue click would override a later cross-session navigation."""
    got = _pick_file_default(_OPTS, _SESS, bridge=None,
                             requested=_req(13, session="/data/other"),
                             current_value=12)
    assert got == 12        # falls to current_value, not the stale request


def test_request_file_not_in_options_is_ignored():
    got = _pick_file_default(_OPTS, _SESS, bridge=None, requested=_req(999),
                             current_value=12)
    assert got == 12


def test_bridge_for_other_session_ignored():
    got = _pick_file_default(_OPTS, _SESS, bridge=_req(11, session="/x"),
                             requested=None, current_value=12)
    assert got == 12


# ------------------------------------------------------------- corners --- #

def test_empty_options_yields_none():
    assert _pick_file_default([], _SESS, None, _req(13), 12) is None


def test_none_current_value_falls_to_oldest():
    assert _pick_file_default(_OPTS, _SESS, None, None, None) == 10


def test_req_file_shape():
    assert _req_file(_SESS, "13") == {"session_dir": _SESS, "file_id": 13}
