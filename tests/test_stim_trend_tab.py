"""Stim Artifact Trend tab: layout id-contract + NAV registration + the figure
builders. Mirrors the id-contract style of tests/test_periictal_explorer.py.

Run: pytest tests/test_stim_trend_tab.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dashboard.tabs import stim_trend as st         # noqa: E402


def _walk_ids(node, out):
    cid = getattr(node, "id", None)
    if isinstance(cid, str):
        out.add(cid)
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for ch in children:
            _walk_ids(ch, out)
    else:
        _walk_ids(children, out)


_REQUIRED_IDS = {
    "st-fb-open", "st-fb-result", "st-fb-modal", "stim-trend-folder",
    "stim-trend-folder-label", "stim-trend-metric", "stim-trend-poll",
    "stim-trend-status", "stim-trend-summary", "stim-trend-overlay",
    "stim-trend-metric-graph",
}


def test_layout_builds_with_id_contract():
    ids: set = set()
    _walk_ids(st.layout(None, {}), ids)
    assert _REQUIRED_IDS <= ids, f"missing: {_REQUIRED_IDS - ids}"


def test_nav_group_has_stim_trend():
    from src.dashboard import app as _app
    grp = next((g for g in _app.NAV_GROUPS if g["id"] == "quality"), None)
    assert grp is not None
    assert "stim_trend" in [s["id"] for s in grp["subs"]]
    assert _app.SUBTAB_TO_GROUP["stim_trend"] == "quality"


def _rec(hours, ptp, ra=None):
    tm = [(-1.0 + 0.05 * i) for i in range(60)]
    mean = [0.0] * 20 + [1.0, -0.5] + [0.0] * 38     # a token artifact shape
    return {"ok": True, "dt": datetime(2026, 9, 11, 15) + timedelta(hours=hours),
            "time_ms": tm, "mean_trace": mean,
            "metrics": {"ptp": ptp, "pos": 1.0, "neg": -0.5, "ra": ra,
                        "zss": None, "sat": 0.0},
            "stim_params": {"source": "report"}}


def test_overlay_and_trend_build():
    recs = [_rec(0, 1.6, ra=0.8), _rec(1, 1.4, ra=0.7), _rec(2, 1.5, ra=0.75)]
    ov = st._build_overlay(recs)
    assert len(ov.data) == 3                          # one trace per recording
    tr = st._build_metric_trend(recs, "ptp")
    assert len(tr.data) == 1 and list(tr.data[0].y) == [1.6, 1.4, 1.5]
    # a metric with no values (all None) -> a friendly empty figure, no traces
    empty = st._build_metric_trend([_rec(0, 1.0)], "zss")
    assert len(empty.data) == 0


def test_summary_flags_missing_gain():
    recs = [_rec(0, 1.6, ra=None)]                    # no Rₐ -> gain note
    s = st._build_summary(recs)
    assert "Rₐ/Z_ss are unavailable" in s


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
