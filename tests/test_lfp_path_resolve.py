"""LFP / Hilbert / AUC renders must resolve the LIVE path, not the stale DB path.

Regression cover for the Training historical-example bug: the initial load was
handed the recursively-resolved path, but a re-render (Hilbert<->AUC toggle, an
auto-refresh) re-derived the recording path from the DB row -- which for a
catalogued example is a stale, un-updatable path (another processed_files row
owns the located path via UNIQUE(file_path)). Result: video played but the LFP
and Hilbert panels failed with "[Errno 2] No such file". The renders now fall
back to Store.live_file_path (the resolver's cached live location).

Run: pytest tests/test_lfp_path_resolve.py -q
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dashboard.tabs import video as V        # noqa: E402
from src.dashboard.tabs import training as T     # noqa: E402


class _SpyStore:
    """Records live_file_path calls; returns a configurable path. Deliberately
    has NO other methods -- the render must consult live_file_path and stop at
    the not-found branch before touching anything else."""

    def __init__(self, path=None):
        self._path = path
        self.live_calls = []

    def live_file_path(self, file_id):
        self.live_calls.append(int(file_id))
        return self._path


def _fig_text(fig):
    return " ".join(str(getattr(a, "text", "") or "")
                    for a in (fig.layout.annotations or []))


# ---------------------------------------------------- resolver is used --- #

def test_hilbert_render_resolves_via_live_file_path():
    s = _SpyStore(path=None)
    fig, _ = V._render_hilbert_trace(s, 42, 0, file_path=None)
    assert s.live_calls == [42]                     # resolver consulted...
    assert "not found" in _fig_text(fig).lower()    # ...and its None honoured


def test_auc_render_resolves_via_live_file_path():
    s = _SpyStore(path=None)
    fig, _ = V._render_auc_trace(s, 7, 0, 60.0, file_path=None)
    assert s.live_calls == [7]
    assert "not found" in _fig_text(fig).lower()


def test_build_figures_resolves_via_live_file_path():
    s = _SpyStore(path=None)
    lfp, filt, dur = T._build_figures(s, 9, 0, file_path=None)
    assert s.live_calls and s.live_calls[0] == 9
    assert dur == 0.0
    assert "not found" in _fig_text(lfp).lower()


# ------------------------------------------ passed path skips resolver --- #

def test_explicit_file_path_bypasses_the_resolver():
    """When the caller already has the resolved path, don't re-resolve."""
    s = _SpyStore(path="/should/not/be/used.mat")
    # A bogus but truthy path: it gets past the not-found guard and fails later
    # trying to read it -- but live_file_path must NOT have been called.
    try:
        V._render_hilbert_trace(s, 1, 0, file_path="/tmp/given.mat")
    except Exception:
        pass
    assert s.live_calls == []
