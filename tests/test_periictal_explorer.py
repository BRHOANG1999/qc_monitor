"""Peri-ictal Explorer tab: pure helpers, the background-job registry, the
figure/reading/details builders, and accessibility guards (radio labels carry a
colour; the lead-up input can't trip the browser :invalid red).

Run with: pytest tests/test_periictal_explorer.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from dash import dcc

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                              # noqa: E402
from src.dashboard.tabs import periictal_explorer as pex    # noqa: E402


def _sub(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "time_to_onset_sec": np.linspace(0, 21600, n),
        "hour_of_day": rng.uniform(0, 24, n),
        "stim_key": ["2nC"] * (n // 2) + ["5nC"] * (n - n // 2),
        "seizure_idx": rng.integers(0, 3, n),
        "channel": ["BCH111SR"] * n,
        "line_length": rng.normal(size=n),
    })


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for c in children:
        if hasattr(c, "children") or isinstance(c, (dcc.RadioItems, dcc.Input)):
            yield from _walk(c)


# ------------------------------------------------------------------ #
#  Accessibility guards (the bugs the user caught)
# ------------------------------------------------------------------ #

def test_radios_have_light_label_colour(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    tree = pex._controls(store, ["BCH111"], "BCH111")
    radios = [c for c in _walk(tree) if isinstance(c, dcc.RadioItems)]
    assert len(radios) == 2                         # window + embedding
    for r in radios:
        ls = getattr(r, "labelStyle", None) or {}
        assert ls.get("color"), "radio option text needs an explicit colour"


def test_leadup_input_cannot_trip_invalid_red(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    tree = pex._controls(store, ["BCH111"], "BCH111")
    inputs = [c for c in _walk(tree) if isinstance(c, dcc.Input)]
    win = [c for c in inputs if getattr(c, "id", "") == "pex-window-h"]
    assert win and win[0].step == "any"             # no step-mismatch :invalid


# ------------------------------------------------------------------ #
#  Job registry + figures
# ------------------------------------------------------------------ #

def test_job_id_stable_and_discriminating():
    a = pex._job_id("BCH111", "chronicStim", "evoked", 6.0, "pca", 50000)
    b = pex._job_id("BCH111", "chronicStim", "evoked", 6.0, "pca", 50000)
    c = pex._job_id("BCH111", "chronicStim", "passive", 6.0, "pca", 50000)
    assert a == b and a != c


def test_figure_continuous_single_trace_categorical_folded():
    sub = _sub()
    emb = np.random.default_rng(1).random((len(sub), 2))
    cont = pex._figure(emb, sub, "time_to_onset_sec", "pca", {"explained_var": [0.3, 0.2]})
    assert len(cont.data) == 1                       # one Scattergl + colourbar
    assert cont.data[0].customdata is not None       # row indices for selection
    cat = pex._figure(emb, sub, "stim_key", "umap", None)
    assert len(cat.data) == 2                         # one trace per fingerprint


def test_axis_titles_report_variance_and_flag_umap():
    assert pex._axis_titles("pca", {"explained_var": [0.35, 0.2]}) == \
        ("PC1 (35% variance)", "PC2 (20% variance)")
    assert pex._axis_titles("umap", {}) == ("UMAP-1 (relative)", "UMAP-2 (relative)")


def test_reading_strip_flags_high_confound():
    res = {"readout": {"time_of_day": 0.67, "stim_fingerprint": None},
           "meta": {"n_points": 100, "n_total": 100, "explained_var": [0.35, 0.2]},
           "n_seizures": 4}
    view = pex._reading_strip(res)
    assert view is not None                          # renders without error


def test_cache_eviction_bound():
    pex._CACHE.clear()
    for i in range(pex._CACHE_MAX + 3):
        pex._finish(f"job{i}", {"empty": True})
    assert len(pex._CACHE) == pex._CACHE_MAX
    assert "job0" not in pex._CACHE
    pex._CACHE.clear()


def test_render_cached_empty_and_ready_paths():
    fig, reading, status, disabled, jid, traj = pex._render_cached(
        {"empty": True}, "time_to_onset_sec", "j", "line_length")
    assert disabled is True and jid == "j" and traj is not None
    sub = _sub()
    ready = {"empty": False, "emb": np.zeros((len(sub), 2)), "sub": sub,
             "readout": {"time_of_day": 0.1, "stim_fingerprint": None},
             "meta": {"n_points": len(sub), "n_total": len(sub),
                      "explained_var": [0.3, 0.2]}, "method": "pca", "n_seizures": 3}
    fig, reading, status, disabled, jid, traj = pex._render_cached(
        ready, "time_to_onset_sec", "j2", "line_length")
    assert disabled is True and len(fig.data) == 1 and traj is not None


def test_trajectory_fig_from_cache():
    sub = _sub()
    cached = {"empty": False, "emb": np.random.default_rng(0).random((len(sub), 2)),
              "sub": sub}
    # a feature column, a PC coordinate, and hour-of-day all render.
    for y in ("line_length", "__pc1__", "__pc2__", "hour_of_day"):
        assert pex._trajectory_fig(cached, y) is not None
    assert pex._trajectory_fig({"empty": True}, "line_length") is not None


def test_resolve_window_modes():
    import pytest
    assert pex._resolve_window("evoked", "full", 1, 200, 1) == ("evoked", None)
    sv, cfg = pex._resolve_window("evoked", "custom", 5, 100, 1)
    assert sv == "evokedw" and cfg.window_start_ms == 5.0 and cfg.window_end_ms == 100.0
    # passive is always windowed, regardless of the (evoked-only) mode toggle.
    sv, cfg = pex._resolve_window("passive", "full", -200, -1, 1)
    assert sv == "passive" and cfg.window_end_ms == -1.0
    with pytest.raises(ValueError):
        pex._resolve_window("evoked", "custom", 200, 1, 1)     # reversed


def test_win_token_distinguishes_windows():
    a = pex._win_token(*pex._resolve_window("evoked", "custom", 1, 200, 1))
    b = pex._win_token(*pex._resolve_window("evoked", "custom", 1, 100, 1))
    full = pex._win_token(*pex._resolve_window("evoked", "full", 1, 200, 1))
    assert a != b and a != full        # window is part of the job/cache key


def test_selected_indices_defensive():
    assert pex._selected_indices(None) == []
    assert pex._selected_indices({"points": [{"customdata": 3},
                                            {"customdata": [7]},
                                            {"foo": 1}]}) == [3, 7]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
