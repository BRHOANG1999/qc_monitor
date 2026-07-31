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


# ------------------------------------------------------------------ #
#  Nav-group split: three sub-tab layouts share a scope bar
# ------------------------------------------------------------------ #

def _all_ids(node, acc):
    i = getattr(node, "id", None)
    if isinstance(i, str):
        acc.append(i)
    kids = getattr(node, "children", None)
    if kids is None:
        return
    if not isinstance(kids, (list, tuple)):
        kids = [kids]
    for k in kids:
        if k is not None:
            _all_ids(k, acc)


def _find(node, target):
    """First component with id==target in the tree (or None)."""
    if getattr(node, "id", None) == target:
        return node
    kids = getattr(node, "children", None)
    if kids is None:
        return None
    if not isinstance(kids, (list, tuple)):
        kids = [kids]
    for k in kids:
        if k is not None:
            hit = _find(k, target)
            if hit is not None:
                return hit
    return None


def test_three_sub_tab_layouts_build_with_scope_and_lens_ids(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    scope = {"pex-animal", "pex-protocol", "pex-variant", "pex-window-h",
             "pex-method", "pex-build", "pex-job", "pex-poll", "pex-preview",
             "pex-status"}
    lens = {"layout_embedding": {"pex-graph-passive", "pex-graph-evoked",
                                 "pex-reading-passive", "pex-reading-evoked",
                                 "pex-colorby", "pex-details-passive",
                                 "pex-details-evoked", "pex-embed2-job",
                                 "pex-embed2-build"},
            "layout_trend": {"pex-traj", "pex-traj-y", "pex-trend-forest",
                             "pex-trend-verdict", "pex-trend-table"},
            "layout_pdfcdf": {"pex-pc-feature", "pex-pc-pdf", "pex-pc-cdf",
                              "pex-pc-verdict", "pex-pc-scan", "pex-pc-nphases",
                              "pex-pc-roc", "pex-pc-phaseauc", "pex-pc-coef",
                              "pex-pc-fverdict"},
            "layout_waveform": {"pex-erp", "pex-erp-wave", "pex-erp-seizure",
                                "pex-erp-job", "pex-erp-col"},
            "layout_slow_dynamics": {"pex-sd-phi", "pex-sd-circ", "pex-sd-feature",
                                     "pex-sd-win", "pex-sd-build", "pex-sd-job",
                                     "pex-sd-readout", "pex-sd-poll"}}
    for name, want in lens.items():
        acc: list = []
        _all_ids(getattr(pex, name)(store), acc)
        ids = set(acc)
        assert len(acc) == len(ids), f"{name} has duplicate ids"
        assert scope <= ids, f"{name} missing scope ids: {scope - ids}"
        assert want <= ids, f"{name} missing lens ids: {want - ids}"


def test_job_store_persists_across_sub_tab_swaps(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    job = _find(pex.scope_bar(store), "pex-job")
    assert job is not None and job.storage_type == "session"


def test_nav_group_promoted_to_top_level():
    from src.dashboard import app as _app
    grp = next((g for g in _app.NAV_GROUPS if g["id"] == "periictal"), None)
    assert grp is not None
    assert [s["id"] for s in grp["subs"]] == [
        "periictal_embedding", "periictal_trend", "periictal_pdfcdf",
        "periictal_waveform", "periictal_slow"]
    # the old single sub-tab is gone from every group
    assert all(s["id"] != "periictal_explorer"
               for g in _app.NAV_GROUPS for s in g["subs"])
    assert _app.SUBTAB_TO_GROUP["periictal_waveform"] == "periictal"


# ------------------------------------------------------------------ #
#  Trend & test lens: forest plot + synchronous views from the cache
# ------------------------------------------------------------------ #

def _cached_full(n_sz=8, per=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_sz):
        onset = 1.7e9 + sid * 86400
        for phase, sign in (("pre", 1.0), ("post", -1.0)):
            tto = np.sort(rng.uniform(1.0, 3600.0, per)) * sign
            for k in range(per):
                rows.append({
                    "seizure_idx": sid, "phase": phase,
                    "time_to_onset_sec": float(tto[k]),
                    "seizure_onset_epoch": onset, "seizure_racine": 3,
                    "hour_of_day": float(rng.uniform(0, 24)),
                    "line_length": float(-np.log10(abs(tto[k]))
                                         + rng.normal(0, 0.05)),
                    "rms_amplitude": float(rng.normal())})
    return {"empty": False, "full": pd.DataFrame(rows),
            "metrics": ["line_length", "rms_amplitude"]}


def test_forest_fig_diverging_with_null_and_median_lines():
    from src.periictal import trendtest as tt
    cached = _cached_full()
    pre = cached["full"][cached["full"]["phase"] == "pre"]
    per = tt.per_seizure_trend(pre, "line_length")
    summ = tt.across_seizure_test([r["rho"] for r in per])
    fig = pex._forest_fig(per, summ, pex._seizure_labels(cached["full"]))
    assert len(fig.data) == 1                              # one seizure-marker trace
    mk = fig.data[0].marker
    assert mk.reversescale is True and mk.cmin == -1 and mk.cmax == 1
    assert len(fig.layout.shapes) >= 2                     # null(0) + median vlines


def test_trend_test_views_render_and_pc_placeholder():
    cached = _cached_full()
    forest, verdict, table = pex._trend_test_views(cached, "line_length")
    assert len(forest.data) == 1 and verdict is not None and table is not None
    # a PC coordinate isn't a matrix column -> forest is a placeholder, table stays
    f2, v2, t2 = pex._trend_test_views(cached, "__pc1__")
    assert t2 is not None and v2 is not None


def _cached_pc(n_sz=10, per=50, sep=1.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_sz):
        onset = 1.7e9 + sid * 7200
        for lo, hi, mu in ((60, 1800, sep), (3600, 5400, -sep)):
            for _ in range(per):
                rows.append({
                    "seizure_idx": sid, "phase": "pre",
                    "time_to_onset_sec": float(rng.uniform(lo, hi)),
                    "seizure_onset_epoch": onset, "hour_of_day": 0.0,
                    "expfit_initial": float(rng.normal(mu, 1)),
                    "sum_power_low": float(rng.normal(mu, 1)),
                    "sum_power_high": float(rng.normal(mu, 1)),
                    "freq_moment_high": float(rng.normal(0, 1)),
                    "freq_moment_low": float(rng.normal(0, 1))})
    return {"empty": False, "full": pd.DataFrame(rows), "n_seizures": n_sz,
            "metrics": ["expfit_initial", "sum_power_low", "sum_power_high",
                        "freq_moment_high", "freq_moment_low"]}


def test_pdfcdf_lens_render_helpers():
    from src.periictal import forecast as F
    cached = _cached_pc()
    lab = F.label_classes(cached["full"])
    pc = F.pdf_cdf(lab, "expfit_initial")
    assert pc["auc_norm"] > 0.7
    assert len(pex._pdf_fig(pc, "expfit_initial").data) >= 1
    assert len(pex._cdf_fig(pc, "expfit_initial").data) == 2
    v = pex._pc_verdict(pc, F.permutation_p(lab, "expfit_initial", n_perm=200),
                        F.paired_seizure_test(lab, "expfit_initial"))
    assert v is not None
    assert pex._pc_scan_table(F.scan_features(lab, cached["metrics"], n_perm=100)) \
        is not None
    res = F.logistic_forecast(lab, n_phases=4)
    assert len(pex._roc_fig(res).data) >= 1
    assert len(pex._phaseauc_fig(res).data) >= 1
    assert pex._coef_fig(res) is not None
    assert pex._forecast_verdict(res, cached["n_seizures"]) is not None


# ------------------------------------------------------------------ #
#  Feature reference (validation): every UMAP input is documented
# ------------------------------------------------------------------ #

def _iter_nodes(node):
    yield node
    kids = getattr(node, "children", None)
    if kids is None:
        return
    if not isinstance(kids, (list, tuple)):
        kids = [kids]
    for k in kids:
        if k is not None and not isinstance(k, str):
            yield from _iter_nodes(k)


def test_column_docs_document_every_feature():
    # drift guard: no feature column may go undocumented (the validation promise).
    from src.utils import evoked_features as ef
    assert [c for c in ef.ALL_COLUMNS if c not in ef.COLUMN_DOCS] == []
    assert all(m in ef.COLUMN_DOCS for m in pex._cfg.CHEAP_METRICS)


def test_feature_reference_lists_all_umap_inputs():
    from dash import html
    ref = pex._feature_reference()
    codes = [n.children for n in _iter_nodes(ref) if isinstance(n, html.Code)]
    for m in pex._cfg.CHEAP_METRICS:
        assert m in codes, f"{m} missing from the feature reference"


# ------------------------------------------------------------------ #
#  Linear time-to-onset colour cap + PCA loadings (vector-space view)
# ------------------------------------------------------------------ #

def test_cap_seconds_units_and_fallback():
    assert pex._cap_seconds(2, "h") == 7200.0
    assert pex._cap_seconds(90, "m") == 5400.0
    assert pex._cap_seconds(45, "s") == 45.0
    assert pex._cap_seconds(None, "h") == pex._pal.DEFAULT_TIME_CAP_SEC  # blank
    assert pex._cap_seconds(0, "h") == pex._pal.DEFAULT_TIME_CAP_SEC     # non-positive


def test_continuous_colour_is_linear_capped_not_log():
    sub = _sub()
    emb = np.zeros((len(sub), 2))
    fig = pex._figure(emb, sub, "time_to_onset_sec", "pca",
                      {"explained_var": [0.3, 0.2]}, cap_sec=3600.0)
    mk = fig.data[0].marker
    assert mk.cmin == 0.0 and mk.cmax == 3600.0        # linear range, capped at 1 h
    assert float(np.max(mk.color)) <= 3600.0           # clamped, no value past cap


def test_loadings_fig_pca_bars_and_umap_note():
    cols = ["line_length", "rms_amplitude", "variance"]
    cached = {"empty": False, "method": "pca", "cols": cols,
              "meta": {"explained_var": [0.4, 0.2],
                       "components": [[0.7, 0.1, -0.5], [0.2, 0.6, 0.3]]}}
    fig = pex._loadings_fig(cached)
    assert len(fig.data) == 2                          # PC1 + PC2 bar traces
    assert set(fig.data[0].y) == set(cols)             # one bar per feature
    # UMAP (no linear loadings) -> an honest note, not bars
    umap_fig = pex._loadings_fig({"empty": False, "method": "umap",
                                  "cols": cols, "meta": {}})
    assert not any(getattr(t, "y", None) is not None and len(t.y) for t in umap_fig.data)


def test_loadings_fig_single_pc_does_not_crash():
    # degenerate build: one surviving feature -> 1 component, explained_var len 1.
    cached = {"empty": False, "method": "pca", "cols": ["peak_to_trough"],
              "meta": {"explained_var": [0.99], "components": [[1.0]]}}
    fig = pex._loadings_fig(cached)                    # must NOT raise IndexError
    assert len(fig.data) == 1                          # only PC1 (no PC2 exists)
    assert pex._pc_name(1, [0.99]) == "PC1 (99%)"
    assert pex._pc_name(2, [0.99]) == "PC2"            # missing ev -> no percent


def test_single_value_categorical_annotates_not_blank():
    # A colour-by with ONE distinct value (e.g. channel/stim on BCH111 chronicStim)
    # must SAY so + force the legend, not silently paint one colour (reads as
    # "colour-by does nothing").
    n = 30
    sub = pd.DataFrame({"channel": ["BCH111SR"] * n,
                        "seizure_idx": np.arange(n) % 4})
    emb = np.random.default_rng(0).random((n, 2))
    one = pex._categorical_fig(emb, sub, "channel")
    assert len(one.data) == 1 and one.data[0].showlegend is True
    assert one.layout.annotations and "Only one" in one.layout.annotations[0].text
    # a genuinely multi-valued category is unaffected (no annotation)
    multi = pex._categorical_fig(emb, sub, "seizure_idx")
    assert len(multi.data) == 4 and not multi.layout.annotations


def test_loadings_fig_rejects_mismatched_components():
    # components width must match cols length, else bail gracefully.
    cached = {"empty": False, "method": "pca", "cols": ["a", "b"],
              "meta": {"explained_var": [0.5, 0.3], "components": [[1.0]]}}
    fig = pex._loadings_fig(cached)
    assert not any(getattr(t, "y", None) for t in fig.data)   # note, not bars


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
