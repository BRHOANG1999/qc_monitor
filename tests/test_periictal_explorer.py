"""Peri-ictal Explorer tab: pure helpers + the background-job registry (job-id
stability, cache eviction bound, ready/render path). No browser -- exercises the
same functions the callbacks call.

Run with: pytest tests/test_periictal_explorer.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dashboard.tabs import periictal_explorer as pex  # noqa: E402


def _sub(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "time_to_onset_sec": np.linspace(0, 3600, n),
        "hour_of_day": rng.uniform(0, 24, n),
        "stim_key": ["2nC"] * (n // 2) + ["5nC"] * (n - n // 2),
        "seizure_idx": rng.integers(0, 3, n),
        "channel": ["BCH111SR"] * n,
        "line_length": rng.normal(size=n),
    })


def test_job_id_stable_and_discriminating():
    a = pex._job_id("BCH111", "chronicStim", "evoked", 6.0, "pca", 50000)
    b = pex._job_id("BCH111", "chronicStim", "evoked", 6.0, "pca", 50000)
    c = pex._job_id("BCH111", "chronicStim", "passive", 6.0, "pca", 50000)
    assert a == b and a != c


def test_continuous_figure_single_trace_and_categorical_multi():
    sub = _sub()
    emb = np.random.default_rng(1).random((len(sub), 2))
    cont = pex._figure(emb, sub, "time_to_onset_sec")
    assert len(cont.data) == 1                     # one Scattergl w/ colorbar
    cat = pex._figure(emb, sub, "stim_key")
    assert len(cat.data) == 2                      # one trace per fingerprint
    empty = pex._figure(np.empty((0, 2)), sub.iloc[:0], "time_to_onset_sec")
    assert empty is not None                       # graceful empty


def test_scope_text_reports_seizure_count_not_stimuli():
    res = {"n_seizures": 4, "meta": {"n_points": 23745, "n_total": 23745}}
    txt = pex._scope_text(res, "evoked")
    assert "n = 4 seizures" in txt and "23,745" in txt
    assert "not a significance test" in txt


def test_readout_view_flags_high_confound():
    res = {"readout": {"time_of_day": 0.67, "stim_fingerprint": None},
           "meta": {"explained_var": [0.35, 0.20]}}
    view = pex._readout_view(res)
    assert view is not None                        # renders without error


def test_cache_eviction_bound():
    pex._CACHE.clear()
    for i in range(pex._CACHE_MAX + 3):
        pex._finish(f"job{i}", {"empty": True})
    assert len(pex._CACHE) == pex._CACHE_MAX        # bounded, oldest evicted
    assert "job0" not in pex._CACHE
    pex._CACHE.clear()


def test_render_cached_empty_and_ready_paths():
    # Empty result -> disabled poll, message.
    fig, ro, status, scope, disabled, jid = pex._render_cached(
        {"empty": True}, "time_to_onset_sec", "evoked", "j")
    assert disabled is True and jid == "j"
    # Ready result -> figure + built status.
    sub = _sub()
    emb = np.zeros((len(sub), 2))
    ready = {"empty": False, "emb": emb, "sub": sub,
             "readout": {"time_of_day": 0.1, "stim_fingerprint": None},
             "meta": {"n_points": len(sub), "n_total": len(sub)}, "n_seizures": 3}
    fig, ro, status, scope, disabled, jid = pex._render_cached(
        ready, "time_to_onset_sec", "evoked", "j2")
    assert disabled is True and "n = 3 seizures" in scope
    assert len(fig.data) == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
