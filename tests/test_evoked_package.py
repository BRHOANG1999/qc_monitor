"""Evoked figure package: band stats, day/week fan-out, and folder+index build.

Run with: pytest tests/test_evoked_package.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.notifications import evoked_package as ep      # noqa: E402
from src.notifications import evoked_digest as ed       # noqa: E402


# ------------------------------------------------------------------ #
#  _BandStats
# ------------------------------------------------------------------ #

def test_bandstats_mean_sd_and_reservoir():
    rng = np.random.default_rng(0)
    st = ep._BandStats(n_bands=2, k_samples=4, rng=rng)
    T = 30
    traces = [rng.normal(size=T) for _ in range(20)]
    for t in traces:
        st.add(0, t, np.arange(T, dtype=float))
    res = st.result()
    assert res[0]["n"] == 20
    assert np.allclose(res[0]["mean"], np.mean(traces, axis=0), atol=1e-9)
    assert np.allclose(res[0]["sd"], np.std(traces, axis=0), atol=1e-9)
    assert len(res[0]["samples"]) == 4                 # reservoir capped at k


def test_bandstats_reservoir_seed_deterministic():
    def run():
        st = ep._BandStats(2, 3, np.random.default_rng(7))
        for i in range(30):
            st.add(0, np.full(20, float(i)), np.arange(20.0))
        return [s[0] for s in st.result()[0]["samples"]]
    assert run() == run()


# ------------------------------------------------------------------ #
#  collect_stats day/week fan-out
# ------------------------------------------------------------------ #

def _rows_for(date_str, stim0, vals, channel="BCH999", feat="amp"):
    base = datetime.fromisoformat(date_str + "T00:00:00")
    rows = []
    for i, v in enumerate(vals):
        stim = stim0 + i * 2.0
        rows.append({"channel": channel, "stim_time_sec": stim,
                     "abs_dt": (base + timedelta(seconds=stim)).isoformat(),
                     feat: float(v)})
    return rows


def test_collect_stats_fans_to_day_and_week(monkeypatch):
    T = 20
    fileA, fileB = "fA.mat", "fB.mat"
    rowsA = _rows_for("2026-09-19", 10.0, [1.0, 2.0])       # 2 on day A
    rowsB = _rows_for("2026-09-20", 10.0, [3.0])            # 1 on day B
    rows_by = {fileA: rowsA, fileB: rowsB}

    def fake_sidecar(fp, a):
        return rows_by[fp]

    def fake_evoked(fp, only_animals=None):
        rows = rows_by[fp]
        times = [r["stim_time_sec"] for r in rows]
        return {"BCH999": {"times": times,
                           "traces": np.ones((len(rows), T)),
                           "time_ms": np.arange(T, dtype=float)}}

    monkeypatch.setattr(ep, "read_feature_sidecar", fake_sidecar)
    monkeypatch.setattr(ep, "read_file_evoked", fake_evoked)

    # every response in one band (band_of_ts -> 0) for its period
    def info_for(rows):
        m = {ed._ts_key(r): 0 for r in rows}
        return {"amp": {"band_of_ts": m, "n_bands": 1, "n_zero": 0,
                        "edges": np.array([0.0, 9.0]), "bands": np.array([]),
                        "counts": np.array([len(rows)])}}
    info_by_period = {"2026-09-19": info_for(rowsA),
                      "2026-09-20": info_for(rowsB),
                      "week": info_for(rowsA + rowsB)}
    stats = ep.collect_stats([fileA, fileB], "BCH999", "BCH999",
                             info_by_period, "week", n_bands=1, k_samples=4,
                             rng=np.random.default_rng(0))
    assert stats["2026-09-19"]["amp"].result()[0]["n"] == 2
    assert stats["2026-09-20"]["amp"].result()[0]["n"] == 1
    assert stats["week"]["amp"].result()[0]["n"] == 3     # both days pooled


# ------------------------------------------------------------------ #
#  end-to-end build -> folder tree + index.html + manifest
# ------------------------------------------------------------------ #

def test_build_package_writes_tree(monkeypatch, tmp_path):
    T = 40
    days = ["2026-09-19", "2026-09-20"]
    files, rows_by, traces_by = [], {}, {}
    rng = np.random.default_rng(1)
    for dstr in days:
        fp = f"s__BCH999_{dstr.replace('-', '_')}__12_00_00_evoked.mat"
        files.append(fp)
        vals = np.linspace(1.0, 10.0, 60)                  # 60 non-zero responses
        rows = _rows_for(dstr, 10.0, vals, feat="peak_to_trough")
        rows_by[fp] = rows
        traces_by[fp] = np.stack([v * rng.normal(1, 0.05, T) for v in vals])

    monkeypatch.setattr(ed, "list_evoked_files", lambda d: files)
    monkeypatch.setattr(ed, "read_feature_sidecar", lambda fp, a: rows_by[fp])
    monkeypatch.setattr(ep, "read_feature_sidecar", lambda fp, a: rows_by[fp])
    monkeypatch.setattr(ep, "read_file_evoked", lambda fp, only_animals=None: {
        "BCH999": {"times": [r["stim_time_sec"] for r in rows_by[fp]],
                   "traces": traces_by[fp], "time_ms": np.arange(T, dtype=float)}})
    # keep compute_bands happy with a low min for the tiny fixture
    monkeypatch.setattr(ed, "_MIN_PER_FEATURE", 20)

    res = ep.build_package("BCH999", "x", datetime(2026, 9, 20), str(tmp_path),
                           features=["peak_to_trough"], n_bands=2, k_samples=4,
                           max_trace_files=10)
    assert not res.get("empty")
    assert os.path.isfile(res["index"]) and os.path.isfile(
        os.path.join(res["pkg_dir"], "manifest.json"))
    # week + 2 days = 3 periods, each with the one feature
    assert len(res["figures"]) == 3
    d0 = res["figures"][0]["dir"]
    assert os.path.isfile(os.path.join(res["pkg_dir"], d0, "bands_mean.png"))
    assert os.path.isfile(os.path.join(res["pkg_dir"], d0, "deciles_grid.png"))
