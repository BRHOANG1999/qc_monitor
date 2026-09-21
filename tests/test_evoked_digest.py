"""Evoked-response digest: percentile banding, per-band waveform accumulation,
and the send/render orchestration.

Run with: pytest tests/test_evoked_digest.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.notifications import evoked_digest as ed        # noqa: E402
from src.notifications import evoked_daily as ev         # noqa: E402


# ------------------------------------------------------------------ #
#  even_band_edges + assign_band
# ------------------------------------------------------------------ #

def test_even_bands_uniform():
    v = np.linspace(0, 100, 500)
    e = ed.even_band_edges(v, 5)
    assert e is not None and e.size == 6
    assert np.all(np.diff(e) > 0)
    # roughly equal population per band
    counts = np.array([np.sum((v >= e[b]) & (v < e[b + 1])) for b in range(5)])
    assert counts.min() > 0.15 * v.size


def test_point_mass_collapses_to_fewer_bands():
    v = np.array([0.0] * 60 + list(range(1, 41)), dtype=float)   # 60% flat zeros
    e = ed.even_band_edges(v, 5)
    assert e is not None and e.size >= 3          # not None; ties merged
    assert np.all(np.diff(e) > 0)
    # the flat mass lands in ONE bottom band, not split across many
    assert ed.assign_band(0.0, e) == 0
    assert ed.assign_band(40.0, e) == ed.n_bands_of(e) - 1


def test_degenerate_and_sparse_return_none():
    assert ed.even_band_edges(np.full(60, 5.0), 5) is None      # single value
    assert ed.even_band_edges(np.arange(10.0), 5) is None       # too few points


def test_assign_band_bounds():
    e = np.array([0.0, 1.0, 2.0, np.nextafter(3.0, np.inf)])
    assert ed.assign_band(0.5, e) == 0
    assert ed.assign_band(2.5, e) == 2
    assert ed.assign_band(3.0, e) == 2                          # max in top band
    assert ed.assign_band(-1.0, e) == -1
    assert ed.assign_band(np.nan, e) == -1


# ------------------------------------------------------------------ #
#  band_waveforms_multi: recovers each band's mean waveform
# ------------------------------------------------------------------ #

def _flat(T):
    return np.zeros(T)


def _bump(T):
    t = np.arange(T)
    return np.exp(-((t - T / 2) ** 2) / (2 * (T / 12) ** 2))    # a gaussian bump


def test_band_waveforms_recovers_groups(monkeypatch):
    T, n = 60, 50
    times = (np.arange(n) * 2.0 + 10.0)
    feats = np.array([0.0] * 25 + [1.0] * 25)                   # low half, high half
    traces = np.stack([_flat(T) if i < 25 else _bump(T) for i in range(n)])
    rows = [{"channel": "BCH999", "stim_time_sec": float(times[i]),
             "amp": float(feats[i])} for i in range(n)]

    monkeypatch.setattr(ed, "read_feature_sidecar", lambda fp, a: rows)
    monkeypatch.setattr(ed, "read_file_evoked", lambda fp, only_animals=None: {
        "BCH999": {"times": times.tolist(), "traces": traces,
                   "time_ms": np.arange(T, dtype=float)}})

    edges = np.array([0.0, 0.5, np.nextafter(1.0, np.inf)])     # 2 bands
    tm, wf = ed.band_waveforms(["f1"], "BCH999", "BCH999", "amp", edges)
    assert tm is not None and set(wf) == {0, 1}
    assert wf[0]["n"] == 25 and wf[1]["n"] == 25
    assert np.allclose(wf[0]["mean"], _flat(T), atol=1e-9)      # band0 = flat
    assert np.allclose(wf[1]["mean"], _bump(T), atol=1e-9)      # band1 = bump


def test_band_waveforms_stim_time_mismatch_skips(monkeypatch):
    # sidecar stim times don't line up with any trace time -> nothing accumulates
    rows = [{"channel": "BCH999", "stim_time_sec": 999.0, "amp": 0.2}]
    monkeypatch.setattr(ed, "read_feature_sidecar", lambda fp, a: rows)
    monkeypatch.setattr(ed, "read_file_evoked", lambda fp, only_animals=None: {
        "BCH999": {"times": [10.0, 12.0], "traces": np.zeros((2, 20)),
                   "time_ms": np.arange(20.0)}})
    edges = np.array([0.0, 0.5, np.nextafter(1.0, np.inf)])
    _tm, wf = ed.band_waveforms(["f1"], "BCH999", "BCH999", "amp", edges)
    assert wf == {}


# ------------------------------------------------------------------ #
#  day scoping + orchestration
# ------------------------------------------------------------------ #

def test_list_day_files_filters_by_date(monkeypatch):
    files = ["s__BCH999_2026_02_01__10_00_00_evoked.mat",
             "s__BCH999_2026_02_01__23_00_00_evoked.mat",
             "s__BCH999_2026_02_02__00_30_00_evoked.mat",
             "s__BCH888_2026_02_01__12_00_00_evoked.mat"]
    monkeypatch.setattr(ed, "list_evoked_files", lambda d: files)
    from datetime import datetime
    got = ed.list_day_files("BCH999", "x", datetime(2026, 2, 1))
    assert len(got) == 2 and all("2026_02_01" in f for f in got)


def test_build_animal_no_files(monkeypatch):
    monkeypatch.setattr(ed, "list_evoked_files", lambda d: [])
    from datetime import datetime
    res = ed.build_animal("BCH999", "x", datetime(2026, 2, 1), work_dir="x")
    assert res.get("empty") and res["reason"] == "no recordings"


def test_send_evoked_daily_dry_run(monkeypatch, tmp_path):
    fake = {"animal": "BCH999", "channel": "BCH999", "date": "2026-02-01",
            "n_responses": 100, "pngs": {"peak_to_trough": "a.png",
                                         "line_length": "b.png"}}
    monkeypatch.setattr(ev._ed, "build_animal", lambda *a, **k: fake)
    monkeypatch.setattr(ev, "_animals_with_data", lambda d, day: ["BCH999"])
    from datetime import datetime
    cfg = {"notifications": {"evoked_daily": {}},
           "chronic_evoked": {"evoked_output_dir": "x"}}
    res = ev.send_evoked_daily(datetime(2026, 2, 2), cfg, None, None,
                               dry_run=True, out_dir=str(tmp_path))
    assert res["dry_run"] and res["animals"] == ["BCH999"] and not res["sent"]


def test_split_attachments_headline_inline():
    res = [{"pngs": {"peak_to_trough": __file__, "recovery_tau": __file__}}]
    inline, files = ev._split_attachments(res, ["peak_to_trough"])
    assert [os.path.basename(p) for p in inline] == [os.path.basename(__file__)]
    assert len(files) == 1                                      # recovery_tau attached
