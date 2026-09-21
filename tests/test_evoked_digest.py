"""Evoked-response digest: rank-based percentile banding, per-band waveform
accumulation, and the send/render orchestration.

Run with: pytest tests/test_evoked_digest.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.notifications import evoked_digest as ed        # noqa: E402
from src.notifications import evoked_daily as ev         # noqa: E402


def _secs(n, t0=1.7e9):
    return np.arange(n, dtype=float) * 2.0 + t0


# ------------------------------------------------------------------ #
#  compute_bands: equal population (deciles), even under a point mass
# ------------------------------------------------------------------ #

def test_deciles_equal_population():
    v = np.random.default_rng(0).normal(size=500)
    info = ed.compute_bands(_secs(500), v, 10)
    assert info is not None and info["n_bands"] == 10
    counts = info["counts"]
    assert counts.sum() == 500
    assert counts.max() - counts.min() <= 1            # equal population


def test_zeros_separated_from_deciles():
    v = np.array([0.0] * 300 + list(range(1, 201)), dtype=float)   # 300 exact-0
    info = ed.compute_bands(_secs(500), v, 10)
    assert info is not None and info["n_bands"] == 10
    assert info["n_zero"] == 300                                   # zeros counted
    # the 200 NON-zero responses split into 10 even deciles
    assert info["counts"].sum() == 200
    assert info["counts"].max() - info["counts"].min() <= 1
    # every exact-0 response is assigned the extra "zero" band (== n_bands)
    assert set(info["bands"][:300].tolist()) == {10}


def test_compute_bands_too_few_returns_none():
    assert ed.compute_bands(_secs(20), np.arange(20.0), 10) is None


# ------------------------------------------------------------------ #
#  band_waveforms: recovers each band's mean waveform (band by timestamp)
# ------------------------------------------------------------------ #

def _flat(T):
    return np.zeros(T)


def _bump(T):
    t = np.arange(T)
    return np.exp(-((t - T / 2) ** 2) / (2 * (T / 12) ** 2))


def test_band_waveforms_recovers_groups(monkeypatch):
    from datetime import timedelta
    T, n = 60, 50
    stim = np.arange(n) * 2.0 + 10.0                   # stim_time_sec into recording
    base = datetime(2026, 2, 1, 0, 0, 0)
    abs_dt = [(base + timedelta(seconds=float(stim[i]))).isoformat() for i in range(n)]
    feats = np.concatenate([np.linspace(1.0, 1.4, 25),   # low group (spread)
                            np.linspace(2.0, 2.4, 25)])  # high group (no pile-up)
    traces = np.stack([_flat(T) if i < 25 else _bump(T) for i in range(n)])
    rows = [{"channel": "BCH999", "stim_time_sec": float(stim[i]),
             "abs_dt": abs_dt[i], "amp": float(feats[i])} for i in range(n)]

    monkeypatch.setattr(ed, "read_feature_sidecar", lambda fp, a: rows)
    monkeypatch.setattr(ed, "read_file_evoked", lambda fp, only_animals=None: {
        "BCH999": {"times": stim.tolist(), "traces": traces,
                   "time_ms": np.arange(T, dtype=float)}})

    secs = np.array([ed._d.parse_iso(x).timestamp() for x in abs_dt])
    info = ed.compute_bands(secs, feats, 2)            # 2 bands: low, high
    assert info is not None
    tm, wf = ed.band_waveforms(["f1"], "BCH999", "BCH999", "amp", info)
    assert tm is not None and set(wf) == {0, 1}
    assert wf[0]["n"] == 25 and wf[1]["n"] == 25
    assert np.allclose(wf[0]["mean"], _flat(T), atol=1e-9)
    assert np.allclose(wf[1]["mean"], _bump(T), atol=1e-9)


# ------------------------------------------------------------------ #
#  day/window scoping + orchestration
# ------------------------------------------------------------------ #

def test_window_files_daily_vs_weekly(monkeypatch):
    files = ["s__BCH999_2026_09_14__10_00_00_evoked.mat",
             "s__BCH999_2026_09_16__10_00_00_evoked.mat",
             "s__BCH999_2026_09_20__10_00_00_evoked.mat",
             "s__BCH999_2026_09_07__10_00_00_evoked.mat"]   # outside the week
    monkeypatch.setattr(ed, "list_evoked_files", lambda d: files)
    end = datetime(2026, 9, 20)
    assert len(ed.list_window_files("BCH999", "x", end, 1)) == 1     # daily = 09-20
    assert len(ed.list_window_files("BCH999", "x", end, 7)) == 3     # 09-14..09-20


def test_build_animal_no_files(monkeypatch):
    monkeypatch.setattr(ed, "list_evoked_files", lambda d: [])
    res = ed.build_animal("BCH999", "x", datetime(2026, 2, 1), work_dir="x")
    assert res.get("empty") and res["reason"] == "no recordings"


def test_send_evoked_daily_dry_run(monkeypatch, tmp_path):
    fake = {"animal": "BCH999", "channel": "BCH999", "date": "2026-02-01",
            "n_responses": 100, "skipped": ["recovery_tau"],
            "pngs": {"peak_to_trough": "a.png", "line_length": "b.png"}}
    monkeypatch.setattr(ev._ed, "build_animal", lambda *a, **k: fake)
    monkeypatch.setattr(ev, "_animals_with_data", lambda d, day, w=1: ["BCH999"])
    cfg = {"notifications": {"evoked_daily": {}},
           "chronic_evoked": {"evoked_output_dir": "x"}}
    res = ev.send_evoked_daily(datetime(2026, 2, 2), cfg, None, None,
                               window_days=7, dry_run=True, out_dir=str(tmp_path))
    assert res["dry_run"] and res["animals"] == ["BCH999"] and not res["sent"]
    assert "–" in res["period"]                    # a date-range label (weekly)


def test_split_attachments_inlines_all():
    res = [{"pngs": {"peak_to_trough": __file__, "recovery_tau": __file__}}]
    inline, files = ev._split_attachments(res)
    assert len(inline) == 2 and files == []            # both under the cap -> inline
