"""Paired-pulse extraction (src/paired_pulse/data.py): per-epoch onset/sign detection
from the stim-copy trace, the 1-49 ms windowing at each pulse's own onset, PPR recovery,
gain invariance, and single-pulse rejection.

Synthetic h5 evokedOutput files (no DB). Run: pytest tests/test_paired_pulse.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.paired_pulse import data as D                   # noqa: E402
from src.paired_pulse import config as C                 # noqa: E402


def _burst(tw_ms, amp):
    """Smooth (Hann-windowed) 1-cycle sine over the window -> p2p ~ 2*amp."""
    n = tw_ms.size
    return amp * np.sin(np.linspace(0, 2 * np.pi, n)) * np.hanning(n)


def _write_file(path, *, n=80, fs=20000.0, a1=1.0, a2=1.25, side="plus",
                gain=1.0, single=False, seed=0):
    """Write a synthetic *_evoked.mat with a 50 ms-ISI pair (or single pulse)."""
    import h5py
    rng = np.random.default_rng(seed)
    t = np.linspace(-500.0, 500.0, int(round(fs)) + 1)   # ms, ±500 ms, fs Hz
    S = t.size

    def _idx(ms):
        return int(np.argmin(np.abs(t - ms)))

    lfp = rng.normal(0, 1e-3, (n, S))
    stim = rng.normal(0, 1e-4, (n, S))
    wlo, whi = _idx(1.0) - _idx(0.0), _idx(49.0) - _idx(0.0)   # window sample span
    tw = t[_idx(1.0):_idx(1.0) + (whi - wlo)]
    s1_on = 0.0 if (side == "plus" or single) else -C.ISI_MS
    s2_on = C.ISI_MS if side == "plus" else 0.0
    for ep in range(n):
        # stim-copy deflections (narrow): anchor at 0 always
        stim[ep, _idx(0.0)] = 1.0
        if not single:
            stim[ep, _idx(C.ISI_MS if side == "plus" else -C.ISI_MS)] = 0.8
        # LFP responses: earlier pulse amp a1, later pulse amp a2
        a = _idx(s1_on + 1.0)
        lfp[ep, a:a + (whi - wlo)] += gain * _burst(tw, a1)
        if not single:
            b = _idx(s2_on + 1.0)
            lfp[ep, b:b + (whi - wlo)] += gain * _burst(tw, a2)
    with h5py.File(path, "w") as g:
        node = g.create_group("allAnimalResults").create_group("BCH111SR")
        node["evokedData"] = lfp
        node["stimulusTraces"] = stim
        node["timeAxis"] = t
        node["stimulusTimes"] = np.arange(n, dtype=float) * 2.0   # 0.5 Hz


def test_detect_partner_plus(tmp_path):
    fp = str(tmp_path / "pp_plus_evoked.mat")
    _write_file(fp, side="plus")
    df = D.extract_pairs(fp)
    assert df is not None and len(df) >= 70
    assert df["anchor_is_s1"].all()                      # partner at +50 => anchor is S1


def test_detect_partner_minus(tmp_path):
    fp = str(tmp_path / "pp_minus_evoked.mat")
    _write_file(fp, side="minus")
    df = D.extract_pairs(fp)
    assert df is not None
    assert not df["anchor_is_s1"].any()                  # partner at -50 => anchor is S2


def test_ppr_recovered(tmp_path):
    fp = str(tmp_path / "pp_fac_evoked.mat")
    _write_file(fp, a1=1.0, a2=1.3, side="plus")
    df = D.extract_pairs(fp)
    ppr = df["ppr_peak_to_trough"].to_numpy(float)
    assert abs(np.median(ppr) - 1.3) < 0.15             # recovers the planted ratio
    assert np.mean(ppr > 1) > 0.8                        # facilitation


def test_ppr_gain_invariant(tmp_path):
    a = str(tmp_path / "g1_evoked.mat"); _write_file(a, a1=1.0, a2=1.2, gain=1.0)
    b = str(tmp_path / "g10_evoked.mat"); _write_file(b, a1=1.0, a2=1.2, gain=10.0)
    ma = np.median(D.extract_pairs(a)["ppr_peak_to_trough"])
    mb = np.median(D.extract_pairs(b)["ppr_peak_to_trough"])
    assert abs(ma - mb) < 0.05                           # ratio cancels gain


def test_single_pulse_rejected(tmp_path):
    fp = str(tmp_path / "single_evoked.mat")
    _write_file(fp, single=True)
    assert D.extract_pairs(fp) is None                   # no 2nd deflection => skipped


def test_minus_side_ppr_direction(tmp_path):
    # side=minus: S1 at -50 (a1), S2 at 0 (a2); PPR should still be a2/a1
    fp = str(tmp_path / "pp_minus_fac_evoked.mat")
    _write_file(fp, a1=1.0, a2=1.25, side="minus")
    df = D.extract_pairs(fp)
    assert abs(np.median(df["ppr_peak_to_trough"]) - 1.25) < 0.15
