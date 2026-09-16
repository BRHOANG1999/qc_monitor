"""stim_artifact_trend: folder -> per-recording stim-artifact metrics, driven
with a fake ChunkData (no real .mat, no off-proc path).

Run: pytest tests/test_stim_artifact_trend.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import stim_artifact_trend as sat        # noqa: E402
from src.utils.mat_loader import ChunkData              # noqa: E402


def _fake_chunk(names, *, n_onsets=30, spacing=2000, fs=20000.0,
                charge_nC=0.2, pw_us=200.0, ratio=3.0, R_kohm=0.5, gain=100.0):
    """A 2-channel ChunkData: ch0 = commanded biphasic current (stimCopy), ch1 =
    a PURE-RESISTOR voltage (V = I·R·gain/1000), tiled at *n_onsets* onsets. The
    ohmic step is exactly R, so metrics['ra'] must recover R_kohm."""
    i_fast = charge_nC / pw_us * 1000.0           # = phase_currents i_pos
    i_slow = i_fast / ratio
    pw_samp = int(round(pw_us / 1000.0 * fs / 1000.0))     # 200 us -> 4 samples
    slow_samp = int(round(pw_samp * ratio))
    n = spacing * (n_onsets + 2)
    stim = np.zeros(n)
    art = np.zeros(n)
    v_fast = i_fast * R_kohm * gain / 1000.0
    v_slow = -i_slow * R_kohm * gain / 1000.0
    for k in range(n_onsets):
        c = spacing * (k + 1)
        stim[c:c + pw_samp] = i_fast
        stim[c + pw_samp:c + pw_samp + slow_samp] = -i_slow
        art[c:c + pw_samp] = v_fast
        art[c + pw_samp:c + pw_samp + slow_samp] = v_slow
    sig = np.column_stack([stim, art])            # ch0 = stim, ch1 = artifact
    return ChunkData(signal=sig, fs=fs, num_channels=2, num_samples=n,
                     duration_sec=n / fs, source_path="fake.mat",
                     channel_names=list(names))


_PARAMS = {"charge_nC": 0.2, "pulse_width_us": 200.0, "ratio": 3.0,
           "gain": 100.0, "freq_hz": 0.5, "source": "report"}


def _patch_params(monkeypatch, params=_PARAMS):
    monkeypatch.setattr(sat, "stim_params_for", lambda p: dict(params))


# ------------------------------------------------------------ channels --- #

def test_pick_channels_ignores_test_keyword():
    # 'electrodeTest' contains 'test' -> is_animal_channel would reject it, but
    # pick_channels must still return it as the artifact channel.
    assert sat.pick_channels(["stimCopy_x", "electrodeTestCh"]) == (0, 1)
    assert sat.pick_channels(["stimCopy", "BCH114SLM"]) == (0, 1)


# ------------------------------------------------------- analyze one --- #

def test_onset_count_and_shape(monkeypatch):
    _patch_params(monkeypatch)
    chunk = _fake_chunk(["stimCopy", "BCH114SLM"], n_onsets=30)
    rec = sat.analyze_recording("x_2026_09_11__15_37_50.mat",
                                get_chunk_fn=lambda p: chunk)
    assert rec["ok"] and rec["error"] is None
    assert rec["n_pulses"] == 30
    assert rec["dt"] is not None
    # window [-1, 2] ms at 20 kHz = 60 samples
    assert len(rec["time_ms"]) == 60 == len(rec["mean_trace"])


def test_access_resistance_recovered(monkeypatch):
    _patch_params(monkeypatch)
    chunk = _fake_chunk(["stimCopy", "BCH114SLM"], R_kohm=0.5, gain=100.0)
    rec = sat.analyze_recording("x_2026_09_11__15_37_50.mat",
                                get_chunk_fn=lambda p: chunk)
    m = rec["metrics"]
    assert m["ra"] == pytest.approx(0.5, abs=1e-3)      # pure resistor -> exact
    assert m["zss"] is not None
    assert m["ptp"] > 0
    assert m["sat"] < 1.0        # square-wave plateau is at the rail by construction


def test_saturation_pct():
    # 10 of 100 samples pinned at the rail -> ~10%.
    col = np.concatenate([np.full(10, 5.0), np.linspace(0.0, 4.0, 90)])
    assert sat.saturation_pct(col) == pytest.approx(10.0, abs=0.5)
    # a smooth ramp barely touches the rail
    assert sat.saturation_pct(np.linspace(-1.0, 1.0, 1000)) < 3.0
    assert sat.saturation_pct(np.zeros(50)) == 0.0      # flat -> guard


def test_channel_order_by_name_not_position(monkeypatch):
    _patch_params(monkeypatch)
    # Put the artifact channel FIRST; pick_channels must still find stim by name.
    chunk = _fake_chunk(["stimCopy", "BCH114SLM"])
    swapped = ChunkData(signal=chunk.signal[:, ::-1], fs=chunk.fs,
                        num_channels=2, num_samples=chunk.num_samples,
                        duration_sec=chunk.duration_sec, source_path="f.mat",
                        channel_names=["BCH114SLM", "stimCopy"])
    rec = sat.analyze_recording("x_2026_09_11__15_37_50.mat",
                                get_chunk_fn=lambda p: swapped)
    assert rec["ok"] and rec["metrics"]["ra"] == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------- stim params --- #

def test_stim_params_from_report(tmp_path):
    mat = tmp_path / "rec_2026_09_11__15_37_50.mat"
    mat.write_bytes(b"x")
    report = tmp_path / "rec_2026_09_11__15_37_50_STIM_REPORT.txt"
    report.write_text(
        "Channel 1: ACTIVE\n"
        "  Charge per phase:     100.0 nC\n"
        "  Pulse width:          200 µs\n"
        "  Neg pulse ratio:      3.0\n"
        "  Frequency:            0.50 Hz\n"
        "  Gain:                 1.00\n"
        "    Total pulses:       0\n"
        "    Total charge:       0.0 nC\n", encoding="utf-8")
    p = sat.stim_params_for(str(mat))
    assert p["source"] == "report"
    assert p["charge_nC"] == 100.0 and p["pulse_width_us"] == 200.0
    assert p["ratio"] == 3.0 and p["gain"] == 1.0


def test_missing_report_skips_ra_keeps_peaks(monkeypatch, tmp_path):
    # No report on disk -> stim_params_for returns source='none' -> Rₐ/Z_ss None,
    # but ptp/pos/neg/sat still compute.
    chunk = _fake_chunk(["stimCopy", "BCH114SLM"])
    rec = sat.analyze_recording(str(tmp_path / "x_2026_09_11__15_37_50.mat"),
                                get_chunk_fn=lambda p: chunk)
    m = rec["metrics"]
    assert rec["stim_params"]["source"] == "none"
    assert m["ra"] is None and m["zss"] is None
    assert m["ptp"] is not None and m["pos"] is not None and m["sat"] is not None


# ------------------------------------------------------ folder --- #

def test_analyze_folder_progress(monkeypatch):
    _patch_params(monkeypatch)
    chunk = _fake_chunk(["stimCopy", "BCH114SLM"])
    monkeypatch.setattr(sat, "list_recordings",
                        lambda folder: ["a_2026_09_11__00_00_00.mat",
                                        "b_2026_09_11__01_00_00.mat"])
    seen = []
    recs = sat.analyze_folder("anyfolder", get_chunk_fn=lambda p: chunk,
                              progress_cb=lambda i, n, p: seen.append((i, n)))
    assert len(recs) == 2 and all(r["ok"] for r in recs)
    assert seen[0] == (0, 2) and seen[-1] == (2, 2)     # start-of-each + final


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
