"""Unit tests for src/utils/impedance.py transfer-impedance math."""

import pytest

from src.utils.impedance import (
    normalize_ratio,
    phase_currents,
    phase_voltages,
    compute_impedances,
    impedance_for_channel,
    find_current_transitions,
    access_resistance,
)
from src.utils.impedance_refresh import stimulated_indices


# --- Access resistance (ohmic step) ------------------------------------ #

def _synthetic_biphasic(fs_khz=20.0, pw_ms=0.15, ratio=3.0,
                        i_fast=1.0, R_kohm=0.5, gain=100.0):
    """Build a synthetic stim-copy + a PURE-RESISTOR animal voltage so the
    ohmic step is exactly R·ΔI. Returns (time_ms, animal_mean_trace,
    stim_mean_trace, i_fast_ua, i_slow_ua)."""
    dt = 1.0 / fs_khz                       # ms per sample
    t = [round(-1.0 + dt * i, 4) for i in range(int(2.0 / dt))]
    i_slow = i_fast / ratio
    stim, volt = [], []
    for tt in t:
        if 0 <= tt < pw_ms:                 # fast phase +I
            cur = i_fast
        elif pw_ms <= tt < pw_ms * (1 + ratio):   # slow phase -I/ratio
            cur = -i_slow
        else:
            cur = 0.0
        stim.append(cur)                    # stim-copy = commanded current
        # Pure resistor with current in µA and R in kΩ: V_mV = I_µA·R_kΩ, and
        # raw = V_mV/1000 * gain (since the code does v_raw/gain*1000 = mV).
        volt.append(cur * R_kohm * gain / 1000.0)
    return t, volt, stim, i_fast, i_slow


def test_find_current_transitions_biphasic():
    t, _v, sm, _if, _is = _synthetic_biphasic()
    tr = find_current_transitions(sm, t)
    assert tr is not None
    onset, reversal, offset = tr
    # onset near t=0, reversal near pw (0.15), offset near pw*(1+ratio)=0.6
    assert abs(t[onset]) <= 0.05
    assert abs(t[reversal] - 0.15) <= 0.05
    assert abs(t[offset] - 0.60) <= 0.05


def test_find_current_transitions_flat_returns_none():
    t = [round(-1 + 0.05 * i, 3) for i in range(40)]
    assert find_current_transitions([0.0] * len(t), t) is None


def test_access_resistance_pure_resistor_exact():
    # A pure resistor must give R exactly at BOTH transitions.
    R, gain, i_fast, ratio = 0.5, 100.0, 1.0, 3.0
    t, v, sm, i_f, i_s = _synthetic_biphasic(
        R_kohm=R, gain=gain, i_fast=i_fast, ratio=ratio)
    out = access_resistance(v, t, sm, gain, i_f, i_s)
    assert out["r_reversal_kohm"] == pytest.approx(R, abs=1e-6)
    assert out["r_offset_kohm"] == pytest.approx(R, abs=1e-6)
    assert out["r_access_kohm"] == pytest.approx(R, abs=1e-6)
    assert out["n_used"] == 2


def test_access_resistance_missing_inputs():
    t, v, sm, i_f, i_s = _synthetic_biphasic()
    assert access_resistance(v, t, sm, 0, i_f, i_s)["r_access_kohm"] is None
    assert access_resistance(v, t, sm, 100, None, i_s)["r_access_kohm"] is None
    assert access_resistance(v, t, [], 100, i_f, i_s)["r_access_kohm"] is None


def test_stimulated_indices_only_channels_after_stimcopy():
    # channel_names: stimCopy(0) BCH110SLM(1) stimCopy(2) BCH062SR(3)
    #                BCH061SLM(4) BCH111SR(5)
    eeg = [1, 3, 4, 5]
    stim_copy = [0, 2]
    # Stimulated = eeg channels preceded by a stimCopy: 1 and 3.
    assert stimulated_indices(eeg, stim_copy) == {1, 3}
    # BCH061SLM(4) and BCH111SR(5) are record-only -> excluded.
    assert 4 not in stimulated_indices(eeg, stim_copy)
    assert 5 not in stimulated_indices(eeg, stim_copy)


def test_stimulated_indices_non_stim_session():
    assert stimulated_indices([1, 2, 3], []) == set()
    assert stimulated_indices([], [0]) == set()


def test_normalize_ratio():
    assert normalize_ratio(3.0) == 3.0
    assert normalize_ratio(1) == 1.0
    # Fractional form inverted so the negative phase is always longer.
    assert normalize_ratio(1 / 3) == pytest.approx(3.0)
    assert normalize_ratio(0) is None
    assert normalize_ratio(-2) is None
    assert normalize_ratio("x") is None
    assert normalize_ratio(None) is None


def test_phase_currents_biphasic():
    # 5 nC over 150 us, 1:3 ratio.
    i_pos, i_neg = phase_currents(5.0, 150.0, 3.0)
    assert i_pos == pytest.approx(33.333, abs=0.01)   # fast phase, uA
    assert i_neg == pytest.approx(11.111, abs=0.01)   # slow phase, uA


def test_phase_currents_bad_inputs():
    assert phase_currents(0, 150, 3) == (None, None)
    assert phase_currents(5, 0, 3) == (None, None)
    assert phase_currents(5, 150, 0) == (None, None)
    assert phase_currents(None, 150, 3) == (None, None)


def test_positive_phase_impedance_matches_pi_worked_examples():
    # BCH110: v_raw 0.770, gain 300, 5 nC / 150 us -> 0.077 kOhm
    i_pos, i_neg = phase_currents(5.0, 150.0, 3.0)
    z_pos, _ = compute_impedances(0.770, None, 300.0, i_pos, i_neg)
    assert z_pos == pytest.approx(0.077, abs=0.001)

    # BCH062: v_raw 4.315, gain 150, 5 nC / 150 us -> 0.864 kOhm
    z_pos2, _ = compute_impedances(4.315, None, 150.0, i_pos, i_neg)
    assert z_pos2 == pytest.approx(0.864, abs=0.002)


def test_negative_phase_uses_slow_current():
    # Same voltage, but the negative phase current is 3x smaller,
    # so its impedance is 3x larger.
    i_pos, i_neg = phase_currents(5.0, 150.0, 3.0)
    z_pos, z_neg = compute_impedances(1.0, 1.0, 100.0, i_pos, i_neg)
    assert z_neg == pytest.approx(z_pos * 3.0, rel=1e-6)


def test_compute_impedances_guards():
    z_pos, z_neg = compute_impedances(1.0, 1.0, 0.0, 33.3, 11.1)  # gain 0
    assert z_pos is None and z_neg is None
    z_pos, z_neg = compute_impedances(None, 1.0, 300.0, 33.3, 11.1)
    assert z_pos is None
    assert z_neg is not None


def test_phase_voltages_windows():
    # 20 kHz axis (0.05 ms/sample) spanning -1..1 ms.
    time_ms = [round(-1.0 + 0.05 * i, 4) for i in range(41)]
    trace = [0.0] * len(time_ms)
    # Positive-phase spike inside [-0.1, 0.15]: put a peak at t=0.1.
    idx_peak = time_ms.index(0.1)
    trace[idx_peak] = 4.4
    # Negative-phase deflection over [0.15, 0.6]: constant -2.0.
    for i, t in enumerate(time_ms):
        if 0.15 <= t <= 0.6:
            trace[i] = -2.0
    v_pos, v_neg = phase_voltages(trace, time_ms, 150.0, 3.0)
    assert v_pos == pytest.approx(4.4)        # peak |deflection|
    assert v_neg == pytest.approx(2.0)        # mean |deflection|


def test_phase_voltages_bad_inputs():
    assert phase_voltages([], [], 150, 3) == (None, None)
    assert phase_voltages([1, 2], [0, 1], 0, 3) == (None, None)
    assert phase_voltages([1, 2], [0, 1], 150, 0) == (None, None)


def test_impedance_for_channel_full_record():
    time_ms = [round(-1.0 + 0.05 * i, 4) for i in range(41)]
    trace = [0.0] * len(time_ms)
    trace[time_ms.index(0.1)] = 0.770
    rec = impedance_for_channel(trace, time_ms, 300.0, 5.0, 150.0, 3.0)
    assert rec["gain"] == 300.0
    assert rec["neg_ratio"] == 3.0
    assert rec["i_pos_ua"] == pytest.approx(33.333, abs=0.01)
    assert rec["v_pos_raw"] == pytest.approx(0.770)
    assert rec["impedance_pos_kohm"] == pytest.approx(0.077, abs=0.001)
    # Negative-phase voltage is 0 here (no deflection) -> 0 impedance.
    assert rec["impedance_neg_kohm"] == pytest.approx(0.0)


def test_impedance_for_channel_missing_gain():
    time_ms = [round(-1.0 + 0.05 * i, 4) for i in range(41)]
    trace = [0.0] * len(time_ms)
    trace[time_ms.index(0.1)] = 0.770
    rec = impedance_for_channel(trace, time_ms, None, 5.0, 150.0, 3.0)
    assert rec["gain"] is None
    assert rec["v_pos_raw"] == pytest.approx(0.770)   # voltage still read
    assert rec["impedance_pos_kohm"] is None          # but no Z without gain
