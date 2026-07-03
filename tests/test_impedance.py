"""Unit tests for src/utils/impedance.py transfer-impedance math."""

import pytest

from src.utils.impedance import (
    normalize_ratio,
    phase_currents,
    phase_voltages,
    compute_impedances,
    impedance_for_channel,
)
from src.utils.impedance_refresh import stimulated_indices


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
