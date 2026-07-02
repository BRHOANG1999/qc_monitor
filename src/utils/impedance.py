"""Transfer-impedance computation for stimulation/recording channels.

Transfer impedance = (recorded voltage response) / (injected current),
computed **per phase** of the charge-balanced asymmetric biphasic pulse:

  * positive (fast) phase -- short, high current
  * negative (slow) phase -- long, low current (the 1:N recharge)

The same charge is injected on each phase, so the two phases have
different currents (I = charge / duration) and therefore different
impedances. We report both.

Derivation (validated against the PI's hand calcs):

  I_uA   = (charge_nC / pulse_width_us) * 1000
  V_mV   = (v_raw / gain) * 1000
  Z_kOhm = V_mV / I_uA

Worked examples:
  * BCH110: v_raw 0.770, gain 300, 5 nC / 150 us  ->  Z_pos ~= 0.077 kOhm
  * BCH062: v_raw 4.315, gain 150, 5 nC / 150 us  ->  Z_pos ~= 0.864 kOhm

Voltage is read phase-resolved from the recorded mean evoked trace
(``evoked_waveforms.mean_trace`` + ``time_axis_ms``): the stimulus
artifact lands in the first fraction of a millisecond around t=0, well
before the 2-200 ms evoked-response window. The positive phase takes the
peak |deflection| over ``[-pos_guard_ms, pulse_width]``; the negative
phase takes the mean |deflection| over ``[pulse_width, pulse_width*(1+ratio)]``.

Pure functions only -- no DB, no I/O. Unit-tested in tests/test_impedance.py.
"""

from __future__ import annotations

# A small pre-onset guard (ms) folded into the positive-phase window:
# stimulus detection sometimes aligns t=0 a sample or two after the
# artifact's leading edge, so the fast-phase peak can sit just before 0.
POS_GUARD_MS = 0.1


def normalize_ratio(ratio) -> float | None:
    """Return the negative-phase duration multiplier as a value >= 1.

    The recorder stores "Neg pulse ratio" as a plain number (3.0 for a
    1:3 asymmetric pulse). Defensive: if a fractional form (0.333) ever
    shows up, invert it so the negative phase is always the *longer* one.
    Returns None for a missing / non-positive ratio.
    """
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return None
    if r <= 0:
        return None
    if r < 1.0:
        r = 1.0 / r
    return r


def phase_currents(charge_nC, pulse_width_us, ratio
                   ) -> tuple[float | None, float | None]:
    """Return ``(i_pos_ua, i_neg_ua)`` from the *commanded* pulse.

    Charge-balanced: both phases inject ``charge_nC``. The positive phase
    lasts ``pulse_width_us``; the negative phase ``pulse_width_us * ratio``.
    Current (uA) = charge_nC / duration_us * 1000. Returns ``(None, None)``
    on missing / non-positive inputs.
    """
    try:
        q = float(charge_nC)
        pw = float(pulse_width_us)
    except (TypeError, ValueError):
        return (None, None)
    r = normalize_ratio(ratio)
    if q <= 0 or pw <= 0 or r is None:
        return (None, None)
    i_pos = (q / pw) * 1000.0
    i_neg = (q / (pw * r)) * 1000.0
    assert i_pos > 0 and i_neg > 0, "currents must be positive"
    return (i_pos, i_neg)


def _window_extremum(mean_trace, time_ms, t0_ms, t1_ms, reducer):
    """Apply *reducer* to |trace| over the sample window ``[t0_ms, t1_ms]``.

    ``reducer`` is ``max`` (positive-phase peak) or ``"mean"`` (negative-
    phase average). Returns None when the window catches no samples or the
    inputs are ragged.
    """
    assert reducer in ("max", "mean"), "reducer must be 'max' or 'mean'"
    n = min(len(mean_trace), len(time_ms))
    mags = []
    max_iter = n + 1
    for i in range(n):
        assert i < max_iter, "window scan runaway"
        t = time_ms[i]
        if t0_ms <= t <= t1_ms:
            mags.append(abs(float(mean_trace[i])))
    if not mags:
        return None
    if reducer == "max":
        return max(mags)
    return sum(mags) / len(mags)


def phase_voltages(mean_trace, time_ms, pulse_width_us, ratio,
                   pos_guard_ms: float = POS_GUARD_MS
                   ) -> tuple[float | None, float | None]:
    """Return ``(v_pos_raw, v_neg_raw)`` from the recorded mean trace.

    Positive phase = peak |deflection| over ``[-pos_guard_ms, pw_ms]``;
    negative phase = mean |deflection| over ``[pw_ms, pw_ms*(1+ratio)]``,
    where ``pw_ms = pulse_width_us / 1000``. Raw units (gain applied later).
    Returns ``(None, None)`` on bad inputs.
    """
    if not mean_trace or not time_ms:
        return (None, None)
    try:
        pw_ms = float(pulse_width_us) / 1000.0
    except (TypeError, ValueError):
        return (None, None)
    r = normalize_ratio(ratio)
    if pw_ms <= 0 or r is None:
        return (None, None)
    v_pos = _window_extremum(mean_trace, time_ms,
                             -abs(pos_guard_ms), pw_ms, "max")
    v_neg = _window_extremum(mean_trace, time_ms,
                             pw_ms, pw_ms * (1.0 + r), "mean")
    return (v_pos, v_neg)


def _one_impedance_kohm(v_raw, gain, i_ua) -> float | None:
    """Z_kOhm = (v_raw/gain*1000) / i_ua. None on missing/bad inputs."""
    if v_raw is None or gain is None or i_ua is None:
        return None
    try:
        g = float(gain)
        i = float(i_ua)
        v = float(v_raw)
    except (TypeError, ValueError):
        return None
    if g <= 0 or i <= 0:
        return None
    v_mv = (v / g) * 1000.0
    return v_mv / i


def compute_impedances(v_pos_raw, v_neg_raw, gain, i_pos_ua, i_neg_ua
                       ) -> tuple[float | None, float | None]:
    """Return ``(z_pos_kohm, z_neg_kohm)``. Each is None if its inputs
    are missing (guards: gain>0, current>0)."""
    z_pos = _one_impedance_kohm(v_pos_raw, gain, i_pos_ua)
    z_neg = _one_impedance_kohm(v_neg_raw, gain, i_neg_ua)
    return (z_pos, z_neg)


def impedance_for_channel(mean_trace, time_ms, gain,
                          charge_nC, pulse_width_us, ratio,
                          pos_guard_ms: float = POS_GUARD_MS) -> dict:
    """Convenience wrapper: full per-channel impedance record for storage.

    Returns a dict with the raw voltages, phase currents, gain, and both
    impedances (kOhm). Any field that can't be computed is ``None`` so the
    caller can still persist a partial row (e.g. gain-unknown channels).
    """
    i_pos, i_neg = phase_currents(charge_nC, pulse_width_us, ratio)
    v_pos, v_neg = phase_voltages(mean_trace, time_ms,
                                  pulse_width_us, ratio, pos_guard_ms)
    z_pos, z_neg = compute_impedances(v_pos, v_neg, gain, i_pos, i_neg)
    return {
        "gain": (float(gain) if gain not in (None, "") else None),
        "charge_nc": _as_float(charge_nC),
        "pulse_width_us": _as_float(pulse_width_us),
        "neg_ratio": normalize_ratio(ratio),
        "v_pos_raw": v_pos,
        "v_neg_raw": v_neg,
        "i_pos_ua": i_pos,
        "i_neg_ua": i_neg,
        "impedance_pos_kohm": z_pos,
        "impedance_neg_kohm": z_neg,
    }


def _as_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
