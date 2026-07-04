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


# ===================================================================== #
#  Access resistance (ohmic step at a current transition) -- the metric.
#
#  Rₐ = |ΔV / ΔI| measured as the INSTANTANEOUS voltage step across a
#  current transition. The electrode double-layer (polarization) voltage
#  cannot change instantaneously, so the step at a transition is purely the
#  ohmic (access/series) drop -- it isolates access resistance from
#  polarization by construction. This replaces the peak-vs-mean phase
#  "impedance", which mixed resistive + capacitive terms and used two
#  non-comparable statistics.
# ===================================================================== #

# The stim-copy excursion must be at least this (raw units) to count as a
# real biphasic pulse rather than a flat/absent command.
STIM_AMP_MIN = 1e-4


def find_current_transitions(stim_mean_trace, time_ms):
    """Locate the biphasic current transitions from the stim-copy average.

    Returns ``(onset, reversal, offset)`` sample indices, or ``None`` if the
    stim-copy doesn't show a clean fast(+) / slow(−) / off pulse. The
    stim-copy is a clean recording of the COMMANDED current, so its
    transitions mark exactly where the recorded electrode voltage shows its
    ohmic steps -- robust to alignment jitter on the animal channel.
    """
    if not stim_mean_trace or not time_ms:
        return None
    n = min(len(stim_mean_trace), len(time_ms))
    if n < 4:
        return None
    sm = [float(stim_mean_trace[i]) for i in range(n)]
    t = [float(time_ms[i]) for i in range(n)]
    pre = [sm[i] for i in range(n) if t[i] < -0.15]
    if len(pre) < 2:
        pre = sm[:max(2, n // 10)]
    base = sorted(pre)[len(pre) // 2]              # median pre-stim level
    fast_amp = max(sm) - base                      # positive (fast) excursion
    slow_amp = base - min(sm)                       # negative (slow) excursion
    if fast_amp < STIM_AMP_MIN or slow_amp < STIM_AMP_MIN:
        return None
    onset = reversal = offset = None
    max_iter = n + 1
    for i in range(n):
        assert i < max_iter, "transition scan runaway"
        d = sm[i] - base
        if onset is None:
            if d > 0.5 * fast_amp:
                onset = i
        elif reversal is None:
            if d < -0.5 * slow_amp:
                reversal = i
        elif d > -0.3 * slow_amp:               # risen back toward baseline
            offset = i
            break
    if onset is None or reversal is None or offset is None:
        return None
    return (onset, reversal, offset)


def _step_kohm(mean_trace, i_after, gain, delta_i_ua) -> float | None:
    """Ohmic ``|ΔV/ΔI|`` in kΩ for the step landing on sample ``i_after``
    (jump from ``i_after-1`` → ``i_after``). ΔV gain-corrected to mV, ΔI in
    µA → kΩ (mV/µA = kΩ). None on missing / bad inputs."""
    if i_after is None or i_after < 1 or gain in (None, 0) or not delta_i_ua:
        return None
    try:
        g = float(gain)
        di = float(delta_i_ua)
        dv_mv = (float(mean_trace[i_after])
                 - float(mean_trace[i_after - 1])) / g * 1000.0
    except (TypeError, ValueError, IndexError):
        return None
    if g <= 0 or di <= 0:
        return None
    return abs(dv_mv) / di


# Two independent transitions must agree within this fraction for the
# measurement to pass the LINEARITY check (rejects asymmetric clipping /
# noise). NB: for an ohmic interface both transitions give the same R by
# Ohm's law, so agreement confirms linearity, not the absolute value.
ACCESS_R_AGREE_TOL = 0.35

# Saturation/collapse guard: the reversal step must be at least this fraction
# of the fast-phase PEAK deflection. When the amp saturates on the fast-phase
# spike and recovers before the reversal (response collapses to ~baseline by
# then), the reversal step is spuriously tiny → nonphysical R≈0. Such rows
# are rejected rather than plotted as a real drop.
ACCESS_R_MIN_SUSTAIN = 0.15


def access_resistance(mean_trace, time_ms, stim_mean_trace, gain,
                      i_fast_ua, i_slow_ua,
                      agree_tol: float = ACCESS_R_AGREE_TOL,
                      min_sustain: float = ACCESS_R_MIN_SUSTAIN) -> dict:
    """Effective electrode SERIES resistance from the ohmic voltage step (kΩ).

    Rₐ = |ΔV|/ΔI at a current transition. NB: at 20 kHz one sample is 50 µs,
    so ΔV folds in ≤50 µs of double-layer charging on top of the pure ohmic
    drop — this is an *effective* series resistance at ~single-sample
    bandwidth (great for tracking drift, not an absolute access resistance).

    Uses the fast→slow REVERSAL (largest ΔI, best SNR) and the slow→OFF step
    as two estimates. Two QC gates: (1) a saturation/collapse guard — reject
    when the fast-phase response has decayed before the reversal (amp railed
    and recovered → spurious R≈0); (2) a linearity check — the two transitions
    must agree within ``agree_tol`` (else clipping/noise). Reports the mean as
    ``r_access_kohm`` when both pass; individual steps always returned.
    """
    out = {"r_reversal_kohm": None, "r_offset_kohm": None,
           "r_access_kohm": None, "n_used": 0}
    tr = find_current_transitions(stim_mean_trace, time_ms)
    if tr is None or gain in (None, 0) or not i_fast_ua or not i_slow_ua:
        return out
    onset, reversal, offset = tr
    rev = _step_kohm(mean_trace, reversal, gain,
                     float(i_fast_ua) + float(i_slow_ua))
    off = _step_kohm(mean_trace, offset, gain, float(i_slow_ua))
    out["r_reversal_kohm"] = rev
    out["r_offset_kohm"] = off
    # (1) saturation/collapse guard.
    if not _reversal_sustained(mean_trace, onset, reversal, min_sustain):
        return out                               # spurious R≈0 -> reject
    # (2) linearity check.
    if rev is not None and off is not None:
        mean = (rev + off) / 2.0
        if mean > 0 and abs(rev - off) / mean <= agree_tol:
            out["r_access_kohm"] = mean
            out["n_used"] = 2
    return out


# Slow-phase steady-state window (fractions of the slow phase, measured from
# the reversal to the offset): skip the reversal transient at the start and
# the offset transient at the end -> the settled plateau in between.
SLOW_SS_LO = 0.4
SLOW_SS_HI = 0.9


def slow_steady_state(mean_trace, time_ms, stim_mean_trace, gain,
                      i_slow_ua, lo: float = SLOW_SS_LO,
                      hi: float = SLOW_SS_HI) -> dict:
    """Steady-state IMPEDANCE of the long slow (negative) phase (kΩ).

    A stim-consistency signal complementary to Rₐ: the settled plateau voltage
    of the slow phase, gain-corrected and divided by the COMMANDED slow current
    -> ``Z_ss = |V_ss|(mV) / i_slow(µA)`` [kΩ]. Because the gain is divided
    out, Z_ss is comparable BETWEEN animals. V_ss = mean of ``mean_trace`` over
    the mid-late slow window ``[reversal+lo·span, reversal+hi·span]`` (span =
    offset-reversal), which avoids the reversal + offset transients. Returns
    ``{slow_ss_raw, slow_ss_kohm}`` (None on missing/bad inputs). NB: at 20 kHz
    the slow phase is still decaying, so this is a settled-plateau estimate
    (window-dependent absolute value; valid for drift + like-for-like
    cross-animal comparison)."""
    out = {"slow_ss_raw": None, "slow_ss_kohm": None}
    tr = find_current_transitions(stim_mean_trace, time_ms)
    if tr is None or gain in (None, 0) or not i_slow_ua:
        return out
    _onset, reversal, offset = tr
    span = offset - reversal
    if span <= 1:
        return out
    a = reversal + int(span * lo)
    b = max(a + 1, reversal + int(span * hi))
    try:
        g = float(gain)
        i_s = float(i_slow_ua)
        win = [float(mean_trace[i]) for i in range(a, min(b, len(mean_trace)))]
    except (TypeError, ValueError, IndexError):
        return out
    if not win or g <= 0 or i_s <= 0:
        return out
    v_ss_raw = sum(win) / len(win)
    out["slow_ss_raw"] = v_ss_raw
    out["slow_ss_kohm"] = abs(v_ss_raw / g * 1000.0) / i_s
    return out


def _reversal_sustained(mean_trace, onset, reversal, min_sustain) -> bool:
    """True when the reversal step is a real fraction of the fast-phase peak
    deflection (i.e. the response hadn't already collapsed to baseline)."""
    if reversal is None or reversal < 1 or onset is None:
        return False
    try:
        peak = max(abs(float(mean_trace[i]))
                   for i in range(onset, reversal + 1))
        step = abs(float(mean_trace[reversal]) - float(mean_trace[reversal - 1]))
    except (TypeError, ValueError, IndexError):
        return False
    return peak > 0 and (step / peak) >= min_sustain


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
