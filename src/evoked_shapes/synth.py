"""Stage 0: synthetic evoked traces with KNOWN structure, the test oracle.

Every later stage is validated against these generators: if a stage cannot recover
structure we planted here (with realistic gain scaling, latency jitter, and noise),
it will not be trusted on real data. Three regimes:

  discrete   each trace is one of ``n_templates`` fixed shapes  -> stages should
             report DISCRETE modes and recover the templates (ARI near 1).
  continuum  each trace is a smooth blend between two shapes, blend coordinate
             drawn uniformly -> stages should report a CONTINUUM (unimodal).
  mixture    part discrete, part continuum -> the hard, realistic case.

Each trace additionally gets a random gain (x0.3 to x3), an integer latency shift
(+/- jitter samples), and additive white noise. Because the analysis amplitude-
normalizes first, a correct pipeline is invariant to the gain; the jitter is what a
latency-robust method (alignment, DTW) should absorb.

Pure numpy. Explicit seed. Returns ``(X[n, n_samples], meta)`` with the ground truth
in ``meta`` so tests can assert recovery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The synthetic window is a unit interval; real windows are ms post-stim, but the
# shape math is scale-free so the templates transfer.
_T0, _T1 = 0.0, 1.0


def _gauss(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def make_templates(n_samples: int, n_templates: int = 3) -> np.ndarray:
    """``n_templates`` distinct, evoked-response-like unit-peak shapes over
    *n_samples* points. Deterministic (no randomness): the templates are the fixed
    ground-truth shapes every regime is built from."""
    assert n_samples >= 16, "n_samples must be >= 16"
    assert 2 <= n_templates <= 5, "n_templates must be in [2, 5]"
    t = np.linspace(_T0, _T1, n_samples)
    shapes = [
        # A: sharp early trough then a positive rebound (N1-P2 like).
        -_gauss(t, 0.12, 0.035) + 0.75 * _gauss(t, 0.28, 0.07),
        # B: single slow broad positive deflection.
        _gauss(t, 0.32, 0.15),
        # C: biphasic damped oscillation.
        np.sin(2.0 * np.pi * 3.0 * t) * np.exp(-t / 0.35),
        # D: late positive with a small early notch.
        0.9 * _gauss(t, 0.55, 0.12) - 0.3 * _gauss(t, 0.15, 0.04),
        # E: sustained plateau then decay.
        np.clip(1.0 - np.abs(t - 0.4) / 0.4, 0.0, 1.0),
    ][:n_templates]
    out = np.asarray(shapes, dtype=np.float64)
    peak = np.nanmax(np.abs(out), axis=1, keepdims=True)
    peak[peak == 0] = 1.0
    return out / peak


def _apply_jitter(y: np.ndarray, shift: int) -> np.ndarray:
    """Shift *y* by an integer number of samples with edge fill (no wraparound).
    Positive *shift* delays the response."""
    n = y.size
    out = np.empty_like(y)
    if shift == 0:
        return y.copy()
    if shift > 0:
        s = min(shift, n)
        out[:s] = y[0]
        out[s:] = y[:n - s]
    else:
        s = min(-shift, n)
        out[n - s:] = y[-1]
        out[:n - s] = y[s:]
    return out


def _stamp(y: np.ndarray, gain: float, shift: int, noise: float,
           rng: np.random.Generator) -> np.ndarray:
    """Apply gain, latency jitter, and additive noise to one unit-scale shape."""
    yj = _apply_jitter(y, shift)
    return gain * yj + noise * rng.standard_normal(yj.size)


def generate(mode: str, n: int, n_samples: int = 128, *,
             n_templates: int = 3, gain_range: tuple[float, float] = (0.3, 3.0),
             jitter: int = 4, noise: float = 0.05, n_sessions: int = 6,
             seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate *n* synthetic traces in ``{"discrete", "continuum", "mixture"}``.

    Returns ``(X[n, n_samples], meta)``. ``meta`` mirrors the real Stage-1 frame
    (``animal, session, trial_time, time_to_seizure``) and adds the ground truth:
    ``kind`` (discrete/continuum), ``true_label`` (template index, -1 for continuum
    rows), ``true_coord`` (blend coordinate in [0, 1], NaN for discrete rows),
    ``true_gain``, ``true_shift``.
    """
    assert mode in ("discrete", "continuum", "mixture"), \
        "mode must be discrete, continuum, or mixture"
    assert n >= 1 and n_samples >= 16, "need n>=1 and n_samples>=16"
    assert gain_range[0] > 0 and gain_range[1] >= gain_range[0], "bad gain_range"
    assert jitter >= 0 and noise >= 0 and n_sessions >= 1, "bad params"
    rng = np.random.default_rng(seed)
    tpl = make_templates(n_samples, n_templates)

    X = np.empty((n, n_samples), dtype=np.float64)
    kind = np.empty(n, dtype=object)
    true_label = np.full(n, -1, dtype=int)
    true_coord = np.full(n, np.nan, dtype=np.float64)
    gains = rng.uniform(gain_range[0], gain_range[1], size=n)
    shifts = rng.integers(-jitter, jitter + 1, size=n) if jitter > 0 else np.zeros(n, int)

    # In mixture mode, the first half is discrete and the second half continuum.
    discrete_mask = np.ones(n, dtype=bool)
    if mode == "continuum":
        discrete_mask[:] = False
    elif mode == "mixture":
        discrete_mask[n // 2:] = False

    for i in range(n):
        if discrete_mask[i]:
            lab = int(rng.integers(0, n_templates))
            base = tpl[lab]
            true_label[i] = lab
            kind[i] = "discrete"
        else:
            c = float(rng.uniform(0.0, 1.0))
            base = (1.0 - c) * tpl[0] + c * tpl[1]
            pk = np.max(np.abs(base))
            base = base / (pk if pk > 0 else 1.0)
            true_coord[i] = c
            kind[i] = "continuum"
        X[i] = _stamp(base, float(gains[i]), int(shifts[i]), noise, rng)

    sessions = rng.integers(0, n_sessions, size=n)
    # Give trials a plausible time axis: 2 s apart, so block-bootstrap sees order.
    trial_time = 1_700_000_000.0 + 2.0 * np.arange(n)
    meta = pd.DataFrame({
        "animal": "SYN",
        "session": [f"syn_sess_{int(s):02d}" for s in sessions],
        "rec": [f"syn_rec_{int(s):02d}" for s in sessions],
        "trial_time": trial_time,
        "time_to_seizure": np.nan,
        "stim_status": "mapped",
        "kind": kind,
        "true_label": true_label,
        "true_coord": true_coord,
        "true_gain": gains,
        "true_shift": shifts,
    })
    return X, meta


def time_ms(n_samples: int, window_ms: tuple[float, float] = (1.0, 200.0)) -> np.ndarray:
    """A ms time axis matching a generated trace, so synthetic data can flow through
    the same plotting/preprocess paths as real data."""
    assert n_samples >= 2, "n_samples must be >= 2"
    return np.linspace(window_ms[0], window_ms[1], n_samples)
