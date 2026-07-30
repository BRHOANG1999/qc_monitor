"""Compose the Part B slow-dynamics analyses into one UI summary (pure).

Takes a TrialSeries (B0) + the animal's onsets and returns every headline number
the Slow-dynamics lens shows: the eigenvalue phi/lambda near vs far from onset
(B1), the variance/phi coupling (B2), the infraslow band fraction (B3), and the
circadian-matched vs same-day AUC (B4). No Dash, no Store -- unit-testable; the
tab worker just builds the TrialSeries and calls this.

Every readout is bounded by seizure count, not trial count -- the caller prints
that caveat. Reports phi/lambda, never tau.
"""

from __future__ import annotations

import numpy as np

from src.periictal import circadian as _cz
from src.periictal import slow_dynamics as _sd
from src.periictal import spectral as _sp

_DEFAULT_CAP_H = 6.0


def phi_vs_leadtime(series, phi: np.ndarray, *, cap_h: float = _DEFAULT_CAP_H,
                    n_bins: int = 8) -> dict:
    """Mean sliding-phi in linear lead-time bins over the pre-onset window (tto in
    (0, cap_h h]). Returns bin centres (hours before onset) + mean phi + counts;
    near = the smallest-lead bin, far = the largest."""
    tto = np.asarray(series.time_to_onset, dtype=float)
    phi = np.asarray(phi, dtype=float)
    edges = np.linspace(0.0, cap_h * 3600.0, int(n_bins) + 1)
    centers_h, phi_bin, counts = [], [], []
    for i in range(int(n_bins)):
        m = (tto >= edges[i]) & (tto < edges[i + 1]) & np.isfinite(phi)
        centers_h.append(0.5 * (edges[i] + edges[i + 1]) / 3600.0)
        phi_bin.append(float(np.nanmean(phi[m])) if m.any() else float("nan"))
        counts.append(int(m.sum()))
    return {"centers_h": centers_h, "phi": phi_bin, "counts": counts}


def summarize(series, *, win: int = 300, band=(0.001, 0.01),
              cap_h: float = _DEFAULT_CAP_H, n_bins: int = 8,
              min_n: int = 20) -> dict:
    """Full Slow-dynamics summary for a TrialSeries. Empty-safe."""
    n = series.n
    base = {"n": n, "channel": series.channel, "feature": series.feature,
            "dt_med": float(series.dt_med), "win": int(win),
            "n_seizures": int(np.isfinite(series.onsets).sum())}
    if n < max(win, 32):
        return {**base, "insufficient": True}
    phi = _sd.phi_series(series, win=win)
    lead = phi_vs_leadtime(series, phi, cap_h=cap_h, n_bins=n_bins)
    phi_near = next((p for p in lead["phi"] if np.isfinite(p)), float("nan"))
    finite = [p for p in lead["phi"] if np.isfinite(p)]
    phi_far = finite[-1] if finite else float("nan")
    dec = _sd.decoupling(series, win=win)
    spec = _sp.series_band_power(series, band[0], band[1])
    cz = _cz.circadian_matched_auc(series, min_n=min_n)
    return {**base, "insufficient": False,
            "phi_leadtime": lead,
            "phi_near": float(phi_near), "phi_far": float(phi_far),
            "lambda_near": float(_sd.lambda_from_phi(phi_near, series.dt_med)),
            "lambda_far": float(_sd.lambda_from_phi(phi_far, series.dt_med)),
            "coupling_rho": dec["coupling_rho"],
            "band_frac": spec["band_frac"], "band": tuple(band),
            "matched_auc": cz["matched_pooled_auc"],
            "sameday_auc": cz["sameday_pooled_auc"],
            "circadian_per_seizure": cz["per_seizure"]}
