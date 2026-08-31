"""Rₐ flat-run detection for chronic-stim stability mapping.

Detects the LONGEST contiguous 'flat' run in an access-resistance (Rₐ) time
series. 'Flat' = low rolling drift AND low rolling dispersion, both measured
RELATIVE TO A GLOBAL ROBUST SCALE of the series (not the local median).

Scale for 'small': the irreducible sample-to-sample NOISE FLOOR
``nf = 1.4826*MAD(diff(r)) / sqrt(2)`` (trend-removed via first differences),
NOT %-of-local-median and NOT the whole-series MAD. Rationale:

* %-of-local-median is unstable when Rₐ dips near zero (BCH111's flattest
  stretch sits at ~0.13 kΩ with ~0.012 kΩ wiggle): dividing by the tiny median
  inflates that to ~10% and the detector picks a higher-magnitude drifting run.
* whole-series MAD works when flat and non-flat regions coexist, but collapses
  to the point noise on a globally-flat series (disp/scale ~1 -> nothing flat).

The noise floor is the flat-region wiggle itself, so 'flat' == 'local wiggle and
drift within a few noise floors', which is well-defined even when the whole
series is flat. Validated: recovers Jul 22->29 against the eyeballed Jul 21-31.

Dropouts are ABSENT rows (the DB query drops NULL Rₐ), so gaps are found from
timestamp deltas, not NaNs. Pure + numpy-only so it is unit-testable.
"""

from __future__ import annotations

import numpy as np

_MAX_PTS = 1_000_000        # NASA Rule 2: explicit bound on series scans.
_MAD_K = 1.4826             # MAD -> Gaussian-sigma consistency factor.
_EPS = 1e-9


def robust_scale(r) -> float:
    """Whole-series robust scale ``S = 1.4826*MAD(r)`` (kΩ), floored off zero.
    A convenience descriptor of overall spread (the flatness gate and spike
    detector use the sample-to-sample :func:`noise_floor`, not this)."""
    r = np.asarray(r, dtype=float)
    assert r.ndim == 1 and r.size >= 1, "r must be a non-empty 1-D array"
    med = float(np.median(r))
    s = _MAD_K * float(np.median(np.abs(r - med)))
    return float(max(s, _EPS))


def noise_floor(r) -> float:
    """Irreducible sample-to-sample noise ``1.4826*MAD(diff(r))/sqrt(2)`` (kΩ).
    First-differencing removes any slow trend, so this is the flat-region
    wiggle -- the scale against which 'flat' is judged. Floored off zero."""
    r = np.asarray(r, dtype=float)
    assert r.ndim == 1 and r.size >= 1, "r must be a non-empty 1-D array"
    if r.size < 2:
        return _EPS
    d = np.diff(r)
    nf = _MAD_K * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0)
    return float(max(nf, _EPS))


def _rolling_slope_disp(t_h, r, win_h: float, min_pts: int, valid=None):
    """Per-index centered-rolling LS slope (kΩ/h) and robust SD (kΩ).

    Computed on the ACTUAL ``t_h`` inside each window so uneven spacing / gaps
    do not distort the fit. Points where ``valid`` is False (e.g. flagged
    spikes) are EXCLUDED from every window's fit, so a lone spike does not
    inflate its neighbours' slope. NaN where the window holds < *min_pts*
    valid points.
    """
    t_h = np.asarray(t_h, float)
    r = np.asarray(r, float)
    n = r.size
    assert n <= _MAX_PTS, "series too long"
    ok = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    slope = np.full(n, np.nan)
    disp = np.full(n, np.nan)
    half = win_h / 2.0
    for i in range(n):
        m = (t_h >= t_h[i] - half) & (t_h <= t_h[i] + half) & ok
        if int(m.sum()) < min_pts:
            continue
        x = t_h[m]
        y = r[m]
        slope[i] = float(np.polyfit(x - x.mean(), y, 1)[0])
        disp[i] = _MAD_K * float(np.median(np.abs(y - np.median(y))))
    return slope, disp


def flat_mask(t_h, r, *, win_h: float = 24.0, disp_k: float = 4.0,
              slope_k: float = 6.0, spike_k: float = 5.0,
              nf: float | None = None) -> dict:
    """Per-point flatness + spike flags for a Rₐ series.

    ``flat`` = rolling dispersion <= ``disp_k`` noise floors AND rolling drift
    (|slope|*win) <= ``slope_k`` noise floors. ``spike`` = a point whose
    deviation from its local median exceeds ``spike_k`` local robust-SDs
    (floored by the noise floor so a flat neighbourhood doesn't over-flag).
    """
    t_h = np.asarray(t_h, float)
    r = np.asarray(r, float)
    assert t_h.shape == r.shape and r.ndim == 1, "t_h, r must be equal 1-D"
    NF = float(nf) if nf is not None else noise_floor(r)
    min_pts = max(4, int(round(win_h / 2.0)))
    spike = _spike_mask(t_h, r, win_h, spike_k, NF)
    slope, disp = _rolling_slope_disp(t_h, r, win_h, min_pts, valid=~spike)
    disp_n = disp / NF
    slope_n = np.abs(slope) * win_h / NF
    flat = np.isfinite(disp_n) & np.isfinite(slope_n) \
        & (disp_n <= disp_k) & (slope_n <= slope_k)
    return {"flat": flat, "spike": spike, "disp_n": disp_n,
            "slope_n": slope_n, "noise_floor": NF}


def _spike_mask(t_h, r, win_h, spike_k, NF) -> np.ndarray:
    """Point-level artifact flag: |r_i - local_median| > spike_k * local scale
    (local robust SD, floored by the noise floor ``NF``)."""
    n = r.size
    half = win_h / 2.0
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        m = (t_h >= t_h[i] - half) & (t_h <= t_h[i] + half)
        y = r[m]
        if y.size < 4:
            continue
        loc = _MAD_K * float(np.median(np.abs(y - np.median(y))))
        thr = spike_k * max(loc, NF)
        out[i] = abs(r[i] - float(np.median(y))) > thr
    return out


def detect_longest_flat_run(t_h, r, t_epoch=None, *, win_h: float = 24.0,
                            gap_tol_h: float = 6.0, disp_k: float = 4.0,
                            slope_k: float = 6.0, spike_k: float = 5.0,
                            nf: float | None = None) -> dict:
    """Longest contiguous flat run (by DURATION). A run breaks on a time gap
    > ``gap_tol_h`` or on >=2 ADJACENT spikes; a lone spike (both neighbours
    flat) is flagged but tolerated. Returns boundary indices/epochs, duration,
    tolerated-gap stats, flagged spikes, and summary Rₐ stats."""
    t_h = np.asarray(t_h, float)
    r = np.asarray(r, float)
    assert r.ndim == 1 and r.size >= 1, "r must be a non-empty 1-D array"
    fm = flat_mask(t_h, r, win_h=win_h, disp_k=disp_k,
                   slope_k=slope_k, spike_k=spike_k, nf=nf)
    included, lone = _included_points(fm["flat"], fm["spike"])
    a, b = _longest_included_run(t_h, included, gap_tol_h)
    a, b = _trim_to_flat(fm["flat"], a, b)
    return _summarize_run(t_h, r, t_epoch, a, b, gap_tol_h, fm, lone)


def _included_points(flat, spike):
    """A point joins a run when its neighbourhood is flat AND it is not part of
    a spike cluster: ``flat AND (not spike OR lone)``. A LONE spike (neither
    neighbour is a spike) in a flat neighbourhood is tolerated; two ADJACENT
    spikes are both excluded, breaking the run. Rolling stats are already
    computed with spikes excluded, so a lone spike leaves its neighbours flat."""
    n = flat.size
    lone = np.zeros(n, dtype=bool)
    for i in range(n):
        if not spike[i]:
            continue
        left = spike[i - 1] if i > 0 else False
        right = spike[i + 1] if i < n - 1 else False
        lone[i] = (not left) and (not right)
    included = flat & (~spike | lone)
    return included, lone


def _longest_included_run(t_h, included, gap_tol_h):
    """Longest run (by duration) over included points, broken by a Δt gap
    exceeding ``gap_tol_h``."""
    n = included.size
    best = (-1.0, 0, -1)
    i = 0
    while i < n:
        assert i <= n, "run scan runaway"
        if not included[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and included[j + 1] and (t_h[j + 1] - t_h[j]) <= gap_tol_h:
            j += 1
        dur = t_h[j] - t_h[i]
        if dur > best[0]:
            best = (dur, i, j)
        i = j + 1
    return best[1], best[2]


def _trim_to_flat(flat, a, b):
    """Trim leading/trailing tolerated spikes so the run's endpoints are
    genuine flat points."""
    if b < a:
        return a, b
    while a <= b and not flat[a]:
        a += 1
    while b >= a and not flat[b]:
        b -= 1
    return a, b


def _summarize_run(t_h, r, t_epoch, a, b, gap_tol_h, fm, lone) -> dict:
    """Assemble the run's descriptive record."""
    if b < a:
        return {"found": False, "flat_mask": fm}
    seg_t = t_h[a:b + 1]
    seg_r = r[a:b + 1]
    dt = np.diff(seg_t)
    med = float(np.median(seg_r))
    cv = 100.0 * _MAD_K * float(np.median(np.abs(seg_r - med))) / med \
        if med > 0 else float("nan")
    out = {
        "found": True, "idx_start": int(a), "idx_end": int(b),
        "n_samples": int(b - a + 1),
        "duration_h": float(t_h[b] - t_h[a]),
        "median_ra_kohm": med, "robust_cv_pct": cv,
        "n_gaps_tolerated": int((dt > 1.5 * float(np.median(dt))).sum())
        if dt.size else 0,
        "max_gap_h": float(dt.max()) if dt.size else 0.0,
        "flagged_spikes_in_run": [int(k) for k in range(a, b + 1) if lone[k]],
        "flat_mask": fm,
    }
    if t_epoch is not None:
        te = np.asarray(t_epoch, float)
        out["epoch_start"] = float(te[a])
        out["epoch_end"] = float(te[b])
    return out


def validate_against_window(run: dict, t_epoch, eye_start_epoch: float,
                            eye_end_epoch: float, *, iou_pass: float = 0.6,
                            bound_tol_h: float = 24.0) -> dict:
    """Compare the detected run to an eyeballed [start, end] window on the time
    axis: IoU, %-overlap each way, signed boundary deltas (h), PASS verdict.
    A REPORTING AID -- never used to tune thresholds."""
    assert run.get("found"), "no run to validate"
    te = np.asarray(t_epoch, float)
    d0 = te[run["idx_start"]]
    d1 = te[run["idx_end"]]
    inter = max(0.0, min(d1, eye_end_epoch) - max(d0, eye_start_epoch))
    union = max(d1, eye_end_epoch) - min(d0, eye_start_epoch)
    det = max(_EPS, d1 - d0)
    eye = max(_EPS, eye_end_epoch - eye_start_epoch)
    iou = inter / union if union > 0 else 0.0
    d_start = (d0 - eye_start_epoch) / 3600.0
    d_end = (d1 - eye_end_epoch) / 3600.0
    verdict = (iou >= iou_pass and abs(d_start) <= bound_tol_h
               and abs(d_end) <= bound_tol_h)
    return {"iou": iou, "pct_eye_covered": 100.0 * inter / eye,
            "pct_det_covered": 100.0 * inter / det,
            "delta_start_h": d_start, "delta_end_h": d_end,
            "verdict_pass": bool(verdict)}


def sensitivity_sweep(t_h, r, t_epoch, grid: list[dict]) -> list[dict]:
    """Run the detector across a small parameter grid so the chosen window's
    robustness is shown, not asserted. Each grid entry overrides defaults."""
    assert isinstance(grid, list) and grid, "grid must be a non-empty list"
    out = []
    for i, g in enumerate(grid):
        assert i < 10_000, "sweep runaway"
        run = detect_longest_flat_run(t_h, r, t_epoch, **g)
        rec = {**g, "found": run.get("found", False)}
        if run.get("found"):
            rec.update({"duration_h": run["duration_h"],
                        "n_samples": run["n_samples"],
                        "epoch_start": run.get("epoch_start"),
                        "epoch_end": run.get("epoch_end")})
        out.append(rec)
    return out
