"""Across-trial feature series -- the substrate for slow-dynamics analysis.

The evoked FEATURES (`autocorrelation`, `variance`, ...) are within-waveform
(ms-scale, computed along the 20 kHz samples of one <=200 ms trace), so they
cannot see a minute-scale mode. The slow eigenvalue lives in the trial-to-trial
SERIES of a feature: at ~0.5 Hz, one scalar every ~2 s for days is a discretely
sampled dynamical series whose AR(1) coefficient phi = exp(lambda*dt) carries the
eigenvalue (phi -> 1 as lambda -> 0 near a bifurcation). This module builds that
series over the FULL timeline (not just the peri-onset rows the matrix keeps),
gap-aware, anchored to seizure onsets for lead-time labelling.

Reuses the whole-timeline reader src.evoked_figures.data (animal_series /
primary_channel / finite_series); adds dt estimation, gap flags, and
time_to_onset. The dt is measured from the data -- there is no hardcoded stim
rate (0.5 Hz appears only in comments). Gaps (file boundaries / dropouts) are
flagged, NOT interpolated, so downstream AR/spectral steps can segment at them.

See docs/periictal_registered_analyses.md and the Part B plan. Report phi/lambda,
never tau (tau is ill-conditioned near phi=1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MAX_STIMULI = 50_000_000


@dataclass
class TrialSeries:
    animal: str
    feature: str
    channel: str
    secs: np.ndarray          # absolute epoch seconds, ascending, finite
    values: np.ndarray        # feature value per stimulus (finite), aligned to secs
    dt_med: float             # median inter-stimulus interval (s)
    gap_after: np.ndarray     # bool[n]: a gap (> gap threshold) follows sample i
    time_to_onset: np.ndarray # signed s to the NEAREST included onset (+ = before)
    onsets: np.ndarray        # included-seizure onset epochs used for anchoring

    @property
    def n(self) -> int:
        return int(self.secs.size)

    def segments(self):
        """Index slices between gaps -- contiguous runs safe for AR/spectral fits."""
        if self.n == 0:
            return []
        cut = np.flatnonzero(self.gap_after[:-1]) + 1 if self.n > 1 else np.array([], int)
        bounds = np.concatenate(([0], cut, [self.n]))
        return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])
                if b - a >= 1]


def nearest_onset_delta(secs: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """Signed seconds from each stimulus to the NEAREST seizure onset, POSITIVE =
    the stimulus is that many seconds BEFORE the onset. Vectorised. NaN when there
    are no onsets."""
    secs = np.asarray(secs, dtype=float)
    onsets = np.asarray(onsets, dtype=float)
    onsets = onsets[np.isfinite(onsets)]
    if onsets.size == 0 or secs.size == 0:
        return np.full(secs.size, np.nan)
    so = np.sort(onsets)
    j = np.searchsorted(so, secs, side="left")
    up = np.full(secs.size, np.inf)                 # nearest upcoming onset (>=0)
    has_up = j < so.size
    up[has_up] = so[j[has_up]] - secs[has_up]
    past = np.full(secs.size, -np.inf)              # nearest past onset (<0)
    has_past = j > 0
    past[has_past] = so[j[has_past] - 1] - secs[has_past]
    tto = np.where(np.abs(up) <= np.abs(past), up, past)
    return np.where(np.isfinite(tto), tto, np.nan)


def assemble_trial_series(animal: str, feature: str, channel: str,
                          secs: np.ndarray, values: np.ndarray, onsets: np.ndarray,
                          *, gap_factor: float = 5.0,
                          gap_min_sec: float = 120.0) -> TrialSeries:
    """Pure builder from arrays (testable without a Store / sidecars). *secs* and
    *values* must be finite and time-ascending. A gap is flagged after sample i
    when the interval to i+1 exceeds max(gap_min_sec, gap_factor * median dt)."""
    secs = np.asarray(secs, dtype=float)
    values = np.asarray(values, dtype=float)
    assert secs.shape == values.shape, "secs/values length mismatch"
    assert secs.size < _MAX_STIMULI, "stimulus count runaway"
    diffs = np.diff(secs)
    dt_med = float(np.median(diffs)) if diffs.size else float("nan")
    thr = (max(gap_min_sec, gap_factor * dt_med)
           if np.isfinite(dt_med) and dt_med > 0 else gap_min_sec)
    gap_after = np.zeros(secs.size, dtype=bool)
    if diffs.size:
        gap_after[:-1] = diffs > thr
    tto = nearest_onset_delta(secs, onsets)
    return TrialSeries(animal=animal, feature=feature, channel=channel,
                       secs=secs, values=values, dt_med=dt_med,
                       gap_after=gap_after, time_to_onset=tto,
                       onsets=np.sort(np.asarray(onsets, dtype=float)))


def build_trial_series(store, animal: str, feature: str, evoked_dir: str, *,
                       channel: str | None = None, gap_factor: float = 5.0,
                       gap_min_sec: float = 120.0) -> TrialSeries:
    """Full-timeline across-trial series for (animal, feature). Reads every
    sidecar via src.evoked_figures.data, picks the primary channel (override with
    *channel*), and anchors to ``isi.included_seizures``. Empty TrialSeries when
    the animal has no finite points."""
    from src.evoked_figures.data import (animal_series, finite_series,
                                         primary_channel)
    from src.preictal.isi import included_seizures
    assert animal and evoked_dir, "animal and evoked_dir required"
    series, _inputs = animal_series(animal, evoked_dir, [feature])
    ch = primary_channel(animal, series, [feature], override=channel)
    onsets = np.array([s.onset_epoch for s in included_seizures(store, animal)],
                      dtype=float)
    if ch is None:
        return assemble_trial_series(animal, feature, "", np.empty(0),
                                     np.empty(0), onsets)
    _dts, secs, vals = finite_series(series, ch, feature)
    return assemble_trial_series(animal, feature, ch, secs, vals, onsets,
                                 gap_factor=gap_factor, gap_min_sec=gap_min_sec)
