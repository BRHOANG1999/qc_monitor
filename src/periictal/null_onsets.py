"""Deep-interictal NULL onsets for the nonstationarity control.

The sliding-window AUC scores a preictal window (0-30 min before a seizure) vs
interictal reference windows (1-6 h before). An AUC well above 0.5 could be a real
pre-ictal signal OR just feature drift over hours (nonstationarity). To separate
them we place FAKE ("null") onsets in deep-interictal time and run the SAME
machinery: if the null AUC matches the seizure AUC, the effect is drift.

A valid null onset:
  * matches the COUNT of real seizures (n = len(included_seizures));
  * sits on a calendar day that HAS >=1 real seizure (match the circadian/state
    distribution) and, softly, in proportion to that day's real-seizure count;
  * is >= buffer_sec (default band_hi + 1 h) clear of EVERY real seizure on BOTH
    sides -- so its whole 1-6 h lookback is seizure-free AND it doesn't sit inside
    the next seizure's own lookback (i.e. it isn't a disguised real preictal
    sample); and
  * has real stimulus coverage in both its (0, preictal_max] preictal window and
    its [band_lo, band_hi] interictal band (else no window is scorable).

Pure + seeded (deterministic per seed) so a null DISTRIBUTION is K independent
draws. The seizure-relative matrix columns for these fake onsets are then built by
``matrix.build_null_matrix`` and scored by the unchanged ``sliding_window_auc``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from src.periictal import config as _cfg
from src.preictal.isi import included_seizures

_MAX_DAYS = 100_000                 # NASA Rule 2 bounds
_MAX_CAND_PER_DAY = 100_000


@dataclass
class NullDraw:
    """One seeded set of fake onsets. ``onsets`` is sorted epoch seconds;
    ``n_placed < n_requested`` (with a ``reason``) when deep-interictal coverage
    is too thin to match the seizure count."""
    onsets: np.ndarray
    n_requested: int
    n_placed: int
    reason: str | None
    seed: int


def _clear_of_onsets(cands: np.ndarray, real: np.ndarray,
                     buffer_sec: float) -> np.ndarray:
    """Mask of candidates whose NEAREST real onset (either side) is >= buffer."""
    if real.size == 0:
        return np.ones(cands.shape, dtype=bool)
    idx = np.searchsorted(real, cands)
    left = np.where(idx > 0, cands - real[np.clip(idx - 1, 0, real.size - 1)],
                    np.inf)
    right = np.where(idx < real.size, real[np.clip(idx, 0, real.size - 1)] - cands,
                     np.inf)
    return np.minimum(left, right) >= buffer_sec


def _count_after(cov: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Count of coverage epochs in (lo, hi], elementwise for arrays lo/hi."""
    return (np.searchsorted(cov, hi, side="right")
            - np.searchsorted(cov, lo, side="right"))


def _has_coverage(cands: np.ndarray, cov: np.ndarray, *, preictal_max_sec: float,
                  band_lo_sec: float, band_hi_sec: float,
                  min_coverage: int) -> np.ndarray:
    """Mask of candidates with >= min_coverage stimuli in BOTH the preictal
    window (a-preictal_max, a] and the interictal band [a-band_hi, a-band_lo]."""
    if cov.size == 0:
        return np.zeros(cands.shape, dtype=bool)
    pre = _count_after(cov, cands - preictal_max_sec, cands)
    band = _count_after(cov, cands - band_hi_sec, cands - band_lo_sec)
    return (pre >= min_coverage) & (band >= min_coverage)


def _day_candidates(day: date, real: np.ndarray, cov: np.ndarray, *,
                    grid_step_sec: float, buffer_sec: float, preictal_max_sec: float,
                    band_lo_sec: float, band_hi_sec: float,
                    min_coverage: int) -> np.ndarray:
    """Valid anchor epochs on *day*: on a fixed grid, buffer-clear of real
    seizures, and stimulus-backed. Deterministic (no RNG)."""
    start = datetime(day.year, day.month, day.day).timestamp()
    cands = np.arange(start, start + 86400.0, grid_step_sec)
    assert cands.size < _MAX_CAND_PER_DAY, "candidate grid runaway"
    cands = cands[_clear_of_onsets(cands, real, buffer_sec)]
    if cands.size == 0:
        return cands
    return cands[_has_coverage(cands, cov, preictal_max_sec=preictal_max_sec,
                               band_lo_sec=band_lo_sec, band_hi_sec=band_hi_sec,
                               min_coverage=min_coverage)]


def _greedy_pick(chosen: list, cands: np.ndarray, want: int, n: int, *,
                 min_spacing_sec: float) -> None:
    """Append up to *want* candidates (and never past *n* total) to *chosen*,
    keeping every accepted onset >= min_spacing_sec from all prior picks. *cands*
    is pre-shuffled; mutates *chosen* in place."""
    placed = 0
    for a in cands:
        if placed >= want or len(chosen) >= n:
            return
        a = float(a)
        if all(abs(a - c) >= min_spacing_sec for c in chosen):
            chosen.append(a)
            placed += 1


def _place_null_onsets(real_onsets, day_counts, coverage, n, *, rng,
                       buffer_sec, window_sec, preictal_max_sec, band_lo_sec,
                       band_hi_sec, min_coverage, min_spacing_sec,
                       grid_step_sec) -> np.ndarray:
    """Pure core: choose up to *n* fake onsets from the given real onsets, seizure
    day->count map, and sorted stimulus-coverage array. Stratified by day
    (per-day target ~ that day's real count), then a fill pass; spacing enforced
    globally. Returns sorted epochs (may be < n)."""
    real = np.sort(np.asarray(real_onsets, dtype=float))
    cov = np.sort(np.asarray(coverage, dtype=float))
    cov = cov[np.isfinite(cov)]
    days = sorted(day_counts)
    assert len(days) < _MAX_DAYS, "seizure-day count runaway"

    # Precompute valid candidates per day once (deterministic).
    per_day: dict = {}
    for d in days:
        c = _day_candidates(d, real, cov, grid_step_sec=grid_step_sec,
                            buffer_sec=buffer_sec, preictal_max_sec=preictal_max_sec,
                            band_lo_sec=band_lo_sec, band_hi_sec=band_hi_sec,
                            min_coverage=min_coverage)
        if c.size:
            per_day[d] = c

    chosen: list = []
    # Pass 1: stratified -- per-day target ~ that day's real-seizure count.
    for d in days:
        if len(chosen) >= n:
            break
        c = per_day.get(d)
        if c is None:
            continue
        c = c.copy()
        rng.shuffle(c)
        _greedy_pick(chosen, c, int(max(1, day_counts.get(d, 1))), n,
                     min_spacing_sec=min_spacing_sec)
    # Pass 2: fill the remainder from all valid candidates, ignoring the per-day
    # cap (so a few busy days can make up a shortfall on sparse ones).
    if len(chosen) < n:
        allc = np.concatenate([per_day[d] for d in days if d in per_day]) \
            if per_day else np.array([])
        rng.shuffle(allc)
        _greedy_pick(chosen, allc, n - len(chosen), n,
                     min_spacing_sec=min_spacing_sec)
    return np.sort(np.asarray(chosen, dtype=float))


def _seizure_day_counts(store, animal: str) -> dict:
    """{date: n_real_seizures} for *animal* from the seizure calendar."""
    per = store.seizure_days_per_animal().get(animal, {}) or {}
    out: dict = {}
    for daystr, cnt in per.items():
        try:
            out[date.fromisoformat(daystr)] = int(cnt)
        except (ValueError, TypeError):
            continue
    return out


def _coverage_epochs(animal: str, evoked_dir: str) -> np.ndarray:
    """Sorted absolute epoch of EVERY stimulus for *animal*, unioned across
    channels (one cheap metric read; only the timestamps are used)."""
    from src.evoked_figures.data import animal_series
    series, _ = animal_series(animal, evoked_dir, [_cfg.CHEAP_METRICS[0]])
    if not series:
        return np.array([])
    allsecs = np.concatenate([d["secs"] for d in series.values()])
    allsecs = allsecs[np.isfinite(allsecs)]
    return np.sort(allsecs)


def draw_null_onsets(store, animal: str, evoked_dir: str, *, seed: int,
                     n: int | None = None, buffer_sec: float | None = None,
                     window_sec: float = _cfg.SLIDING_BAND_HI_SEC,
                     preictal_max_sec: float = _cfg.PREICTAL_MAX_SEC,
                     band_lo_sec: float = _cfg.SLIDING_BAND_LO_SEC,
                     band_hi_sec: float = _cfg.SLIDING_BAND_HI_SEC,
                     min_coverage: int = _cfg.SLIDING_MIN_N,
                     min_spacing_sec: float | None = None,
                     grid_step_sec: float = 300.0,
                     _real=None, _day_counts=None, _coverage=None) -> NullDraw:
    """Draw a seeded set of matched deep-interictal fake onsets for *animal*. The
    ``_real`` / ``_day_counts`` / ``_coverage`` hooks let tests inject synthetic
    inputs without a store. Default ``buffer_sec = band_hi + 1 h`` and
    ``min_spacing_sec = window_sec`` (non-overlapping lookback bands)."""
    assert animal, "animal required"
    real = (np.asarray(_real, dtype=float) if _real is not None
            else np.array([s.onset_epoch for s in included_seizures(store, animal)],
                          dtype=float))
    N = int(n) if n is not None else int(real.size)
    if buffer_sec is None:
        buffer_sec = band_hi_sec + 3600.0
    if min_spacing_sec is None:
        min_spacing_sec = window_sec
    if N <= 0 or real.size == 0:
        return NullDraw(np.array([]), N, 0, "no scored seizures", seed)
    day_counts = (_day_counts if _day_counts is not None
                  else _seizure_day_counts(store, animal))
    if not day_counts:
        return NullDraw(np.array([]), N, 0, "no seizure-days", seed)
    cov = (np.asarray(_coverage, dtype=float) if _coverage is not None
           else _coverage_epochs(animal, evoked_dir))
    if cov.size == 0:
        return NullDraw(np.array([]), N, 0, "no stimulus coverage", seed)

    onsets = _place_null_onsets(
        real, day_counts, cov, N, rng=np.random.default_rng(seed),
        buffer_sec=buffer_sec, window_sec=window_sec,
        preictal_max_sec=preictal_max_sec, band_lo_sec=band_lo_sec,
        band_hi_sec=band_hi_sec, min_coverage=min_coverage,
        min_spacing_sec=min_spacing_sec, grid_step_sec=grid_step_sec)
    reason = (None if onsets.size >= N else
              f"only {onsets.size}/{N} deep-interictal anchors with stim coverage")
    return NullDraw(onsets, N, int(onsets.size), reason, seed)


def draw_null_set(store, animal: str, evoked_dir: str, *, seeds, **kw
                  ) -> list[NullDraw]:
    """K independent matched draws, one per seed -- the null distribution."""
    return [draw_null_onsets(store, animal, evoked_dir, seed=int(s), **kw)
            for s in seeds]
