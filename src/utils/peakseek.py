"""1:1 Python port of ``peakseek.m`` from BHZ_DETECTOR.

The MATLAB original (``C:\\Users\\hoang392\\Downloads\\src\\
peakseek.m``) is the seizure-event detector ``BHZ_DETECTOR``
applies to the smoothed Hilbert envelope produced by
``tay_preprocess``. Pipeline:

  hilbert_trace = tay_preprocess(lfp, fs)       # 20-200 Hz env
  found_events  = peakseek(hilbert_trace,
                            minpeakdist=fs*30*5,  # 150 s apart
                            minpeakh=cutoff)      # e.g. 0.05

  -> integer sample indices, one per candidate seizure event.

Algorithm:
  1. Find every local maximum: ``x[i] >= x[i-1] && x[i] >=
     x[i+1]``. Endpoints excluded.
  2. Drop peaks shorter than ``minpeakh``.
  3. Iteratively prune neighboring peaks closer than
     ``minpeakdist``: for each adjacent pair too close, kill
     the smaller one. Loop until no pair is too close.

This is much faster than scipy ``find_peaks`` for big signals
and handles ties (MATLAB's stock ``findpeaks`` erases both
peaks in a tie -- BHZ deliberately avoids that).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("qc_monitor.utils.peakseek")


def peakseek(x: np.ndarray,
              minpeakdist: int = 1,
              minpeakh: float | None = None,
              ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(locs, pks)`` for *x*.

    Mirrors the MATLAB signature exactly. ``locs`` is a 0-indexed
    int array of peak positions; ``pks`` is the float array of
    peak heights at those positions.

    Empty signal -> empty arrays.
    """
    assert isinstance(x, np.ndarray), "x must be ndarray"
    assert x.ndim == 1, "x must be 1-D"
    assert isinstance(minpeakdist, int) and minpeakdist >= 1, (
        "minpeakdist must be int >= 1")
    n = x.shape[0]
    if n < 3:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=x.dtype))

    # Step 1: local maxima. MATLAB uses 1-indexed locs (+1 in
    # the source); we return 0-indexed which is the natural
    # Python convention. The +1 in MATLAB compensates for the
    # `find(x(2:end-1) >= ...)` shifting the indices by 1; in
    # Python that's the same shift since we slice [1:-1].
    middle = x[1:-1]
    locs = np.flatnonzero(
        (middle >= x[:-2]) & (middle >= x[2:])
    ) + 1

    # Step 2: height filter.
    if minpeakh is not None and len(locs):
        locs = locs[x[locs] > float(minpeakh)]

    # Step 3: iterative min-distance pruning. The MATLAB version
    # uses a while loop with a fixed-point break; we mirror that
    # with a NASA Rule 2 cap.
    if minpeakdist > 1 and len(locs) > 1:
        max_iter = len(locs) + 1  # each iter removes >=1 peak
        for it in range(max_iter):
            if len(locs) < 2:
                break
            diffs = np.diff(locs)
            too_close = diffs < minpeakdist
            if not too_close.any():
                break
            # For every (i, i+1) pair that's too close, drop the
            # smaller peak. The MATLAB version vectorizes this
            # by indexing into the heights at the two ends of
            # each "too close" gap and minimizing -- we follow
            # the same logic.
            pks = x[locs]
            left_h = pks[:-1][too_close]
            right_h = pks[1:][too_close]
            # Pair indices in `locs` of the smaller in each pair.
            del_left_mask = left_h < right_h
            gap_idxs = np.flatnonzero(too_close)
            to_drop = np.where(
                del_left_mask,
                gap_idxs,        # drop the LEFT peak of the pair
                gap_idxs + 1,    # drop the RIGHT peak
            )
            # Unique + sorted so np.delete works cleanly.
            to_drop = np.unique(to_drop)
            locs = np.delete(locs, to_drop)
        else:
            # Hit the iteration cap; shouldn't happen with
            # correct invariants but log if it does.
            logger.warning("peakseek hit max_iter prune cap; "
                            "n_peaks=%d remaining", len(locs))

    pks = x[locs] if len(locs) else np.empty(0, dtype=x.dtype)
    return (locs.astype(np.int64), pks)
