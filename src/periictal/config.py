"""Shared constants for the peri-ictal evoked explorer."""

from __future__ import annotations

from src.utils import evoked_features as ef

# The usable feature set: the 22 cheap columns. The 6 expensive columns are
# all-null in every shipped sidecar today (compute_expensive=False), so a
# drop-NaN matrix over them would drop every row.
CHEAP_METRICS: list[str] = list(ef.CHEAP_COLUMNS)

# Features that are meaningless on a PRE-stim (passive) window: _band_area
# hard-codes the POST-stim ms bands [0,50] and [50,200], so on a negative
# window they silently integrate nothing and return a constant 0.
PASSIVE_INVALID: set[str] = {"early_area", "late_area", "early_late_ratio"}


def metrics_for_variant(variant: str) -> list[str]:
    """The metric columns to embed for *variant* ('evoked' | 'passive')."""
    if variant == "passive":
        return [m for m in CHEAP_METRICS if m not in PASSIVE_INVALID]
    return list(CHEAP_METRICS)


# Lead-up defaults. window_sec caps how far before an onset a stimulus counts as
# "pre-ictal"; post_ictal_buffer_sec keeps the window clear of the previous
# seizure's post-ictal tail (both feed src.preictal.isi.lookback_ceilings).
DEFAULT_WINDOW_SEC = 6 * 3600.0
DEFAULT_POST_ICTAL_BUFFER_SEC = 300.0
DEFAULT_MIN_LEADTIME_SEC = 2.0        # one stim ISI (0.5 Hz)

# Stim-artifact guard half-width (ms). The passive/evoked windows exclude
# +/- guard_ms around t=0; the guard widens with smoothing because filtering is
# applied to the whole trace BEFORE the window crop (see passive.guard_ms).
DEFAULT_ARTIFACT_HALF_MS = 1.0

# Default feature windows (ms) offered in the explorer's "Feature window" panel.
# Evoked = post-stim (excludes the artifact at t=0); passive = pre-stim.
DEFAULT_EVOKED_WINDOW_MS = (1.0, 200.0)
DEFAULT_PASSIVE_WINDOW_MS = (-200.0, -1.0)

# Above this many points an interactive embedding is stratified-subsampled
# (UMAP at ~50k is ~20s; PCA is cheap at any N but the browser Scattergl
# ceiling is ~100k).
INTERACTIVE_POINT_CAP = 50_000
