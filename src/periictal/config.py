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

# --- Chang et al. 2026 preictal / interictal class windows ----------------
# Seconds before the NEXT seizure onset (time_to_onset_sec is signed positive
# for pre-onset rows). Preictal = the 30 min before a seizure; interictal =
# 60-90 min before (the user's bounded window; the paper used ">60 min"). The
# 30-60 min band is the redacted buffer. A seizure's preictal rows are dropped
# when its inter-seizure interval is shorter than MIN_PREICTAL_SEC (clustered
# seizures have no clean 30-min preictal period). The interictal side is
# implicitly ISI-guarded by src.preictal.isi.lookback_ceilings.
PREICTAL_MAX_SEC = 1800.0
INTERICTAL_LO_SEC = 3600.0
INTERICTAL_HI_SEC = 5400.0
MIN_PREICTAL_SEC = 1800.0
DEFAULT_N_PHASES = 10            # epileptogenesis phases for the forecaster

# --- Sliding-window ROC-AUC test ------------------------------------------
# The preictal (positive) window stays fixed at (0, PREICTAL_MAX_SEC]. The
# interictal reference is no longer a single fixed window: N evenly-spaced
# WIDTH-wide windows are sampled from a fixed lookback BAND before onset, and
# any window whose far edge falls within POSTICTAL_GUARD of the PREVIOUS seizure
# is dropped (o <= isi - WIDTH - GUARD). Geometry tunable here.
SLIDING_N_WINDOWS = 12
SLIDING_BAND_LO_SEC = 3600.0       # windows start no closer than 1 h to onset
SLIDING_BAND_HI_SEC = 21600.0      # ... and no further than 6 h before onset
SLIDING_WIDTH_SEC = 1800.0         # 30-min windows, matching the preictal width
SLIDING_POSTICTAL_GUARD_SEC = 3600.0   # keep windows >= 1 h after the prev seizure
SLIDING_MIN_N = 20                 # min finite stimuli per class to score a window

# --- Matrix-build speedup: near-seizure file prefilter ---------------------
# A cold matrix build reads EVERY one of an animal's evoked sidecars off the
# share, but only files whose recording could contain a stimulus within
# window_sec of a seizure onset can contribute a kept row (see matrix.build_
# matrix: pre rows within a per-seizure ceiling <= window_sec, post rows within
# window_sec). Filtering by filename time BEFORE reading is a PROVABLE superset
# of contributing files, so the matrix is unchanged -- it just skips files whose
# every stimulus is dropped anyway (BCH111: ~127 of 351). SLACK guards the file
# that STARTS before the window but CONTAINS its start (a recording is a short
# chunk; matches erpimage._candidate_files' validated 2 h guard).
PERIICTAL_PREFILTER_NEAR_SEIZURE = True
PERIICTAL_PREFILTER_SLACK_SEC = 7200.0

# The paper's five best perturbed features (Fig 3 / Table 1), in our column
# names: SumPower[1-64], SumPower[256-1024], exp-fit initial factor a,
# 1st-moment[64-256], 1st-moment[1-64].
PAPER_BEST5 = ["sum_power_low", "sum_power_high", "expfit_initial",
               "freq_moment_high", "freq_moment_low"]
