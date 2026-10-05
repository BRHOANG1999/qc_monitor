"""Config for the paired-pulse PPR analysis. Absolute paths (resolved from __file__)
so a notebook/cron run from any cwd still finds the cache."""

from __future__ import annotations

import datetime as _dt
import os as _os

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))

ANIMAL = "BCH111"
CHANNEL = "BCH111SR"

# --- paired-pulse geometry ---------------------------------------------------
ISI_MS = 50.0                       # inter-pulse interval (the second pulse offset)
WIN = (3.0, 45.0)                   # per-pulse analysis window (ms from its onset).
#   EXCLUDES the stim artifact at BOTH edges (measured on real BCH111 data): the onset
#   artifact rings through ~0.5-2 ms (a -0.25 then +0.09 transient) and only settles into
#   the real evoked descent by ~3 ms; the NEXT pulse's onset artifact bleeds back as a
#   +0.1 spike at ~49 ms. So start at 3 ms (clear of the onset ring) and end at 45 ms
#   (>=4 ms before the next-pulse spike). S1/S2 windows stay equal-length (42 ms) and
#   non-overlapping within the 50 ms ISI.
SECOND_PULSE_TOL_MS = 5.0           # accept a stim-copy deflection at ISI +/- tol
SECOND_PULSE_FRAC = 0.2             # a partner deflection must be >= this * anchor amp

# Pre-ictal binning excludes epochs AT/AFTER an onset: an epoch is pre-ictal only if
# the next onset is ahead (tto > 0) AND it is > this buffer since the PREVIOUS onset
# (so post-ictal recovery from a prior seizure doesn't leak into pre-ictal bins).
POSTICTAL_BUFFER_SEC = 3600.0       # 1 h post-ictal exclusion

# 500 Hz LOW-PASS ONLY (no high-pass): a high-pass corner rings off the sharp stim
# artifact and droops the baseline across the window, bleeding the artifact into the
# evoked response. A plain low-pass avoids that. Each window is then baseline-subtracted
# (demeaned) so the low-pass's retained DC does not distort DC-sensitive features (rms);
# the primary p2p / line-length / slope features are DC-invariant anyway.
BANDPASS = True                     # (apply a filter at all)
FILTER_MODE = "lowpass"            # "lowpass" (500 Hz only) | "bandpass" (BP_LOW..HIGH)
BASELINE_SUBTRACT = True           # demean each window after filtering
BP_LOW_HZ = 1.0                     # only used when FILTER_MODE == "bandpass"
BP_HIGH_HZ = 500.0                  # low-pass cutoff
FILTER_LABEL = "≤500 Hz low-pass"  # for figure footnotes

# Features measured on each pulse; PPR = S2/S1 per feature. Only LINEAR, positive-
# magnitude features (a ratio of logs is meaningless, so log_auc is excluded).
# peak_to_trough is the headline (matches KMRecorder's live p2p PPR).
FEATURES = ["peak_to_trough", "rms_amplitude", "line_length", "max_slope"]
PRIMARY = "peak_to_trough"

# Scope the file scan: paired-pulse began on/after this date (auto-detection still
# skips any non-paired epoch, so a loose bound is safe). Override with --since.
PP_START_DATE = _dt.datetime(2026, 10, 1)

# --- paths -------------------------------------------------------------------
# New-methodology output (500 Hz low-pass only, 3-45 ms window, NO pHFO) lives in its own
# folder + cache so the earlier 1-500 Hz / pHFO figures in data/BCH111_paired_pulse/ stay
# intact for comparison.
OUT_DIR = _os.path.join(_ROOT, "data", "BCH111_paired_pulse_lp500")
CACHE_DIR = _os.path.join(_ROOT, "data", "derivatives", "paired_pulse")
PP_CACHE = _os.path.join(CACHE_DIR, "BCH111_paired_pulse_lp500_matrix.pkl")

# --- dark theme (matches the project dashboards) -----------------------------
BG = "#15151f"
PANEL = "#1e1e2f"
TEXT = "#f0f0f5"
MUTED = "#9aa0b4"
ACCENT = "#5e7ce2"
S1_COLOR = "#2663c4"                # pulse 1 = blue   (KMRecorder P1 [0.15 0.35 0.75])
S2_COLOR = "#e6800f"               # pulse 2 = orange (KMRecorder P2 [0.90 0.50 0.10])
SEIZURE_COLOR = "#ffffff"
NODATA_COLOR = "#3a3a46"
FACIL_COLOR = "#2ee6a6"            # PPR > 1 (facilitation)
DEPR_COLOR = "#d62f2f"             # PPR < 1 (depression)
