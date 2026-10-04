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
WIN = (1.0, 49.0)                   # per-pulse analysis window (ms from its onset):
#                                     1 ms start excludes the stim artifact; 49 ms end
#                                     fits inside the 50 ms ISI so S1/S2 windows are
#                                     equal-length and non-overlapping.
SECOND_PULSE_TOL_MS = 5.0           # accept a stim-copy deflection at ISI +/- tol
SECOND_PULSE_FRAC = 0.2             # a partner deflection must be >= this * anchor amp

# 1-500 Hz band, matching the single-pulse evoked feature pipeline.
BANDPASS = True
BP_LOW_HZ = 1.0
BP_HIGH_HZ = 500.0

# Features measured on each pulse; PPR = S2/S1 per feature. Only LINEAR, positive-
# magnitude features (a ratio of logs is meaningless, so log_auc is excluded).
# peak_to_trough is the headline (matches KMRecorder's live p2p PPR).
FEATURES = ["peak_to_trough", "rms_amplitude", "line_length", "max_slope"]
PRIMARY = "peak_to_trough"

# Scope the file scan: paired-pulse began on/after this date (auto-detection still
# skips any non-paired epoch, so a loose bound is safe). Override with --since.
PP_START_DATE = _dt.datetime(2026, 10, 1)

# --- paths -------------------------------------------------------------------
OUT_DIR = _os.path.join(_ROOT, "data", "BCH111_paired_pulse")
CACHE_DIR = _os.path.join(_ROOT, "data", "derivatives", "paired_pulse")
PP_CACHE = _os.path.join(CACHE_DIR, "BCH111_paired_pulse_matrix.pkl")

# --- dark theme (matches the project dashboards) -----------------------------
BG = "#15151f"
PANEL = "#1e1e2f"
TEXT = "#f0f0f5"
MUTED = "#9aa0b4"
ACCENT = "#5e7ce2"
S1_COLOR = "#2663c4"                # pulse 1 = blue   (KMRecorder P1 [0.15 0.35 0.75])
S2_COLOR = "#e6800f"               # pulse 2 = orange (KMRecorder P2 [0.90 0.50 0.10])
SEIZURE_COLOR = "#ffffff"
FACIL_COLOR = "#2ee6a6"            # PPR > 1 (facilitation)
DEPR_COLOR = "#d62f2f"             # PPR < 1 (depression)
