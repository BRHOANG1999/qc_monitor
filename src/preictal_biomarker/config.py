"""Central, documented configuration for the pre-ictal biomarker rebuild.

Every analysis choice that a reviewer would ask about lives here (window, filter,
detrend target, PC count, k, lead gap, date scope), so the pipeline is one
glance from reproducible and nothing is buried in a scratch script.
"""

from __future__ import annotations

import datetime as _dt

ANIMAL = "BCH111"
CHANNEL = "BCH111SR"

# --- evoked feature window (non-overlapping with the stim artifact) ----------
WINDOW_MS = (2.0, 50.0)       # analysis window, post-stim
BANDPASS = True               # 1-500 Hz Butterworth before windowing ("LP500")
BP_LOW_HZ = 1.0
BP_HIGH_HZ = 500.0

# --- date scope (the gain-constant 300x window; seizures >= 2026-09-13) -------
ANALYSIS_START = _dt.datetime(2026, 9, 13)
ANALYSIS_END = _dt.datetime(2026, 9, 23)        # exclusive

# --- seizure handling --------------------------------------------------------
LEAD_GAP_H = 6.0              # cluster-leader gap: drop followers within 6 h

# --- peri-ictal / baseline windows (seconds) ---------------------------------
LOOKBACK_SEC = 21600.0        # 6 h peri-ictal lookback (matrix window_sec)
PREICTAL_SEC = 1800.0         # pre-ictal = 30 min before onset
BASELINE_MIN_SEC = 7200.0     # clean baseline = >= 2 h from any seizure

# --- state space -------------------------------------------------------------
N_PCS = 6                     # PCs retained (~95% var in the original figures)
K_STATES = 4                  # k-means clusters
SEED = 0

# The 8 features behind the figures. Six are per-waveform; two are across-trial
# critical-slowing-down (CSD) statistics of a primary feature's time series.
WAVEFORM_FEATURES = ["line_length", "rms_amplitude", "variance",
                     "autocorrelation", "peak_to_trough", "max_slope"]
CSD_FEATURES = ["csd_variance", "csd_ar1"]
FEATURES8 = WAVEFORM_FEATURES + CSD_FEATURES
CSD_PRIMARY = "peak_to_trough"   # feature whose rolling var / AR1 = the CSD pair
CSD_WIN = 20                     # trials per rolling CSD window

# amplitude-scaled features (gain-corrected in the matrix); the rest are shape.
AMPL_FEATURES = ["peak_to_trough", "rms_amplitude", "line_length", "max_slope"]

# --- paths -------------------------------------------------------------------
OUT_DIR = "data/BCH111_preictal_biomarker"
CACHE_DIR = "data/derivatives/preictal_biomarker"
FEATURE_CACHE = CACHE_DIR + "/BCH111_feature_matrix.pkl"

# state display palette (matches the existing figures: 0 blue,1 teal,2 gold,3 red)
STATE_COLORS = ["#5b6ee1", "#2ee6a6", "#f2c744", "#d62f2f"]
NODATA_COLOR = "#3a3a46"
SEIZURE_COLOR = "#ffffff"

# dark theme
BG = "#15151f"
PANEL = "#1e1e2f"
TEXT = "#f0f0f5"
MUTED = "#9aa0b4"
ACCENT = "#5e7ce2"
