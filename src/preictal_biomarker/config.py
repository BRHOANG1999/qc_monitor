"""Central, documented configuration for the pre-ictal biomarker rebuild.

Every analysis choice that a reviewer would ask about lives here (window, filter,
detrend target, PC count, k, lead gap, date scope), so the pipeline is one
glance from reproducible and nothing is buried in a scratch script.
"""

from __future__ import annotations

import datetime as _dt
import os as _os

# Repo root (…/qc_monitor), so data paths resolve the same from any cwd — the
# CLI, a notebook run from notebooks/, or an import elsewhere.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

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

# ---- Expanded per-waveform metric set (all cheap -> already in the matrix) ----
# Grouped so the state space is not energy-dominated. All computed on the 2-50 ms
# evoked (feature branch) or residual (residual branch) waveform.
AMPL_ENERGY = ["line_length", "rms_amplitude", "variance", "peak_to_trough",
               "max_slope", "log_auc", "early_area", "late_area"]
SHAPE_FEATURES = ["curvature", "skewness", "early_late_ratio", "autocorrelation"]
LATENCY_FEATURES = ["peak_latency_ms", "trough_latency_ms", "tp_latency_ms",
                    "max_slope_time_ms"]
SPECTRAL_FEATURES = ["sum_power_low", "sum_power_mid", "sum_power_high",
                     "freq_moment_low", "freq_moment_high", "freq_moment_vhigh",
                     "wavelet_power_slow_gamma", "wavelet_power_gamma",
                     "wavelet_power_high_gamma"]
# Across-trial critical-slowing-down stats of the primary feature's time series.
CSD_FEATURES = ["csd_variance", "csd_ar1", "csd_skew", "csd_cv", "csd_redden"]
CSD_PRIMARY = "peak_to_trough"   # feature whose rolling stats = the CSD set
CSD_WIN = 20                     # trials per rolling CSD window
# pHFO occurrence (riding_event), joined per epoch. Added by features when present.
PHFO_FEATURES = ["phfo_frac", "phfo_prom", "phfo_snr"]

# The full feature set fed to PCA (pHFO appended when the join is available).
FEATURES = (AMPL_ENERGY + SHAPE_FEATURES + LATENCY_FEATURES + SPECTRAL_FEATURES
            + CSD_FEATURES)

# Impedance (Z_ss) scales response MAGNITUDE -> detrend the amplitude/energy/power
# features; leave latency / frequency-moment / ratio / shape features raw.
DETREND_FEATURES = (["line_length", "rms_amplitude", "variance", "peak_to_trough",
                     "max_slope", "log_auc", "early_area", "late_area"]
                    + ["sum_power_low", "sum_power_mid", "sum_power_high",
                       "wavelet_power_slow_gamma", "wavelet_power_gamma",
                       "wavelet_power_high_gamma"])

# amplitude-scaled features (gain-corrected in the matrix); kept for reference.
AMPL_FEATURES = ["peak_to_trough", "rms_amplitude", "line_length", "max_slope"]
# legacy 8-feature set (pre-expansion), kept so old caches/calls still resolve.
WAVEFORM_FEATURES = ["line_length", "rms_amplitude", "variance",
                     "autocorrelation", "peak_to_trough", "max_slope"]
FEATURES8 = WAVEFORM_FEATURES + ["csd_variance", "csd_ar1"]

# --- paths -------------------------------------------------------------------
OUT_DIR = _os.path.join(_ROOT, "data", "BCH111_preictal_biomarker")
CACHE_DIR = _os.path.join(_ROOT, "data", "derivatives", "preictal_biomarker")
FEATURE_CACHE = _os.path.join(CACHE_DIR, "BCH111_feature_matrix.pkl")

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
