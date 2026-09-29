"""Resolve the evoked_shapes config: paths, analysis window, normalization mode,
QC threshold, the k range to explore, bootstrap counts, and the random seed.

All accessors take the already-loaded global config dict (from
``src.dashboard.data_helpers.load_config``) and fall back to safe defaults, so the
package runs even before an ``evoked_shapes:`` section is added to config/config.yaml.
The evoked directory and DB path are read from the SHARED sections
(``chronic_evoked`` / ``database``) so this package tracks the rest of the repo.
"""

from __future__ import annotations

import os

from src.utils.evoked_output import DEFAULT_EVOKED_DIR

_DEFAULT_OUT = os.path.join("data", "derivatives", "evoked_shapes")

# Analysis window (ms post-stim). Matches the shared evoked default; the 1 ms start
# excludes the stim artifact at t=0 by cropping (not blanking), as everywhere else.
_DEFAULT_WINDOW_MS = (1.0, 200.0)

# QC: drop a trial whose lag-1 autocorrelation (smoothness) is below this (a
# no-response noise trial). Shape-agnostic, so it does not delete distinct minority
# shapes the way a grand-mean correlation would. Logged, never silent.
_DEFAULT_QC_MIN_SMOOTHNESS = 0.3

# Cluster-leader gap: a seizure within this many hours of a prior one is a follower
# and dropped (its pre-onset baseline would overlap the prior seizure). Matches the
# 6 h rule used by the riding_event / seizure_lfp analyses.
_DEFAULT_LEAD_GAP_H = 6.0

# Post-ictal buffer feeding isi.lookback_ceilings (keeps a pre-ictal window clear of
# the previous seizure's tail).
_DEFAULT_POST_ICTAL_BUFFER_SEC = 300.0

# How far before an onset a stimulus still counts as pre-ictal (the lookback cap).
_DEFAULT_WINDOW_SEC = 6 * 3600.0


def _cfg(config: dict) -> dict:
    return (config or {}).get("evoked_shapes", {}) or {}


def evoked_dir(config: dict) -> str:
    ce = (config or {}).get("chronic_evoked", {}) or {}
    return ce.get("evoked_output_dir") or DEFAULT_EVOKED_DIR


def db_path(config: dict) -> str:
    db = (config or {}).get("database", {}) or {}
    return db.get("path") or os.path.join("data", "monitor.db")


def out_root(config: dict) -> str:
    return _cfg(config).get("out_root") or _DEFAULT_OUT


def window_ms(config: dict) -> tuple[float, float]:
    w = _cfg(config).get("window_ms")
    if isinstance(w, (list, tuple)) and len(w) == 2:
        return float(w[0]), float(w[1])
    return _DEFAULT_WINDOW_MS


def norm_mode(config: dict) -> str:
    """'l2' (default) or 'peak'. L2 unit-norm is the default because it weights the
    whole shape rather than the single largest deflection."""
    m = str(_cfg(config).get("norm", "l2")).lower()
    return m if m in ("l2", "peak") else "l2"


def qc_min_smoothness(config: dict) -> float:
    try:
        return float(_cfg(config).get("qc_min_smoothness", _DEFAULT_QC_MIN_SMOOTHNESS))
    except (TypeError, ValueError):
        return _DEFAULT_QC_MIN_SMOOTHNESS


def k_range(config: dict) -> tuple[int, int]:
    r = _cfg(config).get("k_range")
    if isinstance(r, (list, tuple)) and len(r) == 2:
        lo, hi = int(r[0]), int(r[1])
        if 2 <= lo <= hi:
            return lo, hi
    return 2, 10


def n_boot(config: dict) -> int:
    try:
        return max(10, int(_cfg(config).get("n_boot", 100)))
    except (TypeError, ValueError):
        return 100


def cap(config: dict) -> int:
    """Max trials to gather (reservoir-subsampled). Keeps the O(n^2) pairwise
    correlation and the bootstrap tractable."""
    try:
        return max(100, int(_cfg(config).get("cap", 8000)))
    except (TypeError, ValueError):
        return 8000


def seed(config: dict) -> int:
    try:
        return int(_cfg(config).get("seed", 0))
    except (TypeError, ValueError):
        return 0


def lead_gap_sec(config: dict) -> float:
    try:
        return max(0.0, float(_cfg(config).get("lead_gap_h", _DEFAULT_LEAD_GAP_H))) * 3600.0
    except (TypeError, ValueError):
        return _DEFAULT_LEAD_GAP_H * 3600.0


def post_ictal_buffer_sec(config: dict) -> float:
    try:
        return max(0.0, float(_cfg(config).get("post_ictal_buffer_sec",
                                               _DEFAULT_POST_ICTAL_BUFFER_SEC)))
    except (TypeError, ValueError):
        return _DEFAULT_POST_ICTAL_BUFFER_SEC


def lookback_window_sec(config: dict) -> float:
    try:
        return max(60.0, float(_cfg(config).get("lookback_window_sec",
                                                _DEFAULT_WINDOW_SEC)))
    except (TypeError, ValueError):
        return _DEFAULT_WINDOW_SEC


def enabled(config: dict) -> bool:
    return bool(_cfg(config).get("enabled", False))
