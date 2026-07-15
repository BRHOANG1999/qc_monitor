"""Passive (pre-stim) and evoked (post-stim) feature windows + their sidecars.

The evoked ``*_evoked.mat`` stores a window around each stimulus (t=0 = stim,
+/- 200-500 ms). The SAME 22 cheap features become PASSIVE pre-stim features by
cropping to ``[-pre_ms, -guard]`` and evoked features by cropping to
``[+guard, +post_ms]`` -- no new feature math. ``guard`` is the stim-artifact
half-width and WIDENS with smoothing/filtering, because filtering is applied to
the whole trace BEFORE the crop, so a boxcar of width S smears the artifact by
~S/2 ms each side (verified). Three post-stim-band features
(``early_area``/``late_area``/``early_late_ratio``) are meaningless on a
pre-stim window (they hard-code [0,50]/[50,200] ms) and are excluded from the
passive metric set (see ``config.PASSIVE_INVALID``).

Passive features live in their OWN sidecar variant (``.features.<animal>.passive.json``)
stamped with a config signature so a window/guard change rebuilds it.
"""

from __future__ import annotations

import json

from src.periictal import config as _cfg
from src.utils import evoked_output as eo
from src.utils.evoked_features import FeatureConfig

_MAX_FILES = 2_000_000        # NASA Rule 2.


def guard_ms(artifact_half_ms: float = _cfg.DEFAULT_ARTIFACT_HALF_MS,
             smoothing: bool = False, smooth_ms: float = 5.0) -> float:
    """Half-width (ms) to exclude on each side of the stim: the intrinsic
    artifact half-width plus half the smoothing window (a zero-phase boxcar of
    width S spreads a point by ~S/2 ms each side)."""
    assert artifact_half_ms >= 0, "artifact_half_ms must be >= 0"
    return artifact_half_ms + (smooth_ms / 2.0 if smoothing else 0.0)


def _validate_window(start_ms: float, end_ms: float) -> None:
    """preprocess() SILENTLY returns the full uncropped trace when a window
    keeps <2 samples (reversed / out-of-range), which would pull the stim
    artifact into a 'passive' feature. Refuse such windows loudly."""
    assert start_ms < end_ms, f"window start {start_ms} must be < end {end_ms}"
    assert abs(start_ms) <= 1000.0 and abs(end_ms) <= 1000.0, \
        "window bounds must be within the +/-500 ms trace (<=1000 ms)"


def passive_config(pre_ms: float = 200.0,
                   artifact_half_ms: float = _cfg.DEFAULT_ARTIFACT_HALF_MS,
                   smoothing: bool = False, smooth_ms: float = 5.0,
                   bandpass: bool = False, bp_low_hz: float = 1.0,
                   bp_high_hz: float = 100.0) -> FeatureConfig:
    """Pre-stim window ``[-pre_ms, -guard]`` with the given filtering."""
    g = guard_ms(artifact_half_ms, smoothing, smooth_ms)
    _validate_window(-pre_ms, -g)
    return FeatureConfig(window_start_ms=-pre_ms, window_end_ms=-g,
                         smoothing=smoothing, smooth_ms=smooth_ms,
                         bandpass=bandpass, bp_low_hz=bp_low_hz,
                         bp_high_hz=bp_high_hz)


def evoked_config(post_ms: float = 200.0,
                  artifact_half_ms: float = _cfg.DEFAULT_ARTIFACT_HALF_MS,
                  smoothing: bool = False, smooth_ms: float = 5.0,
                  bandpass: bool = False, bp_low_hz: float = 1.0,
                  bp_high_hz: float = 100.0) -> FeatureConfig:
    """Post-stim window ``[+guard, +post_ms]`` with the given filtering."""
    g = guard_ms(artifact_half_ms, smoothing, smooth_ms)
    _validate_window(g, post_ms)
    return FeatureConfig(window_start_ms=g, window_end_ms=post_ms,
                         smoothing=smoothing, smooth_ms=smooth_ms,
                         bandpass=bandpass, bp_low_hz=bp_low_hz,
                         bp_high_hz=bp_high_hz)


def config_sig(cfg: FeatureConfig) -> str:
    """Stable signature of the windowing/filter fields, so a passive sidecar is
    rebuilt when the window or guard changes."""
    d = {k: getattr(cfg, k) for k in FeatureConfig.__dataclass_fields__}
    return json.dumps(d, sort_keys=True)


def build_passive_sidecar(mat_path: str, animal: str,
                          cfg: FeatureConfig) -> int:
    """Compute *animal*'s pre-stim features for one .mat and write the passive
    variant sidecar. Returns rows written (0 if none / unreadable)."""
    assert mat_path and animal, "mat_path and animal required"
    rows = eo.compute_feature_rows(mat_path, animal, cfg, expensive=False)
    if not rows:
        return 0
    eo.write_feature_sidecar(mat_path, animal, rows, variant="passive",
                             config_sig=config_sig(cfg))
    return len(rows)


def warm_passive(animal: str, evoked_dir: str, cfg: FeatureConfig,
                 progress=None, protocol: str | None = None) -> tuple[int, int]:
    """Build every fresh passive sidecar for *animal* under *evoked_dir* (skips
    ones already fresh for this config). *protocol* restricts to recordings
    whose session token contains that substring (the passive feature read
    re-parses the traces, ~15 s/file, so scoping to one protocol matters).
    Returns (built, total). *progress* is an optional callback(done, total, path)."""
    assert animal and evoked_dir, "animal and evoked_dir required"
    sig = config_sig(cfg)
    files = [f for f in eo.list_evoked_files(evoked_dir)
             if animal in eo.animals_in_filename(f)
             and (not protocol or protocol in eo.parse_session(f))]
    built = 0
    for i, fp in enumerate(files):
        assert i < _MAX_FILES, "passive warm runaway"
        if eo.read_feature_sidecar(fp, animal, "passive", sig) is None:
            built += build_passive_sidecar(fp, animal, cfg) and 1 or 0
        if progress:
            progress(i + 1, len(files), fp)
    return built, len(files)


def iter_passive_sidecars(animal: str, evoked_dir: str, cfg: FeatureConfig):
    """Yield (mat_path, sidecar_path, rows) for each FRESH passive sidecar of
    *animal* under the given config -- the readiness gate for the passive
    matrix. Mirrors evoked_figures.data.iter_animal_sidecars."""
    assert animal, "animal required"
    sig = config_sig(cfg)
    for i, fp in enumerate(eo.list_evoked_files(evoked_dir)):
        assert i < _MAX_FILES, "passive sidecar scan runaway"
        if animal not in eo.animals_in_filename(fp):
            continue
        rows = eo.read_feature_sidecar(fp, animal, "passive", sig)
        if rows:
            yield fp, eo.feature_sidecar_path(fp, animal, "passive"), rows
