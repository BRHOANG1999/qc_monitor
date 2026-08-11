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


def window_config(start_ms: float, end_ms: float,
                  artifact_half_ms: float = _cfg.DEFAULT_ARTIFACT_HALF_MS,
                  smoothing: bool = False, smooth_ms: float = 5.0,
                  bandpass: bool = False, bp_low_hz: float = 1.0,
                  bp_high_hz: float = 100.0) -> FeatureConfig:
    """FeatureConfig for an explicit ``[start, end]`` ms window. The bound
    nearest the stim (t=0) is pushed out by the artifact guard, so a post-stim
    (evoked) window never starts before +g and a pre-stim (passive) window never
    ends after -g -- the stim artifact is always excluded."""
    g = guard_ms(artifact_half_ms, smoothing, smooth_ms)
    s, e = float(start_ms), float(end_ms)
    if s >= 0:                          # post-stim window
        s = max(s, g)
    if e <= 0:                          # pre-stim window
        e = min(e, -g)
    _validate_window(s, e)
    return FeatureConfig(window_start_ms=s, window_end_ms=e,
                         smoothing=smoothing, smooth_ms=smooth_ms,
                         bandpass=bandpass, bp_low_hz=bp_low_hz,
                         bp_high_hz=bp_high_hz)


def config_sig(cfg: FeatureConfig) -> str:
    """Stable signature of the windowing/filter fields, so a passive sidecar is
    rebuilt when the window or guard changes."""
    d = {k: getattr(cfg, k) for k in FeatureConfig.__dataclass_fields__}
    return json.dumps(d, sort_keys=True)


def build_variant_sidecar(mat_path: str, animal: str, variant: str,
                          cfg: FeatureConfig) -> int:
    """Compute *animal*'s windowed features for one .mat and write the *variant*
    sidecar (config-signed). Returns rows written (0 if none / unreadable).
    NOTE: *variant* must NOT be the shared default 'evoked' -- use a distinct key
    (e.g. 'passive', 'evokedw') so this never overwrites the toolkit-default
    sidecar that chronic_evoked / evoked_figures read."""
    assert mat_path and animal and variant, "mat_path, animal, variant required"
    assert variant != "evoked", "'evoked' is the shared default sidecar"
    rows = eo.compute_feature_rows(mat_path, animal, cfg, expensive=False)
    if not rows:
        return 0
    eo.write_feature_sidecar(mat_path, animal, rows, variant=variant,
                             config_sig=config_sig(cfg))
    return len(rows)


def warm_variant(animal: str, evoked_dir: str, variant: str, cfg: FeatureConfig,
                 progress=None, protocol: str | None = None,
                 file_filter=None) -> tuple[int, int]:
    """Build every fresh *variant* sidecar for *animal* (skips ones already fresh
    for this config). *protocol* restricts to recordings whose session token
    contains that substring (the windowed read re-parses the traces, ~15 s/file,
    so scoping to one protocol matters). *file_filter* (``fp -> bool``) prunes the
    file list first -- the near-seizure prefilter, so a custom-window build only
    recomputes files that can contribute a row. Returns (built, total)."""
    assert animal and evoked_dir and variant, "animal, evoked_dir, variant required"
    sig = config_sig(cfg)
    files = [f for f in eo.list_evoked_files(evoked_dir)
             if animal in eo.animals_in_filename(f)
             and (not protocol or protocol in eo.parse_session(f))]
    if file_filter is not None:
        files = [f for f in files if file_filter(f)]
    import gc
    built = 0
    for i, fp in enumerate(files):
        assert i < _MAX_FILES, "variant warm runaway"
        if eo.read_feature_sidecar(fp, animal, variant, sig) is None:
            built += build_variant_sidecar(fp, animal, variant, cfg) and 1 or 0
            # Each file reads a ~600 MB .mat; force a collection so a long
            # custom-window warm can't let per-file garbage stack up in a
            # long-lived, memory-pressured server (the build thread's process
            # also holds the dashboard's caches). The compute itself is clean;
            # this is cheap insurance, not a fix for a leak.
            gc.collect()
        if progress:
            progress(i + 1, len(files), fp)
    return built, len(files)


def iter_variant_sidecars(animal: str, evoked_dir: str, variant: str,
                          cfg: FeatureConfig, file_filter=None):
    """Yield (mat_path, sidecar_path, rows) for each FRESH *variant* sidecar of
    *animal* under the given config -- the readiness gate for the windowed
    matrix. Mirrors evoked_figures.data.iter_animal_sidecars. *file_filter*
    (``fp -> bool``) prunes the file list first (near-seizure prefilter)."""
    assert animal and variant, "animal and variant required"
    sig = config_sig(cfg)
    for i, fp in enumerate(eo.list_evoked_files(evoked_dir)):
        assert i < _MAX_FILES, "variant sidecar scan runaway"
        if animal not in eo.animals_in_filename(fp):
            continue
        if file_filter is not None and not file_filter(fp):
            continue
        rows = eo.read_feature_sidecar(fp, animal, variant, sig)
        if rows:
            yield fp, eo.feature_sidecar_path(fp, animal, variant), rows


# Backwards-compatible passive wrappers (the passive variant is one instance).
def build_passive_sidecar(mat_path: str, animal: str, cfg: FeatureConfig) -> int:
    return build_variant_sidecar(mat_path, animal, "passive", cfg)


def warm_passive(animal: str, evoked_dir: str, cfg: FeatureConfig,
                 progress=None, protocol: str | None = None) -> tuple[int, int]:
    return warm_variant(animal, evoked_dir, "passive", cfg, progress, protocol)


def iter_passive_sidecars(animal: str, evoked_dir: str, cfg: FeatureConfig):
    return iter_variant_sidecars(animal, evoked_dir, "passive", cfg)
