"""Frozen data types for the pre-ictal engine.

Small, immutable value objects passed between the stages
(events -> trajectories -> CWT -> per-scale summaries) so each module has a
stable contract and can be unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Seizure:
    """One reviewer-confirmed behavioral seizure, with an ABSOLUTE onset time
    so inter-seizure intervals can be computed across chunk-file boundaries."""
    animal_id: str
    file_id: int
    file_path: str
    session_dir: str
    chunk_datetime: str
    eo_sec: float                 # electrographic onset, seconds INTO the chunk
    bb_sec: float | None          # back-to-baseline, seconds into the chunk
    racine: int | None
    seizure_type: str | None      # "LVF" | "HYP" | None
    onset_epoch: float            # absolute onset, seconds since the unix epoch


@dataclass(frozen=True)
class FeatureSpec:
    """A base feature: a pure fn mapping a 1-D signal window + fs -> one scalar.
    *params* lets one builder register several parameterized variants (e.g. the
    evoked windows). ``name`` is the registry key + the stored feature label."""
    name: str
    fn: Callable[..., float]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LeadTimeBins:
    """Log/dyadic lead-time bins for ONE seizure, bounded by its ISI-derived
    ceiling. ``edges_sec`` are seconds-before-onset, ascending; bin k spans
    ``[edges_sec[k], edges_sec[k+1]]``. Bins beyond the ceiling are undefined
    and simply absent (no padding). Empty edges => the seizure's preceding ISI
    was too short to define any pre-ictal bin."""
    edges_sec: tuple[float, ...]
    centers_sec: tuple[float, ...]
    ceiling_sec: float


@dataclass(frozen=True)
class ScaleResult:
    """Per-CWT-scale scalar summary -- the persisted deliverable row (Stage 1
    fills the coefficient stats; collapse/null/forecast arrive in stages 2-3)."""
    scale_index: int
    scale: float
    pseudo_freq_hz: float
    coeff_mean: float
    coeff_std: float
    coeff_max: float
    n_seizures: int


@dataclass
class RunResult:
    """The engine's output for one (feature, channel_role): the per-scale
    summaries + accounting for how many seizures + the ISI ceiling stats."""
    feature: str
    channel_role: str
    scales: list[ScaleResult] = field(default_factory=list)
    n_seizures: int = 0
    ceiling_stats: dict[str, float] = field(default_factory=dict)
