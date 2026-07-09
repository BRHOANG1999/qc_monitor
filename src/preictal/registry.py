"""Open-ended base-feature registry.

A base feature is a pure fn mapping a 1-D signal window + fs -> one scalar.
New features (bandpowers, criticality, evoked-window variants) register via
``@register_feature`` and are picked up by the engine automatically, so
answering "which feature/window carries the pre-ictal signal" is just running
the same CWT+scoring over each registered trajectory.
"""

from __future__ import annotations

from typing import Any, Callable

from src.preictal.types import FeatureSpec

_FEATURES: dict[str, FeatureSpec] = {}


def register_feature(name: str, **params: Any) -> Callable:
    """Decorator: register a feature fn ``(window, fs, **params) -> float``."""
    def deco(fn: Callable[..., float]) -> Callable[..., float]:
        assert name not in _FEATURES, f"duplicate feature {name!r}"
        _FEATURES[name] = FeatureSpec(name=name, fn=fn, params=dict(params))
        return fn
    return deco


def get_feature(name: str) -> FeatureSpec:
    if name not in _FEATURES:
        raise KeyError(f"unknown feature {name!r}; "
                       f"registered: {sorted(_FEATURES)}")
    return _FEATURES[name]


def feature_names() -> list[str]:
    return sorted(_FEATURES)


def resolve_features(patterns: list[str] | None) -> list[FeatureSpec]:
    """Resolve config patterns (``['*']`` or explicit names) to FeatureSpecs."""
    # Import so the decorators run and populate the registry.
    from src import preictal  # noqa: F401
    import src.preictal.features  # noqa: F401
    if not patterns or patterns == ["*"]:
        return [_FEATURES[n] for n in feature_names()]
    return [get_feature(n) for n in patterns]
