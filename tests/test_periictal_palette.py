"""Peri-ictal colour rules: cyclic map for phase (wrap-safe), perceptually-
uniform sequential for magnitude, colourblind-safe categorical capped at 8.

Run with: pytest tests/test_periictal_palette.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import palette as pal  # noqa: E402


def test_cyclic_scale_is_wrap_safe():
    scale = pal.cyclic_scale()
    assert len(scale) >= 2
    assert scale[0][0] == 0.0 and scale[-1][0] == 1.0
    # Endpoints must coincide so hour 0 and hour 24 render identically.
    assert scale[0][1] == scale[-1][1]


def test_hour_of_day_uses_cyclic_not_hsv():
    spec = pal.continuous_spec("hour_of_day", [0, 6, 12, 23])
    assert isinstance(spec["scale"], list)          # a cyclic colorscale, not "HSV"
    assert spec["cmin"] == 0.0 and spec["cmax"] == 24.0
    assert spec["ticks"][0] == [0, 6, 12, 18, 24]


def test_time_to_onset_is_reversed_sequential_in_hours():
    spec = pal.continuous_spec("time_to_onset_sec", [0.0, 3600.0, 7200.0])
    assert spec["reverse"] is True                  # near-onset = salient end
    assert list(spec["vals"]) == [0.0, 1.0, 2.0]    # seconds -> hours
    assert spec["scale"] == pal.SEQUENTIAL_SCALE


def test_is_categorical_seizure_switches_past_the_cap():
    assert pal.is_categorical("stim_key", 20) is True    # identity always
    assert pal.is_categorical("seizure_idx", pal.MAX_CATEGORIES) is True
    assert pal.is_categorical("seizure_idx", pal.MAX_CATEGORIES + 1) is False
    assert pal.is_categorical("time_to_onset_sec", 3) is False


def test_fold_categories_caps_at_eight_plus_other():
    labels = ([f"c{i}" for i in range(10) for _ in range(10 - i)])  # 10 distinct
    order, disp = pal.fold_categories(labels)
    assert len(order) == pal.MAX_CATEGORIES + 1
    assert order[-1].startswith("other (")
    # Every raw label maps to a drawn category; the 2 rarest fold into "other".
    assert len({disp[x] for x in labels}) == pal.MAX_CATEGORIES + 1
    assert pal.hue_for(order[-1], 8) == pal.OTHER_HUE   # folded bucket is muted
    assert pal.hue_for(order[0], 0) in pal.CATEGORICAL_HUES


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
