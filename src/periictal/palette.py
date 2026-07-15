"""Colour rules for the peri-ictal explorer, per scientific-visualization
standards (Crameri et al. 2020, *Nat. Commun.*; Kovesi 2015).

- Magnitude  -> a **perceptually-uniform sequential** map (Plasma / Viridis):
  equal data steps look equally different; no rainbow/jet.
- Circular phase (hour-of-day) -> a **cyclic perceptually-uniform** map
  (Twilight) whose endpoints match, so the 24h->0h wrap has no false seam (a
  linear map would draw a hard discontinuity where midnight meets midnight).
- Identity  -> a **fixed-order colourblind-safe categorical** palette, capped at
  8 (colour-vision deficiency affects ~8% of men); extras fold into "other (k)"
  so the drawn-colour count never exceeds what a viewer can tell apart.

Pure + Dash-free so the routing can be unit-tested.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import plotly.colors as _pcolors

# Colourblind-safe categorical hues, dark-surface stepped, FIXED ORDER --
# identity is assigned by position, never cycled (a 9th series folds to
# "other", it is never a newly-invented hue). Validated set (dataviz palette,
# dark column: worst adjacent CVD dE 8.4, >= 3:1 on the dark surface).
CATEGORICAL_HUES = ["#3987e5", "#008300", "#d55181", "#c98500",
                    "#199e70", "#d95926", "#9085e9", "#e66767"]
MAX_CATEGORIES = 8
OTHER_HUE = "#8a8d99"                # muted grey for the folded "other" bucket

SEQUENTIAL_SCALE = "Plasma"         # perceptually uniform (time-to-onset)
METRIC_SCALE = "Viridis"            # perceptually uniform (feature magnitude)

# Colour-by columns that are always identity (categorical), regardless of count.
_ALWAYS_CATEGORICAL = ("stim_key", "channel", "stim_status")


def cyclic_scale() -> list:
    """Perceptually-uniform CYCLIC colourscale (Twilight) as a plotly colorscale
    list. Endpoints coincide, so hour 0 and hour 24 render identically."""
    cols = _pcolors.cyclical.Twilight
    n = len(cols)
    assert n >= 2, "cyclic colourscale needs >= 2 stops"
    return [[i / (n - 1), c] for i, c in enumerate(cols)]


def is_categorical(color_by: str, n_unique: int) -> bool:
    """Whether *color_by* should be drawn as discrete identity colours. Seizure
    is categorical only while it fits the CVD-safe cap; past that it becomes a
    sequential 'onset order' gradient (19 discrete hues are unreadable)."""
    if color_by in _ALWAYS_CATEGORICAL:
        return True
    if color_by == "seizure_idx":
        return n_unique <= MAX_CATEGORIES
    return False


def fold_categories(labels) -> tuple[list, dict]:
    """Order categories by frequency, keep the top MAX_CATEGORIES, fold the rest
    into a single 'other (k)' bucket. Returns (ordered_display_categories,
    raw_label -> display_label). Keeps drawn colours <= MAX_CATEGORIES + 1."""
    counts = Counter(str(x) for x in labels)
    ordered = [c for c, _ in counts.most_common()]
    keep, folded = ordered[:MAX_CATEGORIES], ordered[MAX_CATEGORIES:]
    disp = {c: c for c in keep}
    display_order = list(keep)
    if folded:
        other = f"other ({len(folded)})"
        for c in folded:
            disp[c] = other
        display_order.append(other)
    return display_order, disp


def hue_for(display_category: str, index: int) -> str:
    """Fixed-order categorical hue for a display category; the folded 'other'
    bucket is drawn in the muted grey so it never impersonates a real series."""
    if display_category.startswith("other ("):
        return OTHER_HUE
    return CATEGORICAL_HUES[index % len(CATEGORICAL_HUES)]


def continuous_spec(color_by: str, values) -> dict:
    """Declarative continuous colour spec: {vals, scale, reverse, label, cmin,
    cmax, ticks}. hour_of_day is CYCLIC (0..24, wrap-safe); time_to_onset is a
    reversed PU sequential (near-onset = the salient bright end); a metric or a
    large seizure set is a PU sequential."""
    v = np.asarray(values, dtype=float)
    if color_by == "time_to_onset_sec":
        # LOG scale: pre-ictal dynamics of interest span seconds..hours, and a
        # linear 0-6h ramp crushes all the near-onset detail into one colour.
        # log10(seconds) spreads 1s / 10s / 1min / 10min / 1h across the map.
        sec = np.maximum(v, 1.0)                         # floor at 1 s (no log 0)
        hi = float(sec.max()) if sec.size else 1.0
        marks = [(1, "1s"), (10, "10s"), (60, "1m"), (600, "10m"),
                 (3600, "1h"), (21600, "6h"), (86400, "1d"), (604800, "1w")]
        tv = [np.log10(s) for s, _ in marks if s <= hi * 1.2]
        tt = [lbl for s, lbl in marks if s <= hi * 1.2]
        return {"vals": np.log10(sec), "scale": SEQUENTIAL_SCALE, "reverse": True,
                "label": "time to onset (log)", "cmin": None, "cmax": None,
                "ticks": (tv, tt) if tv else None}
    if color_by == "hour_of_day":
        return {"vals": v, "scale": cyclic_scale(), "reverse": False,
                "label": "hour of day", "cmin": 0.0, "cmax": 24.0,
                "ticks": ([0, 6, 12, 18, 24], ["0", "6", "12", "18", "24"])}
    if color_by == "seizure_idx":
        return {"vals": v, "scale": METRIC_SCALE, "reverse": False,
                "label": "seizure (onset order)", "cmin": None, "cmax": None,
                "ticks": None}
    return {"vals": v, "scale": METRIC_SCALE, "reverse": False,
            "label": color_by, "cmin": None, "cmax": None, "ticks": None}
