"""``qc_dark`` Plotly template -- registered as the global default at
import time so every chart in the dashboard picks up the same palette
without each callsite having to spell it out.

Import this module from ``src.dashboard.app`` once (top of the file)
before any tab module builds a figure. Importing has the side effect
of registering the template and setting it as default.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from src.dashboard.design import (
    COLOR_ACCENT, COLOR_DANGER, COLOR_DIVIDER, COLOR_SUCCESS,
    COLOR_SURFACE_1, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY,
    COLOR_WARNING, FONT_STACK,
)


# Trace color cycle -- accent first, then semantic colors, then a
# couple of complementary tones for >4-series charts.
_COLORWAY = [
    COLOR_ACCENT,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER,
    "#bf5af2",  # purple
    "#ff6482",  # pink
    "#5ac8fa",  # cyan
    "#ffd60a",  # yellow
]


QC_DARK = go.layout.Template(
    layout=dict(
        paper_bgcolor=COLOR_SURFACE_1,
        plot_bgcolor=COLOR_SURFACE_1,
        font=dict(family=FONT_STACK, color=COLOR_TEXT_PRIMARY, size=12),
        colorway=_COLORWAY,
        xaxis=dict(
            gridcolor=COLOR_DIVIDER, zerolinecolor=COLOR_DIVIDER,
            linecolor=COLOR_DIVIDER,
            tickfont=dict(color=COLOR_TEXT_SECONDARY, size=11),
            title=dict(font=dict(color=COLOR_TEXT_SECONDARY, size=12)),
        ),
        yaxis=dict(
            gridcolor=COLOR_DIVIDER, zerolinecolor=COLOR_DIVIDER,
            linecolor=COLOR_DIVIDER,
            tickfont=dict(color=COLOR_TEXT_SECONDARY, size=11),
            title=dict(font=dict(color=COLOR_TEXT_SECONDARY, size=12)),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_SECONDARY, size=11),
        ),
        margin=dict(l=50, r=20, t=40, b=40),
        hoverlabel=dict(
            bgcolor=COLOR_SURFACE_1,
            bordercolor=COLOR_DIVIDER,
            font=dict(family=FONT_STACK, color=COLOR_TEXT_PRIMARY),
        ),
    ),
)


pio.templates["qc_dark"] = QC_DARK
pio.templates.default = "qc_dark"
