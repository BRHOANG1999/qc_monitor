"""Reusable UI primitives.

Tab modules should compose from these instead of building bespoke
inline-styled divs each time. Anything stylistic that has appeared more
than twice across the dashboard probably belongs here.

Imports only from ``src.dashboard.design`` -- no circular import with
``src.dashboard.app``. Safe to call at module import time.
"""

from __future__ import annotations

from typing import Any

from dash import dash_table, html

from src.dashboard.design import (
    COLOR_ACCENT, COLOR_DANGER, COLOR_DIVIDER, COLOR_SUCCESS,
    COLOR_SURFACE_1, COLOR_SURFACE_2, COLOR_SURFACE_3,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_TERTIARY,
    COLOR_WARNING, FONT_SIZE_BODY, FONT_SIZE_CAPTION, FONT_SIZE_HEADER,
    FONT_STACK, RADIUS_MD, RADIUS_SM,
    SPACE_1, SPACE_2, SPACE_3, SPACE_4, SPACE_5,
)
from src.dashboard.icons import icon

# --------------------------------------------------------------------- #
#  Card -- the universal "panel" wrapper used everywhere on the home
#  page, in tab headers, etc. Replaces every ad-hoc html.Div(style=
#  {"backgroundColor": "#1e1e2f", "borderRadius": "8px", ...}).
# --------------------------------------------------------------------- #

_CARD_BASE: dict = {
    "backgroundColor": COLOR_SURFACE_1,
    "borderRadius": RADIUS_MD,
    "border": f"1px solid {COLOR_DIVIDER}",
}


def card(*children: Any, padded: bool = True, id_: str | None = None,
          style: dict | None = None) -> html.Div:
    """Standard panel. *padded* controls inner padding (default 12px);
    *style* merges on top of the base."""
    merged: dict = {**_CARD_BASE}
    if padded:
        merged["padding"] = f"{SPACE_4} {SPACE_5}"
    if style:
        merged.update(style)
    kwargs: dict = {"style": merged, "children": list(children)}
    if id_ is not None:
        kwargs["id"] = id_
    return html.Div(**kwargs)


# --------------------------------------------------------------------- #
#  Section header -- title row inside a card, optional "Open >" link.
# --------------------------------------------------------------------- #

def section_header(title: str, on_open_id: str | None = None,
                    on_open_label: str = "Open >",
                    extras: list | None = None) -> html.Div:
    """A title strip. When *on_open_id* is given the right side gets a
    clickable Button with that id (a callback elsewhere consumes the
    n_clicks). *extras* slots additional children (e.g. a refresh button)
    between the title and the open link."""
    parts: list = [
        html.Div(title, style={
            "fontSize": FONT_SIZE_CAPTION, "color": COLOR_TEXT_SECONDARY,
            "textTransform": "uppercase", "letterSpacing": "0.6px",
            "fontWeight": "600",
        }),
        html.Div(extras or [], style={"flex": "1"}),
    ]
    if on_open_id:
        # Strip the trailing " >" from the legacy default label; the
        # icon does that job now.
        clean_label = on_open_label.rstrip(" >")
        parts.append(html.Button([
            clean_label,
            icon("arrow-right", size=12, color=COLOR_ACCENT,
                 style={"marginLeft": SPACE_1}),
        ], id=on_open_id, n_clicks=0, style={
            "backgroundColor": "transparent",
            "color": COLOR_ACCENT,
            "border": "none", "cursor": "pointer",
            "fontSize": FONT_SIZE_CAPTION, "fontWeight": "600",
            "letterSpacing": "0.3px", "padding": "0",
            "display": "inline-flex", "alignItems": "center",
            "gap": SPACE_1,
        }))
    return html.Div(parts, style={
        "display": "flex", "alignItems": "center",
        "gap": SPACE_3,
        "marginBottom": SPACE_2,
    })


# --------------------------------------------------------------------- #
#  Status pill -- one colored chip per canonical status string.
# --------------------------------------------------------------------- #

_STATUS_COLOR: dict = {
    "ok": COLOR_SUCCESS,
    "done": COLOR_SUCCESS,
    "success": COLOR_SUCCESS,
    "pending": COLOR_WARNING,
    "warning": COLOR_WARNING,
    "overdue": COLOR_DANGER,
    "danger": COLOR_DANGER,
    "never": COLOR_TEXT_TERTIARY,
    "not_scheduled": COLOR_ACCENT,
    "not_active": COLOR_TEXT_TERTIARY,
    "info": COLOR_ACCENT,
    "neutral": COLOR_TEXT_TERTIARY,
}
_STATUS_LABEL: dict = {
    "ok": "OK", "done": "DONE", "success": "OK",
    "pending": "PENDING", "warning": "WARNING",
    "overdue": "OVERDUE", "danger": "ERROR",
    "never": "NEVER",
    "not_scheduled": "NOT SCHEDULED",
    "not_active": "NOT ACTIVE",
    "info": "INFO", "neutral": "—",
}


def pill(status: str, label: str | None = None) -> html.Span:
    """Colored chip. *status* picks the color (see _STATUS_COLOR keys);
    *label* overrides the default text."""
    color = _STATUS_COLOR.get(status, COLOR_TEXT_TERTIARY)
    text = label if label is not None else _STATUS_LABEL.get(
        status, status.upper())
    classes = ["status-pill"]
    if status in ("overdue", "danger"):
        classes.append("pulse-bad")
    return html.Span(text, className=" ".join(classes), style={
        "backgroundColor": color, "color": "white",
        "padding": f"2px {SPACE_3}", "borderRadius": "12px",
        "fontSize": FONT_SIZE_CAPTION, "fontWeight": "600",
        "letterSpacing": "0.5px",
        "whiteSpace": "nowrap",
    })


def status_dot(status: str, label: str | None = None) -> html.Span:
    """Small filled circle, optionally followed by a label."""
    color = _STATUS_COLOR.get(status, COLOR_TEXT_TERTIARY)
    children: list = [html.Span("", className="status-dot-mark", style={
        "display": "inline-block", "width": "8px", "height": "8px",
        "borderRadius": "50%", "backgroundColor": color,
        "marginRight": SPACE_2 if label else "0",
    })]
    if label:
        children.append(html.Span(label, style={
            "fontSize": FONT_SIZE_CAPTION, "color": COLOR_TEXT_SECONDARY,
        }))
    return html.Span(children, style={
        "display": "inline-flex", "alignItems": "center",
    })


# --------------------------------------------------------------------- #
#  KPI card -- big number + caption. Used for the data-log diff counts.
# --------------------------------------------------------------------- #

def kpi(label: str, value: Any, color: str | None = None) -> html.Div:
    return html.Div([
        html.Div(label, style={
            "fontSize": FONT_SIZE_CAPTION, "color": COLOR_TEXT_SECONDARY,
            "textTransform": "uppercase", "letterSpacing": "0.5px",
            "marginBottom": SPACE_1,
        }),
        html.Div(str(value), style={
            "fontSize": "26px", "fontWeight": "700",
            "color": color or COLOR_TEXT_PRIMARY,
        }),
    ], style={
        "backgroundColor": COLOR_SURFACE_1,
        "padding": f"{SPACE_4} {SPACE_5}",
        "borderRadius": RADIUS_MD,
        "border": f"1px solid {COLOR_DIVIDER}",
        "flex": "0 0 200px",
    })


# --------------------------------------------------------------------- #
#  Refresh button -- canonical small secondary button.
# --------------------------------------------------------------------- #

def refresh_button(id_: str, label: str = "Refresh") -> html.Button:
    return html.Button([
        icon("refresh", size=13, color=COLOR_TEXT_PRIMARY,
             style={"marginRight": SPACE_2}),
        label,
    ], id=id_, n_clicks=0, style={
        "backgroundColor": COLOR_SURFACE_3, "color": COLOR_TEXT_PRIMARY,
        "border": f"1px solid {COLOR_DIVIDER}",
        "padding": f"{SPACE_2} {SPACE_4}",
        "borderRadius": RADIUS_SM,
        "cursor": "pointer", "fontSize": FONT_SIZE_BODY,
        "display": "inline-flex", "alignItems": "center",
    })


def refresh_bar(refresh_btn_id: str) -> html.Div:
    """Right-aligned bar with a Refresh button. Used at the top of
    every Sheets-backed tab so layout is consistent."""
    return html.Div(
        [refresh_button(refresh_btn_id)],
        style={"display": "flex", "justifyContent": "flex-end",
                "marginBottom": SPACE_3},
    )


# --------------------------------------------------------------------- #
#  Empty state -- replace italic-gray "Nothing to show" text everywhere.
# --------------------------------------------------------------------- #

def empty_state(label: str, hint: str | None = None,
                 icon_name: str = "inbox") -> html.Div:
    """Friendly empty placeholder with a faded icon + label.

    *icon_name* picks the SVG -- choose "inbox" / "clock" / "check-
    circle" / "file-text" based on what's empty.
    """
    children: list = [
        icon(icon_name, size=22, color=COLOR_TEXT_TERTIARY,
             style={"marginBottom": SPACE_2, "opacity": "0.7"}),
        html.Div(label, style={
            "fontSize": FONT_SIZE_BODY, "color": COLOR_TEXT_SECONDARY,
        }),
    ]
    if hint:
        children.append(html.Div(hint, style={
            "fontSize": FONT_SIZE_CAPTION, "color": COLOR_TEXT_TERTIARY,
            "marginTop": SPACE_1,
        }))
    return html.Div(children, style={
        "padding": f"{SPACE_5} 0",
        "textAlign": "center",
        "display": "flex", "flexDirection": "column",
        "alignItems": "center",
    })


# --------------------------------------------------------------------- #
#  Skeleton -- shimmer placeholder for cold-cache renders.
# --------------------------------------------------------------------- #

def skeleton(width: str = "100%", height: str = "1em") -> html.Div:
    return html.Div(className="skeleton", style={
        "width": width, "height": height,
        "marginBottom": SPACE_2,
    })


def skeleton_rows(n: int = 3) -> html.Div:
    return html.Div(
        [skeleton(width=("80%" if i % 2 else "100%")) for i in range(n)],
        style={"padding": f"{SPACE_3} 0"},
    )


# --------------------------------------------------------------------- #
#  TABLE_STYLE -- canonical DataTable styling, imported by every tab.
# --------------------------------------------------------------------- #

TABLE_STYLE: dict = {
    "style_table": {"overflowX": "auto"},
    "style_header": {
        "backgroundColor": COLOR_SURFACE_2,
        "color": COLOR_TEXT_PRIMARY,
        "fontWeight": "600",
        "border": f"1px solid {COLOR_DIVIDER}",
        "fontSize": FONT_SIZE_BODY,
    },
    "style_cell": {
        "backgroundColor": COLOR_SURFACE_1,
        "color": COLOR_TEXT_PRIMARY,
        "padding": f"{SPACE_2} {SPACE_3}",
        "border": f"1px solid {COLOR_DIVIDER}",
        "fontSize": FONT_SIZE_BODY,
        "textAlign": "left",
        "whiteSpace": "normal", "height": "auto",
    },
}


# --------------------------------------------------------------------- #
#  DARK_TABLE_STYLE -- read-only "log table" variant with uppercase
#  caption-sized headers, smaller padding, and a filter row. Used by
#  tabs/alerts, tabs/stim, and the in-app Activity Log + Annotations
#  tables.
# --------------------------------------------------------------------- #

DARK_TABLE_STYLE: dict = {
    "style_header": {
        "backgroundColor": COLOR_SURFACE_2,
        "color": COLOR_TEXT_PRIMARY,
        "fontWeight": "600",
        "border": "none",
        "borderBottom": f"1px solid {COLOR_DIVIDER}",
        "fontSize": FONT_SIZE_CAPTION,
        "textTransform": "uppercase",
        "letterSpacing": "0.5px",
    },
    "style_data": {
        "backgroundColor": COLOR_SURFACE_1,
        "color": COLOR_TEXT_SECONDARY,
        "border": "none",
        "borderBottom": f"1px solid {COLOR_DIVIDER}",
        "fontSize": FONT_SIZE_BODY,
    },
    "style_cell": {
        "textAlign": "left",
        "padding": f"{SPACE_3} {SPACE_4}",
        "fontSize": FONT_SIZE_BODY,
        "fontFamily": FONT_STACK,
    },
    "style_filter": {
        "backgroundColor": COLOR_SURFACE_2,
        "color": COLOR_TEXT_PRIMARY,
    },
}

# Zebra stripe -- very subtle, just a lighter surface. Pair with
# DARK_TABLE_STYLE via style_data_conditional=[ZEBRA_STRIPE, ...].
ZEBRA_STRIPE: dict = {
    "if": {"row_index": "odd"},
    "backgroundColor": COLOR_SURFACE_2,
}


# --------------------------------------------------------------------- #
#  Form-control styles -- every dropdown / labelled input across the
#  dashboard uses these. Lifted out of app.py so tab modules carved
#  out of it (criticality, session_compare, etc.) can share one
#  source of truth.
# --------------------------------------------------------------------- #

LABEL_STYLE: dict = {
    "color": COLOR_TEXT_TERTIARY,
    "fontSize": FONT_SIZE_CAPTION,
    "marginBottom": SPACE_2,
    "display": "block",
    "letterSpacing": "0.4px",
    "textTransform": "uppercase",
    "fontWeight": "600",
}
INPUT_STYLE: dict = {
    "backgroundColor": COLOR_SURFACE_3,
    "color": COLOR_TEXT_PRIMARY,
    "border": f"1px solid {COLOR_DIVIDER}",
    "borderRadius": RADIUS_SM,
    "padding": f"{SPACE_2} {SPACE_3}",
    "width": "100%",
    "fontSize": FONT_SIZE_BODY,
    "fontFamily": FONT_STACK,
    "transition": "border-color 0.15s ease, box-shadow 0.15s ease",
}
FIELD_STYLE: dict = {"flex": "1", "minWidth": "200px"}
DROPDOWN_STYLE: dict = {
    "backgroundColor": COLOR_SURFACE_3,
    "color": COLOR_TEXT_PRIMARY,
}


# --------------------------------------------------------------------- #
#  Row -- horizontal flex helper, used by status grids.
# --------------------------------------------------------------------- #

def row(*children: Any, gap: str | None = None,
         padding: str | None = None,
         align: str = "center") -> html.Div:
    style: dict = {"display": "flex", "alignItems": align}
    if gap:
        style["gap"] = gap
    if padding:
        style["padding"] = padding
    return html.Div(list(children), style=style)
