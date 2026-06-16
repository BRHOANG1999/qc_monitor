"""Inline SVG icons (Lucide-flavored).

Each icon is a string of `<path>`/`<circle>`/etc. fragments lifted
from Lucide. ``icon(name, size, color)`` wraps the fragments in a
24×24 SVG and returns it as an ``html.Img`` with a data: URL so Dash
will actually render the markup (a plain ``html.Span(children=svg)``
escapes the angle brackets).

Adding an icon: paste Lucide's path content into ``_PATHS``. Keep
stroke="currentColor" out of the fragments -- color is baked in here
because data: URLs don't inherit CSS color.
"""

from __future__ import annotations

from urllib.parse import quote

from dash import html

# Each entry is the inner-SVG markup (no outer <svg>). Default stroke
# style: stroke=#fff (overridden in icon()), fill=none, width=2,
# linecap=round, linejoin=round -- all Lucide defaults.
_PATHS: dict[str, str] = {
    "refresh": (
        '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>'
        '<path d="M21 3v5h-5"/>'
        '<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>'
        '<path d="M3 21v-5h5"/>'
    ),
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "alert-triangle": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16'
        'a2 2 0 0 0 1.73-3Z"/>'
        '<path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    "check-circle": (
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="m9 12 2 2 4-4"/>'
    ),
    "clock": (
        '<circle cx="12" cy="12" r="10"/>'
        '<polyline points="12 6 12 12 16 14"/>'
    ),
    "calendar": (
        '<rect width="18" height="18" x="3" y="4" rx="2"/>'
        '<path d="M16 2v4"/><path d="M8 2v4"/>'
        '<path d="M3 10h18"/>'
    ),
    "external": (
        '<path d="M15 3h6v6"/>'
        '<path d="M10 14 21 3"/>'
        '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 '
        '2-2h6"/>'
    ),
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "database": (
        '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
        '<path d="M3 5v14a9 3 0 0 0 18 0V5"/>'
        '<path d="M3 12a9 3 0 0 0 18 0"/>'
    ),
    "file-text": (
        '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12'
        'a2 2 0 0 0 2-2V7.5L14.5 2Z"/>'
        '<polyline points="14 2 14 8 20 8"/>'
        '<line x1="16" x2="8" y1="13" y2="13"/>'
        '<line x1="16" x2="8" y1="17" y2="17"/>'
        '<line x1="10" x2="8" y1="9" y2="9"/>'
    ),
    "inbox": (
        '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/>'
        '<path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6'
        'l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11Z"/>'
    ),
    "sliders": (
        '<line x1="4" x2="4" y1="21" y2="14"/>'
        '<line x1="4" x2="4" y1="10" y2="3"/>'
        '<line x1="12" x2="12" y1="21" y2="12"/>'
        '<line x1="12" x2="12" y1="8" y2="3"/>'
        '<line x1="20" x2="20" y1="21" y2="16"/>'
        '<line x1="20" x2="20" y1="12" y2="3"/>'
        '<line x1="2" x2="6" y1="14" y2="14"/>'
        '<line x1="10" x2="14" y1="8" y2="8"/>'
        '<line x1="18" x2="22" y1="16" y2="16"/>'
    ),
    "dot": '<circle cx="12" cy="12" r="4" fill="currentColor"/>',
    "video": (
        '<path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 '
        '0-.752-.432L16 10.5"/>'
        '<rect x="2" y="6" width="14" height="12" rx="2"/>'
    ),
    "maximize": (
        '<path d="M8 3H5a2 2 0 0 0-2 2v3"/>'
        '<path d="M21 8V5a2 2 0 0 0-2-2h-3"/>'
        '<path d="M3 16v3a2 2 0 0 0 2 2h3"/>'
        '<path d="M16 21h3a2 2 0 0 0 2-2v-3"/>'
    ),
    "info": (
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M12 16v-4"/><path d="M12 8h.01"/>'
    ),
    "settings": (
        '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25'
        'a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 '
        '2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 '
        '0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2'
        ' 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 '
        '1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2'
        ' 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l'
        '.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2'
        ' 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2Z"/>'
        '<circle cx="12" cy="12" r="3"/>'
    ),
}


def _svg_data_url(name: str, color: str) -> str:
    path = _PATHS.get(name) or '<circle cx="12" cy="12" r="3"/>'
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f'{path}</svg>'
    )
    return "data:image/svg+xml;utf8," + quote(svg, safe="")


def icon(name: str, size: int = 14, color: str = "#a0a0b0",
          style: dict | None = None) -> html.Img:
    """Render the named icon at *size* px, stroke *color*.

    Color must be specified at call time (data: URLs don't inherit
    CSS color). Default is the design-system secondary-text gray.
    Pass an explicit color (e.g. ``COLOR_ACCENT``) when the icon
    should pop, or pass the current ``COLOR_TEXT_TERTIARY`` to fade.
    """
    assert isinstance(name, str) and name, "icon name required"
    assert size > 0, "size must be positive"
    merged: dict = {
        "width": f"{size}px", "height": f"{size}px",
        "display": "inline-block", "verticalAlign": "middle",
    }
    if style:
        merged.update(style)
    return html.Img(src=_svg_data_url(name, color), style=merged)
