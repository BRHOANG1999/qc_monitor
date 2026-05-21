"""Design tokens -- single source of truth for the QC Monitor's
visual language.

Python modules import the constants here; CSS reads the same values via
custom properties injected at app startup (see ``as_css_root_block``).
Keep the palette and spacing scale narrow on purpose -- if you find
yourself needing a new color or a new spacing step, prefer to compose
the existing ones first.

Apple HIG flavored: deference (quiet chrome), clarity (consistent
ramp), depth (flat layers separated only by hairline dividers).
"""

from __future__ import annotations

# --------------------------------------------------------------------- #
#  Palette
# --------------------------------------------------------------------- #

# Surfaces -- flat layers, darker behind
COLOR_SURFACE_0 = "#0a0a14"   # page background (deepest)
COLOR_SURFACE_1 = "#13131f"   # primary card / section background
COLOR_SURFACE_2 = "#1c1c2c"   # elevated card / hover state
COLOR_SURFACE_3 = "#262638"   # input field background

# Text -- three-step ramp; nothing pure white (easier on the eyes)
COLOR_TEXT_PRIMARY = "#f0f0f5"
COLOR_TEXT_SECONDARY = "#a0a0b0"
COLOR_TEXT_TERTIARY = "#6c6c80"

# Hairline divider -- softer than #333; honors "deference"
COLOR_DIVIDER = "rgba(255,255,255,0.07)"

# Semantic accents -- four meanings, used consistently
COLOR_ACCENT = "#5e7ce2"      # interactive: buttons, focus, selected
COLOR_SUCCESS = "#30d158"     # ok / confirmed (Apple's systemGreen)
COLOR_WARNING = "#ff9f0a"     # warning (Apple's systemOrange)
COLOR_DANGER = "#ff453a"      # error / critical (Apple's systemRed)

# Channel role uses the accent + a neutral so plots stay quiet
ROLE_COLORS = {
    "eeg": COLOR_ACCENT,
    "stim_copy": COLOR_TEXT_TERTIARY,
    "reference": COLOR_SUCCESS,
}

# --------------------------------------------------------------------- #
#  Typography
# --------------------------------------------------------------------- #

# SF first, then Windows variable fonts, then web-safe.
FONT_STACK = ('-apple-system, BlinkMacSystemFont, "SF Pro Text", '
              '"Segoe UI Variable", "Segoe UI", "Helvetica Neue", '
              "Helvetica, Arial, sans-serif")

# Type ramp (px). Resist the urge to invent more steps.
FONT_SIZE_TITLE = "20px"     # was 22px -- tighter density pass
FONT_SIZE_HEADER = "14px"    # was 15px
FONT_SIZE_BODY = "13px"
FONT_SIZE_CAPTION = "11px"

# --------------------------------------------------------------------- #
#  Spacing
# --------------------------------------------------------------------- #
# Multiples of 4 but tightened (~25% less padding than the old pass) so
# more fits on screen without feeling cramped.
SPACE_1 = "4px"
SPACE_2 = "6px"              # was 8px
SPACE_3 = "10px"             # was 12px
SPACE_4 = "12px"             # was 16px
SPACE_5 = "16px"             # was 24px  (page padding)
SPACE_6 = "24px"             # was 32px  (big gaps)

# --------------------------------------------------------------------- #
#  Corner radius
# --------------------------------------------------------------------- #
RADIUS_SM = "5px"             # was 6px
RADIUS_MD = "8px"             # was 10px


# --------------------------------------------------------------------- #
#  CSS custom-property bridge
# --------------------------------------------------------------------- #

_CSS_VARS: dict[str, str] = {
    # Surfaces
    "--surface-0": COLOR_SURFACE_0,
    "--surface-1": COLOR_SURFACE_1,
    "--surface-2": COLOR_SURFACE_2,
    "--surface-3": COLOR_SURFACE_3,
    # Text
    "--text-primary": COLOR_TEXT_PRIMARY,
    "--text-secondary": COLOR_TEXT_SECONDARY,
    "--text-tertiary": COLOR_TEXT_TERTIARY,
    # Divider + accents
    "--divider": COLOR_DIVIDER,
    "--accent": COLOR_ACCENT,
    "--success": COLOR_SUCCESS,
    "--warning": COLOR_WARNING,
    "--danger": COLOR_DANGER,
    # Typography
    "--font-stack": FONT_STACK,
    "--font-size-title": FONT_SIZE_TITLE,
    "--font-size-header": FONT_SIZE_HEADER,
    "--font-size-body": FONT_SIZE_BODY,
    "--font-size-caption": FONT_SIZE_CAPTION,
    # Spacing
    "--space-1": SPACE_1,
    "--space-2": SPACE_2,
    "--space-3": SPACE_3,
    "--space-4": SPACE_4,
    "--space-5": SPACE_5,
    "--space-6": SPACE_6,
    # Radius
    "--radius-sm": RADIUS_SM,
    "--radius-md": RADIUS_MD,
}


def as_css_root_block() -> str:
    """Return a ``<style>`` block defining all tokens as CSS custom
    properties on ``:root``. Embed this in Dash's index_string so the
    CSS theme can use ``var(--surface-1)`` etc."""
    body = "\n".join(f"  {k}: {v};" for k, v in _CSS_VARS.items())
    return f"<style>:root {{\n{body}\n}}</style>"
