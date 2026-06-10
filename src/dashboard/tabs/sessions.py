"""Sessions tab -- one row per discovered session with file counts +
auto-discovered channel / stim config.

Read-only table. Clicking a session row jumps to Evoked Waveforms
with that session pre-filled; that navigation callback lives in
``app.py`` (``jump_to_session``) because it writes app-shell outputs
(group-tabs / tabs / selected-session-dir), not anything owned by
this tab. The table here just exposes the ``sessions-table`` id and
its ``dir`` column that the shell callback reads.

Lifted out of ``src/dashboard/app.py`` following the per-tab pattern.
"""

from __future__ import annotations

from dash import dash_table, html

from src.dashboard.components import (
    DARK_TABLE_STYLE, ZEBRA_STRIPE, tab_empty_state,
)
from src.dashboard.data_helpers import parse_json_field
from src.dashboard.design import (
    COLOR_ACCENT, COLOR_DANGER, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_TERTIARY, FONT_SIZE_CAPTION, FONT_SIZE_HEADER,
    SPACE_3, SPACE_4,
)
from src.db.store import Store

_COLUMNS = [
    {"name": "Session", "id": "name"},
    {"name": "Directory", "id": "dir"},
    {"name": "Files", "id": "files"},
    {"name": "Processed", "id": "processed"},
    {"name": "Errors", "id": "errors"},
    {"name": "First Chunk", "id": "first"},
    {"name": "Last Chunk", "id": "last"},
    {"name": "Channels", "id": "channels"},
    {"name": "SR (Hz)", "id": "sr"},
    {"name": "Stim Freq", "id": "stim_freq"},
    {"name": "Stim Charge (nC)", "id": "stim_charge"},
]


def layout(store: Store):
    """Build the Sessions table. Empty-state when no sessions have
    been discovered yet."""
    sessions = store.get_sessions()
    if not sessions:
        return tab_empty_state(
            "No sessions yet",
            "Sessions appear here once the watcher discovers .mat "
            "files in the network share configured under watch.paths.",
        )

    config_by_dir = {c["session_dir"]: c
                     for c in store.get_all_session_configs()}

    rows = []
    for s in sessions:
        cfg = config_by_dir.get(s["session_dir"], {})
        ch_names = parse_json_field(cfg.get("channel_names"))
        num_ch = cfg.get("num_channels") or (
            len(ch_names) if ch_names else "")
        stim_freq = cfg.get("stim_frequency_hz", "")
        stim_charge = cfg.get("stim_charge_nC", "")
        rows.append({
            "name": s["session_name"],
            "dir": s["session_dir"],
            "files": s["num_files"],
            "processed": s["processed"],
            "errors": s["errors"],
            "first": s["first_chunk"][:16] if s["first_chunk"] else "",
            "last": s["last_chunk"][:16] if s["last_chunk"] else "",
            "channels": num_ch,
            "sr": cfg.get("sampling_rate", ""),
            "stim_freq": f"{stim_freq}" if stim_freq else "",
            "stim_charge": f"{stim_charge}" if stim_charge else "",
        })

    return html.Div([
        html.Div([
            html.H3(f"Sessions ({len(sessions)})",
                    style={"color": COLOR_TEXT_PRIMARY,
                           "fontSize": FONT_SIZE_HEADER,
                           "fontWeight": "600", "margin": "0"}),
            html.Span(
                "Click any row to open that session in Evoked "
                "Waveforms.",
                style={"color": COLOR_TEXT_TERTIARY,
                       "fontSize": FONT_SIZE_CAPTION,
                       "marginLeft": SPACE_3},
            ),
        ], style={"display": "flex", "alignItems": "baseline",
                  "marginBottom": SPACE_4}),
        dash_table.DataTable(
            id="sessions-table",
            data=rows,
            columns=_COLUMNS,
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{errors} > 0"},
                 "backgroundColor": "rgba(255,69,58,0.10)",
                 "color": COLOR_DANGER},
                {"if": {"state": "active"},
                 "backgroundColor": "rgba(94,124,226,0.18)",
                 "border": f"1px solid {COLOR_ACCENT}"},
            ],
            style_cell_conditional=[
                {"if": {"column_id": "name"},
                 "cursor": "pointer", "fontWeight": "600",
                 "color": COLOR_ACCENT},
            ],
            page_size=20,
            sort_action="native",
            filter_action="native",
            cell_selectable=True,
            active_cell=None,
        ),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """No tab-owned callbacks. The row-click navigation
    (``jump_to_session``) lives in app.py because it drives the app
    shell, not this tab's own state."""
    return
