"""Stim QC tab -- most recent 500 stim_qc rows joined to their
processed_files row.

Read-only, no callbacks. The table is rendered once per tab open from
a single read of ``stim_qc + processed_files``. Failed-delivery rows
(``delivery_% contains '0.0'``) are highlighted with the red
``filter_query`` rule.

Lifted out of ``src/dashboard/app.py`` -- shares the same shared
``DARK_TABLE_STYLE`` / ``ZEBRA_STRIPE`` as ``tabs/alerts`` (a single
copy now lives in ``src/dashboard/components.py``).
"""

from __future__ import annotations

from dash import dash_table, html

from src.dashboard.components import DARK_TABLE_STYLE, ZEBRA_STRIPE
from src.db.store import Store


def layout(store: Store):
    """Build the Stim QC tab. Returns a Div containing one dash_table
    with the latest 500 stim deliveries."""
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT pf.chunk_datetime, pf.session_name, sq.*
               FROM stim_qc sq
               JOIN processed_files pf ON sq.file_id = pf.id
               ORDER BY pf.chunk_datetime DESC LIMIT 500"""
        ).fetchall()

    if not rows:
        return html.Div("No stimulation data yet.",
                         style={"color": "#888"})

    data = [dict(r) for r in rows]
    return html.Div([
        html.H3("Stimulation Delivery QC",
                style={"color": "white", "marginBottom": "12px"}),
        dash_table.DataTable(
            data=[{
                "time": d["chunk_datetime"][:16],
                "session": d.get("session_name", ""),
                "channel": d["stim_channel"],
                "charge_nC": (f"{d['charge_nC']:.1f}"
                                if d["charge_nC"] else ""),
                "freq_Hz": (f"{d['frequency_hz']:.1f}"
                             if d["frequency_hz"] else ""),
                "pulses": d["total_pulses"],
                "expected": d["expected_pulses"],
                "delivery_%": (f"{d['delivery_pct']:.1f}"
                                if d["delivery_pct"] else "N/A"),
            } for d in data],
            columns=[{"name": c, "id": c} for c in
                     ["time", "session", "channel", "charge_nC",
                      "freq_Hz", "pulses", "expected", "delivery_%"]],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{delivery_%} contains '0.0'"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
            ],
            page_size=25,
            filter_action="native",
            sort_action="native",
        ),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """No callbacks on this tab."""
    return
