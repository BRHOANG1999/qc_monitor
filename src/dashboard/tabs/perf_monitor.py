"""Performance tab: what is eating computation, ranked.

Reads the in-process perf ledger (src/dashboard/perf.py). Every Dash callback is
timed by a Flask after_request hook; each tab render and each Overview stage adds
its own labelled sample. Labels read:

  * ``cb:<output>``       -- a Dash callback, keyed by its output spec
  * ``tab:<Tab>``         -- one whole tab render (render_tab)
  * ``overview:<stage>``  -- one Overview build stage

So a "QC Monitor is slow" report becomes a sorted table: sort by Total to find
what dominates cumulatively, or by Max to find the worst single stall. "Reset"
zeroes the counters so you can time one specific reproduction. Auto-refresh 2 s.
"""

from __future__ import annotations

from dash import Input, Output, State, dash_table, dcc, html

from src.dashboard.components import DARK_TABLE_STYLE
from src.dashboard import perf as _perf

_COLUMNS = [
    {"name": "Component (label)", "id": "label"},
    {"name": "Calls", "id": "n"},
    {"name": "Total (ms)", "id": "total_ms", "type": "numeric",
     "format": {"specifier": ",.0f"}},
    {"name": "Avg (ms)", "id": "avg_ms", "type": "numeric",
     "format": {"specifier": ",.0f"}},
    {"name": "Max (ms)", "id": "max_ms", "type": "numeric",
     "format": {"specifier": ",.0f"}},
    {"name": "Last (ms)", "id": "last_ms", "type": "numeric",
     "format": {"specifier": ",.0f"}},
]

# Red-tint the rows whose worst single call crossed a "you'd feel this" bar.
_ROW_STYLE = [
    {"if": {"filter_query": "{max_ms} >= 2000"},
     "backgroundColor": "rgba(255,69,58,0.18)"},
    {"if": {"filter_query": "{max_ms} >= 800 && {max_ms} < 2000"},
     "backgroundColor": "rgba(255,159,10,0.15)"},
]


def _rows(by: str = "total"):
    return _perf.top(n=60, by=by)


def layout(store=None, config=None):
    return html.Div([
        html.H3("Performance", style={"color": "white", "marginBottom": "4px"}),
        html.Div("Where the dashboard spends compute, ranked. 'cb:' = a Dash "
                 "callback, 'tab:' = a whole tab render, 'overview:' = one "
                 "Overview build stage. Sort by Total for cumulative cost, Max "
                 "for the worst single stall. Auto-refreshes every 2 s.",
                 style={"color": "#a0a0b0", "fontSize": "12px",
                        "marginBottom": "10px"}),
        html.Div([
            html.Span("Rank by: ", style={"color": "#888",
                                          "fontSize": "12px",
                                          "marginRight": "8px"}),
            dcc.RadioItems(
                id="perf-sort",
                options=[{"label": " Total", "value": "total"},
                         {"label": " Max", "value": "max"},
                         {"label": " Avg", "value": "avg"},
                         {"label": " Last", "value": "last"},
                         {"label": " Calls", "value": "n"}],
                value="total", inline=True,
                labelStyle={"color": "#ddd", "fontSize": "12px",
                            "marginRight": "14px"},
                inputStyle={"marginRight": "4px"}),
            html.Button("Reset counters", id="perf-reset", n_clicks=0,
                        style={"marginLeft": "16px", "fontSize": "12px",
                               "cursor": "pointer", "padding": "4px 10px",
                               "borderRadius": "5px", "color": "#f0f0f5",
                               "background": "#3a3a4a",
                               "border": "1px solid #555"}),
            html.Span(id="perf-uptime", style={"color": "#6c6c80",
                                               "fontSize": "11px",
                                               "marginLeft": "14px"}),
        ], style={"display": "flex", "alignItems": "center",
                  "marginBottom": "8px"}),
        dcc.Interval(id="perf-poll", interval=2000, disabled=False),
        dash_table.DataTable(
            id="perf-table",
            columns=_COLUMNS,
            data=_rows("total"),
            sort_action="native",
            **DARK_TABLE_STYLE,
            style_data_conditional=_ROW_STYLE),
    ], style={"padding": "4px 2px"})


def register_callbacks(app, store, config: dict) -> None:
    @app.callback(
        Output("perf-table", "data"),
        Output("perf-uptime", "children"),
        Input("perf-poll", "n_intervals"),
        Input("perf-sort", "value"),
        Input("perf-reset", "n_clicks"),
        State("perf-sort", "value"),
        prevent_initial_call=True,
    )
    def _refresh(_n, sort_by, _reset_clicks, sort_state):
        from dash import ctx
        if ctx.triggered_id == "perf-reset":
            _perf.reset()
        rows = _rows(sort_by or sort_state or "total")
        up = _perf.uptime_sec()
        mins = up / 60.0
        return rows, f"measured over {mins:,.0f} min uptime"
