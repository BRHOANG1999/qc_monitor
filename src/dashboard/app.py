"""Dash web dashboard for the QC Monitor."""

import logging
from dash import Dash, html, dcc, dash_table, callback_context
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from src.db.store import Store

logger = logging.getLogger("qc_monitor.dashboard")


def create_app(config: dict, store: Store) -> Dash:
    refresh_sec = config.get("dashboard", {}).get("refresh_interval_sec", 10)

    app = Dash(__name__, title="QC Monitor", suppress_callback_exceptions=True)

    app.layout = html.Div([
        html.H1("QC Monitor — LFP Signal Quality Dashboard",
                style={"textAlign": "center", "padding": "10px", "backgroundColor": "#1a1a2e", "color": "white"}),

        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Live Overview", value="overview"),
            dcc.Tab(label="Signal Quality", value="signal"),
            dcc.Tab(label="Spectral", value="spectral"),
            dcc.Tab(label="Stim QC", value="stim"),
            dcc.Tab(label="Alerts", value="alerts"),
            dcc.Tab(label="Sessions", value="sessions"),
        ]),

        html.Div(id="tab-content"),
        dcc.Interval(id="refresh", interval=refresh_sec * 1000, n_intervals=0),
    ])

    @app.callback(
        Output("tab-content", "children"),
        [Input("tabs", "value"), Input("refresh", "n_intervals")]
    )
    def render_tab(tab, _n):
        try:
            if tab == "overview":
                return _overview_tab(store)
            elif tab == "signal":
                return _signal_quality_tab(store)
            elif tab == "spectral":
                return _spectral_tab(store)
            elif tab == "stim":
                return _stim_tab(store)
            elif tab == "alerts":
                return _alerts_tab(store)
            elif tab == "sessions":
                return _sessions_tab(store)
        except Exception as e:
            logger.error("Dashboard render error: %s", e, exc_info=True)
            return html.Div(f"Error: {e}", style={"color": "red"})

    return app


def _overview_tab(store: Store):
    health = store.get_health_history(hours=1)
    latest = health[-1] if health else {}

    sessions = store.get_sessions()
    active_session = sessions[0] if sessions else {}

    recent_alerts = store.get_recent_alerts(hours=24)

    return html.Div([
        html.Div([
            _status_card("Network", "OK" if latest.get("network_share_accessible") else "DOWN",
                        "green" if latest.get("network_share_accessible") else "red"),
            _status_card("CPU", f"{latest.get('cpu_pct', 0):.0f}%",
                        "green" if latest.get("cpu_pct", 0) < 80 else "orange"),
            _status_card("Memory", f"{latest.get('memory_pct', 0):.0f}%",
                        "green" if latest.get("memory_pct", 0) < 85 else "orange"),
            _status_card("Disk Free", f"{latest.get('disk_free_gb', 0):.1f} GB",
                        "green" if latest.get("disk_free_gb", 0) > 50 else "red"),
            _status_card("Files/Hour", str(latest.get("files_processed_last_hour", 0)), "blue"),
            _status_card("Queue", str(latest.get("queue_depth", 0)), "blue"),
        ], style={"display": "flex", "gap": "10px", "padding": "10px", "flexWrap": "wrap"}),

        html.H3("Active Session"),
        html.P(f"{active_session.get('session_name', 'None')} — "
               f"{active_session.get('num_files', 0)} files, "
               f"{active_session.get('processed', 0)} processed") if active_session else html.P("No sessions"),

        html.H3(f"Recent Alerts ({len(recent_alerts)})"),
        dash_table.DataTable(
            data=[{"time": a["sent_at"][:19], "severity": a["severity"],
                   "type": a["alert_type"], "message": a["message"][:100]}
                  for a in recent_alerts[:10]],
            columns=[{"name": c, "id": c} for c in ["time", "severity", "type", "message"]],
            style_data_conditional=[
                {"if": {"filter_query": "{severity} = critical"}, "backgroundColor": "#ffcccc"},
                {"if": {"filter_query": "{severity} = warning"}, "backgroundColor": "#fff3cd"},
            ],
            page_size=10,
        ) if recent_alerts else html.P("No alerts in the last 24 hours"),
    ], style={"padding": "20px"})


def _signal_quality_tab(store: Store):
    data = store.get_qc_timeseries(hours=48)
    if not data:
        return html.Div("No QC data yet.", style={"padding": "20px"})

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["RMS Amplitude", "Artifact %", "Line Noise Ratio"])

    times = [d["chunk_datetime"] for d in data]
    channels = sorted(set(d["channel"] for d in data))

    for ch in channels:
        ch_data = [d for d in data if d["channel"] == ch]
        ch_times = [d["chunk_datetime"] for d in ch_data]

        fig.add_trace(go.Scatter(x=ch_times, y=[d["rms_amplitude"] for d in ch_data],
                                 name=f"Ch{ch}", legendgroup=f"ch{ch}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ch_times, y=[d["artifact_pct"] for d in ch_data],
                                 name=f"Ch{ch}", legendgroup=f"ch{ch}", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=ch_times, y=[d["line_noise_ratio"] for d in ch_data],
                                 name=f"Ch{ch}", legendgroup=f"ch{ch}", showlegend=False), row=3, col=1)

    fig.update_layout(height=700, template="plotly_dark")
    return html.Div([dcc.Graph(figure=fig)], style={"padding": "20px"})


def _spectral_tab(store: Store):
    data = store.get_qc_timeseries(hours=48)
    if not data:
        return html.Div("No spectral data yet.", style={"padding": "20px"})

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Band Powers Over Time", "Line Noise Power"])

    ch0 = [d for d in data if d["channel"] == 0]
    times = [d["chunk_datetime"] for d in ch0]

    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    for band, color in zip(bands, colors):
        key = f"{band}_power"
        vals = [d.get(key, 0) for d in ch0]
        fig.add_trace(go.Scatter(x=times, y=vals, name=band,
                                 line=dict(color=color)), row=1, col=1)

    fig.add_trace(go.Scatter(x=times, y=[d.get("line_noise_power", 0) for d in ch0],
                             name="60Hz", line=dict(color="red")), row=2, col=1)

    fig.update_layout(height=600, template="plotly_dark")
    return html.Div([dcc.Graph(figure=fig)], style={"padding": "20px"})


def _stim_tab(store: Store):
    conn = store._connect()
    rows = conn.execute(
        """SELECT pf.chunk_datetime, sq.*
           FROM stim_qc sq JOIN processed_files pf ON sq.file_id = pf.id
           ORDER BY pf.chunk_datetime DESC LIMIT 200"""
    ).fetchall()
    conn.close()

    if not rows:
        return html.Div("No stimulation data yet.", style={"padding": "20px"})

    data = [dict(r) for r in rows]
    return html.Div([
        html.H3("Stimulation Delivery QC"),
        dash_table.DataTable(
            data=[{
                "time": d["chunk_datetime"][:16],
                "channel": d["stim_channel"],
                "charge_nC": d["charge_nC"],
                "freq_Hz": d["frequency_hz"],
                "pulses": d["total_pulses"],
                "expected": d["expected_pulses"],
                "delivery_%": f"{d['delivery_pct']:.1f}" if d["delivery_pct"] else "N/A",
            } for d in data],
            columns=[{"name": c, "id": c} for c in
                     ["time", "channel", "charge_nC", "freq_Hz", "pulses", "expected", "delivery_%"]],
            style_data_conditional=[
                {"if": {"filter_query": "{delivery_%} = '0.0'"}, "backgroundColor": "#ffcccc"},
            ],
            page_size=20,
        ),
    ], style={"padding": "20px"})


def _alerts_tab(store: Store):
    alerts = store.get_recent_alerts(hours=168)  # 7 days
    return html.Div([
        html.H3(f"Alert History (last 7 days) — {len(alerts)} alerts"),
        dash_table.DataTable(
            data=[{"time": a["sent_at"][:19], "severity": a["severity"],
                   "type": a["alert_type"], "message": a["message"]}
                  for a in alerts],
            columns=[{"name": c, "id": c} for c in ["time", "severity", "type", "message"]],
            style_data_conditional=[
                {"if": {"filter_query": "{severity} = critical"}, "backgroundColor": "#ffcccc"},
                {"if": {"filter_query": "{severity} = warning"}, "backgroundColor": "#fff3cd"},
            ],
            page_size=25,
            filter_action="native",
            sort_action="native",
        ),
    ], style={"padding": "20px"})


def _sessions_tab(store: Store):
    sessions = store.get_sessions()
    return html.Div([
        html.H3(f"Sessions ({len(sessions)})"),
        dash_table.DataTable(
            data=[{
                "name": s["session_name"],
                "files": s["num_files"],
                "processed": s["processed"],
                "errors": s["errors"],
                "first": s["first_chunk"][:16] if s["first_chunk"] else "",
                "last": s["last_chunk"][:16] if s["last_chunk"] else "",
            } for s in sessions],
            columns=[{"name": c, "id": c} for c in
                     ["name", "files", "processed", "errors", "first", "last"]],
            page_size=20,
            sort_action="native",
        ),
    ], style={"padding": "20px"})


def _status_card(title: str, value: str, color: str = "blue"):
    return html.Div([
        html.Div(title, style={"fontSize": "12px", "color": "#888"}),
        html.Div(value, style={"fontSize": "24px", "fontWeight": "bold", "color": color}),
    ], style={
        "border": f"2px solid {color}", "borderRadius": "8px", "padding": "10px 20px",
        "minWidth": "120px", "textAlign": "center", "backgroundColor": "#f8f9fa",
    })
