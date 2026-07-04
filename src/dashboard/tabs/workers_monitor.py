"""Workers monitor — a live view of the background daemons so you can see, at
a glance, whether each one is actually running.

Two panels, auto-refreshing every 3 s:

* **Pipeline workers** — liveness inferred from the footprint each worker
  leaves in the DB (``store.worker_liveness``): file processing, the health
  loop, impedance refresh, the auto-filter sweep, alerts, mass-analyze. Each
  shows "last activity … ago" + a status pill (running / stale / idle). The
  health loop is the daemon's true heartbeat — if it's stale the whole
  processing daemon is down.
* **Threads in this process** — ``threading.enumerate()`` filtered to the
  app's named daemons (``qc-*``) + the main thread, so you can confirm the
  in-process workers (assignments/operator-log warmers, event-clip +
  mass-analyze workers, chronic warms) are alive. Process-local: if the
  dashboard and the processing daemon run as separate processes, this lists
  only the ones in the dashboard process — the Pipeline panel above covers
  the rest cross-process.
"""

from __future__ import annotations

import threading
from datetime import datetime

from dash import Input, Output, dash_table, dcc, html

from src.dashboard.components import DARK_TABLE_STYLE, ZEBRA_STRIPE

# Each pipeline worker: (key in worker_liveness, label, expected cadence sec,
# event_driven). event_driven workers are idle-when-nothing-to-do, so an old
# timestamp reads as "idle" (grey) rather than "stale" (red).
_WORKERS = [
    ("health", "Health loop (daemon heartbeat)", 60, False),
    ("processing", "File processing (dispatcher)", 3600, True),
    ("impedance", "Impedance refresh", 1800, False),
    ("auto_filter", "Auto-filter sweep", 300, False),
    ("mass_analyze", "Mass-analyze scans", 300, True),
    ("alerts", "Alerts", 3600, True),
]

_COLUMNS = ["Worker", "Status", "Last activity", "Expected"]


def _parse_dt(s):
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%Y_%m_%d__%H_%M_%S"):
        try:
            return datetime.strptime(str(s), fmt)
        except (ValueError, TypeError):
            continue
    try:
        return datetime.fromisoformat(str(s))
    except (ValueError, TypeError):
        return None


def _ago(dt) -> str:
    if dt is None:
        return "never"
    secs = (datetime.now() - dt).total_seconds()
    if secs < 0:
        return "just now"
    if secs < 90:
        return f"{int(secs)}s ago"
    if secs < 5400:
        return f"{int(secs / 60)}m ago"
    if secs < 172800:
        return f"{secs / 3600:.1f}h ago"
    return f"{secs / 86400:.1f}d ago"


def _fmt_cadence(sec: int) -> str:
    if sec < 90:
        return f"~{sec}s"
    if sec < 5400:
        return f"~{int(sec / 60)}m"
    return f"~{int(sec / 3600)}h"


def _status(dt, expected: int, event_driven: bool) -> str:
    """running (fresh) / stale (should be fresh but isn't) / idle (event-
    driven, nothing to do) / down (never seen)."""
    if dt is None:
        return "idle" if event_driven else "down"
    age = (datetime.now() - dt).total_seconds()
    if age <= expected * 2:
        return "running"
    return "idle" if event_driven else "stale"


def _pipeline_rows(store) -> list[dict]:
    try:
        live = store.worker_liveness()
    except Exception:  # noqa: BLE001
        live = {}
    rows = []
    for key, label, expected, event_driven in _WORKERS:
        dt = _parse_dt(live.get(key))
        rows.append({
            "Worker": label,
            "Status": _status(dt, expected, event_driven),
            "Last activity": _ago(dt),
            "Expected": _fmt_cadence(expected),
        })
    return rows


def _thread_rows() -> list[dict]:
    """The app's named daemon threads alive in THIS process."""
    out = []
    for t in threading.enumerate():
        name = t.name or ""
        if not (name.startswith("qc-") or name == "MainThread"
                or "worker" in name.lower() or "warmer" in name.lower()):
            continue
        out.append({
            "Thread": name,
            "Alive": "yes" if t.is_alive() else "no",
            "Daemon": "yes" if t.daemon else "no",
        })
    out.sort(key=lambda r: r["Thread"])
    return out


_STATUS_STYLE = [
    ZEBRA_STRIPE,
    {"if": {"filter_query": "{Status} = running", "column_id": "Status"},
     "color": "#30d158", "fontWeight": "600"},
    {"if": {"filter_query": "{Status} = stale", "column_id": "Status"},
     "color": "#ff453a", "fontWeight": "600"},
    {"if": {"filter_query": "{Status} = down", "column_id": "Status"},
     "color": "#ff453a", "fontWeight": "600"},
    {"if": {"filter_query": "{Status} = idle", "column_id": "Status"},
     "color": "#8a8a99"},
    {"if": {"filter_query": "{Alive} = no", "column_id": "Alive"},
     "color": "#ff453a", "fontWeight": "600"},
]


def layout(store):
    return html.Div([
        html.H3("Workers", style={"color": "white", "marginBottom": "4px"}),
        html.Div("Background daemons and whether they're running. The health "
                 "loop is the daemon's heartbeat — if it's stale, processing "
                 "is down. Auto-refreshes every 3 s.",
                 style={"color": "#a0a0b0", "fontSize": "12px",
                        "marginBottom": "10px"}),
        dcc.Interval(id="workers-monitor-poll", interval=3000, disabled=False),
        html.Div("Pipeline workers (liveness from DB footprint)",
                 style={"color": "#cfd0d6", "fontSize": "13px",
                        "fontWeight": "600", "margin": "6px 0 4px"}),
        dash_table.DataTable(
            id="workers-monitor-pipeline",
            columns=[{"name": c, "id": c} for c in _COLUMNS],
            data=_pipeline_rows(store),
            **DARK_TABLE_STYLE,
            style_data_conditional=_STATUS_STYLE),
        html.Div("Threads in this process",
                 style={"color": "#cfd0d6", "fontSize": "13px",
                        "fontWeight": "600", "margin": "14px 0 4px"}),
        dash_table.DataTable(
            id="workers-monitor-threads",
            columns=[{"name": c, "id": c}
                     for c in ("Thread", "Alive", "Daemon")],
            data=_thread_rows(),
            **DARK_TABLE_STYLE,
            style_data_conditional=_STATUS_STYLE),
    ], style={"padding": "4px 2px"})


def register_callbacks(app, store, config: dict) -> None:
    @app.callback(
        Output("workers-monitor-pipeline", "data"),
        Output("workers-monitor-threads", "data"),
        Input("workers-monitor-poll", "n_intervals"),
    )
    def _refresh(_n):
        return _pipeline_rows(store), _thread_rows()
