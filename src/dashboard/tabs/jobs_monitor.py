"""Jobs monitor -- a live view of every background job so the user can see
what's running vs queued.

Mass Analyze scans + screening benchmarks share ONE FIFO worker thread, so
two animals' scans run one at a time (the second is queued, not parallel).
Chronic Evoked warms run in parallel (one daemon thread per animal). Event-
clip extraction has its own worker. This tab surfaces all of them with a
2 s auto-poll, including each pending job's queue position.
"""

from __future__ import annotations

from dash import Input, Output, dash_table, dcc, html

from src.dashboard.components import (
    DARK_TABLE_STYLE, ZEBRA_STRIPE, loading_icon)
from src.utils import mass_analyze as _ma
from src.dashboard.tabs import chronic_evoked as _chronic

_KIND_LABELS = {
    "mass_analyze": "Mass Analyze scan",
    "screen_eval": "Screening benchmark",
    "event_clip": "Video clip",
    "chronic_warm": "Chronic warm",
}
_COLUMNS = ["Kind", "Scope", "Status", "Progress", "Queue",
            "Created", "Started", "Finished"]


def layout(store):
    return html.Div([
        html.H3("Background jobs",
                style={"color": "white", "marginBottom": "4px"}),
        html.Div("Scans run one at a time (single worker); chronic warms "
                 "run in parallel, one per animal. Auto-refreshes every 2s.",
                 style={"color": "#a0a0b0", "fontSize": "12px",
                        "marginBottom": "10px"}),
        html.Div(id="jobs-monitor-summary",
                 style={"color": "#cfd0d6", "fontSize": "13px",
                        "fontWeight": "600", "marginBottom": "10px"}),
        dcc.Interval(id="jobs-monitor-poll", interval=2000, disabled=False),
        dcc.Loading(
            custom_spinner=loading_icon("Loading jobs…", small=True),
            delay_show=150,
            overlay_style={"visibility": "visible", "opacity": 0.45},
            children=dash_table.DataTable(
                id="jobs-monitor-table",
                columns=[{"name": c, "id": c} for c in _COLUMNS],
                data=[], page_size=25, sort_action="native",
                style_data_conditional=[ZEBRA_STRIPE], **DARK_TABLE_STYLE),
        ),
    ], style={"padding": "20px 24px"})


def register_callbacks(app, store, config: dict) -> None:

    @app.callback(
        Output("jobs-monitor-table", "data"),
        Output("jobs-monitor-summary", "children"),
        Input("jobs-monitor-poll", "n_intervals"),
    )
    def _refresh(_n):
        try:
            jobs = _ma.list_active_jobs(store)
        except Exception as e:  # noqa: BLE001 -- never crash the monitor
            return [], f"Error reading jobs: {e}"
        warming = _warming_rows()
        return _rows_for_table(jobs) + warming, _summary(jobs, warming)


# --------------------------------------------------------------------- #
#  Pure helpers
# --------------------------------------------------------------------- #

def _warming_rows() -> list[dict]:
    """Synthetic rows for the in-memory chronic warm threads."""
    rows = []
    for animal in _chronic.list_warming():
        p = _chronic.warm_progress(animal) or {}
        rows.append({"kind": "chronic_warm", "scope": animal,
                     "status": "warming",
                     "scanned": p.get("done"), "total": p.get("total"),
                     "queue_pos": 0, "created_at": "", "started_at": "",
                     "finished_at": ""})
    return rows


def _rows_for_table(entries: list[dict]) -> list[dict]:
    return [{
        "Kind": _KIND_LABELS.get(e["kind"], e["kind"]),
        "Scope": e.get("scope") or "?",
        "Status": e.get("status") or "",
        "Progress": _progress(e),
        "Queue": _queue(e),
        "Created": _short_dt(e.get("created_at")),
        "Started": _short_dt(e.get("started_at")),
        "Finished": _short_dt(e.get("finished_at")),
    } for e in entries]


def _progress(e: dict) -> str:
    total, scanned = e.get("total"), e.get("scanned")
    if total:
        return f"{scanned or 0}/{total}"
    return "—"


def _queue(e: dict) -> str:
    qp = e.get("queue_pos")
    if e.get("status") == "running" or qp == 0:
        return "running"
    if qp:
        return f"#{qp} in queue"
    return "—"


def _short_dt(s) -> str:
    if not s:
        return ""
    return str(s)[:19].replace("T", " ")


def _summary(jobs: list[dict], warming: list[dict]) -> str:
    running = sum(1 for e in jobs if e.get("status") == "running")
    queued = sum(1 for e in jobs if e.get("status") == "pending")
    n_warm = len(warming)
    parts = [f"{running} running", f"{queued} queued"]
    if n_warm:
        parts.append(f"{n_warm} chronic warm{'' if n_warm == 1 else 's'}")
    if not running and not queued and not n_warm:
        return "Nothing running — all background workers idle."
    return " · ".join(parts) + "."
