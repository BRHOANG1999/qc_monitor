"""PI Event Verification tab.

Reviewers (undergrads) score events; their saves land in
``status='pending_pi_review'``. This tab is where the PI:
* sees the queue of pending files,
* opens a detail panel with the stitched 5-min-before-EO to
  5-min-after-BB video clip + LFP for each event,
* clicks Approve (single or bulk) to flip status to
  ``pi_approved`` and queue a CSV write,
* clicks Flag with a note to send the file back to the
  undergrad's queue (``pi_flagged``).

Gated to ``config.review_queue.pi_emails`` -- everyone else
sees the same "Not authorised" card the Review status tab
uses.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

from dash import ALL, Input, Output, State, callback_context
from dash import dcc, html, no_update

import numpy as np
import plotly.graph_objects as go

from src.db.store import Store
from src.dashboard.auth import current_user_email
from src.dashboard.tabs.review_status import _is_pi
from src.utils import bhz_csv as _bhz_csv
from src.utils import event_clip as _event_clip
from src.utils import mass_analyze as _mass_analyze
from src.utils.animal import split_animal_electrode, is_animal_channel
from src.utils.decimate import envelope, choose_target_bins

logger = logging.getLogger("qc_monitor.tabs.event_verification")


_NOT_AUTHORISED = html.Div([
    html.Div("🔒  Not authorised",
              style={"fontSize": "18px", "fontWeight": "600",
                      "color": "#f0f0f5",
                      "marginBottom": "8px"}),
    html.Div(
        "This tab is for PIs only. Ask the lab lead to add "
        "your email to config.review_queue.pi_emails.",
        style={"color": "#a0a0b0", "fontSize": "13px",
                "maxWidth": "520px"}),
], style={"padding": "40px 24px"})


def layout(store: Store, config: dict | None = None):
    email = current_user_email()
    if not _is_pi(config or {}, email):
        return _NOT_AUTHORISED
    ffmpeg_ok = _event_clip.probe_ffmpeg(config or {})
    return html.Div([
        html.H3("Event Verification",
                 style={"color": "#f0f0f5",
                         "marginBottom": "4px"}),
        html.Div(
            "PI review gate. Approve flips files to "
            "'pi_approved' and queues a CSV write; Flag "
            "sends them back to the undergrad's queue "
            "with a note.",
            style={"color": "#a0a0b0", "fontSize": "12px",
                    "marginBottom": "16px"}),
        # ffmpeg health.
        html.Div(
            "ffmpeg not on PATH -- event clips will not "
            "extract until it's installed (see "
            "config.event_clip.ffmpeg_bin)."
            if not ffmpeg_ok else
            "ffmpeg available. Cold clip extraction ~30-90 s "
            "per event; cache hits are instant.",
            style={"color": ("#ff453a" if not ffmpeg_ok
                              else "#30d158"),
                    "fontSize": "11px",
                    "padding": "6px 10px",
                    "marginBottom": "14px",
                    "background":
                        "rgba(255,69,58,0.08)"
                        if not ffmpeg_ok
                        else "rgba(48,209,88,0.08)",
                    "border":
                        "1px solid rgba(255,69,58,0.30)"
                        if not ffmpeg_ok
                        else "1px solid rgba(48,209,88,0.30)",
                    "borderRadius": "6px"}),
        # Mass Analyze panel (collapsed-by-default details
        # card). Runs hilbert_envelope_20_200 + peakseek over
        # every pending file for an animal at the chosen
        # cutoff; zero-peak files get auto-cleared to
        # pending_pi_review on Confirm.
        _mass_analyze_panel(store),
        # Bulk-action bar.
        html.Div([
            html.Button(
                "Refresh", id="evtv-refresh-btn", n_clicks=0,
                style=_btn_style()),
            html.Button(
                "Select all", id="evtv-select-all-btn",
                n_clicks=0, style=_btn_style()),
            html.Button(
                "Clear selection", id="evtv-clear-sel-btn",
                n_clicks=0, style=_btn_style(secondary=True)),
            html.Button(
                "Approve selected",
                id="evtv-approve-sel-btn", n_clicks=0,
                style=_btn_style(accent=True)),
            html.Button(
                "Flag selected",
                id="evtv-flag-sel-btn", n_clicks=0,
                style=_btn_style(warning=True)),
            html.Div(id="evtv-sel-status",
                      style={"color": "#a0a0b0",
                              "fontSize": "11px",
                              "marginLeft": "12px"}),
        ], style={"display": "flex", "gap": "8px",
                   "alignItems": "center",
                   "marginBottom": "12px",
                   "flexWrap": "wrap"}),
        # The bulk-flag note + finalize controls.
        html.Div([
            dcc.Input(
                id="evtv-flag-note",
                type="text",
                placeholder="Note for the undergrad (used on "
                             "Flag selected)",
                style={"flex": "1 1 360px",
                        "padding": "6px 10px",
                        "background": "#262638",
                        "color": "#f0f0f5",
                        "border":
                            "1px solid rgba(255,255,255,0.10)",
                        "borderRadius": "4px",
                        "fontSize": "12px"}),
            dcc.Checklist(
                id="evtv-overwrite-mode",
                options=[{"label": " Overwrite mode "
                                    "(warns on removed rows)",
                            "value": "ow"}],
                value=[],
                labelStyle={"color": "#cfd0d6",
                             "fontSize": "11px"}),
            html.Button(
                "Finalize approved → CSV",
                id="evtv-finalize-csv-btn", n_clicks=0,
                style=_btn_style(accent=True)),
        ], style={"display": "flex", "gap": "10px",
                   "alignItems": "center",
                   "marginBottom": "12px",
                   "flexWrap": "wrap"}),
        # Status / diff toast for finalize.
        html.Div(id="evtv-finalize-status",
                  style={"color": "#a0a0b0",
                          "fontSize": "12px",
                          "minHeight": "16px",
                          "marginBottom": "12px"}),
        # Pending overwrite plan (filled by the Finalize
        # callback when overwrite mode would remove rows).
        # The modal below renders from it.
        dcc.Store(id="evtv-overwrite-plan", data=None),
        _overwrite_modal(),
        # File list. Each row carries a checkbox + per-file
        # Approve/Flag buttons + the event summary.
        dcc.Loading(
            id="evtv-pending-loading",
            type="circle",
            color="#5e7ce2",
            delay_show=180,
            children=html.Div(id="evtv-pending-list"),
        ),
        # Selection store: list of file_ids.
        dcc.Store(id="evtv-selection", data=[]),
        # Signature of the currently-rendered pending set so the
        # 10s auto-refresh can skip re-rendering the (now large)
        # list when nothing actually changed -- the redundant
        # redraw was making the list visibly blink on and off.
        dcc.Store(id="evtv-list-sig", data=None),
        # Detail panel's currently-open (file_id, event_idx)
        # or None. Surfaces under the list when set.
        dcc.Store(id="evtv-detail-target", data=None),
        # Detail view container -- populated by callback.
        html.Div(id="evtv-detail-panel",
                  style={"marginTop": "20px"}),
        # Auto-poll: ticks every 3 s; only fires the detail
        # re-render when a clip is still extracting (status
        # in pending/running). The clientside guard below
        # bumps n_intervals only when the panel is open.
        dcc.Interval(id="evtv-detail-poll", interval=3000,
                       n_intervals=0, disabled=True),
    ], style={"padding": "20px 24px"})


# --------------------------------------------------------------- #
# Style helpers
# --------------------------------------------------------------- #

def _btn_style(*, accent: bool = False, warning: bool = False,
                 secondary: bool = False) -> dict:
    base = {
        "border": "1px solid rgba(255,255,255,0.15)",
        "borderRadius": "5px", "padding": "6px 12px",
        "cursor": "pointer", "fontSize": "12px",
        "fontWeight": "600",
    }
    if accent:
        base["background"] = "#5e7ce2"
        base["color"] = "white"
        base["border"] = "none"
    elif warning:
        base["background"] = "#ff9f0a"
        base["color"] = "white"
        base["border"] = "none"
    elif secondary:
        base["background"] = "transparent"
        base["color"] = "#a0a0b0"
    else:
        base["background"] = "#262638"
        base["color"] = "#cfd0d6"
    return base


# --------------------------------------------------------------- #
# Mass Analyze panel (PI bulk Hilbert/peakseek pre-screen)
# --------------------------------------------------------------- #

def _mass_analyze_panel(store) -> html.Details:
    """Collapsed-by-default details card. The PI picks animal +
    cutoff, clicks Scan, watches progress, and clicks Confirm
    to auto-clear zero-peak files."""
    animals = sorted(store.list_all_animals())
    return html.Details([
        html.Summary([
            html.Span("Mass Analyze pending files ",
                       style={"color": "#f0f0f5",
                               "fontWeight": "600"}),
            html.Span("(PI bulk pre-screen)",
                       style={"color": "#a0a0b0",
                               "fontSize": "11px",
                               "marginLeft": "6px"}),
        ], style={"cursor": "pointer",
                   "marginBottom": "8px"}),
        html.Div([
            html.Div(
                "Hilbert peakseek runs on every pending "
                "file's first animal channel at the "
                "threshold below. Files with zero peaks "
                "above the threshold get an automatic "
                "'no events seen' submission you can "
                "bulk-approve in the list. Files with any "
                "peaks above the threshold stay in the "
                "queue for normal scoring. The threshold "
                "is the same BHZ cutoff column the CSV "
                "records.",
                style={"color": "#a0a0b0",
                        "fontSize": "11px",
                        "padding": "8px 12px",
                        "background": "rgba(94,124,226,0.04)",
                        "border":
                            "1px solid rgba(94,124,226,0.18)",
                        "borderRadius": "6px",
                        "marginBottom": "10px"}),
            html.Div([
                html.Label("Animal:",
                            style={"color": "#a0a0b0",
                                    "fontSize": "12px",
                                    "marginRight": "6px"}),
                dcc.Dropdown(
                    id="evtv-ma-animal-dropdown",
                    options=[{"label": a, "value": a}
                             for a in animals],
                    placeholder="Pick an animal",
                    style={"flex": "1 1 220px",
                            "minWidth": "200px"},
                    className="dark-dropdown",
                ),
                html.Label("Cutoff:",
                            style={"color": "#a0a0b0",
                                    "fontSize": "12px",
                                    "marginLeft": "14px",
                                    "marginRight": "6px"}),
                dcc.Input(
                    id="evtv-ma-cutoff-input",
                    type="number", min=0, step=0.005,
                    value=0.05,
                    style={"flex": "0 0 110px",
                            "padding": "6px 10px",
                            "background": "#262638",
                            "color": "#f0f0f5",
                            "border":
                                "1px solid rgba(255,255,255,0.10)",
                            "borderRadius": "4px",
                            "fontSize": "12px"}),
                html.Button(
                    "Scan files",
                    id="evtv-ma-scan-btn", n_clicks=0,
                    style=_btn_style(accent=True)),
                html.Button(
                    "Cancel scan",
                    id="evtv-ma-cancel-btn", n_clicks=0,
                    style=_btn_style(secondary=True)),
            ], style={"display": "flex",
                       "alignItems": "center",
                       "gap": "8px",
                       "flexWrap": "wrap",
                       "marginBottom": "10px"}),
            html.Div(id="evtv-ma-progress",
                      style={"color": "#a0a0b0",
                              "fontSize": "12px",
                              "minHeight": "16px",
                              "marginBottom": "8px"}),
            html.Div(id="evtv-ma-summary",
                      style={"marginBottom": "10px"}),
            html.Div(id="evtv-ma-confirm-row",
                      style={"display": "flex", "gap": "8px",
                              "alignItems": "center"}),
            # State stores + polling.
            dcc.Store(id="evtv-ma-job-id", data=None),
            dcc.Interval(id="evtv-ma-poll",
                           interval=1500, n_intervals=0,
                           disabled=True),
        ]),
    ], open=False,
       style={"padding": "12px 14px",
               "background": "#13131f",
               "border":
                   "1px solid rgba(255,255,255,0.06)",
               "borderRadius": "8px",
               "marginBottom": "14px"})


# --------------------------------------------------------------- #
# Destructive-overwrite confirmation modal
# --------------------------------------------------------------- #

def _overwrite_modal() -> html.Div:
    """The destructive-overwrite confirmation modal.

    Hidden by default. The Finalize callback sets style.display
    to 'flex' when the proposed overwrite would remove rows
    from one or more existing CSVs. The modal body is rendered
    by the plan-watcher callback from the
    ``evtv-overwrite-plan`` Store; Confirm executes the writes;
    Cancel discards them.
    """
    return html.Div(
        id="evtv-overwrite-modal",
        style={"position": "fixed", "inset": "0",
                "display": "none",
                "alignItems": "center",
                "justifyContent": "center",
                "background": "rgba(10,10,20,0.65)",
                "backdropFilter": "blur(6px)",
                "zIndex": "10001"},
        children=html.Div(
            id="evtv-overwrite-modal-card",
            style={"width": "min(640px, 92vw)",
                    "maxHeight": "82vh",
                    "overflowY": "auto",
                    "background": "#13131f",
                    "border":
                        "1px solid rgba(255,159,10,0.30)",
                    "borderRadius": "10px",
                    "padding": "24px",
                    "boxShadow":
                        "0 24px 48px rgba(0,0,0,0.55)"},
            children=[
                html.Div(
                    "⚠️  Destructive overwrite — confirm",
                    style={"color": "#ff9f0a",
                            "fontWeight": "600",
                            "fontSize": "16px",
                            "marginBottom": "6px"}),
                html.Div(
                    "Overwrite mode rebuilds the CSV from "
                    "the current pi_approved set. The "
                    "following rows would be REMOVED from "
                    "existing files. Continue?",
                    style={"color": "#a0a0b0",
                            "fontSize": "12px",
                            "marginBottom": "14px"}),
                html.Div(id="evtv-overwrite-modal-body",
                          style={"color": "#f0f0f5",
                                  "fontSize": "12px",
                                  "fontFamily":
                                      "ui-monospace, monospace",
                                  "background": "#0a0a14",
                                  "border":
                                      "1px solid rgba(255,255,255,0.08)",
                                  "borderRadius": "6px",
                                  "padding": "10px 12px",
                                  "marginBottom": "16px"}),
                html.Div([
                    html.Button(
                        "Cancel",
                        id="evtv-overwrite-cancel-btn",
                        n_clicks=0,
                        style=_btn_style(secondary=True)),
                    html.Button(
                        "Confirm overwrite",
                        id="evtv-overwrite-confirm-btn",
                        n_clicks=0,
                        style=_btn_style(warning=True)),
                ], style={"display": "flex",
                           "gap": "10px",
                           "justifyContent": "flex-end"}),
            ]))


def _render_overwrite_modal_body(plan: list[dict]
                                    ) -> list:
    """Per-group breakdown of which rows would be lost."""
    if not plan:
        return [html.Div("Nothing pending.")]
    blocks: list = []
    for entry in plan:
        if not entry.get("removed"):
            continue
        blocks.append(html.Div([
            html.Div(
                f"{entry['animal']}  ·  {entry['day']}",
                style={"color": "#ff9f0a",
                        "fontWeight": "600",
                        "marginBottom": "2px"}),
            html.Div(
                f"  {entry['csv_path']}",
                style={"color": "#666", "fontSize": "10px",
                        "marginBottom": "4px"}),
            html.Div(
                [html.Div(f"  − {fn}",
                            style={"color": "#ff453a"})
                 for fn in entry["removed"]]),
            html.Div(
                f"  ({len(entry.get('added') or [])} added, "
                f"{len(entry.get('modified') or [])} modified)",
                style={"color": "#a0a0b0",
                        "marginTop": "4px"}),
        ], style={"marginBottom": "10px"}))
    if not blocks:
        return [html.Div("No removals proposed.")]
    return blocks


# --------------------------------------------------------------- #
# File list rendering
# --------------------------------------------------------------- #

def _animal_for_session(store, session_dir: str) -> str:
    names = store._channel_names_for_session(session_dir)
    for n in names:
        if isinstance(n, str) and is_animal_channel(n):
            a, _ = split_animal_electrode(n)
            return a
    return "unknown"


def _first_animal_channel_index(store, session_dir: str) -> int:
    """Pick the first animal-bearing channel for the LFP stitch.

    The undergrad's review payload doesn't carry the channel
    index; for PI replay we default to the same heuristic the
    Video Review tab uses -- the first ``is_animal_channel``
    contact in session_config. Falls back to channel 0.
    """
    names = store._channel_names_for_session(session_dir)
    for i, n in enumerate(names):
        if isinstance(n, str) and is_animal_channel(n):
            return i
    return 0


def _build_lfp_stitch_figure(spec: '_event_clip.ClipSpec',
                                channel: int,
                                eo_sec: float, bb_sec: float,
                                store,
                                ) -> go.Figure:
    """Stitched LFP trace with EO/BB markers + boundary line.

    The decimation is the same envelope/min-max pattern the
    Video Review tab uses so the visual density matches.
    Boundaries between segments are visible as NaN gaps (set
    by ``extract_lfp_clip``); we also annotate the gap with a
    dashed vertical so the PI can see exactly where the
    stitch happened.
    """
    t, signal, fs = _event_clip.extract_lfp_clip(
        spec, int(channel), store)
    if len(signal) < 4 or fs <= 0:
        return _empty_fig(
            "Could not load LFP for this event.")
    # Decimate for visual density (no point in pushing 6 M
    # samples through Plotly).
    target = choose_target_bins(len(signal)) or 4000
    td, display, _decim = envelope(signal, fs, target,
                                       t_start=0.0)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=td, y=display, mode="lines",
        line=dict(color="#5e7ce2", width=0.9),
        hovertemplate="t=%{x:.2f}s<br>%{y:.1f} μV<extra></extra>",
    ))
    # The window starts at clip_start = max(0, eo - pre); EO
    # within the stitched clip is at (eo - clip_start), same
    # for BB.
    clip_start = max(0.0, eo_sec - spec.pre_sec)
    eo_x = max(0.0, eo_sec - clip_start)
    bb_x = max(0.0, bb_sec - clip_start)
    shapes = [
        dict(type="line", xref="x", yref="paper",
              x0=eo_x, x1=eo_x, y0=0, y1=1,
              line=dict(color="#ff453a", width=2, dash="solid"),
              opacity=0.85),
        dict(type="line", xref="x", yref="paper",
              x0=bb_x, x1=bb_x, y0=0, y1=1,
              line=dict(color="#30d158", width=2, dash="solid"),
              opacity=0.85),
    ]
    # Boundary annotations. Find NaN runs in the signal --
    # extract_lfp_clip inserts one NaN at each file boundary.
    nan_idx = np.where(np.isnan(signal))[0]
    if len(nan_idx):
        for idx in nan_idx[:8]:  # NASA Rule 3 cap
            t_boundary = float(idx) / float(fs)
            shapes.append(dict(
                type="line", xref="x", yref="paper",
                x0=t_boundary, x1=t_boundary, y0=0, y1=1,
                line=dict(color="#ff9f0a", width=1, dash="dot"),
                opacity=0.55,
            ))
    fig.update_layout(
        plot_bgcolor="#13131f", paper_bgcolor="#13131f",
        height=220,
        margin=dict(l=60, r=20, t=10, b=40),
        dragmode="pan",
        xaxis=dict(title="Time within clip (s)",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                    zeroline=False),
        yaxis=dict(title="μV", showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                    zeroline=False),
        shapes=shapes,
        showlegend=False,
        hovermode="x unified",
        annotations=[
            dict(x=eo_x, y=1.02, xref="x", yref="paper",
                  text="EO", showarrow=False,
                  font=dict(color="#ff453a", size=10)),
            dict(x=bb_x, y=1.02, xref="x", yref="paper",
                  text="BB", showarrow=False,
                  font=dict(color="#30d158", size=10)),
        ],
    )
    return fig


def _empty_fig(text: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=220,
        plot_bgcolor="#13131f", paper_bgcolor="#13131f",
        annotations=[dict(text=text, showarrow=False,
                            x=0.5, y=0.5, xref="paper",
                            yref="paper",
                            font=dict(size=12,
                                       color="#a0a0b0"))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def _format_event_row(idx: int, ev: dict) -> html.Div:
    et = ev.get("type") or "?"
    eo = ev.get("EO_sec")
    bb = ev.get("BB_sec")
    racine = ev.get("racine")
    parts = [f"Event {idx + 1}: {et}"]
    if eo is not None:
        parts.append(f"EO {float(eo):.1f}s")
    if bb is not None:
        parts.append(f"BB {float(bb):.1f}s")
    if racine is not None:
        parts.append(f"Racine {racine}")
    return html.Div(
        " · ".join(parts),
        style={"color": "#a0a0b0", "fontSize": "11px"})


def _has_real_click(triggered: list) -> bool:
    """True iff a callback was driven by an actual button click.

    Dash recreates the per-row approve/flag pattern buttons every time
    the pending list re-renders, which re-fires the ALL-pattern
    callback with ``n_clicks=None``. A real click carries a truthy
    n_clicks; recreation carries None/0. Guarding on this stops a
    redraw from silently approving/flagging a file."""
    return any((t.get("value") or 0) for t in (triggered or []))


def _pending_signature(rows: list[dict]) -> str:
    """Cheap content signature of the pending set. Each submission
    appends a new review_state row with a unique state_id, so the
    sorted state_id tuple uniquely identifies which files are pending
    -- a new submission, an approval, or a flag all change it. Used to
    skip the redundant 10s re-render that made the list blink."""
    ids = sorted(int(r.get("state_id") or 0) for r in rows)
    return ",".join(str(i) for i in ids)


def _render_pending_list(rows: list[dict], selection: list[int]
                          ) -> list:
    """Top-level: one card per pending file."""
    if not rows:
        return [html.Div(
            "🎉  No pending files. Undergrads are caught up.",
            style={"color": "#a0a0b0", "fontSize": "13px",
                    "padding": "24px",
                    "textAlign": "center"})]
    sel_set = set(int(x) for x in (selection or []))
    cards: list = []
    for r in rows:
        cards.append(_render_pending_card(r,
                                             r["file_id"] in sel_set))
    return cards


def _render_pending_card(row: dict, is_selected: bool
                          ) -> html.Div:
    file_id = int(row["file_id"])
    chunk_dt_raw = row.get("chunk_datetime") or ""
    try:
        ts = datetime.strptime(chunk_dt_raw,
                                 "%Y_%m_%d__%H_%M_%S")
        ts_label = ts.strftime("%Y-%m-%d  %H:%M")
    except ValueError:
        ts_label = chunk_dt_raw
    events = row.get("events") or []
    n_events = len(events)
    submitter = row.get("user_email") or "—"
    title_parts = [
        html.Span(ts_label,
                   style={"color": "#f0f0f5",
                           "fontWeight": "600",
                           "fontSize": "13px"}),
        html.Span(" · ", style={"color": "#666"}),
        html.Span(f"{n_events} event"
                   f"{'' if n_events == 1 else 's'}",
                   style={"color": "#5e7ce2",
                           "fontSize": "12px"}),
        html.Span(" · ", style={"color": "#666"}),
        html.Span(submitter,
                   style={"color": "#a0a0b0",
                           "fontSize": "11px"}),
    ]
    event_rows = [_format_event_row(i, e)
                   for i, e in enumerate(events)]
    if n_events == 0:
        event_rows = [html.Div(
            "No events scored (undergrad picked "
            "'No events seen').",
            style={"color": "#a0a0b0", "fontSize": "11px",
                    "fontStyle": "italic"})]
    # Per-event detail-open buttons (only when there are events).
    open_buttons = []
    for i, ev in enumerate(events):
        if ev.get("EO_sec") is None or ev.get("BB_sec") is None:
            continue
        open_buttons.append(html.Button(
            f"Open detail · event {i + 1}",
            id={"type": "evtv-open-detail",
                 "file_id": file_id, "idx": i},
            n_clicks=0,
            style={"background": "transparent",
                    "color": "#5e7ce2",
                    "border": "1px solid #5e7ce2",
                    "borderRadius": "4px",
                    "padding": "3px 10px",
                    "fontSize": "11px",
                    "cursor": "pointer",
                    "marginRight": "6px",
                    "marginTop": "4px"}))
    return html.Div([
        html.Div([
            dcc.Checklist(
                id={"type": "evtv-row-check",
                     "file_id": file_id},
                options=[{"label": "", "value": "sel"}],
                value=["sel"] if is_selected else [],
                style={"display": "inline-block",
                        "marginRight": "8px"},
                inputStyle={"transform": "scale(1.2)"}),
            html.Div(title_parts,
                      style={"display": "inline-block",
                              "verticalAlign": "middle"}),
            html.Div([
                html.Button(
                    "Approve",
                    id={"type": "evtv-approve-one-btn",
                         "file_id": file_id},
                    n_clicks=0,
                    style=_btn_style(accent=True)),
                html.Button(
                    "Flag",
                    id={"type": "evtv-flag-one-btn",
                         "file_id": file_id},
                    n_clicks=0,
                    style=_btn_style(warning=True)),
            ], style={"display": "flex", "gap": "6px",
                       "marginLeft": "auto"}),
        ], style={"display": "flex",
                   "alignItems": "center",
                   "marginBottom": "8px"}),
        html.Div(event_rows,
                  style={"marginLeft": "40px"}),
        html.Div(open_buttons,
                  style={"marginLeft": "40px"}),
    ], style={"padding": "10px 14px",
               "marginBottom": "8px",
               "background": ("rgba(94,124,226,0.08)"
                              if is_selected
                              else "#13131f"),
               "border": "1px solid rgba(255,255,255,0.06)",
               "borderRadius": "6px"})


# --------------------------------------------------------------- #
# Detail panel
# --------------------------------------------------------------- #

def _render_event_detail(target: dict, store,
                          config: dict) -> html.Div:
    """Build the per-event detail panel: stitched video +
    LFP placeholder + Approve/Flag/Edit toolbar.

    Video clip is lazy-loaded: this function only kicks off
    the extraction job; the actual <video src=...> source is
    the /media/clip/<spec_hash> route which 202s while the
    ffmpeg job runs.
    """
    if not target:
        return html.Div()
    file_id = int(target.get("file_id") or 0)
    idx = int(target.get("idx") or 0)
    # Look up the pending file + its events.
    rows = store.pi_pending_files(limit=500)
    row = next((r for r in rows
                  if int(r["file_id"]) == file_id), None)
    if not row:
        return html.Div(
            "This file is no longer pending PI review.",
            style={"color": "#a0a0b0",
                    "padding": "16px"})
    events = row.get("events") or []
    if idx < 0 or idx >= len(events):
        return html.Div(
            "Event index out of range.",
            style={"color": "#a0a0b0",
                    "padding": "16px"})
    ev = events[idx]
    eo = float(ev.get("EO_sec") or 0)
    bb = float(ev.get("BB_sec") or 0)
    spec = _event_clip.ClipSpec(
        file_id=file_id, eo_sec=eo, bb_sec=bb,
    )
    # Submit the job (cache hit returns instantly).
    try:
        h = _event_clip.submit_video_clip_job(spec, store,
                                                 config or {})
        clip_status = _event_clip.poll_video_clip_status(
            h, store)
    except Exception as e:
        logger.warning("clip submit failed: %s", e)
        h = ""
        clip_status = {"status": "failed", "error": str(e)}
    video_url = (f"/media/clip/{h}"
                  if h and clip_status.get("status") == "done"
                  else None)
    header = html.Div([
        html.Div([
            html.Span(
                f"Event {idx + 1} of {len(events)} · "
                f"{ev.get('type', '?')} · "
                f"EO {eo:.1f}s · BB {bb:.1f}s · "
                f"Racine {ev.get('racine', '?')}",
                style={"color": "#f0f0f5",
                        "fontWeight": "600"}),
            html.Button("× Close detail",
                         id="evtv-close-detail-btn",
                         n_clicks=0,
                         style=_btn_style(secondary=True)),
        ], style={"display": "flex",
                   "justifyContent": "space-between",
                   "alignItems": "center"}),
        html.Div(
            f"Video clip status: {clip_status.get('status')}"
            + (f" — {clip_status.get('error')}"
                if clip_status.get('error') else ""),
            style={"color": "#a0a0b0",
                    "fontSize": "11px",
                    "marginTop": "4px"}),
    ], style={"marginBottom": "10px"})
    if video_url:
        video_block = html.Video(
            src=video_url,
            controls=True,
            preload="metadata",
            style={"width": "100%",
                    "maxHeight": "440px",
                    "background": "#000",
                    "borderRadius": "8px"})
    else:
        msg = ("Extracting clip…  this panel auto-refreshes "
                "every 3 s while the worker runs."
                if clip_status.get("status")
                    in ("pending", "running")
                else f"Clip not ready: "
                      f"{clip_status.get('status')}")
        video_block = html.Div(
            msg,
            style={"color": "#a0a0b0",
                    "background": "#13131f",
                    "padding": "60px 20px",
                    "textAlign": "center",
                    "borderRadius": "8px"})
    # Build the stitched LFP figure for this window.
    channel = _first_animal_channel_index(
        store, row.get("session_dir") or "")
    try:
        lfp_fig = _build_lfp_stitch_figure(
            spec, channel, eo, bb, store)
    except Exception as e:
        logger.exception("LFP stitch render failed")
        lfp_fig = _empty_fig(f"LFP load error: {e}")
    return html.Div([
        header,
        video_block,
        html.Div([
            html.Div(
                f"LFP for channel {channel} "
                f"(stitched across file boundaries when "
                f"the EO−5min / BB+5min window crosses "
                f"recording boundaries)",
                style={"color": "#a0a0b0",
                        "fontSize": "11px",
                        "marginTop": "10px",
                        "marginBottom": "4px"}),
            dcc.Graph(
                figure=lfp_fig,
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "doubleClick": "reset",
                    "modeBarButtonsToRemove": [
                        "select2d", "lasso2d", "autoScale2d",
                    ],
                    "scrollZoom": True,
                },
            ),
        ]),
        # Inline marker chip strip so the PI can see every
        # landmark this event carries without scrolling back
        # to the list.
        html.Div(
            [_marker_chip(k, ev.get(f"{k}_sec"))
             for k in ("EO", "LAS", "BO", "PID", "BB")
             if ev.get(f"{k}_sec") is not None],
            style={"display": "flex", "gap": "8px",
                    "flexWrap": "wrap",
                    "marginTop": "10px"}),
    ], style={"padding": "14px",
               "background": "rgba(94,124,226,0.04)",
               "border":
                   "1px solid rgba(94,124,226,0.18)",
               "borderRadius": "8px"})


def _marker_chip(label: str, sec: float | None) -> html.Span:
    if sec is None:
        return html.Span()
    return html.Span(
        f"{label}  {float(sec):.1f}s",
        style={"padding": "3px 8px",
                "background": "rgba(255,255,255,0.05)",
                "border":
                    "1px solid rgba(255,255,255,0.10)",
                "borderRadius": "4px",
                "color": "#cfd0d6",
                "fontSize": "11px",
                "fontFamily": "ui-monospace, monospace"})


# --------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------- #

def register_callbacks(app, store, config: dict) -> None:
    """Wire up the tab. Idempotent across reloads."""
    assert app is not None, "app required"

    @app.callback(
        Output("evtv-pending-list", "children"),
        Output("evtv-sel-status", "children"),
        Output("evtv-list-sig", "data"),
        Input("evtv-refresh-btn", "n_clicks"),
        Input("evtv-selection", "data"),
        Input("refresh-trigger", "data"),
        State("evtv-list-sig", "data"),
    )
    def _render_list(_n, selection, _refresh, prev_sig):
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return [], "", no_update
        rows = store.pi_pending_files(limit=500)
        sig = _pending_signature(rows)
        # The 10s refresh-trigger fires whether or not the pending
        # set changed. Re-rendering hundreds of rows on every tick
        # made the list blink on and off. When the ONLY trigger is
        # the periodic tick and the signature is unchanged, skip the
        # redraw entirely. User actions (refresh button, selection
        # change) always re-render so highlights stay responsive.
        trig = callback_context.triggered_id
        if trig == "refresh-trigger" and sig == prev_sig:
            return no_update, no_update, no_update
        sel = list(selection or [])
        sel_count = len([f for f in sel
                          if f in {r["file_id"] for r in rows}])
        sel_label = (f"{sel_count} selected of "
                      f"{len(rows)} pending"
                      if sel_count
                      else f"{len(rows)} pending")
        return _render_pending_list(rows, sel), sel_label, sig

    @app.callback(
        Output("evtv-selection", "data",
                allow_duplicate=True),
        Input({"type": "evtv-row-check",
                "file_id": ALL}, "value"),
        State("evtv-selection", "data"),
        prevent_initial_call=True,
    )
    def _on_row_check(values, current):
        ctx = callback_context.triggered_id
        if not isinstance(ctx, dict):
            return no_update
        # Same recreation trap as _on_action: when the list re-renders,
        # every checkbox is recreated and the ALL-pattern callback fires
        # with many entries in `triggered` at once. A genuine user toggle
        # changes exactly one checkbox, so ignore multi-entry fires to
        # avoid churning the selection (which would re-trigger the list).
        if len(callback_context.triggered or []) != 1:
            return no_update
        file_id = int(ctx.get("file_id") or 0)
        # The triggered_id check tells us WHICH row was toggled.
        # Find its current value in the list.
        new_sel = list(int(x) for x in (current or []))
        # The values list mirrors the DOM order of every
        # evtv-row-check (ALL); we need the triggered one's
        # state. Use callback_context.triggered to get it.
        triggered = callback_context.triggered or []
        toggled_value = None
        for t in triggered:
            toggled_value = t.get("value") or []
        if toggled_value and file_id not in new_sel:
            new_sel.append(file_id)
        elif not toggled_value and file_id in new_sel:
            new_sel = [f for f in new_sel if f != file_id]
        return new_sel

    @app.callback(
        Output("evtv-selection", "data",
                allow_duplicate=True),
        Input("evtv-select-all-btn", "n_clicks"),
        Input("evtv-clear-sel-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_select_all(_a, _b):
        trig = callback_context.triggered_id
        if trig == "evtv-clear-sel-btn":
            return []
        rows = store.pi_pending_files(limit=500)
        return [int(r["file_id"]) for r in rows]

    @app.callback(
        Output("evtv-pending-list", "children",
                allow_duplicate=True),
        Output("evtv-selection", "data",
                allow_duplicate=True),
        Output("evtv-sel-status", "children",
                allow_duplicate=True),
        Input("evtv-approve-sel-btn", "n_clicks"),
        Input("evtv-flag-sel-btn", "n_clicks"),
        Input({"type": "evtv-approve-one-btn",
                "file_id": ALL}, "n_clicks"),
        Input({"type": "evtv-flag-one-btn",
                "file_id": ALL}, "n_clicks"),
        State("evtv-selection", "data"),
        State("evtv-flag-note", "value"),
        prevent_initial_call=True,
    )
    def _on_action(_bulk_a, _bulk_f, _one_a, _one_f,
                    selection, note):
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return no_update, no_update, no_update
        # CRITICAL: the per-row approve/flag buttons are pattern-
        # matching components recreated on every list re-render, which
        # re-fires this callback with n_clicks=None. Without this guard
        # a redraw silently approves a file (and self-triggers a loop
        # that approves one per cycle). Only proceed on a real click.
        if not _has_real_click(callback_context.triggered):
            return no_update, no_update, no_update
        trig = callback_context.triggered_id
        targets: list[int] = []
        action = None
        if isinstance(trig, dict):
            tt = trig.get("type")
            if tt == "evtv-approve-one-btn":
                targets = [int(trig.get("file_id"))]
                action = "approve"
            elif tt == "evtv-flag-one-btn":
                targets = [int(trig.get("file_id"))]
                action = "flag"
        elif trig == "evtv-approve-sel-btn":
            targets = [int(x) for x in (selection or [])]
            action = "approve"
        elif trig == "evtv-flag-sel-btn":
            targets = [int(x) for x in (selection or [])]
            action = "flag"
        if not targets or action is None:
            return no_update, no_update, no_update
        if action == "approve":
            n = store.pi_bulk_approve(targets, email)
            msg = (f"Approved {n} file"
                    f"{'' if n == 1 else 's'}.")
        else:
            n = store.pi_bulk_flag(targets, email,
                                     note=(note or ""))
            msg = (f"Flagged {n} file"
                    f"{'' if n == 1 else 's'}.")
        rows = store.pi_pending_files(limit=500)
        return (_render_pending_list(rows, []), [], msg)

    @app.callback(
        Output("evtv-detail-target", "data"),
        Input({"type": "evtv-open-detail",
                "file_id": ALL, "idx": ALL}, "n_clicks"),
        Input("evtv-close-detail-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_detail_open(_clicks, _close):
        trig = callback_context.triggered_id
        if trig == "evtv-close-detail-btn":
            return None
        if not isinstance(trig, dict):
            return no_update
        return {"file_id": int(trig.get("file_id")),
                 "idx": int(trig.get("idx"))}

    @app.callback(
        Output("evtv-detail-panel", "children"),
        Input("evtv-detail-target", "data"),
        Input("refresh-trigger", "data"),
        Input("evtv-detail-poll", "n_intervals"),
    )
    def _render_detail(target, _refresh, _tick):
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return []
        if not target:
            return []
        return _render_event_detail(target, store, config)

    # Auto-poll controller: enable the Interval ONLY while a
    # detail panel is open AND its clip is still extracting.
    # Saves bandwidth + DB queries the rest of the time.
    @app.callback(
        Output("evtv-detail-poll", "disabled"),
        Input("evtv-detail-target", "data"),
        Input("evtv-detail-poll", "n_intervals"),
    )
    def _toggle_detail_poll(target, _tick):
        if not target:
            return True
        # Cheap check: peek the clip status from the
        # cache/DB; only keep polling while pending/running.
        file_id = int(target.get("file_id") or 0)
        idx = int(target.get("idx") or 0)
        rows = store.pi_pending_files(limit=500)
        row = next((r for r in rows
                      if int(r["file_id"]) == file_id), None)
        if not row:
            return True
        events = row.get("events") or []
        if idx < 0 or idx >= len(events):
            return True
        ev = events[idx]
        eo = ev.get("EO_sec")
        bb = ev.get("BB_sec")
        if eo is None or bb is None:
            return True
        spec = _event_clip.ClipSpec(
            file_id=file_id, eo_sec=float(eo),
            bb_sec=float(bb))
        try:
            segs = _event_clip.resolve_clip_segments(
                spec, store)
            h = _event_clip.spec_hash(spec, segs)
            info = _event_clip.poll_video_clip_status(h, store)
        except Exception:
            return True  # silence the loop on lookup errors
        return info.get("status") not in ("pending", "running")

    @app.callback(
        Output("evtv-finalize-status", "children"),
        Output("evtv-overwrite-plan", "data"),
        Output("evtv-overwrite-modal", "style"),
        Input("evtv-finalize-csv-btn", "n_clicks"),
        State("evtv-overwrite-mode", "value"),
        State("evtv-overwrite-modal", "style"),
        prevent_initial_call=True,
    )
    def _on_finalize_click(n_clicks, ow_value, modal_style):
        """Two-phase: append mode writes immediately;
        overwrite mode does a dry-run first and surfaces the
        modal when any rows would be lost.
        """
        nope = (no_update, no_update, no_update)
        if not n_clicks:
            return nope
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return ("Not authorised.", None, no_update)
        overwrite = bool(ow_value and "ow" in ow_value)
        if not overwrite:
            # Append mode is non-destructive; write through.
            try:
                summary = _finalize_approved_to_csv(
                    store, config, overwrite=False)
            except Exception as e:
                logger.exception("Finalize (append) failed")
                return (f"Finalize failed: {e}",
                         None, no_update)
            return (summary, None, no_update)
        # Overwrite mode: dry-run + maybe modal.
        try:
            plan = _finalize_dry_run(store, config)
        except Exception as e:
            logger.exception("Finalize dry-run failed")
            return (f"Dry-run failed: {e}", None, no_update)
        if not plan:
            return ("Nothing to finalize -- no pi_approved "
                     "files.", None, no_update)
        any_removed = any(p.get("removed") for p in plan)
        if not any_removed:
            # Safe overwrite (no rows lost). Write through.
            try:
                summary = _finalize_approved_to_csv(
                    store, config, overwrite=True)
            except Exception as e:
                logger.exception(
                    "Finalize (safe-overwrite) failed")
                return (f"Finalize failed: {e}",
                         None, no_update)
            return (summary, None, no_update)
        # Destructive overwrite: stash the plan + show modal.
        new_style = dict(modal_style or {})
        new_style["display"] = "flex"
        return ("Confirm the destructive overwrite to write.",
                 plan, new_style)

    @app.callback(
        Output("evtv-overwrite-modal-body", "children"),
        Input("evtv-overwrite-plan", "data"),
    )
    def _render_modal_body(plan):
        return _render_overwrite_modal_body(plan or [])

    @app.callback(
        Output("evtv-finalize-status", "children",
                allow_duplicate=True),
        Output("evtv-overwrite-plan", "data",
                allow_duplicate=True),
        Output("evtv-overwrite-modal", "style",
                allow_duplicate=True),
        Input("evtv-overwrite-cancel-btn", "n_clicks"),
        Input("evtv-overwrite-confirm-btn", "n_clicks"),
        State("evtv-overwrite-modal", "style"),
        prevent_initial_call=True,
    )
    def _on_modal_action(_cancel, _confirm, modal_style):
        trig = callback_context.triggered_id
        new_style = dict(modal_style or {})
        new_style["display"] = "none"
        if trig == "evtv-overwrite-cancel-btn":
            return ("Overwrite cancelled. No CSV changes.",
                     None, new_style)
        if trig == "evtv-overwrite-confirm-btn":
            email = (current_user_email() or "").lower()
            if not _is_pi(config or {}, email):
                return ("Not authorised.", None, new_style)
            try:
                summary = _finalize_approved_to_csv(
                    store, config, overwrite=True)
            except Exception as e:
                logger.exception(
                    "Confirmed overwrite failed")
                return (f"Confirmed overwrite failed: {e}",
                         None, new_style)
            return (summary, None, new_style)
        return (no_update, no_update, no_update)

    # ---- Mass Analyze: 4 callbacks ---- #
    @app.callback(
        Output("evtv-ma-job-id", "data"),
        Output("evtv-ma-poll", "disabled"),
        Output("evtv-ma-progress", "children",
                allow_duplicate=True),
        Input("evtv-ma-scan-btn", "n_clicks"),
        State("evtv-ma-animal-dropdown", "value"),
        State("evtv-ma-cutoff-input", "value"),
        prevent_initial_call=True,
    )
    def _on_ma_scan(n_clicks, animal_id, cutoff):
        if not n_clicks:
            return no_update, no_update, no_update
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return None, True, "Not authorised."
        if not animal_id:
            return (None, True,
                     "Pick an animal first.")
        try:
            cutoff = float(cutoff)
        except (TypeError, ValueError):
            return (None, True,
                     f"Invalid cutoff: {cutoff!r}")
        if cutoff <= 0:
            return (None, True,
                     "Cutoff must be > 0.")
        try:
            job_id = _mass_analyze.create_job(
                store, email, animal_id, cutoff)
        except Exception as e:
            logger.exception("create_job failed")
            return (None, True, f"Failed to start: {e}")
        return job_id, False, "Scan queued…"

    @app.callback(
        Output("evtv-ma-progress", "children"),
        Output("evtv-ma-summary", "children"),
        Output("evtv-ma-confirm-row", "children"),
        Output("evtv-ma-poll", "disabled",
                allow_duplicate=True),
        Input("evtv-ma-poll", "n_intervals"),
        Input("evtv-ma-job-id", "data"),
        prevent_initial_call=True,
    )
    def _on_ma_poll(_n, job_id):
        if not job_id:
            return ("", "", [], True)
        job = _mass_analyze.get_job(store, int(job_id))
        if not job:
            return ("Job vanished.", "", [], True)
        status = job.get("status") or "?"
        scanned = int(job.get("scanned_files") or 0)
        total = int(job.get("total_files") or 0)
        n_zero = int(job.get("n_zero_peaks") or 0)
        n_with = int(job.get("n_with_peaks") or 0)
        progress = (
            f"{status}  ·  scanned {scanned}"
            + (f" of {total}" if total else "")
            + f"  ·  zero-peak: {n_zero}  ·  "
              f"≥1 peak: {n_with}"
        )
        summary = []
        confirm_row = []
        polling_disabled = status in ("done", "failed",
                                         "cancelled")
        if status == "done":
            summary = html.Div([
                html.Div([
                    html.Span("✓",
                               style={"color": "#30d158",
                                       "marginRight": "6px"}),
                    html.Span(
                        f"{n_zero} files with 0 peaks above "
                        f"{job['cutoff']:g}  ·  will mark as "
                        "no-events pending PI review",
                        style={"color": "#30d158",
                                "fontWeight": "600"}),
                ], style={"fontSize": "12px",
                           "marginBottom": "4px"}),
                html.Div([
                    html.Span("•",
                               style={"color": "#a0a0b0",
                                       "marginRight": "6px"}),
                    html.Span(
                        f"{n_with} files with ≥1 peak  ·  "
                        "stay in queue for normal review",
                        style={"color": "#a0a0b0"}),
                ], style={"fontSize": "12px"}),
            ])
            confirm_row = [
                html.Button(
                    f"Confirm threshold → mark {n_zero} files "
                    "as no-events pending PI review",
                    id="evtv-ma-confirm-btn", n_clicks=0,
                    style=_btn_style(accent=True)
                    if n_zero > 0
                    else _btn_style(secondary=True),
                    disabled=(n_zero == 0)),
                html.Button(
                    "Discard scan",
                    id="evtv-ma-discard-btn", n_clicks=0,
                    style=_btn_style(secondary=True)),
            ]
        elif status == "failed":
            err = job.get("error") or "unknown"
            summary = html.Div(f"Scan failed: {err}",
                                 style={"color": "#ff453a",
                                         "fontSize": "12px"})
        elif status == "cancelled":
            summary = html.Div(
                f"Cancelled at scanned {scanned} of {total}.",
                style={"color": "#ff9f0a",
                        "fontSize": "12px"})
        return (progress, summary, confirm_row,
                polling_disabled)

    @app.callback(
        Output("evtv-ma-job-id", "data",
                allow_duplicate=True),
        Output("evtv-ma-progress", "children",
                allow_duplicate=True),
        Output("evtv-ma-poll", "disabled",
                allow_duplicate=True),
        Input("evtv-ma-cancel-btn", "n_clicks"),
        State("evtv-ma-job-id", "data"),
        prevent_initial_call=True,
    )
    def _on_ma_cancel(n_clicks, job_id):
        if not n_clicks or not job_id:
            return no_update, no_update, no_update
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return no_update, "Not authorised.", no_update
        _mass_analyze.cancel_job(store, int(job_id))
        return no_update, "Cancel requested.", no_update

    @app.callback(
        Output("evtv-ma-progress", "children",
                allow_duplicate=True),
        Output("evtv-ma-summary", "children",
                allow_duplicate=True),
        Output("evtv-ma-confirm-row", "children",
                allow_duplicate=True),
        Output("evtv-ma-job-id", "data",
                allow_duplicate=True),
        Output("evtv-pending-list", "children",
                allow_duplicate=True),
        Input("evtv-ma-confirm-btn", "n_clicks"),
        Input("evtv-ma-discard-btn", "n_clicks"),
        State("evtv-ma-animal-dropdown", "value"),
        State("evtv-ma-cutoff-input", "value"),
        prevent_initial_call=True,
    )
    def _on_ma_commit(_confirm_n, _discard_n,
                       animal_id, cutoff):
        trig = callback_context.triggered_id
        if trig == "evtv-ma-discard-btn":
            return ("", "", [], None, no_update)
        if trig != "evtv-ma-confirm-btn":
            return (no_update, no_update, no_update,
                     no_update, no_update)
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return ("Not authorised.", "", [], None,
                     no_update)
        try:
            cutoff = float(cutoff)
        except (TypeError, ValueError):
            return (f"Invalid cutoff: {cutoff!r}",
                     "", [], None, no_update)
        try:
            result = _mass_analyze.commit_threshold(
                store, animal_id, cutoff, email)
        except Exception as e:
            logger.exception("commit_threshold failed")
            return (f"Commit failed: {e}", "", [],
                     None, no_update)
        msg = (f"✓ {result['n_cleared']} files moved to "
                "pending_pi_review. Approve them in the list "
                "below.")
        # Force a list re-render so the new pending files
        # show up immediately.
        rows = store.pi_pending_files(limit=500)
        return (msg, "", [], None,
                 _render_pending_list(rows, []))


# --------------------------------------------------------------- #
# Finalize: write all pi_approved rows to CSV
# --------------------------------------------------------------- #

def _finalize_dry_run(store, config: dict) -> list[dict]:
    """Return one row per (animal, day) group with the
    proposed CsvDiff. Caller decides whether to show the
    confirmation modal (only when any group's
    ``removed_filenames`` is non-empty).

    Each dict is JSON-safe so it can ride in a dcc.Store while
    the modal is open. Re-runs are cheap; the actual write
    happens on Confirm.
    """
    bhz_cfg = (config or {}).get("bhz_csv", {}) or {}
    if not bhz_cfg.get("enabled"):
        return []
    rows = _fetch_approved_rows(store)
    if not rows:
        return []
    bucket: dict[tuple[str, date], list[dict]] = {}
    for r in rows:
        animal, chunk_dt = _animal_and_date(store, r)
        if not chunk_dt:
            continue
        bucket.setdefault((animal, chunk_dt.date()), []).append(r)
    plan: list[dict] = []
    for (animal, day), items in sorted(bucket.items()):
        csv_path = _bhz_csv.resolve_csv_path(
            bhz_cfg.get("base_dir", ""),
            bhz_cfg.get("filename_template",
                         "{date}_{animal}.csv"),
            day, animal,
        )
        events_by_fn: dict[str, list[dict]] = {}
        meta_by_fn: dict[str, dict] = {}
        for r in items:
            file_meta = _file_meta_for_row(store, r, bhz_cfg)
            if file_meta is None:
                continue
            fn = file_meta["filename"]
            meta_by_fn[fn] = file_meta
            events_by_fn[fn] = r.get("events") or []
        try:
            diff = _bhz_csv.overwrite_day_csv(
                csv_path, events_by_fn, meta_by_fn,
                fs=_pick_fs(items), dry_run=True,
            )
        except Exception as e:
            logger.warning("dry_run failed for %s/%s: %s",
                            animal, day, e)
            continue
        plan.append({
            "animal": animal,
            "day": day.isoformat(),
            "csv_path": str(csv_path),
            "n_files": len(events_by_fn),
            "added": list(diff.added_filenames),
            "removed": list(diff.removed_filenames),
            "modified": list(diff.modified_filenames),
        })
    return plan


def _finalize_approved_to_csv(store, config: dict,
                                *, overwrite: bool) -> str:
    """Walk every ``status='pi_approved'`` row that hasn't yet
    been written to its (animal, day) CSV.

    Append mode: each file's events flow through
    ``bhz_csv.write_event_rows`` with the existing dedup-by-
    EventEO contract.

    Overwrite mode: groups events by (animal, day) and calls
    ``bhz_csv.overwrite_day_csv``. The destructive-overwrite
    confirmation is gated upstream by the modal; by the time
    this function runs, the PI has approved the diff.
    """
    bhz_cfg = (config or {}).get("bhz_csv", {}) or {}
    if not bhz_cfg.get("enabled"):
        return "BHZ CSV export disabled in config."
    # Pull every pi_approved row + its file meta.
    rows = _fetch_approved_rows(store)
    if not rows:
        return "Nothing to finalize -- no pi_approved files."
    bucket: dict[tuple[str, date], list[dict]] = {}
    for r in rows:
        animal, chunk_dt = _animal_and_date(store, r)
        if not chunk_dt:
            continue
        bucket.setdefault((animal, chunk_dt.date()), []).append(r)
    parts: list[str] = []
    for (animal, day), items in sorted(bucket.items()):
        csv_path = _bhz_csv.resolve_csv_path(
            bhz_cfg.get("base_dir", ""),
            bhz_cfg.get("filename_template",
                         "{date}_{animal}.csv"),
            day, animal,
        )
        events_by_fn: dict[str, list[dict]] = {}
        meta_by_fn: dict[str, dict] = {}
        for r in items:
            file_meta = _file_meta_for_row(store, r,
                                              bhz_cfg)
            if file_meta is None:
                continue
            fn = file_meta["filename"]
            meta_by_fn[fn] = file_meta
            events_by_fn[fn] = r.get("events") or []
        if overwrite:
            diff = _bhz_csv.overwrite_day_csv(
                csv_path, events_by_fn, meta_by_fn,
                fs=_pick_fs(items),
            )
            parts.append(
                f"{animal} {day}: "
                f"+{len(diff.added_filenames)} "
                f"−{len(diff.removed_filenames)} "
                f"~{len(diff.modified_filenames)}")
            if diff.removed_filenames:
                parts.append(
                    f"  ⚠ removed rows for: "
                    f"{', '.join(diff.removed_filenames)}")
        else:
            n_total = 0
            for fn, evs in events_by_fn.items():
                n_total += _bhz_csv.write_event_rows(
                    csv_path, meta_by_fn[fn], evs,
                    fs=meta_by_fn[fn].get("fs") or 20000.0,
                )
            parts.append(
                f"{animal} {day}: {n_total} row"
                f"{'' if n_total == 1 else 's'} appended")
    return "  •  ".join(parts)


def _fetch_approved_rows(store) -> list[dict]:
    """Every pi_approved review_state row + its deserialised
    events list."""
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT rs.id AS state_id, rs.file_id,
                      rs.user_email, rs.markers_json,
                      pf.file_path, pf.session_dir,
                      pf.chunk_datetime, pf.sampling_rate
               FROM review_state rs
               JOIN processed_files pf
                 ON pf.id = rs.file_id
               WHERE rs.status = 'pi_approved'
               ORDER BY rs.updated_at ASC"""
        ).fetchall()
    import json as _json
    out: list[dict] = []
    for r in rows:
        try:
            evs = _json.loads(r["markers_json"] or "[]")
        except _json.JSONDecodeError:
            evs = []
        d = dict(r)
        d["events"] = evs
        out.append(d)
    return out


def _animal_and_date(store, row: dict
                       ) -> tuple[str, datetime | None]:
    """Resolve (animal_prefix, chunk_datetime) for a row."""
    session_dir = row.get("session_dir") or ""
    animal = _animal_for_session(store, session_dir)
    chunk_dt = None
    raw = row.get("chunk_datetime") or ""
    try:
        chunk_dt = datetime.strptime(raw,
                                       "%Y_%m_%d__%H_%M_%S")
    except ValueError:
        pass
    return animal, chunk_dt


def _file_meta_for_row(store, row: dict, bhz_cfg: dict
                         ) -> dict | None:
    """Build a bhz_csv file_meta for one pi_approved row."""
    import os as _os
    fp = row.get("file_path") or ""
    if not fp:
        return None
    fs = float(row.get("sampling_rate") or 20000.0)
    chunk_dt_raw = row.get("chunk_datetime") or ""
    try:
        peak_dt = datetime.strptime(chunk_dt_raw,
                                      "%Y_%m_%d__%H_%M_%S")
    except ValueError:
        peak_dt = None
    return _bhz_csv.build_file_meta(
        folder=_os.path.dirname(fp) + _os.sep,
        filename=_os.path.basename(fp),
        fs=fs, cutoff=0.05, channel=2,
        peak_index=None, peak_stamp=None, peak_dt=peak_dt,
    )


def _pick_fs(rows: list[dict]) -> float:
    """Median fs across a group; falls back to 20 kHz."""
    fss = [float(r.get("sampling_rate") or 0) for r in rows]
    fss = [f for f in fss if f > 0]
    if not fss:
        return 20000.0
    return sorted(fss)[len(fss) // 2]
