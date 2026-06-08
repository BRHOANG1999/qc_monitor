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

from src.db.store import Store
from src.dashboard.auth import current_user_email
from src.dashboard.tabs.review_status import _is_pi
from src.utils import bhz_csv as _bhz_csv
from src.utils import event_clip as _event_clip
from src.utils.animal import split_animal_electrode, is_animal_channel

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
        # Detail panel's currently-open (file_id, event_idx)
        # or None. Surfaces under the list when set.
        dcc.Store(id="evtv-detail-target", data=None),
        # Detail view container -- populated by callback.
        html.Div(id="evtv-detail-panel",
                  style={"marginTop": "20px"}),
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
# File list rendering
# --------------------------------------------------------------- #

def _animal_for_session(store, session_dir: str) -> str:
    names = store._channel_names_for_session(session_dir)
    for n in names:
        if isinstance(n, str) and is_animal_channel(n):
            a, _ = split_animal_electrode(n)
            return a
    return "unknown"


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
        msg = ("Extracting clip… the PI tab will poll. "
                "Re-open this detail panel to refresh." if
                clip_status.get("status") in ("pending",
                                              "running")
                else f"Clip not ready: "
                      f"{clip_status.get('status')}")
        video_block = html.Div(
            msg,
            style={"color": "#a0a0b0",
                    "background": "#13131f",
                    "padding": "60px 20px",
                    "textAlign": "center",
                    "borderRadius": "8px"})
    return html.Div([
        header,
        video_block,
        html.Div("LFP stitch for this window renders in a "
                  "follow-up; the video clip is the primary "
                  "PI surface today.",
                  style={"color": "#666",
                          "fontStyle": "italic",
                          "fontSize": "11px",
                          "marginTop": "6px"}),
    ], style={"padding": "14px",
               "background": "rgba(94,124,226,0.04)",
               "border":
                   "1px solid rgba(94,124,226,0.18)",
               "borderRadius": "8px"})


# --------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------- #

def register_callbacks(app, store, config: dict) -> None:
    """Wire up the tab. Idempotent across reloads."""
    assert app is not None, "app required"

    @app.callback(
        Output("evtv-pending-list", "children"),
        Output("evtv-sel-status", "children"),
        Input("evtv-refresh-btn", "n_clicks"),
        Input("evtv-selection", "data"),
        Input("refresh-trigger", "data"),
    )
    def _render_list(_n, selection, _refresh):
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return [], ""
        rows = store.pi_pending_files(limit=500)
        sel = list(selection or [])
        sel_count = len([f for f in sel
                          if f in {r["file_id"] for r in rows}])
        sel_label = (f"{sel_count} selected of "
                      f"{len(rows)} pending"
                      if sel_count
                      else f"{len(rows)} pending")
        return _render_pending_list(rows, sel), sel_label

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
    )
    def _render_detail(target, _refresh):
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return []
        if not target:
            return []
        return _render_event_detail(target, store, config)

    @app.callback(
        Output("evtv-finalize-status", "children"),
        Input("evtv-finalize-csv-btn", "n_clicks"),
        State("evtv-overwrite-mode", "value"),
        prevent_initial_call=True,
    )
    def _on_finalize(n_clicks, ow_value):
        if not n_clicks:
            return no_update
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return "Not authorised."
        overwrite = bool(ow_value and "ow" in ow_value)
        try:
            summary = _finalize_approved_to_csv(
                store, config, overwrite=overwrite)
        except Exception as e:
            logger.exception("Finalize failed")
            return f"Finalize failed: {e}"
        return summary


# --------------------------------------------------------------- #
# Finalize: write all pi_approved rows to CSV
# --------------------------------------------------------------- #

def _finalize_approved_to_csv(store, config: dict,
                                *, overwrite: bool) -> str:
    """Walk every ``status='pi_approved'`` row that hasn't yet
    been written to its (animal, day) CSV.

    Append mode: each file's events flow through
    ``bhz_csv.write_event_rows`` with the existing dedup-by-
    EventEO contract.

    Overwrite mode: groups events by (animal, day) and calls
    ``bhz_csv.overwrite_day_csv``. Returns a summary that
    includes any rows that would be removed (the PI sees this
    in the toast and re-runs without overwrite if they want
    to back out).
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
    conn = store._connect()
    try:
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
    finally:
        conn.close()
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
