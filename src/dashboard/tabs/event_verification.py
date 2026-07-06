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

from dash import Input, Output, State, callback_context
from dash import dash_table, dcc, html, no_update

from src.db.store import Store
from src.dashboard import activity as _activity
from src.dashboard.auth import current_user_email
from src.dashboard.components import (
    DARK_TABLE_STYLE, DROPDOWN_STYLE, ZEBRA_STRIPE)
from src.dashboard.tabs.review_status import _is_pi
from src.utils import bhz_csv as _bhz_csv
from src.utils import mass_analyze as _mass_analyze
from src.utils.animal import split_animal_electrode, is_animal_channel

logger = logging.getLogger("qc_monitor.tabs.event_verification")

# Which of the scanned animal's electrodes the screen runs on (ordinal).
_ELECTRODE_OPTIONS = [
    {"label": "First electrode", "value": 0},
    {"label": "Second", "value": 1},
    {"label": "Third", "value": 2},
    {"label": "Fourth", "value": 3},
]


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
    return html.Div([
        html.H3("Event Verification",
                 style={"color": "#f0f0f5",
                         "marginBottom": "4px"}),
        html.Div(
            "PI review gate. Tick rows and Approve to flip them "
            "to 'pi_approved' (queues a CSV write); Flag sends "
            "them back to the undergrad's queue with a note. "
            "Click a row's Open to inspect the recording in "
            "Video Review.",
            style={"color": "#a0a0b0", "fontSize": "12px",
                    "marginBottom": "16px"}),
        # Mass Analyze panel (collapsed-by-default details
        # card). Runs hilbert_envelope_20_200 + peakseek over
        # every pending file for an animal at the chosen
        # cutoff; zero-peak files get auto-cleared to
        # pending_pi_review on Confirm.
        _mass_analyze_panel(store),
        # Screen-comparison benchmark: run both screens over the
        # animal's human-labeled files and compare TP/FP/TN/FN.
        _screen_compare_panel(store),
        # Per-animal auto-filter thresholds (background sweep).
        _auto_filter_panel(store),
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
                "Approve ALL pending",
                id="evtv-approve-all-btn", n_clicks=0,
                style=_btn_style(accent=True),
                title="Approve every pending file across all pages "
                      "(not just the visible/selected ones), so the "
                      "finalize CSV covers all of them."),
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
        html.Div(
            "Each CSV row's onset time is in EventEO (sample), "
            "EventEO_WallClock (date + time) and EO_HourOfDay (0-23.99, for "
            "circadian binning). Peak_Index / Peak_Stamp are blank for "
            "manually-scored events (they come only from the automated "
            "detector).",
            style={"color": "#808090", "fontSize": "11px",
                   "marginBottom": "10px", "maxWidth": "680px"}),
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
        # Pending files table. Native multi-select (clientside, no
        # server redraw on select -> no flash), native sort by
        # Animal/Date, paginated for large backlogs. Each row's id
        # is the file_id so selection survives sort/paging. Click a
        # row's "Open" cell to inspect it in Video Review.
        dcc.Loading(
            id="evtv-pending-loading",
            type="circle",
            color="#5e7ce2",
            delay_show=180,
            children=dash_table.DataTable(
                id="evtv-pending-table",
                columns=[
                    {"name": "Animal", "id": "animal"},
                    {"name": "Date", "id": "date"},
                    {"name": "Events", "id": "n_events",
                     "type": "numeric"},
                    {"name": "Submitter", "id": "submitter"},
                    {"name": "", "id": "view"},
                ],
                data=[],
                row_selectable="multi",
                sort_action="native",
                sort_by=[{"column_id": "date",
                           "direction": "asc"}],
                page_action="native",
                page_size=25,
                cell_selectable=True,
                style_as_list_view=True,
                **DARK_TABLE_STYLE,
                style_data_conditional=[
                    ZEBRA_STRIPE,
                    {"if": {"column_id": "view"},
                     "color": "#5e7ce2", "cursor": "pointer",
                     "fontWeight": "600"},
                ],
            ),
        ),
        # Signature of the currently-rendered pending set so the
        # 10s auto-refresh can skip pushing identical data.
        dcc.Store(id="evtv-list-sig", data=None),
        # Flagged & scored -> submit section (needs_scoring pool with onsets).
        _flagged_scored_section(),
    ], style={"padding": "20px 24px"})


def _flagged_scored_section() -> html.Details:
    """Collapsible section: the needs_scoring (flagged) files that carry an EEG
    onset, with a Scored (onset+Racine) flag. The PI ticks the scored ones and
    Submits -> they flip to pi_approved and their events are written to the
    per-animal day CSV (reusing the finalize path). Files still missing a
    Racine stay in Needs scoring."""
    return html.Details([
        html.Summary(
            "Flagged & scored — ready to submit",
            style={"cursor": "pointer", "color": "#f0f0f5",
                   "fontSize": "14px", "fontWeight": "700",
                   "padding": "8px 0"}),
        html.Div(
            "Files you flagged with an EEG onset. A file is ready when every "
            "onset also has a Racine score ('✓ scored'); onset-without-Racine "
            "stays in Needs scoring. Tick the scored ones and Submit — they "
            "flip to pi_approved and their onsets are written to the per-animal "
            "day CSV. Click Open to finish scoring one in Video Review.",
            style={"color": "#a0a0b0", "fontSize": "12px",
                   "margin": "4px 0 12px", "maxWidth": "720px"}),
        html.Div([
            html.Button("Select all scored", id="evtv-fs-select-scored-btn",
                        n_clicks=0, style=_btn_style()),
            html.Button("Clear selection", id="evtv-fs-clear-btn",
                        n_clicks=0, style=_btn_style(secondary=True)),
            html.Button("Submit selected → CSV", id="evtv-fs-submit-btn",
                        n_clicks=0, style=_btn_style(accent=True),
                        title="Approve the selected scored files and write "
                              "their onsets to the per-animal BHZ day CSV."),
            html.Div(id="evtv-fs-status",
                     style={"color": "#a0a0b0", "fontSize": "11px",
                            "marginLeft": "12px"}),
        ], style={"display": "flex", "gap": "8px", "alignItems": "center",
                  "marginBottom": "10px", "flexWrap": "wrap"}),
        # Bulk re-attribution: fix multi-animal mis-filing. Tick rows filed
        # under the wrong animal, pick the correct one, Move. Only animals
        # actually in each recording are accepted (others are skipped).
        html.Div([
            html.Span("Re-attribute:",
                      style={"color": "#cfd0d6", "fontSize": "12px",
                             "fontWeight": "600"}),
            html.Span("move selected onsets to",
                      style={"color": "#a0a0b0", "fontSize": "11px"}),
            dcc.Dropdown(id="evtv-fs-move-animal", options=[], value=None,
                         placeholder="animal", clearable=False,
                         style={**DROPDOWN_STYLE, "minWidth": "140px"},
                         className="dark-dropdown"),
            html.Button("Move selected", id="evtv-fs-move-btn", n_clicks=0,
                        style=_btn_style(warning=True),
                        title="Re-file the selected onsets under the chosen "
                              "animal (must be one of that recording's "
                              "animals). Retires the mis-filed row."),
        ], style={"display": "flex", "gap": "8px", "alignItems": "center",
                  "marginBottom": "10px", "flexWrap": "wrap"}),
        html.Div(id="evtv-fs-finalize-status",
                 style={"color": "#a0a0b0", "fontSize": "12px",
                        "minHeight": "16px", "marginBottom": "10px"}),
        dcc.Loading(
            type="circle", color="#5e7ce2", delay_show=180,
            children=dash_table.DataTable(
                id="evtv-flagged-table",
                columns=[
                    {"name": "Animal", "id": "animal"},
                    {"name": "Date", "id": "date"},
                    {"name": "Onsets", "id": "n_events", "type": "numeric"},
                    {"name": "Status", "id": "scored"},
                    {"name": "Flagged by", "id": "submitter"},
                    {"name": "", "id": "view"},
                ],
                data=[], row_selectable="multi", sort_action="native",
                sort_by=[{"column_id": "scored", "direction": "asc"}],
                page_action="native", page_size=25, cell_selectable=True,
                style_as_list_view=True, **DARK_TABLE_STYLE,
                style_data_conditional=[
                    ZEBRA_STRIPE,
                    {"if": {"filter_query": "{scored} = '✓ scored'",
                            "column_id": "scored"},
                     "color": "#30d158", "fontWeight": "600"},
                    {"if": {"filter_query": "{scored} = 'needs Racine'",
                            "column_id": "scored"},
                     "color": "#ff9f0a"},
                    {"if": {"column_id": "view"},
                     "color": "#5e7ce2", "cursor": "pointer",
                     "fontWeight": "600"},
                ],
            ),
        ),
        dcc.Store(id="evtv-fs-sig", data=None),
    ], open=False, style={"marginTop": "20px",
                          "borderTop": "1px solid #2a2a3a",
                          "paddingTop": "12px"})


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
# Per-animal auto-filter thresholds
# --------------------------------------------------------------- #

_AF_NUM = {"width": "92px", "background": "#262638", "color": "#f0f0f5",
           "border": "1px solid rgba(255,255,255,0.15)",
           "borderRadius": "4px", "padding": "4px 6px", "fontSize": "12px"}
_AF_LAB = {"color": "#a0a0b0", "fontSize": "12px"}
_AF_COLUMNS = [
    {"name": "Animal", "id": "animal_id"},
    {"name": "Envelope", "id": "peak_cutoff"},
    {"name": "AUC", "id": "auc_threshold"},
    {"name": "Win(s)", "id": "auc_window_sec"},
    {"name": "Elec", "id": "electrode"},
    {"name": "Screens", "id": "screen_mode"},
    {"name": "On", "id": "enabled"},
    {"name": "Auto-cleared", "id": "auto_cleared", "type": "numeric"},
    {"name": "Updated by", "id": "updated_by"},
    {"name": "Updated", "id": "updated_at"},
]


def _af_table_rows(store) -> list[dict]:
    """Saved per-animal auto-filter configs + each one's auto-cleared count."""
    out = []
    for c in store.list_animal_screen_configs():
        out.append({
            "animal_id": c["animal_id"],
            "peak_cutoff": c.get("peak_cutoff"),
            "auc_threshold": c.get("auc_threshold"),
            "auc_window_sec": c.get("auc_window_sec"),
            "electrode": c.get("electrode"),
            "screen_mode": c.get("screen_mode"),
            "enabled": "yes" if c.get("enabled") else "no",
            "auto_cleared": store.count_auto_filter_cleared(c["animal_id"]),
            "updated_by": c.get("updated_by") or "",
            "updated_at": (c.get("updated_at") or "")[:16].replace("T", " "),
        })
    return out


def _auto_filter_panel(store) -> html.Details:
    """PI sets a per-animal envelope/AUC threshold + which screens apply; a
    background sweep auto-clears pending files that cross none of them."""
    animals = sorted(store.list_all_animals())
    dd = {"width": "150px"}
    return html.Details([
        html.Summary("⚙  Per-animal auto-filter thresholds",
                     style={"cursor": "pointer", "fontWeight": "600",
                            "color": "#cfd0d6", "fontSize": "13px",
                            "padding": "6px 0"}),
        html.Div(
            "Set a raw-envelope cutoff and/or AUC threshold per animal. A "
            "background sweep marks any of that animal's pending files that "
            "cross NONE of the active screens as pending_pi_review (you still "
            "rubber-stamp them below). 'Both' = keep if it crosses EITHER "
            "screen. Animals with no saved+enabled threshold are never "
            "auto-filtered.",
            style={"color": "#8a8d99", "fontSize": "11px",
                   "margin": "4px 0 10px", "maxWidth": "760px"}),
        html.Div([
            html.Span("Animal:", style=_AF_LAB),
            dcc.Dropdown(id="evtv-af-animal",
                         options=[{"label": a, "value": a} for a in animals],
                         style=dd, className="dark-dropdown"),
            html.Span("Envelope cutoff:", style=_AF_LAB),
            dcc.Input(id="evtv-af-cutoff", type="number", step="any",
                      style=_AF_NUM),
            html.Span("AUC thresh:", style=_AF_LAB),
            dcc.Input(id="evtv-af-auc-thr", type="number", step="any",
                      style=_AF_NUM),
            html.Span("AUC win (s):", style=_AF_LAB),
            dcc.Input(id="evtv-af-auc-win", type="number", value=5,
                      step="any", style=_AF_NUM),
            html.Span("Electrode:", style=_AF_LAB),
            dcc.Dropdown(id="evtv-af-electrode", options=_ELECTRODE_OPTIONS,
                         value=0, style={"width": "78px"},
                         className="dark-dropdown"),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px",
                  "flexWrap": "wrap", "marginBottom": "8px"}),
        html.Div([
            html.Span("Screens:", style=_AF_LAB),
            dcc.RadioItems(id="evtv-af-mode",
                           options=[{"label": " Both (OR)", "value": "both"},
                                    {"label": " Envelope", "value": "envelope"},
                                    {"label": " AUC", "value": "auc"}],
                           value="both", inline=True,
                           labelStyle={"color": "#ddd", "fontSize": "12px",
                                       "marginRight": "12px"},
                           inputStyle={"marginRight": "4px"}),
            dcc.Checklist(id="evtv-af-enabled",
                          options=[{"label": " Enabled", "value": "on"}],
                          value=["on"], style={"display": "inline-block"},
                          labelStyle={"color": "#ddd", "fontSize": "12px"}),
            html.Button("Save", id="evtv-af-save-btn", n_clicks=0,
                        style=_btn_style(accent=True)),
            html.Button("Sweep now", id="evtv-af-sweep-btn", n_clicks=0,
                        style=_btn_style(secondary=True)),
            html.Span(id="evtv-af-status",
                      style={"color": "#a0a0b0", "fontSize": "12px",
                             "marginLeft": "8px"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px",
                  "flexWrap": "wrap", "marginBottom": "10px"}),
        dash_table.DataTable(
            id="evtv-af-table", columns=_AF_COLUMNS,
            data=_af_table_rows(store), page_size=10, sort_action="native",
            style_as_list_view=True, **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE]),
    ], style={"border": "1px solid rgba(255,255,255,0.08)",
              "borderRadius": "6px", "padding": "8px 12px",
              "marginBottom": "16px"})


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
                    type="number", min=0, step="any",
                    value=0.05,
                    style={"flex": "0 0 110px",
                            "padding": "6px 10px",
                            "background": "#262638",
                            "color": "#f0f0f5",
                            "border":
                                "1px solid rgba(255,255,255,0.10)",
                            "borderRadius": "4px",
                            "fontSize": "12px"}),
                html.Label("Electrode:",
                            style={"color": "#a0a0b0",
                                    "fontSize": "12px",
                                    "marginLeft": "14px",
                                    "marginRight": "6px"}),
                dcc.Dropdown(
                    id="evtv-ma-electrode",
                    options=_ELECTRODE_OPTIONS, value=0,
                    clearable=False,
                    style={"flex": "0 0 140px", "minWidth": "120px"},
                    className="dark-dropdown"),
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


def _screen_compare_panel(store) -> html.Details:
    """Benchmark the two file-screens against human labels.

    Runs the envelope screen (peak height) and the AUC screen
    (sliding-window area) over every file this animal has a human
    verdict for, then reports each screen's confusion matrix +
    sensitivity / false-positive rate so the PI can pick the
    screen that wastes the least review time."""
    animals = sorted(store.list_all_animals())
    num_style = {"flex": "0 0 110px", "padding": "6px 10px",
                  "background": "#262638", "color": "#f0f0f5",
                  "border": "1px solid rgba(255,255,255,0.10)",
                  "borderRadius": "4px", "fontSize": "12px"}
    lab = {"color": "#a0a0b0", "fontSize": "12px",
            "marginLeft": "14px", "marginRight": "6px"}
    return html.Details([
        html.Summary([
            html.Span("Compare screening methods ",
                       style={"color": "#f0f0f5",
                               "fontWeight": "600"}),
            html.Span("(envelope vs AUC, scored on human labels)",
                       style={"color": "#a0a0b0",
                               "fontSize": "11px",
                               "marginLeft": "6px"}),
        ], style={"cursor": "pointer", "marginBottom": "8px"}),
        html.Div([
            html.Div(
                "Runs both screens over every file with a human "
                "verdict (has-events / no-events / PI-approved) "
                "for this animal, then compares each screen's "
                "true/false positives. The AUC screen integrates "
                "the envelope over a sliding window, so it should "
                "raise fewer false positives on transient noise "
                "while keeping the real events. Set the AUC "
                "threshold after eyeballing the Hilbert-AUC trace "
                "in Video Review.",
                style={"color": "#a0a0b0", "fontSize": "11px",
                        "padding": "8px 12px",
                        "background": "rgba(94,124,226,0.04)",
                        "border": "1px solid rgba(94,124,226,0.18)",
                        "borderRadius": "6px",
                        "marginBottom": "10px"}),
            html.Div([
                html.Label("Animal:",
                            style={"color": "#a0a0b0",
                                    "fontSize": "12px",
                                    "marginRight": "6px"}),
                dcc.Dropdown(
                    id="evtv-sc-animal-dropdown",
                    options=[{"label": a, "value": a}
                             for a in animals],
                    placeholder="Pick an animal",
                    style={"flex": "1 1 200px", "minWidth": "180px"},
                    className="dark-dropdown",
                ),
                html.Label("Peak cutoff:", style=lab),
                dcc.Input(id="evtv-sc-cutoff-input", type="number",
                           min=0, step="any", value=0.05,
                           style=num_style),
                html.Label("AUC thresh:", style=lab),
                dcc.Input(id="evtv-sc-auc-threshold-input",
                           type="number", min=0, step="any",
                           placeholder="e.g. 0.5", style=num_style),
                html.Label("AUC win (s):", style=lab),
                dcc.Input(id="evtv-sc-auc-window-input",
                           type="number", min=0, step="any",
                           value=5, style={**num_style,
                                            "flex": "0 0 80px"}),
                html.Label("Electrode:", style=lab),
                dcc.Dropdown(
                    id="evtv-sc-electrode",
                    options=_ELECTRODE_OPTIONS, value=0,
                    clearable=False,
                    style={"flex": "0 0 140px", "minWidth": "120px"},
                    className="dark-dropdown"),
                html.Button("Compare screens",
                             id="evtv-sc-run-btn", n_clicks=0,
                             style=_btn_style(accent=True)),
                html.Button("Cancel",
                             id="evtv-sc-cancel-btn", n_clicks=0,
                             style=_btn_style(secondary=True)),
            ], style={"display": "flex", "alignItems": "center",
                       "gap": "8px", "flexWrap": "wrap",
                       "marginBottom": "10px"}),
            html.Div(id="evtv-sc-progress",
                      style={"color": "#a0a0b0", "fontSize": "12px",
                              "minHeight": "16px",
                              "marginBottom": "8px"}),
            html.Div(id="evtv-sc-results",
                      style={"marginBottom": "10px"}),
            dcc.Store(id="evtv-sc-job-id", data=None),
            dcc.Interval(id="evtv-sc-poll", interval=1500,
                           n_intervals=0, disabled=True),
        ]),
    ], open=False,
       style={"padding": "12px 14px", "background": "#13131f",
               "border": "1px solid rgba(255,255,255,0.06)",
               "borderRadius": "8px", "marginBottom": "14px"})


def _screen_compare_table(job: dict) -> html.Div:
    """Render the two-screen confusion comparison from a done
    screen_eval_job row."""
    def _rate(num: int, den: int) -> str:
        return f"{num / den:.3f}" if den else "--"
    p1 = {k: int(job.get(f"p1_{k}") or 0)
          for k in ("tp", "fp", "tn", "fn")}
    p2 = {k: int(job.get(f"p2_{k}") or 0)
          for k in ("tp", "fp", "tn", "fn")}
    n_labeled = int(job.get("scanned_files") or 0)
    rows = [
        {"metric": "Files labeled", "env": n_labeled, "auc": n_labeled},
        {"metric": "True positives (events caught)",
         "env": p1["tp"], "auc": p2["tp"]},
        {"metric": "False positives (event-free flagged)",
         "env": p1["fp"], "auc": p2["fp"]},
        {"metric": "True negatives", "env": p1["tn"], "auc": p2["tn"]},
        {"metric": "False negatives (events missed)",
         "env": p1["fn"], "auc": p2["fn"]},
        {"metric": "Sensitivity TP/(TP+FN)",
         "env": _rate(p1["tp"], p1["tp"] + p1["fn"]),
         "auc": _rate(p2["tp"], p2["tp"] + p2["fn"])},
        {"metric": "False-positive rate FP/(FP+TN)",
         "env": _rate(p1["fp"], p1["fp"] + p1["tn"]),
         "auc": _rate(p2["fp"], p2["fp"] + p2["tn"])},
    ]
    eot = int(job.get("env_only_true") or 0)
    eof = int(job.get("env_only_false") or 0)
    aot = int(job.get("auc_only_true") or 0)
    aof = int(job.get("auc_only_false") or 0)
    return html.Div([
        dash_table.DataTable(
            data=rows,
            columns=[{"name": "Metric", "id": "metric"},
                      {"name": "Envelope screen", "id": "env"},
                      {"name": "AUC screen", "id": "auc"}],
            **DARK_TABLE_STYLE,
        ),
        html.Div([
            html.Div("Disagreement (flagged by exactly one screen):",
                      style={"color": "#cfd0d6", "fontSize": "12px",
                              "fontWeight": "600",
                              "margin": "10px 0 4px"}),
            html.Div(
                f"Envelope-only: {eot + eof}  "
                f"({eot} real events, {eof} event-free false alarms)  ·  "
                f"AUC-only: {aot + aof}  "
                f"({aot} real events, {aof} event-free false alarms)",
                style={"color": "#a0a0b0", "fontSize": "12px"}),
        ]),
    ])


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


def _format_chunk_dt(raw: str | None) -> str:
    """KMrecorder chunk_datetime (``YYYY_MM_DD__HH_MM_SS``) to a
    sortable human label. Falls back to the raw string."""
    raw = raw or ""
    try:
        ts = datetime.strptime(raw, "%Y_%m_%d__%H_%M_%S")
        return ts.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return raw


def _pending_table_rows(store, rows: list[dict]) -> list[dict]:
    """One DataTable record per pending (file, animal). ``id`` is the
    review_state state_id (unique per (file, animal)) so native selection
    (selected_row_ids) survives sort + pagination AND distinguishes the
    animals of a shared recording. ``file_id`` is carried for "Open >"."""
    out: list[dict] = []
    max_iter = len(rows) + 1
    for i, r in enumerate(rows):
        assert i < max_iter, "row scan runaway"
        animal = r.get("animal_id") or _animal_for_session(
            store, r["session_dir"])
        out.append({
            "id": int(r["state_id"]),
            "file_id": int(r["file_id"]),
            "animal": animal,
            "date": _format_chunk_dt(r.get("chunk_datetime")),
            "n_events": len(r.get("events") or []),
            "submitter": r.get("user_email") or "-",
            "view": "Open >",
        })
    return out


def _row_is_scored(events) -> bool:
    """A flagged file is 'scored' (submittable) when it has at least one event
    with an EEG onset AND every onset event also carries a Racine score. Onset
    without Racine => still needs scoring (per the review contract)."""
    onset_evs = [e for e in (events or [])
                 if e.get("EO_sec") not in (None, "")]
    return bool(onset_evs) and all(
        e.get("racine") is not None for e in onset_evs)


def _flagged_scored_rows(store) -> list[dict]:
    """Rows for the 'Flagged & scored — ready to submit' table: every
    needs_scoring (flagged) file that has at least one EEG onset, with a Scored
    flag (onset+Racine). ``id`` = ``"<file_id>:<animal>"`` so multi-select
    survives sort/paging and distinguishes a shared recording's animals."""
    try:
        rows = store.flagged_files(statuses=("needs_scoring",), limit=500)
    except Exception as e:  # noqa: BLE001
        logger.warning("flagged_scored_rows failed: %s", e)
        return []
    out: list[dict] = []
    for r in rows:
        events = r.get("events") or []
        onset_evs = [e for e in events if e.get("EO_sec") not in (None, "")]
        if not onset_evs:
            continue                      # no onset dropped yet -> not "flagged"
        animal = r.get("animal") or r.get("animal_id") or ""
        scored = _row_is_scored(events)
        out.append({
            "id": f"{int(r['file_id'])}:{animal}",
            "file_id": int(r["file_id"]),
            "animal": animal,
            "date": _format_chunk_dt(r.get("chunk_datetime")),
            "n_events": len(onset_evs),
            "scored": "✓ scored" if scored else "needs Racine",
            "submitter": r.get("user_email") or "-",
            "view": "Open >",
        })
    # Ready-to-submit (scored) first, then oldest recording first.
    out.sort(key=lambda x: (x["scored"] != "✓ scored", x["date"]))
    return out


# --------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------- #

def register_callbacks(app, store, config: dict) -> None:
    """Wire up the tab. Idempotent across reloads."""
    assert app is not None, "app required"

    @app.callback(
        Output("evtv-pending-table", "data"),
        Output("evtv-sel-status", "children"),
        Output("evtv-list-sig", "data"),
        Input("evtv-refresh-btn", "n_clicks"),
        Input("refresh-trigger", "data"),
        State("evtv-list-sig", "data"),
    )
    def _render_list(_n, _refresh, prev_sig):
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return [], "", no_update
        rows = store.pi_pending_files(limit=500)
        sig = _pending_signature(rows)
        # The 10s refresh-trigger fires whether or not the pending
        # set changed. Skip pushing identical data so the table
        # doesn't churn. User-driven refresh always rebuilds.
        trig = callback_context.triggered_id
        if trig == "refresh-trigger" and sig == prev_sig:
            return no_update, no_update, no_update
        return (_pending_table_rows(store, rows),
                f"{len(rows)} pending", sig)

    @app.callback(
        Output("evtv-pending-table", "selected_row_ids",
                allow_duplicate=True),
        Input("evtv-select-all-btn", "n_clicks"),
        Input("evtv-clear-sel-btn", "n_clicks"),
        State("evtv-pending-table", "data"),
        prevent_initial_call=True,
    )
    def _on_select_all(_a, _b, data):
        trig = callback_context.triggered_id
        if trig == "evtv-clear-sel-btn":
            return []
        return [r["id"] for r in (data or [])]

    @app.callback(
        Output("evtv-pending-table", "data",
                allow_duplicate=True),
        Output("evtv-pending-table", "selected_row_ids",
                allow_duplicate=True),
        Output("evtv-sel-status", "children",
                allow_duplicate=True),
        Output("evtv-list-sig", "data",
                allow_duplicate=True),
        Input("evtv-approve-sel-btn", "n_clicks"),
        Input("evtv-approve-all-btn", "n_clicks"),
        Input("evtv-flag-sel-btn", "n_clicks"),
        State("evtv-pending-table", "selected_row_ids"),
        State("evtv-flag-note", "value"),
        prevent_initial_call=True,
    )
    def _on_action(_bulk_a, _all_a, _bulk_f, selected_ids, note):
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return no_update, no_update, no_update, no_update
        # Guard against the bulk buttons firing with no real click
        # (cheap; the per-row pattern buttons that used to cause
        # silent approvals are gone entirely now).
        if not _has_real_click(callback_context.triggered):
            return no_update, no_update, no_update, no_update
        trig = callback_context.triggered_id
        # "Approve ALL pending" ignores the table selection/pagination.
        if trig == "evtv-approve-all-btn":
            n = store.pi_approve_all_pending(email)
            msg = f"Approved ALL {n} pending file{'' if n == 1 else 's'}."
        else:
            targets = [int(x) for x in (selected_ids or [])]
            if not targets:
                return (no_update, no_update,
                         "Tick at least one row first.", no_update)
            if trig == "evtv-approve-sel-btn":
                n = store.pi_bulk_approve(targets, email)
                msg = f"Approved {n} file{'' if n == 1 else 's'}."
            elif trig == "evtv-flag-sel-btn":
                n = store.pi_bulk_flag(targets, email,
                                         note=(note or ""))
                msg = f"Flagged {n} file{'' if n == 1 else 's'}."
            else:
                return no_update, no_update, no_update, no_update
        rows = store.pi_pending_files(limit=500)
        return (_pending_table_rows(store, rows), [], msg,
                _pending_signature(rows))

    # ---- Flagged & scored -> submit section ---- #
    @app.callback(
        Output("evtv-flagged-table", "data"),
        Output("evtv-fs-sig", "data"),
        Output("evtv-fs-move-animal", "options"),
        Input("evtv-refresh-btn", "n_clicks"),
        Input("refresh-trigger", "data"),
        State("evtv-fs-sig", "data"),
    )
    def _render_flagged(_n, _refresh, prev_sig):
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return [], no_update, no_update
        rows = _flagged_scored_rows(store)
        sig = "|".join(f"{r['id']}={r['scored']}" for r in rows)
        if callback_context.triggered_id == "refresh-trigger" \
                and sig == prev_sig:
            return no_update, no_update, no_update
        # Re-attribution targets = every animal present in the flagged
        # recordings (so a mis-filed onset can be moved to the right one).
        animals: set = set()
        try:
            for r in store.flagged_files(statuses=("needs_scoring",),
                                          limit=500):
                for a in store.session_animals(r.get("session_dir") or ""):
                    animals.add(a)
        except Exception:  # noqa: BLE001
            pass
        opts = [{"label": a, "value": a} for a in sorted(animals)]
        return rows, sig, opts

    @app.callback(
        Output("evtv-flagged-table", "selected_row_ids",
                allow_duplicate=True),
        Input("evtv-fs-select-scored-btn", "n_clicks"),
        Input("evtv-fs-clear-btn", "n_clicks"),
        State("evtv-flagged-table", "data"),
        prevent_initial_call=True,
    )
    def _fs_select(_a, _b, data):
        if callback_context.triggered_id == "evtv-fs-clear-btn":
            return []
        return [r["id"] for r in (data or [])
                if r.get("scored") == "✓ scored"]

    @app.callback(
        Output("evtv-flagged-table", "data", allow_duplicate=True),
        Output("evtv-flagged-table", "selected_row_ids",
                allow_duplicate=True),
        Output("evtv-fs-status", "children"),
        Output("evtv-fs-finalize-status", "children"),
        Output("evtv-fs-sig", "data", allow_duplicate=True),
        Input("evtv-fs-submit-btn", "n_clicks"),
        State("evtv-flagged-table", "selected_row_ids"),
        prevent_initial_call=True,
    )
    def _fs_submit(_n, selected_ids):
        email = (current_user_email() or "").lower()
        blank = (no_update,) * 5
        if not _is_pi(config or {}, email) \
                or not _has_real_click(callback_context.triggered):
            return blank
        if not selected_ids:
            return (no_update, no_update,
                    "Tick at least one ✓ scored file first.",
                    no_update, no_update)
        # Re-fetch the flagged pool as the source of truth for the events.
        by_id: dict = {}
        for r in store.flagged_files(statuses=("needs_scoring",), limit=1000):
            animal = r.get("animal") or r.get("animal_id") or ""
            by_id[f"{int(r['file_id'])}:{animal}"] = r
        submitted = skipped = 0
        for sid in (selected_ids or []):
            r = by_id.get(sid)
            if not r:
                continue
            events = r.get("events") or []
            if not _row_is_scored(events):
                skipped += 1           # onset without Racine -> stays flagged
                continue
            animal = r.get("animal") or r.get("animal_id") or None
            try:
                store.mark_review(int(r["file_id"]), email, "pi_approved",
                                  markers=events, animal_id=animal)
                submitted += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("flagged submit mark_review failed: %s", e)
        fin = ""
        if submitted:
            fin = _finalize_approved_to_csv(store, config, overwrite=False)
        msg = f"Submitted {submitted} file(s) → pi_approved + CSV."
        if skipped:
            msg += (f" Skipped {skipped} still missing a Racine "
                    "(left in Needs scoring).")
        rows = _flagged_scored_rows(store)
        sig = "|".join(f"{x['id']}={x['scored']}" for x in rows)
        return rows, [], msg, fin, sig

    @app.callback(
        Output("evtv-flagged-table", "data", allow_duplicate=True),
        Output("evtv-flagged-table", "selected_row_ids",
                allow_duplicate=True),
        Output("evtv-fs-status", "children", allow_duplicate=True),
        Output("evtv-fs-sig", "data", allow_duplicate=True),
        Input("evtv-fs-move-btn", "n_clicks"),
        State("evtv-flagged-table", "selected_row_ids"),
        State("evtv-fs-move-animal", "value"),
        prevent_initial_call=True,
    )
    def _fs_move(_n, selected_ids, to_animal):
        email = (current_user_email() or "").lower()
        blank = (no_update,) * 4
        if not _is_pi(config or {}, email) \
                or not _has_real_click(callback_context.triggered):
            return blank
        if not selected_ids:
            return (no_update, no_update,
                    "Tick at least one row to re-attribute.", no_update)
        if not to_animal:
            return (no_update, no_update,
                    "Pick a target animal first.", no_update)
        moved = skipped = 0
        for sid in (selected_ids or []):
            # Row id is "<file_id>:<from_animal>".
            try:
                fid_s, from_animal = str(sid).split(":", 1)
                fid = int(fid_s)
            except (ValueError, AttributeError):
                skipped += 1
                continue
            try:
                ok = store.reattribute_review(fid, from_animal, to_animal,
                                              email)
            except Exception as e:  # noqa: BLE001
                logger.warning("reattribute %s failed: %s", sid, e)
                ok = False
            moved += 1 if ok else 0
            skipped += 0 if ok else 1
        rows = _flagged_scored_rows(store)
        sig = "|".join(f"{x['id']}={x['scored']}" for x in rows)
        msg = f"Moved {moved} onset file(s) → {to_animal}."
        if skipped:
            msg += (f" Skipped {skipped} (target not in that recording, or "
                    "already that animal).")
        return rows, [], msg, sig

    # Open a flagged row in Video Review (mirrors _open_in_video_review).
    @app.callback(
        Output("lfp-to-video-bridge", "data", allow_duplicate=True),
        Output("group-tabs", "value", allow_duplicate=True),
        Output("tabs", "value", allow_duplicate=True),
        Input("evtv-flagged-table", "active_cell"),
        State("evtv-flagged-table", "data"),
        prevent_initial_call=True,
    )
    def _fs_open(active_cell, data):
        if not active_cell or active_cell.get("column_id") != "view":
            return no_update, no_update, no_update
        row = next((r for r in (data or [])
                    if r.get("id") == active_cell.get("row_id")), None)
        if not row:
            return no_update, no_update, no_update
        file_id = int(row["file_id"])
        with store.connection() as conn:
            pr = conn.execute(
                "SELECT session_dir, duration_sec "
                "FROM processed_files WHERE id = ?", (file_id,)).fetchone()
        if not pr or not pr["session_dir"]:
            return no_update, no_update, no_update
        session_dir = pr["session_dir"]
        bridge = {
            "session_dir": session_dir, "file_id": file_id,
            "channel": int(_first_animal_channel_index(store, session_dir)),
            "hp": 0, "lp": 0, "notch": 0, "smooth": 0, "start_sec": 0.0,
            "lfp_dur": float(pr["duration_sec"] or 0.0),
            "seq": int(datetime.now().timestamp() * 1000),
        }
        return bridge, "analysis", "video"

    # ---- Per-animal auto-filter thresholds ---- #
    @app.callback(
        Output("evtv-af-cutoff", "value"),
        Output("evtv-af-auc-thr", "value"),
        Output("evtv-af-auc-win", "value"),
        Output("evtv-af-electrode", "value"),
        Output("evtv-af-mode", "value"),
        Output("evtv-af-enabled", "value"),
        Input("evtv-af-animal", "value"),
        prevent_initial_call=True,
    )
    def _af_load(animal):
        if not animal:
            return None, None, 5, 0, "both", ["on"]
        cfg = store.get_animal_screen_config(animal) or {}
        return (cfg.get("peak_cutoff"), cfg.get("auc_threshold"),
                cfg.get("auc_window_sec") or 5, cfg.get("electrode") or 0,
                cfg.get("screen_mode") or "both",
                (["on"] if cfg.get("enabled", 1) else []))

    @app.callback(
        Output("evtv-af-status", "children"),
        Output("evtv-af-table", "data"),
        Input("evtv-af-save-btn", "n_clicks"),
        Input("evtv-af-sweep-btn", "n_clicks"),
        State("evtv-af-animal", "value"),
        State("evtv-af-cutoff", "value"),
        State("evtv-af-auc-thr", "value"),
        State("evtv-af-auc-win", "value"),
        State("evtv-af-electrode", "value"),
        State("evtv-af-mode", "value"),
        State("evtv-af-enabled", "value"),
        prevent_initial_call=True,
    )
    def _af_save_or_sweep(_s, _w, animal, cutoff, auc_thr, auc_win,
                          electrode, mode, enabled):
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return "PI only.", no_update
        if not animal:
            return "Pick an animal first.", no_update
        trig = callback_context.triggered_id
        mode = mode or "both"
        need_env = mode in ("both", "envelope")
        need_auc = mode in ("both", "auc")
        if trig == "evtv-af-save-btn":
            has_env = cutoff not in (None, "")
            has_auc = auc_thr not in (None, "")
            if (mode == "envelope" and not has_env) or \
               (mode == "auc" and not has_auc) or \
               (mode == "both" and not has_env and not has_auc):
                return ("Set a threshold for the chosen screen(s).", no_update)
            store.upsert_animal_screen_config(
                animal, peak_cutoff=cutoff, auc_threshold=auc_thr,
                auc_window_sec=auc_win, electrode=int(electrode or 0),
                screen_mode=mode, enabled=("on" in (enabled or [])),
                updated_by=email)
            _mass_analyze.kick_auto_screen_for_animal(store, animal)
            _activity.track(store, "event_verification", "autofilter_save",
                            animal, {"screen_mode": mode})
            msg = (f"Saved {animal} — sweeping its backlog in the "
                   f"background…")
        elif trig == "evtv-af-sweep-btn":
            r = _mass_analyze.auto_screen_for_animal(store, animal)
            _activity.track(store, "event_verification", "autofilter_sweep",
                            animal, {"n_cleared": r.get("n_cleared")})
            if r.get("reason"):
                msg = f"{animal}: {r['reason'].replace('_', ' ')}."
            else:
                msg = (f"{animal}: cleared {r['n_cleared']}, kept "
                       f"{r['n_kept']}, errors {r['n_error']}.")
        else:
            return no_update, no_update
        return msg, _af_table_rows(store)

    @app.callback(
        Output("evtv-af-table", "data", allow_duplicate=True),
        Input("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def _af_refresh_table(_n):
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return no_update
        return _af_table_rows(store)

    # Click a row's "Open >" cell -> hand the recording to Video
    # Review via the same lfp-to-video-bridge the LFP Browser uses
    # (lfp_browser._on_view_video). Reuses the whole synchronized
    # player; zero changes on the video side.
    @app.callback(
        Output("lfp-to-video-bridge", "data"),
        Output("group-tabs", "value", allow_duplicate=True),
        Output("tabs", "value", allow_duplicate=True),
        Input("evtv-pending-table", "active_cell"),
        prevent_initial_call=True,
    )
    def _open_in_video_review(active_cell):
        if (not active_cell
                or active_cell.get("column_id") != "view"):
            return no_update, no_update, no_update
        # row_id is the review_state state_id (one per (file, animal)); map
        # it back to the file_id to open in Video Review.
        sid = active_cell.get("row_id")
        if sid is None:
            return no_update, no_update, no_update
        file_id = store._file_id_for_state(int(sid))
        if file_id is None:
            return no_update, no_update, no_update
        with store.connection() as conn:
            row = conn.execute(
                "SELECT session_dir, duration_sec "
                "FROM processed_files WHERE id = ?",
                (file_id,),
            ).fetchone()
        if not row or not row["session_dir"]:
            return no_update, no_update, no_update
        session_dir = row["session_dir"]
        channel = _first_animal_channel_index(store, session_dir)
        bridge = {
            "session_dir": session_dir,
            "file_id": file_id,
            "channel": int(channel),
            "hp": 0, "lp": 0, "notch": 0, "smooth": 0,
            "start_sec": 0.0,
            "lfp_dur": float(row["duration_sec"] or 0.0),
            "seq": int(datetime.now().timestamp() * 1000),
        }
        return bridge, "analysis", "video"

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
            _activity.track(store, "event_verification", "finalize_csv",
                            "append")
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
        State("evtv-ma-electrode", "value"),
        prevent_initial_call=True,
    )
    def _on_ma_scan(n_clicks, animal_id, cutoff, electrode):
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
                store, email, animal_id, cutoff,
                electrode=int(electrode or 0))
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
        n_auc = int(job.get("n_auc_pos") or 0)
        n_dis = int(job.get("n_disagree") or 0)
        ran_auc = bool(job.get("auc_threshold")) and \
            bool(job.get("auc_window_sec"))
        progress = (
            f"{status}  ·  scanned {scanned}"
            + (f" of {total}" if total else "")
            + f"  ·  Pool 1 (envelope): {n_with}"
            + (f"  ·  Pool 2 (AUC): {n_auc}  ·  "
               f"Pool 3 (disagree): {n_dis}" if ran_auc else "")
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

    # ---- Screen comparison: start / poll / cancel ---- #
    @app.callback(
        Output("evtv-sc-job-id", "data"),
        Output("evtv-sc-poll", "disabled"),
        Output("evtv-sc-progress", "children",
                allow_duplicate=True),
        Input("evtv-sc-run-btn", "n_clicks"),
        State("evtv-sc-animal-dropdown", "value"),
        State("evtv-sc-cutoff-input", "value"),
        State("evtv-sc-auc-threshold-input", "value"),
        State("evtv-sc-auc-window-input", "value"),
        State("evtv-sc-electrode", "value"),
        prevent_initial_call=True,
    )
    def _on_sc_run(n_clicks, animal_id, cutoff,
                    auc_threshold, auc_window, electrode):
        if not n_clicks:
            return no_update, no_update, no_update
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return None, True, "Not authorised."
        if not animal_id:
            return None, True, "Pick an animal first."
        try:
            cutoff = float(cutoff)
            auc_t = float(auc_threshold)
            auc_w = float(auc_window) if auc_window else 5.0
        except (TypeError, ValueError):
            return (None, True,
                     "Peak cutoff and AUC threshold must be numbers.")
        if cutoff <= 0 or auc_t <= 0 or auc_w <= 0:
            return None, True, "Cutoff, AUC threshold, window > 0."
        try:
            job_id = _mass_analyze.create_screen_eval_job(
                store, email, animal_id, cutoff, auc_t, auc_w,
                electrode=int(electrode or 0))
        except Exception as e:
            logger.exception("create_screen_eval_job failed")
            return None, True, f"Failed to start: {e}"
        return job_id, False, "Benchmark queued…"

    @app.callback(
        Output("evtv-sc-progress", "children"),
        Output("evtv-sc-results", "children"),
        Output("evtv-sc-poll", "disabled",
                allow_duplicate=True),
        Input("evtv-sc-poll", "n_intervals"),
        Input("evtv-sc-job-id", "data"),
        prevent_initial_call=True,
    )
    def _on_sc_poll(_n, job_id):
        if not job_id:
            return ("", "", True)
        job = _mass_analyze.get_screen_eval_job(store, int(job_id))
        if not job:
            return ("Job vanished.", "", True)
        status = job.get("status") or "?"
        scanned = int(job.get("scanned_files") or 0)
        total = int(job.get("total_files") or 0)
        progress = (
            f"{status}  ·  scored {scanned}"
            + (f" of {total} labeled files" if total else ""))
        polling_disabled = status in ("done", "failed", "cancelled")
        results = []
        if status == "done":
            if total == 0:
                results = html.Div(
                    "No human-labeled files for this animal yet "
                    "-- nothing to score the screens against.",
                    style={"color": "#a0a0b0", "fontSize": "12px"})
            else:
                results = _screen_compare_table(job)
        elif status == "failed":
            results = html.Div(
                f"Benchmark failed: {job.get('error') or 'unknown'}",
                style={"color": "#ff453a", "fontSize": "12px"})
        elif status == "cancelled":
            results = html.Div(
                f"Cancelled at {scanned} of {total}.",
                style={"color": "#ff9f0a", "fontSize": "12px"})
        return (progress, results, polling_disabled)

    @app.callback(
        Output("evtv-sc-progress", "children",
                allow_duplicate=True),
        Input("evtv-sc-cancel-btn", "n_clicks"),
        State("evtv-sc-job-id", "data"),
        prevent_initial_call=True,
    )
    def _on_sc_cancel(n_clicks, job_id):
        if not n_clicks or not job_id:
            return no_update
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return "Not authorised."
        _mass_analyze.cancel_screen_eval_job(store, int(job_id))
        return "Cancel requested."

    @app.callback(
        Output("evtv-ma-progress", "children",
                allow_duplicate=True),
        Output("evtv-ma-summary", "children",
                allow_duplicate=True),
        Output("evtv-ma-confirm-row", "children",
                allow_duplicate=True),
        Output("evtv-ma-job-id", "data",
                allow_duplicate=True),
        Output("evtv-pending-table", "data",
                allow_duplicate=True),
        Output("evtv-list-sig", "data",
                allow_duplicate=True),
        Input("evtv-ma-confirm-btn", "n_clicks"),
        Input("evtv-ma-discard-btn", "n_clicks"),
        State("evtv-ma-animal-dropdown", "value"),
        State("evtv-ma-cutoff-input", "value"),
        State("evtv-ma-electrode", "value"),
        prevent_initial_call=True,
    )
    def _on_ma_commit(_confirm_n, _discard_n,
                       animal_id, cutoff, electrode):
        trig = callback_context.triggered_id
        if trig == "evtv-ma-discard-btn":
            return ("", "", [], None, no_update, no_update)
        if trig != "evtv-ma-confirm-btn":
            return (no_update, no_update, no_update,
                     no_update, no_update, no_update)
        email = (current_user_email() or "").lower()
        if not _is_pi(config or {}, email):
            return ("Not authorised.", "", [], None,
                     no_update, no_update)
        try:
            cutoff = float(cutoff)
        except (TypeError, ValueError):
            return (f"Invalid cutoff: {cutoff!r}",
                     "", [], None, no_update, no_update)
        try:
            result = _mass_analyze.commit_threshold(
                store, animal_id, cutoff, email,
                electrode=int(electrode or 0))
        except Exception as e:
            logger.exception("commit_threshold failed")
            return (f"Commit failed: {e}", "", [],
                     None, no_update, no_update)
        msg = (f"✓ {result['n_cleared']} files moved to "
                "pending_pi_review. Approve them in the list "
                "below.")
        # Refresh the table so the new pending files show up.
        rows = store.pi_pending_files(limit=500)
        return (msg, "", [], None,
                 _pending_table_rows(store, rows),
                 _pending_signature(rows))


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
    # One structured entry per CSV file written, so the summary can show
    # exactly WHICH files got what (instead of a giant one-line string).
    entries: list[dict] = []
    base_dir = bhz_cfg.get("base_dir", "")
    # One day-row per animal tab: {animal_tab: [day_stat_dict, ...]}.
    rows_by_tab: dict[str, list[dict]] = {}
    for (animal, day), items in sorted(bucket.items()):
        csv_path = _bhz_csv.resolve_csv_path(
            base_dir,
            bhz_cfg.get("filename_template", "{date}_{animal}.csv"),
            day, animal,
        )
        rows_by_tab.setdefault(animal, []).append(
            _day_stats(store, animal, day, items, csv_path.name))
        events_by_fn: dict[str, list[dict]] = {}
        meta_by_fn: dict[str, dict] = {}
        for r in items:
            file_meta = _file_meta_for_row(store, r, bhz_cfg)
            if file_meta is None:
                continue
            fn = file_meta["filename"]
            meta_by_fn[fn] = file_meta
            events_by_fn[fn] = r.get("events") or []
        entry = {"csv_name": csv_path.name, "animal": animal, "day": str(day),
                 "n_recordings": len(events_by_fn)}
        if overwrite:
            diff = _bhz_csv.overwrite_day_csv(
                csv_path, events_by_fn, meta_by_fn, fs=_pick_fs(items))
            entry.update(added=len(diff.added_filenames),
                         removed=len(diff.removed_filenames),
                         modified=len(diff.modified_filenames),
                         removed_files=list(diff.removed_filenames),
                         n_rows=0)
        else:
            n_total = 0
            for fn, evs in events_by_fn.items():
                n_total += _bhz_csv.write_event_rows(
                    csv_path, meta_by_fn[fn], evs,
                    fs=meta_by_fn[fn].get("fs") or 20000.0)
            entry["n_rows"] = n_total
        entries.append(entry)
    # Daily Google-Sheet upsert (non-fatal): one row per day in each
    # animal's own tab (MouseID). Merge-update never blanks the lab's
    # hand-filled columns.
    sheet_note = ""
    gs = (config or {}).get("google_sheets", {}) or {}
    if gs.get("enabled") and rows_by_tab:
        try:
            from src.utils import sheets_write
            res = sheets_write.upsert_day_rows(
                gs["service_account_file"], gs["spreadsheet_id"],
                rows_by_tab,
                gs.get("key_columns", ["Date"]),
                gs.get("column_map"))
            sheet_note = (f"Google Sheet: {res['updated']} updated, "
                          f"{res['appended']} appended")
            if res.get("skipped_tabs"):
                sheet_note += (f" (no tab for: "
                               f"{', '.join(res['skipped_tabs'])})")
        except Exception as e:
            logger.warning("Google Sheet upsert failed: %s", e)
            sheet_note = f"Google Sheet sync FAILED: {e}"
    return _finalize_summary_block(base_dir, entries, overwrite, sheet_note)


def _finalize_summary_block(base_dir: str, entries: list[dict],
                             overwrite: bool, sheet_note: str):
    """A readable 'what got written where' summary: total, the target folder,
    and one line per CSV file (rows + how many recordings), so the PI can see
    exactly which files were written."""
    if not entries:
        return "Nothing to finalize -- no pi_approved files."
    total_recs = sum(e["n_recordings"] for e in entries)
    total_rows = sum(e.get("n_rows", 0) for e in entries)
    if overwrite:
        head = (f"✓ Rebuilt {len(entries)} CSV file(s) "
                f"· {total_recs} recording(s)")
    else:
        head = (f"✓ Wrote {total_rows} row(s) from {total_recs} "
                f"recording(s) into {len(entries)} CSV file(s)")
    lines = []
    for e in entries:
        if overwrite:
            detail = (f"+{e['added']} −{e['removed']} ~{e['modified']} "
                      f"· {e['n_recordings']} rec")
        else:
            detail = (f"+{e['n_rows']} row{'' if e['n_rows'] == 1 else 's'} "
                      f"· {e['n_recordings']} rec")
        lines.append(html.Div([
            html.Span(e["csv_name"], style={
                "fontFamily": "ui-monospace, monospace", "color": "#cfd0d6",
                "fontSize": "11px"}),
            html.Span(f"   {detail}", style={
                "color": "#808090", "fontSize": "11px", "marginLeft": "auto"}),
        ], style={"display": "flex", "padding": "2px 0"}))
        if overwrite and e.get("removed_files"):
            lines.append(html.Div(
                "⚠ removed rows for: " + ", ".join(e["removed_files"]),
                style={"color": "#ff9f0a", "fontSize": "10px",
                       "padding": "0 0 2px 10px"}))
    children = [
        html.Div(head, style={"color": "#30d158", "fontWeight": "700",
                              "fontSize": "12px", "marginBottom": "2px"}),
        html.Div(f"→ {base_dir}", style={
            "color": "#808090", "fontFamily": "ui-monospace, monospace",
            "fontSize": "11px", "marginBottom": "6px"}),
        html.Div(lines, style={
            "maxHeight": "220px", "overflowY": "auto",
            "border": "1px solid rgba(255,255,255,0.06)",
            "borderRadius": "6px", "padding": "4px 8px"}),
    ]
    if sheet_note:
        children.append(html.Div(sheet_note, style={
            "color": "#808090", "fontSize": "11px", "marginTop": "6px"}))
    return html.Div(children)


def _day_stats(store, animal: str, day, items: list[dict],
                 csv_name: str = "") -> dict:
    """Canonical one-row-per-(animal, day) summary for the Sheet.

    *items* are the pi_approved rows for this (animal, day); each has a
    decoded ``events`` list + ``file_id`` / ``session_dir`` /
    ``user_email``.
    """
    from datetime import datetime as _dt
    from src.utils import mass_analyze as _ma
    n_files = len(items)
    n_with = n_events = max_racine = n_no_event = 0
    n_stim_files = n_during = 0
    reviewers: list[str] = []
    sessions: list[str] = []
    for r in items:
        ue = (r.get("user_email") or "").strip()
        if ue and ue not in reviewers:
            reviewers.append(ue)
        sd = r.get("session_dir")
        if sd and sd not in sessions:
            sessions.append(sd)
        evs = r.get("events") or []
        n_ev = len(evs) if isinstance(evs, list) else 0
        if n_ev > 0:
            n_with += 1
            n_events += n_ev
            for e in evs:
                try:
                    max_racine = max(max_racine,
                                      int((e or {}).get("racine") or 0))
                except (TypeError, ValueError):
                    pass
        else:
            n_no_event += 1
        try:
            if _ma.has_stim_for_file(store, int(r["file_id"]),
                                       r.get("session_dir")):
                n_stim_files += 1
                n_during += n_ev
        except Exception:
            pass

    # Per-animal recording metadata from the session config(s):
    #  Recording Location = electrode suffix(es) (SR / SLM / ...)
    #  Channel(s)         = the animal's electrode channel index(es)
    #  More Settings      = the stim parameters
    locations: list[str] = []
    channels: list[str] = []
    stim_settings = ""
    for sd in sessions:
        try:
            for e in store.electrodes_for_animal_in_session(sd, animal):
                loc = e.get("location")
                if loc and loc not in locations:
                    locations.append(loc)
                ch = str(e.get("channel_index"))
                if ch not in channels:
                    channels.append(ch)
            if not stim_settings:
                cfg = store.get_session_config(sd) or {}
                bits = []
                if cfg.get("stim_charge_nC"):
                    bits.append(f"{float(cfg['stim_charge_nC']):g}nC")
                if cfg.get("stim_pulse_width_us"):
                    bits.append(
                        f"{float(cfg['stim_pulse_width_us']):g}us")
                if cfg.get("stim_frequency_hz"):
                    bits.append(
                        f"{float(cfg['stim_frequency_hz']):g}Hz")
                stim_settings = ", ".join(bits)
        except Exception:
            pass

    is_stim = n_stim_files > 0
    return {
        "date": day.isoformat() if hasattr(day, "isoformat")
                 else str(day),
        "animal": animal,
        "csv_filename": csv_name,
        "n_files": n_files,
        "n_files_with_events": n_with,
        "n_events": n_events,
        "max_racine": max_racine,
        "n_during_stim_events": n_during,
        "n_no_event_files": n_no_event,
        "reviewers": ", ".join(reviewers),
        "recording_location": ", ".join(locations),
        "channels": ", ".join(
            sorted(channels,
                    key=lambda x: int(x) if x.isdigit() else 0)),
        "type_of_recording": "stim" if is_stim else "baseline",
        "more_settings": stim_settings if is_stim else "",
        "exported_at": _dt.now().isoformat(timespec="seconds"),
    }


def _fetch_approved_rows(store) -> list[dict]:
    """Every pi_approved review_state row + its deserialised
    events list."""
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT rs.id AS state_id, rs.file_id, rs.animal_id,
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
    """Resolve (animal, chunk_datetime) for a pi_approved row. Prefer the
    review row's stored animal_id (the animal actually scored); fall back to
    the session's first animal channel only for legacy whole-file (NULL) rows
    — fixing the old multi-animal mis-filing."""
    animal = row.get("animal_id") or _animal_for_session(
        store, row.get("session_dir") or "")
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
        # Peak_Index/Peak_Stamp come only from the automated detector; a
        # PI-approved manual review has no envelope peak, so they stay blank
        # (the onset is in EventEO / EventEO_WallClock / EO_HourOfDay).
        peak_index=None, peak_stamp=None, peak_dt=peak_dt,
    )


def _pick_fs(rows: list[dict]) -> float:
    """Median fs across a group; falls back to 20 kHz."""
    fss = [float(r.get("sampling_rate") or 0) for r in rows]
    fss = [f for f in fss if f > 0]
    if not fss:
        return 20000.0
    return sorted(fss)[len(fss) // 2]
