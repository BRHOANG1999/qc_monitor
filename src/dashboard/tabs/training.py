"""Training tab -- students blind-re-score PI-validated recordings and
get an agreement score vs the validated answer.

Three maturation stages (see src/utils/training.STAGES):
  1 Detection      -- is there a BHZ? (yes/no)
  2 Type + Racine  -- per event: onset type (LVF/HYP) + Racine, by EO
  3 Full           -- type + Racine + all five landmarks

Advancement is auto-flagged (rolling mean >= pass_pct over `window`) but
PI-confirmed. Examples are served unseen-first then recycle oldest-seen
(store.next_training_file). The validated answer is revealed only AFTER
the student submits.

All component ids are ``training-`` / ``tr-`` namespaced so nothing
collides with the Video Review tab, whose figure builders + media route
this tab reuses.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading

from dash import (Input, Output, State, dcc, html, no_update, ALL,
                   callback_context)

from src.dashboard import activity as _activity
from src.dashboard.auth import current_user_email
from src.dashboard.components import (
    button, empty_state, LABEL_STYLE, card)
from src.dashboard.design import (
    COLOR_ACCENT, COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER,
    COLOR_SURFACE_1, COLOR_SURFACE_2, COLOR_DIVIDER, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_TERTIARY, RADIUS_MD, RADIUS_SM,
    SPACE_2, SPACE_3, SPACE_4, FONT_SIZE_BODY, FONT_SIZE_CAPTION)
from src.db.store import Store
from src.utils.animal import (is_animal_channel, recording_channel_index,
                               split_animal_electrode, stim_copy_indices)
from src.utils import training as _grade
from src.utils import past_events as _past

# Reused from the Video Review tab -- the LFP/Hilbert figure builders and
# the file-path lookup. video.py never imports this module, so no cycle.
from src.dashboard.tabs.video import (
    _file_path_for_id, _decimated_lfp, _build_lfp_figure,
    _render_hilbert_trace, _render_auc_trace, _empty_lfp_fig,
    _session_dir_for_file, _stim_times_for_file, _stim_copy_channels)

logger = logging.getLogger("qc_monitor.dashboard.training")

_LANDMARKS = _grade.LANDMARKS  # ("EO","LAS","BO","PID","BB")
_AUC_WINDOW_SEC = 5.0          # sliding-window AUC width for the LFP toggle

# DOM id of the Training <video> element. The cursor-sync clientside
# callbacks find it with document.getElementById(), exactly like the Video
# Review tab does with VIDEO_DOM_ID -- a distinct id so the two tabs never
# collide if both are ever in the DOM.
TRAINING_VIDEO_DOM_ID = "training-lfp-video"


# ===================================================================== #
#  Historical practice-library import (background, with live progress)
# ===================================================================== #
#
# Resolving past scored events across the share roots + every mounted drive
# can take a while, so it runs off-thread; the UI polls _IMPORT_STATE. One
# import at a time (shared library) -- concurrent kicks coalesce. The import
# is NOT optional: it auto-starts once per process the first time anyone
# opens the Training tab (the button only forces a re-scan).
_IMPORT_LOCK = threading.Lock()
_IMPORT_AUTOSTARTED = False
_IMPORT_STATE: dict = {
    "running": False, "done": 0, "total": 0, "imported": 0,
    "phase": "", "error": "", "finished": False,
}


def _import_progress(done: int, total: int, filename: str) -> None:
    _IMPORT_STATE["done"] = int(done)
    _IMPORT_STATE["total"] = int(total)
    _IMPORT_STATE["phase"] = (f"Resolving {filename}" if filename
                              else "Finishing up")


def _import_worker(store: Store, config: dict) -> None:
    try:
        _IMPORT_STATE.update({"running": True, "finished": False,
                              "error": "", "phase": "Reading scored CSVs",
                              "done": 0, "total": 0})
        n = _past.import_into_training(store, config,
                                        progress=_import_progress)
        _IMPORT_STATE["imported"] = int(n)
    except Exception as e:  # noqa: BLE001 -- background import must not crash
        logger.exception("historical practice import failed")
        _IMPORT_STATE["error"] = str(e)
    finally:
        _IMPORT_STATE["running"] = False
        _IMPORT_STATE["finished"] = True
        _IMPORT_STATE["phase"] = ""


def _kick_import(store: Store, config: dict) -> bool:
    """Start the import in a daemon thread if one isn't already running.
    The non-blocking lock makes concurrent clicks coalesce. Returns True if
    this call started it."""
    if not _IMPORT_LOCK.acquire(blocking=False):
        return False
    t = threading.Thread(target=_run_import_then_release,
                          args=(store, config), daemon=True)
    t.start()
    return True


def _run_import_then_release(store: Store, config: dict) -> None:
    try:
        _import_worker(store, config)
    finally:
        _IMPORT_LOCK.release()


def _import_status_text(cfg_enabled: bool) -> str:
    if not cfg_enabled:
        return "Disabled in config (past_events.enabled)."
    s = _IMPORT_STATE
    if s["running"]:
        tot = s["total"] or "?"
        return f"Importing… {s['done']}/{tot} recordings.  {s['phase']}"
    if s["finished"]:
        if s["error"]:
            return f"Import failed: {s['error']}"
        return (f"{s['imported']} past-scored recordings in your practice "
                f"pool. Each one's EEG is located on disk when you load it.")
    return "Cataloging the practice library…"


# ===================================================================== #
#  Helpers
# ===================================================================== #

def _is_pi(config: dict, email: str | None) -> bool:
    if not email:
        return False
    pis = (((config or {}).get("review_queue", {}) or {})
           .get("pi_emails", []) or [])
    return email.lower() in {e.lower() for e in pis}


def _cfg(config: dict) -> dict:
    t = (config or {}).get("training", {}) or {}
    return {
        "pass_pct": float(t.get("pass_pct", 85)),
        "window": int(t.get("window", 10)),
        "onset_tol_s": float(t.get("onset_tol_s", 10)),
        "racine_tol": int(t.get("racine_tol", 1)),
        "examples_per_round": int(t.get("examples_per_round", 10)),
        "min_file_duration_sec": float(t.get("min_file_duration_sec", 1800)),
    }


def _channel_names(store: Store, session_dir: str | None,
                    file_path: str | None = None) -> list:
    """Channel names for the session: from session_config when known, else
    parsed from the filename. Historical practice files often have no
    session_config row, so without the fallback the channel defaults to 0
    (the stimCopy) -- the bug the reviewer hit."""
    try:
        names = store._channel_names_for_session(session_dir) or []
    except Exception:
        names = []
    if names:
        return names
    if file_path:
        try:
            return _past.extract_channel_names(file_path)
        except Exception:
            return []
    return []


def _first_animal_channel(store: Store, session_dir: str | None,
                          file_path: str | None = None) -> int:
    """The recording electrode to show: the animal channel immediately after
    the stimCopy channel (lab layout), else the first animal channel."""
    return recording_channel_index(
        _channel_names(store, session_dir, file_path))


def _channel_for_animal(store: Store, session_dir: str | None, animal: str,
                        file_path: str | None = None) -> int | None:
    """Channel index of *animal*'s electrode, using session_config when known
    and the FILENAME channel list otherwise. Historical CSV examples have no
    session_config row, so the DB-only ``animal_channel_index`` would fall
    back to channel 0 (the wrong animal); this honours the filename fallback
    so the seizure animal's channel resolves either way. None when no channel
    parses to *animal*."""
    if not animal:
        return None
    for i, n in enumerate(_channel_names(store, session_dir, file_path)):
        if isinstance(n, str) and is_animal_channel(n):
            a, _ = split_animal_electrode(n)
            if a == animal:
                return i
    return None


def _channel_animal(store: Store, session_dir: str | None, channel: int,
                    file_path: str | None = None) -> tuple[str, str]:
    """(channel_name, animal_id) for the displayed channel, or ("","") when
    the session's channel names aren't known."""
    names = _channel_names(store, session_dir, file_path)
    if not (0 <= channel < len(names)) or not isinstance(names[channel], str):
        return "", ""
    name = names[channel]
    animal, _elec = split_animal_electrode(name)
    return name, (animal or "")


def _filtered_trace(store: Store, file_id: int, channel: int,
                     mode: str):
    """The student's chosen 2nd-panel trace: raw Hilbert envelope (default)
    or its sliding-window AUC. Falls back to a placeholder on error."""
    try:
        if mode == "auc":
            fig, _s = _render_auc_trace(store, file_id, channel,
                                        _AUC_WINDOW_SEC)
        else:
            fig, _s = _render_hilbert_trace(store, file_id, channel)
        return fig
    except Exception:
        return _empty_lfp_fig("Trace failed to load.")


def _build_figures(store: Store, file_id: int, channel: int,
                    mode: str = "hilbert"):
    """(lfp_fig, filtered_fig, lfp_dur_s) for a recording -- stim-blanked
    like the Video Review default. *mode* picks the 2nd panel (hilbert |
    auc). *lfp_dur_s* is the recording length in seconds (0.0 if unknown);
    the cursor-sync clientside callback maps video time onto it."""
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        ph = _empty_lfp_fig("Recording file not found.")
        return ph, ph, 0.0
    dur = 0.0
    try:
        session_dir = _session_dir_for_file(store, file_id)
        stim_copy = _stim_copy_channels(store, session_dir)
        if not stim_copy:   # historical files have no session_config role tags
            stim_copy = stim_copy_indices(
                _channel_names(store, session_dir, file_path))
        do_blank = channel not in stim_copy
        stim_times = (_stim_times_for_file(store, file_id)
                      if do_blank else None)
        t, sig, dur, _n = _decimated_lfp(
            file_path, channel,
            stim_times=stim_times if (stim_times is not None
                                       and len(stim_times)) else None)
        lfp = _build_lfp_figure(t, sig, label=f"Ch{channel}",
                                 uirevision=f"train:{file_id}:{channel}")
    except Exception:
        lfp = _empty_lfp_fig("LFP failed to load.")
        dur = 0.0
    return lfp, _filtered_trace(store, file_id, channel, mode), float(dur)


def _ensure_round(store: Store, email: str, stage: int, n: int,
                   min_duration_sec: float = 0.0):
    """The open round for (email, stage), creating a balanced one when none
    is open. None when there are no candidate files. Recordings shorter than
    *min_duration_sec* (when measured) are excluded from the pool."""
    rnd = store.current_training_round(email, stage)
    if rnd is not None:
        return rnd
    pool = store.training_candidate_pool(email, min_duration_sec)
    if not pool:
        return None
    seen = {c["file_id"]: (1 if c["last_seen"] else 0) for c in pool}
    files = _grade.select_round(pool, int(stage), n, seen, random.Random())
    if not files:
        return None
    return store.create_training_round(email, int(stage), files)


def _round_idx(store: Store, email: str, stage: int, rnd: dict) -> int:
    """Examples of this round the student has already submitted."""
    return len(store.round_scores(email, int(stage),
                                  rnd["started_after_id"]))


def _history_text(store: Store, email: str, stage: int, rnd=None) -> str:
    life = store.training_lifetime_stats(email, int(stage))
    bits = []
    if rnd is not None:
        idx = min(_round_idx(store, email, stage, rnd), rnd["examples"])
        bits.append(f"round {idx}/{rnd['examples']}")
    if life["n"]:
        bits.append(f"lifetime {life['mean'] * 100:.0f}% / {life['n']}")
    return "  ·  ".join(bits)


def _resolve_for_load(store: Store, config: dict, file_id: int) -> None:
    """Lazily confirm/relocate a Training recording right before it loads.
    For a real DB file (path on disk) this is a quick stat; for a cataloged
    historical example with a stale path it runs the recursive multi-drive
    EEG search once (cached). Failures are non-fatal -- the figure builders
    fall back to a 'file not found' placeholder."""
    try:
        loc = _past.make_locator(store, config or {})
        store.resolve_training_file(int(file_id), loc)
    except Exception as e:  # noqa: BLE001 -- resolution must not break load
        logger.warning("training: lazy resolve failed for #%s: %s",
                       file_id, e)


def _load_next(store: Store, email: str, stage: int, mode: str, n: int,
                config: dict | None = None):
    """('loaded', payload) for the next round example, ('complete', rnd) when
    the round is finished, or None when no candidate files exist."""
    min_dur = _cfg(config or {})["min_file_duration_sec"]
    rnd = _ensure_round(store, email, stage, n, min_dur)
    if rnd is None:
        return None
    idx = _round_idx(store, email, stage, rnd)
    if idx >= rnd["examples"]:
        return "complete", rnd
    file_id = int(rnd["files"][idx])
    # Locate the EEG on disk now, only for this one file (lazy, cached).
    _resolve_for_load(store, config or {}, file_id)
    mat_path = _file_path_for_id(store, file_id)
    fname = os.path.basename(mat_path) if mat_path else f"file #{file_id}"
    session_dir = _session_dir_for_file(store, file_id)
    # Show the channel of the animal whose validated answer carries the
    # seizure -- on a multi-animal recording the seizure may be on the 2nd
    # animal, not the default 'recording' electrode. Falls back to the
    # recording channel for negatives / un-attributed approvals so the LFP,
    # the validated ground truth, and the grading all agree on one animal.
    gt_animal = store.validated_seizure_animal_for_file(file_id)
    gt_channel = _channel_for_animal(store, session_dir, gt_animal, mat_path)
    if gt_channel is not None:
        channel = gt_channel
        validated = store.validated_events_for_file(file_id,
                                                     animal_id=gt_animal)
    else:
        channel = _first_animal_channel(store, session_dir, mat_path)
        validated = store.validated_events_for_file(file_id)
    chan_name, animal = _channel_animal(store, session_dir, channel, mat_path)
    lfp, hil, lfp_dur = _build_figures(store, file_id, channel, mode)
    # Stamp the measured length so the min-duration gate becomes real for
    # historical files once opened (no-op when already known).
    if lfp_dur:
        store.set_file_duration(file_id, lfp_dur)
    cur = {"file_id": file_id, "channel": channel,
           "validated": validated,
           "session_dir": session_dir, "lfp_dur": lfp_dur,
           "filename": fname, "animal": animal, "channel_name": chan_name}
    who = f"{animal} · Ch{channel} {chan_name}" if animal else \
        f"Ch{channel} {chan_name}".strip()
    now = (f"Example {idx + 1} of {rnd['examples']}  ·  {who}  ·  {fname}")
    # The video is rendered by _render_video (camera picker + .avi transcode);
    # show a transient placeholder until that fires.
    return "loaded", (cur, _vid_placeholder("Loading video…"), lfp, hil, now,
                      _history_text(store, email, stage, rnd))


def _build_prompt(store: Store, email: str, stage: int,
                   rnd: dict) -> html.Div:
    scores = store.round_scores(email, int(stage), rnd["started_after_id"])
    rmean = (sum(scores) / len(scores) * 100) if scores else 0.0
    life = store.training_lifetime_stats(email, int(stage))
    lifetxt = (f"{life['mean'] * 100:.0f}% over {life['n']} attempts"
               if life["n"] else "—")
    return card(
        html.Div(f"Round complete — {rmean:.0f}% on these "
                 f"{rnd['examples']} examples (lifetime {lifetxt}).",
                 style={"fontWeight": "700", "color": COLOR_TEXT_PRIMARY,
                        "marginBottom": SPACE_3}),
        html.Div("How confident are you in your scoring now? (1 = unsure, "
                 "5 = very confident)",
                 style={"color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginBottom": SPACE_2}),
        dcc.RadioItems(
            id="training-confidence",
            options=[{"label": f" {i}", "value": i} for i in range(1, 6)],
            value=3, inline=True,
            labelStyle={"marginRight": SPACE_3, "color": COLOR_TEXT_PRIMARY,
                         "fontSize": FONT_SIZE_BODY}),
        html.Div([
            button("Review more examples", "training-review-more-btn",
                   variant="secondary", style={"marginRight": SPACE_3}),
            button("I'm confident — move on", "training-moveon-btn",
                   variant="primary", tone="success"),
        ], style={"marginTop": SPACE_3}),
        html.Div(id="training-prompt-status",
                 style={"color": COLOR_TEXT_TERTIARY,
                        "fontSize": FONT_SIZE_CAPTION, "marginTop": SPACE_2}),
        style={"borderLeft": f"3px solid {COLOR_ACCENT}",
               "marginTop": SPACE_4},
    )


def _notify_pi(store: Store, config: dict, email: str, stage: int,
                rnd: dict) -> bool:
    """Email the PI(s) that a student finished a stage. Non-fatal."""
    pis = (((config or {}).get("review_queue", {}) or {})
           .get("pi_emails", []) or [])
    if not pis:
        return False
    scores = store.round_scores(email, int(stage), rnd["started_after_id"])
    rmean = (sum(scores) / len(scores) * 100) if scores else 0.0
    life = store.training_lifetime_stats(email, int(stage))
    lifetxt = (f"{life['mean'] * 100:.0f}% over {life['n']} attempts"
               if life["n"] else "—")
    label = STAGE_LABEL.get(int(stage), stage)
    subject = f"Training: {email} finished Stage {stage} ({label})"
    body = (f"{email} just finished a round of Stage {stage} ({label}) and "
            f"says they're confident.\n\n"
            f"Round score: {rmean:.0f}%\nLifetime: {lifetxt}\n\n"
            f"Open the Training tab -> Student progress (PI) to review their "
            f"performance and advance them.")
    try:
        from src.alerting.email_alert import EmailAlerter
        return EmailAlerter(config).send(subject, body, recipients=pis,
                                          subject_prefix=False)
    except Exception as e:  # noqa: BLE001 -- email must never break the UI
        logger.warning("PI training-stage email failed: %s", e)
        return False


def _events_table(events: list, title: str, color: str) -> html.Div:
    """Compact read-only listing of an event list for the feedback
    side-by-side (student vs validated)."""
    rows = []
    for i, e in enumerate(_grade._real_events(events)):
        bits = [f"#{i + 1}", e.get("type") or "?",
                f"R{e.get('racine')}" if e.get("racine") is not None
                else "R?"]
        eo = e.get("EO_sec")
        bits.append(f"EO {float(eo):.1f}s" if eo is not None else "EO ?")
        rows.append(html.Div(" · ".join(bits),
                             style={"fontSize": FONT_SIZE_CAPTION,
                                     "color": COLOR_TEXT_SECONDARY,
                                     "padding": "2px 0"}))
    if not rows:
        rows = [html.Div("No events.",
                         style={"fontSize": FONT_SIZE_CAPTION,
                                 "color": COLOR_TEXT_TERTIARY})]
    return html.Div([
        html.Div(title, style={"fontSize": FONT_SIZE_CAPTION,
                                "fontWeight": "700", "color": color,
                                "marginBottom": SPACE_2}),
        *rows,
    ], style={"flex": "1 1 220px", "padding": SPACE_3,
              "background": COLOR_SURFACE_1,
              "border": f"1px solid {COLOR_DIVIDER}",
              "borderRadius": RADIUS_SM})


def _past_enabled(config: dict) -> bool:
    return bool(((config or {}).get("past_events", {}) or {})
                .get("enabled", False))


def _import_card(config: dict) -> html.Div:
    """A slim status strip for the auto-built past-events practice library.
    No button -- the catalog runs automatically (training-import-tick kicks
    it once) and each EEG is located on disk only when its example loads."""
    return card(
        html.Div([
            html.Span("Practice library — past scored events:  ",
                      style={"fontWeight": "600",
                              "color": COLOR_TEXT_SECONDARY}),
            html.Span(id="training-import-status",
                      style={"color": COLOR_TEXT_TERTIARY}),
        ], style={"fontSize": FONT_SIZE_CAPTION}),
        dcc.Interval(id="training-import-tick", interval=1500, n_intervals=0),
        style={"marginBottom": SPACE_4},
    )


# ===================================================================== #
#  Layout
# ===================================================================== #

def layout(store: Store, config: dict | None = None):
    email = current_user_email()
    is_pi = _is_pi(config or {}, email)

    return html.Div([
        html.H3("Training",
                style={"color": COLOR_TEXT_PRIMARY, "marginBottom": "2px"}),
        html.Div("Score recordings that already have a validated answer, "
                 "then see how closely you agreed. Work through the "
                 "stages; your PI signs off when you're consistent enough.",
                 style={"color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_BODY, "marginBottom": SPACE_4}),

        # Stage selector + readiness badge.
        card(
            html.Div([
                html.Span("Stage:", style={**LABEL_STYLE,
                                            "display": "inline",
                                            "marginRight": SPACE_3}),
                dcc.RadioItems(
                    id="training-stage",
                    options=[{"label": f" {s}. {STAGE_LABEL[s]}",
                              "value": s} for s in (1, 2, 3)],
                    value=1, inline=True,
                    labelStyle={"marginRight": SPACE_4,
                                 "color": COLOR_TEXT_PRIMARY,
                                 "fontSize": FONT_SIZE_BODY}),
                html.Span(id="training-history",
                          style={"marginLeft": "auto",
                                  "color": COLOR_TEXT_TERTIARY,
                                  "fontSize": FONT_SIZE_CAPTION}),
                html.Span(id="training-ready-badge",
                          style={"marginLeft": SPACE_3}),
            ], style={"display": "flex", "alignItems": "center",
                       "flexWrap": "wrap", "gap": SPACE_2}),
            html.Div(id="training-stage-hint",
                      style={"color": COLOR_TEXT_TERTIARY,
                              "fontSize": FONT_SIZE_CAPTION,
                              "marginTop": SPACE_2}),
        ),

        # Practice library: import past scored events on demand.
        _import_card(config or {}),

        # Viewer: video | LFP + Hilbert.
        html.Div([
            html.Div([
                # Camera picker (shown only when a recording is multi-camera).
                html.Div([
                    html.Span("Camera:", style={
                        "color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginRight": SPACE_2}),
                    dcc.RadioItems(id="training-cam", inline=True, value=1,
                                    labelStyle={"marginRight": SPACE_3,
                                                 "color": COLOR_TEXT_PRIMARY,
                                                 "fontSize":
                                                     FONT_SIZE_CAPTION}),
                ], id="training-cam-wrap",
                   style={"display": "none", "alignItems": "center",
                           "marginBottom": SPACE_2}),
                dcc.Loading(
                    html.Div(id="training-video",
                              style={"background": "#000",
                                      "borderRadius": RADIUS_MD,
                                      "minHeight": "300px",
                                      "display": "flex",
                                      "alignItems": "center",
                                      "justifyContent": "center"}),
                    type="default"),
            ]),
            dcc.Loading(html.Div([
                dcc.Graph(id="training-lfp",
                          figure=_empty_lfp_fig("Load an example below."),
                          config={"displayModeBar": True,
                                   "displaylogo": False,
                                   "scrollZoom": True}),
                html.Div([
                    html.Span("2nd panel:", style={
                        "color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginRight": SPACE_2}),
                    dcc.RadioItems(
                        id="training-lfp-mode",
                        options=[{"label": " Hilbert envelope",
                                  "value": "hilbert"},
                                 {"label": " AUC", "value": "auc"}],
                        value="hilbert", inline=True,
                        labelStyle={"marginRight": SPACE_3,
                                     "color": COLOR_TEXT_PRIMARY,
                                     "fontSize": FONT_SIZE_CAPTION}),
                ], style={"display": "flex", "alignItems": "center",
                           "margin": f"{SPACE_2} 0"}),
                dcc.Graph(id="training-hilbert",
                          figure=_empty_lfp_fig(""),
                          config={"displayModeBar": True,
                                   "displaylogo": False,
                                   "scrollZoom": True}),
            ]), type="default"),
        ], style={"display": "grid",
                   "gridTemplateColumns": "minmax(320px,1fr) "
                                           "minmax(320px,1fr)",
                   "gap": SPACE_4, "marginTop": SPACE_4,
                   "marginBottom": SPACE_4}),

        html.Div([
            button("Load an example", "training-next-btn",
                   variant="primary", icon_name="video"),
            html.Span(id="training-now",
                      style={"marginLeft": SPACE_4,
                              "color": COLOR_TEXT_SECONDARY,
                              "fontSize": FONT_SIZE_CAPTION}),
        ], style={"display": "flex", "alignItems": "center",
                   "marginBottom": SPACE_4}),

        # Your impression (stage-dependent form).
        card(
            html.Div("Your impression",
                      style={**LABEL_STYLE, "marginBottom": SPACE_3}),
            # Stage-1 detection radio is a PERMANENT component (shown only for
            # stage 1) -- not regenerated by _render_form -- so re-renders
            # can't wipe the student's yes/no selection before they submit.
            html.Div(
                dcc.RadioItems(
                    id="training-detect",
                    options=[{"label": " No BHZ in this recording",
                              "value": "no"},
                             {"label": " Yes — there is at least one BHZ",
                              "value": "yes"}],
                    value=None,
                    labelStyle={"display": "block",
                                 "color": COLOR_TEXT_PRIMARY,
                                 "fontSize": FONT_SIZE_BODY,
                                 "marginBottom": SPACE_2}),
                id="training-detect-wrap"),
            html.Div(id="training-form"),
            html.Div([
                button("Submit", "training-submit-btn", variant="primary",
                       tone="success", icon_name="check-circle"),
            ], style={"marginTop": SPACE_3}),
            style={"marginBottom": SPACE_4},
        ),

        # Feedback (student vs validated), revealed after submit.
        html.Div(id="training-feedback"),

        # Post-round confidence prompt (hidden until a round completes).
        html.Div(id="training-prompt", style={"display": "none"}),

        # PI roster.
        html.Div(
            card(
                html.Div("Student progress (PI)",
                          style={**LABEL_STYLE, "marginBottom": SPACE_3}),
                html.Div(id="training-roster"),
            ) if is_pi else "",
            id="training-roster-wrap",
            style={"marginTop": SPACE_4} if is_pi else {"display": "none"},
        ),

        # State.
        dcc.Store(id="training-current"),          # {file_id, channel,
                                                   #  validated events,
                                                   #  lfp_dur, ...}
        dcc.Store(id="training-events", data=[]),  # student events
        dcc.Store(id="training-refresh", data=0),  # bump -> roster/badge
        dcc.Store(id="training-is-pi", data=bool(is_pi)),

        # Video <-> LFP cursor sync (mirrors the Video Review tab). The
        # interval samples the <video> element's currentTime 10x/s; the
        # store carries it to the clientside callbacks that move the cursor
        # on both panels. The sink swallows the click-to-seek return value.
        dcc.Interval(id="training-time-tick", interval=100, n_intervals=0),
        dcc.Store(id="training-current-time", data=0.0),
        html.Div(id="training-seek-sink", style={"display": "none"}),
        # Camera/video rendering: a 2.5 s tick polls the .avi->mp4 transcode
        # so the player loads when ready; the signature store dedupes renders
        # so a playing <video> isn't reset every tick.
        dcc.Interval(id="training-cam-tick", interval=2500, n_intervals=0,
                     disabled=True),
        dcc.Store(id="training-vidsig"),
        # Sink for the LFP<->Hilbert x-axis lock (the clientside callbacks
        # relayout the sibling graph directly; this just terminates them).
        dcc.Store(id="training-xsync-sink"),
    ])


STAGE_LABEL = {s: _grade.STAGES[s]["label"] for s in (1, 2, 3)}


# ===================================================================== #
#  Callbacks
# ===================================================================== #

def register_callbacks(app, store: Store, config: dict) -> None:
    cfg = _cfg(config)

    # ---- Stage access: disable locked stages for this student ---- #
    @app.callback(
        Output("training-stage", "options"),
        Output("training-stage", "value"),
        Output("training-stage-hint", "children"),
        Output("training-ready-badge", "children"),
        Input("training-refresh", "data"),
        State("training-stage", "value"),
    )
    def _stage_access(_refresh, cur_stage):
        email = current_user_email()
        prog = (store.get_training_progress(email) if email
                else {"unlocked_stage": 1, "certified_at": None})
        unlocked = int(prog.get("unlocked_stage", 1))
        opts = [{"label": f" {s}. {STAGE_LABEL[s]}", "value": s,
                 "disabled": s > unlocked} for s in (1, 2, 3)]
        value = cur_stage if (cur_stage and cur_stage <= unlocked) \
            else unlocked
        hint = _grade.STAGES[value]["scored"]
        badge = ""
        if email:
            # Readiness is round-scoped (resets when the student reviews more).
            rnd = store.current_training_round(email, value)
            scores = (store.round_scores(email, value, rnd["started_after_id"])
                      if rnd else
                      store.recent_training_scores(email, value,
                                                   limit=cfg["window"]))
            if prog.get("certified_at"):
                badge = _pill("Certified ✓", COLOR_SUCCESS)
            elif _grade.rolling_ready(scores, cfg["pass_pct"], cfg["window"]):
                badge = _pill(
                    "Ready to certify — ask your PI" if value == 3
                    else "Ready to advance — ask your PI", COLOR_WARNING)
        return opts, value, hint, badge

    # ---- Load the next example ---- #
    @app.callback(
        Output("training-current", "data"),
        Output("training-video", "children"),
        Output("training-lfp", "figure"),
        Output("training-hilbert", "figure"),
        Output("training-events", "data", allow_duplicate=True),
        Output("training-feedback", "children", allow_duplicate=True),
        Output("training-now", "children"),
        Output("training-prompt", "children", allow_duplicate=True),
        Output("training-prompt", "style", allow_duplicate=True),
        Output("training-history", "children", allow_duplicate=True),
        Input("training-next-btn", "n_clicks"),
        State("training-stage", "value"),
        State("training-lfp-mode", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _load_example(_n, stage, mode):
        hide = {"display": "none"}
        email = current_user_email()
        if not email:
            ph = _empty_lfp_fig("Sign in to start training.")
            return (None, _vid_placeholder("Sign in to start training."),
                    ph, ph, [], "", "", "", hide, "")
        stage = int(stage or 1)
        res = _load_next(store, email, stage, mode or "hilbert",
                         cfg["examples_per_round"], config)
        if res is None:
            ph = _empty_lfp_fig("No validated recordings yet.")
            return (None, _vid_placeholder(
                "No validated recordings to train on yet — check back once "
                "the PI has approved some."), ph, ph, [], "", "", "", hide, "")
        kind, payload = res
        if kind == "complete":
            prompt = _build_prompt(store, email, stage, payload)
            return (no_update, no_update, no_update, no_update, no_update, "",
                    "Round complete — choose below.", prompt,
                    {"display": "block"},
                    _history_text(store, email, stage, payload))
        cur, vid, lfp, hil, now, hist = payload
        return cur, vid, lfp, hil, [], "", now, "", hide, hist

    # ---- 2nd-panel filter: Hilbert envelope vs AUC ---- #
    @app.callback(
        Output("training-hilbert", "figure", allow_duplicate=True),
        Input("training-lfp-mode", "value"),
        State("training-current", "data"),
        prevent_initial_call=True,
    )
    def _toggle_lfp_mode(mode, current):
        if not current or not current.get("file_id"):
            return no_update
        return _filtered_trace(store, int(current["file_id"]),
                               int(current.get("channel") or 0),
                               mode or "hilbert")

    # ---- Stage-dependent scoring form ---- #
    @app.callback(
        Output("training-form", "children"),
        Input("training-stage", "value"),
        Input("training-events", "data"),
    )
    def _render_form(stage, events):
        # The stage-1 detection radio is permanent in the layout (toggled by
        # _toggle_impression), so for stage 1 this form area is empty.
        stage = int(stage or 1)
        if stage == 1:
            return ""
        # Stages 2 / 3: event rows + add button.
        rows = [_event_row(e, i, stage)
                for i, e in enumerate(events or [])]
        return html.Div([
            html.Div(rows or [html.Div(
                "No events added. If you think this recording has no "
                "BHZ, just submit. Otherwise add one.",
                style={"color": COLOR_TEXT_TERTIARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginBottom": SPACE_2})]),
            button("+ Add event", "training-add-btn", variant="secondary",
                   style={"marginTop": SPACE_2}),
        ])

    # ---- Show the detection radio for stage 1 only ---- #
    @app.callback(
        Output("training-detect-wrap", "style"),
        Input("training-stage", "value"),
    )
    def _toggle_impression(stage):
        return ({} if int(stage or 1) == 1 else {"display": "none"})

    # ---- Reset the detection answer when a new example loads ---- #
    @app.callback(
        Output("training-detect", "value"),
        Input("training-current", "data"),
        prevent_initial_call=True,
    )
    def _reset_detect(_current):
        return None

    # ---- Events editor: add / remove / set fields ---- #
    @app.callback(
        Output("training-events", "data", allow_duplicate=True),
        Input("training-add-btn", "n_clicks"),
        State("training-events", "data"),
        prevent_initial_call=True,
    )
    def _add_event(_n, events):
        if not _n:
            return no_update
        events = list(events or [])
        events.append({"type": "", "racine": None, "EO_sec": None,
                       "LAS_sec": None, "BO_sec": None, "PID_sec": None,
                       "BB_sec": None})
        return events

    @app.callback(
        Output("training-events", "data", allow_duplicate=True),
        Input({"type": "tr-remove", "idx": ALL}, "n_clicks"),
        State("training-events", "data"),
        prevent_initial_call=True,
    )
    def _remove_event(_clicks, events):
        if not any((t.get("value") or 0)
                   for t in (callback_context.triggered or [])):
            return no_update
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update
        idx = trig.get("idx")
        events = list(events or [])
        if idx is None or idx < 0 or idx >= len(events):
            return no_update
        events.pop(idx)
        return events

    @app.callback(
        Output("training-events", "data", allow_duplicate=True),
        Input({"type": "tr-type", "idx": ALL}, "value"),
        Input({"type": "tr-racine", "idx": ALL}, "value"),
        Input({"type": "tr-lm", "idx": ALL, "field": ALL}, "value"),
        State("training-events", "data"),
        prevent_initial_call=True,
    )
    def _set_fields(_types, _racines, _lms, events):
        events = list(events or [])
        if not events:
            return no_update
        changed = False
        for group in (callback_context.inputs_list or []):
            for item in group:
                cid = item.get("id") or {}
                idx = cid.get("idx")
                if idx is None or idx < 0 or idx >= len(events):
                    continue
                val = item.get("value")
                kind = cid.get("type")
                ev = dict(events[idx])
                if kind == "tr-type":
                    new = val or ""
                    if new != (ev.get("type") or ""):
                        ev["type"] = new
                        changed = True
                elif kind == "tr-racine":
                    new = int(val) if val not in (None, "") else None
                    if new != ev.get("racine"):
                        ev["racine"] = new
                        changed = True
                elif kind == "tr-lm":
                    field = f"{cid.get('field')}_sec"
                    new = float(val) if val not in (None, "") else None
                    if new != ev.get(field):
                        ev[field] = new
                        changed = True
                if changed:
                    events[idx] = ev
        return events if changed else no_update

    # ---- Submit: grade, persist, reveal feedback ---- #
    @app.callback(
        Output("training-feedback", "children"),
        Output("training-refresh", "data"),
        Output("training-prompt", "children", allow_duplicate=True),
        Output("training-prompt", "style", allow_duplicate=True),
        Output("training-history", "children", allow_duplicate=True),
        Input("training-submit-btn", "n_clicks"),
        State("training-stage", "value"),
        State("training-detect", "value"),
        State("training-events", "data"),
        State("training-current", "data"),
        State("training-refresh", "data"),
        prevent_initial_call=True,
    )
    def _submit(_n, stage, detect, events, current, refresh):
        nope = (no_update,) * 5
        if not _n:
            return nope
        email = current_user_email()
        if not email:
            return (_note("Sign in to submit.", COLOR_DANGER),) + nope[1:]
        if not current:
            return (_note("Load an example first.", COLOR_WARNING),) + nope[1:]
        stage = int(stage or 1)
        validated = current.get("validated") or []
        if stage == 1:
            if detect not in ("yes", "no"):
                return (_note("Pick yes or no first.", COLOR_WARNING),
                        ) + nope[1:]
            answer = {"events_present": detect == "yes", "events": []}
        else:
            answer = {"events_present": bool(events), "events": events or []}
        result = _grade.grade_attempt(
            stage, validated, answer,
            onset_tol_s=cfg["onset_tol_s"], racine_tol=cfg["racine_tol"])
        score = result["score"]
        try:
            store.add_training_attempt(
                email, stage, int(current["file_id"]),
                json.dumps(answer), float(score),
                json.dumps(result.get("breakdown")))
        except Exception as e:
            return (_note(f"Save failed: {e}", COLOR_DANGER),) + nope[1:]
        fb = _feedback(stage, score, result.get("breakdown", {}),
                       answer, validated)
        hide = {"display": "none"}
        rnd = store.current_training_round(email, stage)
        hist = _history_text(store, email, stage, rnd)
        nxt = int(refresh or 0) + 1
        if rnd is not None and _round_idx(store, email, stage,
                                          rnd) >= rnd["examples"]:
            return (fb, nxt, _build_prompt(store, email, stage, rnd),
                    {"display": "block"}, hist)
        return fb, nxt, no_update, hide, hist

    # ---- Post-round prompt: review more (reset grade) ---- #
    @app.callback(
        Output("training-current", "data", allow_duplicate=True),
        Output("training-video", "children", allow_duplicate=True),
        Output("training-lfp", "figure", allow_duplicate=True),
        Output("training-hilbert", "figure", allow_duplicate=True),
        Output("training-events", "data", allow_duplicate=True),
        Output("training-feedback", "children", allow_duplicate=True),
        Output("training-now", "children", allow_duplicate=True),
        Output("training-prompt", "children", allow_duplicate=True),
        Output("training-prompt", "style", allow_duplicate=True),
        Output("training-history", "children", allow_duplicate=True),
        Output("training-refresh", "data", allow_duplicate=True),
        Input("training-review-more-btn", "n_clicks"),
        State("training-stage", "value"),
        State("training-lfp-mode", "value"),
        State("training-confidence", "value"),
        State("training-refresh", "data"),
        prevent_initial_call=True,
    )
    def _review_more(_n, stage, mode, confidence, refresh):
        if not _n:
            return (no_update,) * 11
        email = current_user_email()
        stage = int(stage or 1)
        hide = {"display": "none"}
        bump = int(refresh or 0) + 1
        rnd = store.current_training_round(email, stage)
        if rnd is not None:
            # Close the round -> a fresh round starts a new reset boundary,
            # so the current grade restarts (history is kept).
            store.finish_training_round(rnd["round_id"], confidence,
                                        "review_more")
        res = _load_next(store, email, stage, mode or "hilbert",
                         cfg["examples_per_round"], config)
        if res is None or res[0] != "loaded":
            return ((no_update,) * 7 + ("", hide,
                    _history_text(store, email, stage, None), bump))
        cur, vid, lfp, hil, now, hist = res[1]
        return cur, vid, lfp, hil, [], "", now, "", hide, hist, bump

    # ---- Post-round prompt: move on (notify PI) ---- #
    @app.callback(
        Output("training-prompt-status", "children"),
        Output("training-refresh", "data", allow_duplicate=True),
        Input("training-moveon-btn", "n_clicks"),
        State("training-stage", "value"),
        State("training-confidence", "value"),
        State("training-refresh", "data"),
        prevent_initial_call=True,
    )
    def _move_on(_n, stage, confidence, refresh):
        if not _n:
            return no_update, no_update
        email = current_user_email()
        stage = int(stage or 1)
        rnd = store.current_training_round(email, stage)
        if rnd is None:
            return "No active round.", no_update
        store.finish_training_round(rnd["round_id"], confidence, "move_on")
        sent = (_notify_pi(store, config, email, stage, rnd)
                if store.mark_round_notified(rnd["round_id"]) else False)
        msg = ("Sent to your PI — they'll review your performance and "
               "advance you." if sent else
               "Recorded. Your PI reviews and gates the next stage.")
        return msg, int(refresh or 0) + 1

    # ---- PI roster ---- #
    @app.callback(
        Output("training-roster", "children"),
        Input("training-refresh", "data"),
        State("training-is-pi", "data"),
    )
    def _roster(_refresh, is_pi):
        if not is_pi:
            return no_update
        roster = store.all_training_progress()
        if not roster:
            return empty_state("No students yet",
                                hint="Attempts appear here as students "
                                     "start training.", icon_name="inbox")
        body = []
        for r in roster:
            em = r["student_email"]
            unlocked = int(r.get("unlocked_stage", 1))
            certified = bool(r.get("certified_at"))
            # Current grade is round-scoped (resets when they review more);
            # lifetime is the full history.
            rnd = store.current_training_round(em, unlocked)
            scores = (store.round_scores(em, unlocked,
                                         rnd["started_after_id"]) if rnd else
                      store.recent_training_scores(em, unlocked,
                                                   limit=cfg["window"]))
            mean = (sum(scores) / len(scores)) if scores else 0.0
            ready = _grade.rolling_ready(scores, cfg["pass_pct"],
                                         cfg["window"])
            life = store.training_lifetime_stats(em, unlocked)
            lifetxt = (f"{life['mean'] * 100:.0f}% / {life['n']}"
                       if life["n"] else "—")
            status = ("Certified" if certified
                      else ("Ready" if ready else "In progress"))
            label = ("Certify" if unlocked >= 3 else
                     f"Advance to {unlocked + 1}")
            body.append(html.Tr([
                html.Td(em, style=_TD),
                html.Td(f"{unlocked}. {STAGE_LABEL[unlocked]}", style=_TD),
                html.Td(f"{mean * 100:.0f}% (n={len(scores)})", style=_TD),
                html.Td(lifetxt, style=_TD),
                html.Td(status, style={**_TD, "color": (
                    COLOR_SUCCESS if certified or ready
                    else COLOR_TEXT_SECONDARY)}),
                html.Td(button(
                    label, {"type": "training-advance", "email": em},
                    variant="secondary",
                    style={"padding": "2px 10px",
                            "fontSize": FONT_SIZE_CAPTION},
                    **({"disabled": True} if certified else {})),
                    style=_TD),
                html.Td(html.Div([
                    dcc.Dropdown(
                        id={"type": "training-reset-stage", "email": em},
                        options=[{"label": "Start", "value": 1},
                                 {"label": "Stage 2", "value": 2},
                                 {"label": "Stage 3", "value": 3}],
                        value=1, clearable=False,
                        style={"width": "92px", "fontSize": FONT_SIZE_CAPTION},
                        className="dark-dropdown"),
                    button("Reset", {"type": "training-reset", "email": em},
                           variant="ghost",
                           style={"padding": "2px 10px",
                                   "fontSize": FONT_SIZE_CAPTION}),
                ], style={"display": "flex", "alignItems": "center",
                           "gap": SPACE_2}), style=_TD),
            ]))
        return html.Table([
            html.Thead(html.Tr([
                html.Th(h, style=_TH) for h in
                ("Student", "Stage", "Round grade", "Lifetime", "Status",
                 "", "Reset to")])),
            html.Tbody(body),
        ], style={"width": "100%", "borderCollapse": "collapse"})

    # ---- PI advance ---- #
    @app.callback(
        Output("training-refresh", "data", allow_duplicate=True),
        Input({"type": "training-advance", "email": ALL}, "n_clicks"),
        State("training-refresh", "data"),
        State("training-is-pi", "data"),
        prevent_initial_call=True,
    )
    def _advance(_clicks, refresh, is_pi):
        if not is_pi:
            return no_update
        if not any((t.get("value") or 0)
                   for t in (callback_context.triggered or [])):
            return no_update
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update
        email = trig.get("email")
        if not email:
            return no_update
        prog = store.get_training_progress(email)
        store.advance_student(email, int(prog.get("unlocked_stage", 1)) + 1)
        _activity.track(store, "training", "advance_student", email)
        return int(refresh or 0) + 1

    # ---- PI reset: send a student back to start / a stage ---- #
    @app.callback(
        Output("training-refresh", "data", allow_duplicate=True),
        Input({"type": "training-reset", "email": ALL}, "n_clicks"),
        State({"type": "training-reset-stage", "email": ALL}, "value"),
        State({"type": "training-reset-stage", "email": ALL}, "id"),
        State("training-refresh", "data"),
        State("training-is-pi", "data"),
        prevent_initial_call=True,
    )
    def _reset(_clicks, stages, stage_ids, refresh, is_pi):
        if not is_pi:
            return no_update
        if not any((t.get("value") or 0)
                   for t in (callback_context.triggered or [])):
            return no_update
        trig = callback_context.triggered_id
        if not isinstance(trig, dict) or not trig.get("email"):
            return no_update
        email = trig["email"]
        # Find this student's chosen reset-stage from the matched ALL lists.
        stage = 1
        for sid, val in zip(stage_ids or [], stages or []):
            if isinstance(sid, dict) and sid.get("email") == email:
                stage = int(val or 1)
                break
        store.reset_student(email, stage)
        _activity.track(store, "training", "reset_student", email,
                        {"stage": stage})
        return int(refresh or 0) + 1

    # ---- Practice library: auto-catalog past scored events ---- #
    @app.callback(
        Output("training-import-status", "children"),
        Input("training-import-tick", "n_intervals"),
        prevent_initial_call=True,
    )
    def _poll_import(_n):
        global _IMPORT_AUTOSTARTED
        enabled = _past_enabled(config)
        # Not optional: the first poll after the tab mounts kicks the import
        # automatically (once per process). The button is only a re-scan.
        if (enabled and not _IMPORT_AUTOSTARTED
                and not _IMPORT_STATE["running"]
                and not _IMPORT_STATE["finished"]):
            _IMPORT_AUTOSTARTED = True
            _kick_import(store, config)
        return _import_status_text(enabled)

    # ---- Camera picker: populate from the loaded recording ---- #
    @app.callback(
        Output("training-cam", "options"),
        Output("training-cam", "value"),
        Output("training-cam-wrap", "style"),
        Input("training-current", "data"),
        prevent_initial_call=True,
    )
    def _cam_options(current):
        hide = {"display": "none"}
        if not current or not current.get("file_id"):
            return [], 1, hide
        from src.utils.video import companion_videos
        mat_path = _file_path_for_id(store, int(current["file_id"]))
        cams = companion_videos(mat_path) if mat_path else []
        opts = [{"label": f" Cam {c['cam']}", "value": c["cam"]}
                for c in cams]
        first = cams[0]["cam"] if cams else 1
        if len(cams) < 2:                 # no picker for single/no camera
            return opts, first, hide
        return opts, first, {"display": "flex", "alignItems": "center",
                             "marginBottom": SPACE_2}

    # ---- Render the video (camera + .avi transcode polling) ---- #
    @app.callback(
        Output("training-video", "children", allow_duplicate=True),
        Output("training-vidsig", "data"),
        Output("training-cam-tick", "disabled"),
        Input("training-current", "data"),
        Input("training-cam", "value"),
        Input("training-cam-tick", "n_intervals"),
        State("training-vidsig", "data"),
        prevent_initial_call=True,
    )
    def _render_video(current, cam, _tick, last_sig):
        if not current or not current.get("file_id"):
            return no_update, no_update, no_update
        children, sig = _video_children_and_sig(
            store, config, int(current["file_id"]), int(cam or 1))
        # Only keep the poll alive while an .avi transcode is in progress;
        # once the video is ready/mp4 (or there's none) stop ticking so the
        # dcc.Loading spinner doesn't flash over a playing video every 2.5 s.
        keep_ticking = sig.endswith(":prep")
        if sig == last_sig:
            return no_update, no_update, (not keep_ticking)
        return children, sig, (not keep_ticking)

    _register_cursor_sync(app)


# ===================================================================== #
#  Video <-> LFP cursor sync (clientside; mirrors the Video Review tab)
# ===================================================================== #

# Move the cursor (shapes[0]) on one panel to the video's current time,
# mapped onto the LFP duration so container-FPS drift can't accumulate:
#   cursor_x = currentTime * (lfp_dur / video_duration).
# shapes[1:] (e.g. the Hilbert threshold line) are preserved untouched.
_CURSOR_JS = """
function(currentTime, fig, current) {
    if (fig === undefined || fig === null) {
        return window.dash_clientside.no_update;
    }
    if (currentTime === null || currentTime === undefined) {
        return window.dash_clientside.no_update;
    }
    var t = currentTime;
    var v = document.getElementById('""" + TRAINING_VIDEO_DOM_ID + """');
    var lfp_dur = (current && current.lfp_dur) ? current.lfp_dur : 0;
    if (v && isFinite(v.duration) && v.duration > 0
            && lfp_dur && lfp_dur > 0) {
        t = currentTime * (lfp_dur / v.duration);
    }
    var keep = ((fig.layout && fig.layout.shapes) || []).slice(1);
    var cursor = {
        type: 'line', xref: 'x', yref: 'paper',
        x0: t, x1: t, y0: 0, y1: 1,
        line: {color: '#ff9f0a', width: 2}
    };
    return {
        data: fig.data,
        layout: Object.assign({}, fig.layout, {
            shapes: [cursor].concat(keep)
        })
    };
}
"""


def _xsync_js(target_id: str) -> str:
    """Clientside: when the source graph's x-range (or autorange) changes,
    Plotly.relayout the target graph to match. The echo guard (skip when the
    target already holds that range) stops the two callbacks ping-ponging.
    Same scheme the Video Review tab uses to lock its LFP + analysis x-axes;
    uirevision keeps the zoom across the cursor/figure redraws."""
    return ("""
    function(rel) {
        if (!rel) { return window.dash_clientside.no_update; }
        var hasRange = ('xaxis.range[0]' in rel
                         && 'xaxis.range[1]' in rel);
        var hasAuto = !!rel['xaxis.autorange'];
        if (!hasRange && !hasAuto) {
            return window.dash_clientside.no_update;
        }
        var host = document.getElementById('%s');
        var gd = null;
        if (host) {
            gd = host.classList
                  && host.classList.contains('js-plotly-plot')
                 ? host : host.querySelector('.js-plotly-plot');
        }
        if (!gd || !window.Plotly) {
            return window.dash_clientside.no_update;
        }
        var cur = (gd.layout && gd.layout.xaxis)
                   ? gd.layout.xaxis.range : null;
        if (hasRange) {
            var x0 = rel['xaxis.range[0]'];
            var x1 = rel['xaxis.range[1]'];
            if (cur && Math.abs(cur[0] - x0) < 1e-6
                    && Math.abs(cur[1] - x1) < 1e-6) {
                return window.dash_clientside.no_update;
            }
            window.Plotly.relayout(gd, {
                'xaxis.range[0]': x0, 'xaxis.range[1]': x1});
        } else {
            if (gd.layout && gd.layout.xaxis
                    && gd.layout.xaxis.autorange === true) {
                return window.dash_clientside.no_update;
            }
            window.Plotly.relayout(gd, {'xaxis.autorange': true});
        }
        return window.dash_clientside.no_update;
    }
    """ % target_id)


def _register_cursor_sync(app) -> None:
    """Wire the Training <video> to the cursor on both LFP panels, plus
    click-to-seek and the LFP<->Hilbert x-axis lock -- the same scheme the
    Video Review tab uses."""
    # 1. Poll the <video> element 10 Hz; push currentTime into a Store.
    app.clientside_callback(
        """
        function(_n) {
            const v = document.getElementById('""" +
        TRAINING_VIDEO_DOM_ID + """');
            if (!v || isNaN(v.currentTime)) {
                return window.dash_clientside.no_update;
            }
            return v.currentTime;
        }
        """,
        Output("training-current-time", "data"),
        Input("training-time-tick", "n_intervals"),
        prevent_initial_call=True,
    )

    # 2. Mirror the current time onto the cursor of each panel.
    for graph_id in ("training-lfp", "training-hilbert"):
        app.clientside_callback(
            _CURSOR_JS,
            Output(graph_id, "figure", allow_duplicate=True),
            Input("training-current-time", "data"),
            State(graph_id, "figure"),
            State("training-current", "data"),
            prevent_initial_call=True,
        )

    # 3. Click the LFP trace -> seek the video to that x value.
    #    Inverse mapping: video_t = x * (video_duration / lfp_dur).
    app.clientside_callback(
        """
        function(clickData, current) {
            if (!clickData || !clickData.points || !clickData.points.length) {
                return '';
            }
            const x = clickData.points[0].x;
            const v = document.getElementById('""" +
        TRAINING_VIDEO_DOM_ID + """');
            if (!v || !isFinite(x)) { return ''; }
            var lfp_dur = (current && current.lfp_dur) ? current.lfp_dur : 0;
            var vt = x;
            if (isFinite(v.duration) && v.duration > 0
                    && lfp_dur && lfp_dur > 0) {
                vt = x * (v.duration / lfp_dur);
            }
            if (vt < 0) { vt = 0; }
            if (isFinite(v.duration) && vt > v.duration) { vt = v.duration; }
            v.currentTime = vt;
            return '';
        }
        """,
        Output("training-seek-sink", "children"),
        Input("training-lfp", "clickData"),
        State("training-current", "data"),
        prevent_initial_call=True,
    )

    # 4. Lock the LFP and Hilbert x-axes together: zooming/panning either
    #    pans the other to the same window (echo-guarded against ping-pong).
    app.clientside_callback(
        _xsync_js("training-hilbert"),
        Output("training-xsync-sink", "data", allow_duplicate=True),
        Input("training-lfp", "relayoutData"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        _xsync_js("training-lfp"),
        Output("training-xsync-sink", "data", allow_duplicate=True),
        Input("training-hilbert", "relayoutData"),
        prevent_initial_call=True,
    )


# ===================================================================== #
#  Small render helpers (module-level; no app/store needed)
# ===================================================================== #

_TD = {"padding": f"{SPACE_2} {SPACE_3}",
       "borderBottom": f"1px solid {COLOR_DIVIDER}",
       "fontSize": FONT_SIZE_BODY, "color": COLOR_TEXT_PRIMARY}
_TH = {"padding": f"{SPACE_2} {SPACE_3}",
       "borderBottom": f"1px solid {COLOR_DIVIDER}",
       "fontSize": FONT_SIZE_CAPTION, "color": COLOR_TEXT_SECONDARY,
       "textAlign": "left", "textTransform": "uppercase",
       "letterSpacing": "0.4px"}


def _pill(text: str, color: str) -> html.Span:
    return html.Span(text, style={
        "padding": "3px 10px", "borderRadius": "10px",
        "background": color, "color": "#10120a" if color == COLOR_WARNING
        else "white", "fontSize": FONT_SIZE_CAPTION, "fontWeight": "700"})


def _note(text: str, color: str) -> html.Div:
    return html.Div(text, style={"color": color, "fontSize": FONT_SIZE_BODY,
                                  "padding": SPACE_2})


def _vid_placeholder(text: str) -> html.Div:
    return empty_state("No recording loaded", hint=text, icon_name="video")


def _cam_video(file_id: int, cam: int):
    """The <video> element for one camera (mp4 served directly, .avi served
    as its cached transcode) -- keeps the cursor-sync DOM id."""
    return html.Video(
        id=TRAINING_VIDEO_DOM_ID,
        src=f"/media/video/{file_id}/cam/{cam}",
        controls=True, preload="metadata",
        style={"width": "100%", "maxHeight": "360px",
               "borderRadius": RADIUS_MD, "background": "#000"})


def _video_children_and_sig(store: Store, config: dict, file_id: int,
                             cam: int):
    """Render the video area for *cam* and a signature string. The signature
    lets the render callback skip redundant re-renders (so a playing <video>
    isn't reset) -- it changes only when the file, camera, or transcode
    readiness changes."""
    from src.utils import avi_transcode as _avi
    from src.utils.video import companion_videos
    mat_path = _file_path_for_id(store, file_id)
    cams = companion_videos(mat_path) if mat_path else []
    if not cams:
        return (_vid_placeholder("No companion video for this recording — "
                                 "score from the EEG below."),
                f"{file_id}:none")
    entry = next((e for e in cams if e["cam"] == cam), cams[0])
    cn = entry["cam"]
    if entry["kind"] == "mp4":
        return _cam_video(file_id, cn), f"{file_id}:{cn}:mp4"
    # .avi -> on-demand transcode to a cached mp4 (background).
    if _avi.ready_path(entry["path"], config):
        return _cam_video(file_id, cn), f"{file_id}:{cn}:ready"
    _avi.ensure_async(entry["path"], config)
    if _avi.status(entry["path"], config) == "error":
        return (_vid_placeholder("Couldn't prepare this video — score from "
                                 "the EEG below."), f"{file_id}:{cn}:err")
    return (_vid_placeholder(f"Preparing camera {cn} video… (converting .avi "
                             f"to a playable format, first time only)"),
            f"{file_id}:{cn}:prep")


def _event_row(event: dict, idx: int, stage: int) -> html.Div:
    type_opts = ([{"label": " LVF", "value": "LVF"},
                  {"label": " HYP", "value": "HYP"}]
                 + ([{"label": " Undefined", "value": "Undefined"}]
                    if stage == 3 else []))
    controls = [
        html.Span(f"Event {idx + 1}",
                  style={"fontWeight": "700", "color": COLOR_TEXT_PRIMARY,
                          "fontSize": FONT_SIZE_BODY,
                          "marginRight": SPACE_3}),
        dcc.RadioItems(id={"type": "tr-type", "idx": idx},
                        options=type_opts, value=event.get("type") or None,
                        inline=True,
                        labelStyle={"marginRight": SPACE_3,
                                     "color": COLOR_TEXT_PRIMARY,
                                     "fontSize": FONT_SIZE_CAPTION}),
        html.Span("Racine", style={"color": COLOR_TEXT_SECONDARY,
                                     "fontSize": FONT_SIZE_CAPTION,
                                     "marginRight": SPACE_2}),
        dcc.Dropdown(id={"type": "tr-racine", "idx": idx},
                      options=[{"label": str(n), "value": n}
                               for n in range(1, 9)],
                      value=event.get("racine"),
                      style={"width": "70px"}, className="dark-dropdown"),
    ]
    lms = _LANDMARKS if stage == 3 else ("EO",)
    for lm in lms:
        controls.append(html.Span(
            lm, style={"color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginLeft": SPACE_2, "marginRight": "2px"}))
        controls.append(dcc.Input(
            id={"type": "tr-lm", "idx": idx, "field": lm},
            type="number", value=event.get(f"{lm}_sec"),
            placeholder="s", step="any",
            style={"width": "70px", "backgroundColor": COLOR_SURFACE_2,
                   "color": COLOR_TEXT_PRIMARY,
                   "border": f"1px solid {COLOR_DIVIDER}",
                   "borderRadius": RADIUS_SM, "padding": "3px 6px"}))
    controls.append(button("✕", {"type": "tr-remove", "idx": idx},
                            variant="ghost",
                            style={"marginLeft": "auto",
                                    "padding": "2px 8px"}))
    return html.Div(controls, style={
        "display": "flex", "alignItems": "center", "flexWrap": "wrap",
        "gap": f"{SPACE_2} {SPACE_2}", "padding": f"{SPACE_2} {SPACE_3}",
        "marginBottom": SPACE_2, "background": COLOR_SURFACE_1,
        "border": f"1px solid {COLOR_DIVIDER}", "borderRadius": RADIUS_SM})


def _feedback(stage, score, breakdown, answer, validated) -> html.Div:
    pct = round(score * 100)
    color = (COLOR_SUCCESS if pct >= 85 else
             COLOR_WARNING if pct >= 60 else COLOR_DANGER)
    if stage == 1:
        you = ("events present" if answer.get("events_present")
               else "no events")
        val = ("events present" if _grade._real_events(validated)
               else "no events")
        detail = html.Div(f"You said: {you}  ·  Validated: {val}",
                          style={"color": COLOR_TEXT_SECONDARY,
                                  "fontSize": FONT_SIZE_CAPTION,
                                  "marginTop": SPACE_2})
    else:
        b = breakdown or {}
        detail = html.Div([
            html.Div(
                f"matched {b.get('matched', 0)} · missed "
                f"{b.get('missed', 0)} · extra {b.get('false_pos', 0)} "
                f"(of {b.get('validated', 0)} validated)",
                style={"color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginTop": SPACE_2}),
            html.Div([
                _events_table(answer.get("events") or [], "Your answer",
                              COLOR_ACCENT),
                _events_table(validated, "Validated answer", COLOR_SUCCESS),
            ], style={"display": "flex", "gap": SPACE_3,
                       "flexWrap": "wrap", "marginTop": SPACE_3}),
        ])
    return card(
        html.Div([
            html.Span(f"{pct}%", style={"fontSize": "28px",
                                         "fontWeight": "800",
                                         "color": color}),
            html.Span(" agreement with the validated answer",
                      style={"color": COLOR_TEXT_SECONDARY,
                              "fontSize": FONT_SIZE_BODY,
                              "marginLeft": SPACE_2}),
        ]),
        detail,
        html.Div("Use “Load an example” above for the next recording.",
                 style={"color": COLOR_TEXT_TERTIARY,
                         "fontSize": FONT_SIZE_CAPTION,
                         "marginTop": SPACE_3}),
        style={"borderLeft": f"3px solid {color}", "marginTop": SPACE_4},
    )
