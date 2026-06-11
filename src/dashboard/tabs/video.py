"""Video Review tab — synchronized video + LFP trace.

Lets a logged-in reviewer pick an LFP file that has a companion video,
watch the ``_v1.mp4`` in the browser, and see the corresponding LFP
channel time-locked to the video below. Scrubbing or playing the
video moves a cursor along the LFP trace; clicking the LFP trace
seeks the video to that moment. Reviewers can leave a structured
note that gets written to ``annotations`` tagged with their email
and ``category='video_review'``.

The video stream itself is served by ``src.dashboard.media_routes`` at
``/media/video/<file_id>`` and inherits the Cloudflare Access auth
gate.

Time-locking is done entirely on the client to keep latency low. The
server only renders the static LFP figure once per file/channel
selection; clientside callbacks handle the cursor + click-to-seek.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from collections import OrderedDict
from threading import RLock
from dash import (Input, Output, State, dcc, html, no_update,
                   Patch, ALL, callback_context)
from dash.dependencies import ClientsideFunction

from src.dashboard.auth import current_user_email
from src.utils import assignments as _assignments

from src.db.store import Store
from src.dashboard.auth import current_user_email
from src.utils.chunk_cache import get_chunk
from src.utils.hilbert_envelope import hilbert_envelope_20_200
from src.utils.peakseek import peakseek
from src.utils import bhz_csv as _bhz_csv
from src.utils.animal import split_animal_electrode, is_animal_channel
from src.dashboard.tabs import video_events as _events
from src.utils import mass_analyze as _mass_analyze
from src.utils.decimate import (
    envelope, window_slice, choose_target_bins, parse_relayout,
)
from src.utils.filters import apply_filter

logger = logging.getLogger("qc_monitor.dashboard.video")

# Initial-render decimation target. The zoom callback re-decimates the
# visible window dynamically, so this only governs the first paint and
# the auto-reset view.
INITIAL_TARGET_BINS = 60_000

# Stim-blanked per-channel series cache. Re-blanking on every zoom is
# wasteful when only the visible window changed; keyed by
# (file_path, channel, blank_pre_ms, blank_post_ms, stim_fingerprint).
_BLANKED_MAX = 8
_blanked_cache: "OrderedDict[tuple, tuple[np.ndarray, float]]" = OrderedDict()
_blanked_lock = RLock()

LABEL_STYLE = {
    "color": "#6c6c80", "fontSize": "11px", "marginBottom": "8px",
    "display": "block", "letterSpacing": "0.4px",
    "textTransform": "uppercase", "fontWeight": "600",
}
DROPDOWN_STYLE = {"backgroundColor": "#262638", "color": "#f0f0f5"}
SECTION_STYLE = {
    "background": "#13131f", "padding": "24px",
    "borderRadius": "10px", "border": "1px solid rgba(255,255,255,0.07)",
    "marginBottom": "16px",
}

# DOM id assigned to the <video> element so clientside JS can find it.
VIDEO_DOM_ID = "lfp-video"


# ===================================================================== #
#  Data helpers
# ===================================================================== #

def _sessions_with_video(store: Store) -> list[dict]:
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT session_dir, session_name, COUNT(*) AS n_videos
               FROM processed_files
               WHERE has_video = 1
               GROUP BY session_dir
               ORDER BY MAX(chunk_datetime) DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def _files_with_video(store: Store, session_dir: str) -> list[dict]:
    if not session_dir:
        return []
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT id, file_path, chunk_datetime
               FROM processed_files
               WHERE session_dir = ? AND has_video = 1
               ORDER BY chunk_datetime""",
            (session_dir,),
        ).fetchall()
        return [dict(r) for r in rows]


def _file_path_for_id(store: Store, file_id: int) -> str | None:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT file_path FROM processed_files WHERE id = ?", (file_id,),
        ).fetchone()
        return row["file_path"] if row else None


def _stim_times_for_file(store: Store, file_id: int) -> np.ndarray:
    """Return stim onset times (sec) detected by the MATLAB pipeline.

    Reads ``evoked_features.epoch_time_sec`` — each epoch corresponds
    to one detected stim event, so this is exactly the list of onsets.
    Returns an empty array for files that haven't been MATLAB-processed.
    """
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT DISTINCT epoch_time_sec
               FROM evoked_features
               WHERE file_id = ? AND epoch_time_sec IS NOT NULL
               ORDER BY epoch_time_sec""",
            (file_id,),
        ).fetchall()
    return np.asarray([r["epoch_time_sec"] for r in rows], dtype=np.float64)


def _stim_copy_channels(store: Store, session_dir: str | None) -> set[int]:
    """Return the set of channel indices flagged as stim-copy.

    Stim-copy channels record the stimulator output itself; blanking
    them out would erase the only signal worth seeing, so we leave
    them as-is.
    """
    if not session_dir:
        return set()
    cfg = store.get_session_config(session_dir) or {}
    raw = cfg.get("stim_copy_channels")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    if isinstance(raw, list):
        return {int(c) for c in raw}
    return set()


def _should_stim_blank(channel: int,
                        stim_copy: set[int]) -> bool:
    """Whether the display trace for *channel* should be stim-blanked.

    The lab's recording rig wires each animal's LFP electrode
    immediately after its corresponding stim-copy channel. The shared
    stim artifact bleeds into the immediately-following LFP via
    capacitive coupling but the next LFP (two slots after stim_copy)
    is far enough away to read cleanly. So the rule is: blank iff
    ``channel - 1`` is a stim-copy channel AND ``channel`` itself
    isn't.

    Example: channel_names ``["stimCopy", "BCH061SLM", "stimCopy",
    "BCH062SR", "BCH062SLM"]`` -> blank Ch1 and Ch3 only. Ch4 is
    two slots away from a stim-copy so it stays raw.
    """
    if channel in stim_copy:
        return False
    return (channel - 1) in stim_copy


def _session_dir_for_file(store: Store, file_id: int) -> str | None:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT session_dir FROM processed_files WHERE id = ?", (file_id,),
        ).fetchone()
        return row["session_dir"] if row else None


def _animal_ids_from_picker(animal_value: str | None
                              ) -> list[str]:
    """Resolve the Step-1 animal picker value into the
    ``animal_ids`` list ``store.get_review_queue`` expects.

    The picker holds either a plain animal id, the
    ``__UNASSIGNED__`` sentinel (no real selection), or a
    ``_pool_<id>`` value pointing at an animal from the
    unassigned pool. Returns ``[]`` for None / sentinel.
    """
    if not animal_value or animal_value == "__UNASSIGNED__":
        return []
    if animal_value.startswith("_pool_"):
        return [animal_value[len("_pool_"):]]
    return [animal_value]


def _ma_animal_from_picker(animal_value: str | None
                             ) -> str | None:
    """Single plain animal id from the Step-1 picker value, or
    None. Mass Analyze operates on one animal, so it takes the
    first id ``_animal_ids_from_picker`` resolves (strips the
    ``_pool_`` prefix / drops the ``__UNASSIGNED__`` sentinel)."""
    ids = _animal_ids_from_picker(animal_value)
    return ids[0] if ids else None


def _pi_flag_note_for_file(store, file_id: int) -> str:
    """Return the most-recent ``pi_flag`` note for *file_id* or
    an empty string if there isn't one.

    Used by the queue card to show "🚩 PI flagged: ..." on
    re-queued files so the undergrad sees the PI's instructions
    without leaving the tab.
    """
    assert isinstance(file_id, int), "file_id must be int"
    with store.connection() as conn:
        row = conn.execute(
            """SELECT note FROM review_state
               WHERE file_id = ? AND status = 'pi_flagged'
               ORDER BY updated_at DESC LIMIT 1""",
            (file_id,),
        ).fetchone()
    return (row["note"] or "") if row else ""


def _prefetch_chunk_safe(file_path: str, file_id: int) -> None:
    """Background-thread chunk warm. Swallows + logs errors so a
    bad file doesn't crash the prefetch loop.

    NOT for use outside the prefetch daemon -- the dashboard's
    foreground render paths want exceptions to surface.
    """
    assert isinstance(file_path, str) and file_path, "path required"
    assert isinstance(file_id, int), "file_id must be int"
    try:
        get_chunk(file_path)
        logger.debug("prefetched chunk for file_id=%s", file_id)
    except Exception as e:
        logger.debug("prefetch skipped for file_id=%s: %s",
                      file_id, e)


def _build_csv_file_meta(store, file_id: int, channel: int
                           ) -> tuple[dict, float, str, "date"]:
    """Assemble the file-level meta dict + (fs, animal_id,
    chunk_date) for a BHZ CSV row.

    Returns ``(file_meta, fs, animal_id, chunk_date)``. Raises
    on missing data so the caller can surface a clean error.
    """
    from datetime import datetime, date
    assert isinstance(file_id, int), "file_id must be int"
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        raise ValueError(f"file_id {file_id} has no file_path")
    session_dir = _session_dir_for_file(store, file_id)
    chunk = get_chunk(file_path)
    fs = float(chunk.fs)
    with store.connection() as conn:
        row = conn.execute(
            "SELECT chunk_datetime FROM processed_files "
            "WHERE id = ?", (file_id,),
        ).fetchone()
        chunk_dt_str = (row["chunk_datetime"] if row else "")
        cfg_row = conn.execute(
            "SELECT channel_names FROM session_config "
            "WHERE session_dir = ?", (session_dir,),
        ).fetchone()
    # Animal id = the prefix of the chosen channel's name (e.g.
    # "BCH062SLM" -> "BCH062"). Falls back to "unknown" if the
    # channel doesn't look like an animal channel.
    animal = "unknown"
    if cfg_row and cfg_row["channel_names"]:
        try:
            names = json.loads(cfg_row["channel_names"])
            if 0 <= channel < len(names):
                cname = names[channel]
                if is_animal_channel(cname):
                    animal, _ = split_animal_electrode(cname)
        except (json.JSONDecodeError, IndexError):
            pass
    # Derive Peak_Date / Peak_Time from chunk_datetime when we
    # can. Without an envelope detector running we leave
    # Peak_Index / Peak_Stamp empty (NaN).
    chunk_dt = None
    try:
        chunk_dt = datetime.strptime(chunk_dt_str,
                                      "%Y_%m_%d__%H_%M_%S")
    except (ValueError, TypeError):
        pass
    folder = os.path.dirname(file_path) + os.sep
    filename = os.path.basename(file_path)
    meta = _bhz_csv.build_file_meta(
        folder=folder, filename=filename,
        fs=fs, cutoff=0.05, channel=int(channel),
        peak_index=None, peak_stamp=None, peak_dt=chunk_dt,
    )
    chunk_date = (chunk_dt.date() if chunk_dt
                   else date.today())
    return meta, fs, animal, chunk_date


def _resolve_next_in_queue(store: Store,
                             animal_value: str | None,
                             current_file_id: int,
                             user_email: str,
                             *,
                             queue_limit: int = 100,
                             direction: int = 1,
                             ) -> tuple[str | None, int | None]:
    """Return ``(session_dir, file_id)`` of the next/prev queue
    entry, or ``(None, None)`` if there's nothing to advance to.

    Wraps ``store.neighbor_queue_file`` with the dropdown-value
    parsing and the session_dir lookup so the auto-advance,
    Undo, and the J/K cycle paths can all share one source of
    truth.
    """
    assert direction in (-1, 1), "direction must be -1 or 1"
    animal_ids = _animal_ids_from_picker(animal_value)
    if not animal_ids:
        return (None, None)
    floor = store.review_backlog_floor()
    next_file_id = store.neighbor_queue_file(
        current_file_id=current_file_id,
        animal_ids=animal_ids,
        user_email=(user_email or "").lower(),
        direction=direction,
        since_iso=floor,
        limit=queue_limit,
    )
    if next_file_id is None:
        return (None, None)
    sd = _session_dir_for_file(store, int(next_file_id))
    return (sd, int(next_file_id))


def _channel_options(store: Store, session_dir: str | None,
                     n_channels: int) -> list[dict]:
    """Build channel-name options for the LFP-review dropdown.

    Only EEG channels appear; stim_copy / saline / reference
    channels are never selectable as an LFP-review target because
    they don't carry behaviorally-interpretable signal. Falls back
    to numeric Ch0, Ch1, ... when no session_config row is on file
    (legacy chunks before auto-discovery landed).
    """
    if not session_dir:
        return [{"label": f"Ch{i}", "value": i}
                for i in range(n_channels)]
    cfg = store.get_session_config(session_dir) or {}
    raw_names = cfg.get("channel_names")
    if isinstance(raw_names, str):
        try:
            raw_names = json.loads(raw_names)
        except json.JSONDecodeError:
            raw_names = None
    raw_eeg = cfg.get("eeg_channels")
    if isinstance(raw_eeg, str):
        try:
            raw_eeg = json.loads(raw_eeg)
        except json.JSONDecodeError:
            raw_eeg = None
    # Without an explicit eeg_channels list, fall back to the whole
    # legacy "every channel" dropdown so we don't regress old sessions.
    if not isinstance(raw_eeg, list):
        options = [{"label": f"Ch{i}", "value": i}
                    for i in range(n_channels)]
        if isinstance(raw_names, list):
            for i in range(min(n_channels, len(raw_names))):
                options[i] = {
                    "label": f"{raw_names[i]} (Ch{i})", "value": i,
                }
        return options
    # Build EEG-only options with friendly labels.
    options: list[dict] = []
    for raw_idx in raw_eeg:
        try:
            i = int(raw_idx)
        except (TypeError, ValueError):
            continue
        if not (0 <= i < n_channels):
            continue
        name = (raw_names[i] if isinstance(raw_names, list)
                 and i < len(raw_names) else f"Ch{i}")
        options.append({"label": f"{name} (Ch{i})", "value": i})
    # If session_config exists but eeg_channels was empty for some
    # reason, give the operator every channel rather than nothing.
    if not options:
        return [{"label": f"Ch{i}", "value": i}
                for i in range(n_channels)]
    return options


def _stim_fingerprint(stim_times: np.ndarray | None) -> tuple:
    """Cheap stable fingerprint for use as a cache key."""
    if stim_times is None or len(stim_times) == 0:
        return (0, 0.0, 0.0)
    arr = np.asarray(stim_times, dtype=np.float64)
    return (int(arr.size), float(arr[0]), float(arr[-1]))


def _get_blanked_series(file_path: str, channel: int,
                        stim_times: np.ndarray | None,
                        blank_pre_ms: float,
                        blank_post_ms: float,
                        ) -> tuple[np.ndarray, float, int]:
    """Return the full single-channel series with stim windows NaN-blanked.

    Cached because zoom callbacks would otherwise redo the blanking on
    every relayout event. Returns (series, fs, n_blanked_pulses).
    """
    assert isinstance(file_path, str) and file_path, "file_path required"
    assert channel >= 0, "channel must be non-negative"

    key = (file_path, int(channel), float(blank_pre_ms), float(blank_post_ms),
           _stim_fingerprint(stim_times))

    with _blanked_lock:
        hit = _blanked_cache.pop(key, None)
        if hit is not None:
            _blanked_cache[key] = hit
            series, fs = hit
            # n_blanked is only consumed by the status line; on cache hit
            # report the requested event count (close enough for display).
            n_blanked = 0 if stim_times is None else int(len(stim_times))
            return series, fs, n_blanked

    chunk = get_chunk(file_path)
    fs = float(chunk.fs)
    sig = chunk.signal
    if sig.ndim != 2 or sig.shape[1] <= channel:
        raise ValueError(
            f"Channel {channel} not in file with shape {sig.shape}"
        )
    series = sig[:, channel].astype(np.float32, copy=True)

    n_blanked = 0
    if stim_times is not None and len(stim_times) > 0:
        pre = int(round(blank_pre_ms * 1e-3 * fs))
        post = int(round(blank_post_ms * 1e-3 * fs))
        n = len(series)
        for t_sec in stim_times:
            center = int(round(t_sec * fs))
            lo = max(0, center + pre)
            hi = min(n, center + post)
            if hi > lo:
                series[lo:hi] = np.nan
                n_blanked += 1

    with _blanked_lock:
        _blanked_cache[key] = (series, fs)
        while len(_blanked_cache) > _BLANKED_MAX:
            _blanked_cache.popitem(last=False)

    return series, fs, n_blanked


def _decimated_lfp(file_path: str, channel: int,
                   stim_times: np.ndarray | None = None,
                   blank_pre_ms: float = -5.0,
                   blank_post_ms: float = 15.0,
                   ) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Load one channel of LFP and decimate for the initial render.

    Delegates blanking + caching to ``_get_blanked_series`` and the
    decimation to the shared envelope helper, so the zoom callback can
    use the same path on a sub-window.

    Returns (time_sec, signal_uV, duration_sec, n_blanked_pulses).
    """
    series, fs, n_blanked = _get_blanked_series(
        file_path, channel, stim_times, blank_pre_ms, blank_post_ms,
    )
    target_bins = choose_target_bins(len(series)) or INITIAL_TARGET_BINS
    t, display, _decim = envelope(series, fs, target_bins, t_start=0.0)
    duration = float(len(series) / fs)
    return t, display, duration, n_blanked


def _build_lfp_figure(t: np.ndarray, signal: np.ndarray, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t, y=signal,
        mode="lines",
        line=dict(color="#5e7ce2", width=0.8),
        hovertemplate="t=%{x:.2f}s<br>%{y:.1f} μV<extra></extra>",
        name=label,
    ))
    # Cursor placeholder — clientside callback updates the x position.
    # dragmode='pan' makes click-on-the-trace forgiving: a stray
    # 1-2px drag while attempting to click no longer activates the
    # zoom-box (which would swallow the click event before our
    # marker-add callback sees it). Zoom moves to scrollwheel +
    # modebar; double-click still resets the view.
    fig.update_layout(
        plot_bgcolor="#13131f", paper_bgcolor="#13131f",
        height=220,
        margin=dict(l=60, r=20, t=10, b=40),
        dragmode="pan",
        xaxis=dict(title="Time (s)", showgrid=True,
                   gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(title="μV", showgrid=True,
                   gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        shapes=[dict(
            type="line", xref="x", yref="paper",
            x0=0, x1=0, y0=0, y1=1,
            line=dict(color="#ff9f0a", width=2),
        )],
        showlegend=False,
        hovermode="x unified",
    )
    return fig


# BHZ_DETECTOR peak-detector defaults (behavioral_seizure_
# detection.m:237). minpeakdist = fs * 30 * 5 samples (= 150
# s); cutoff defaults to 0.05 to match the lab's CSV examples.
_BHZ_CUTOFF: float = 0.05
_BHZ_MIN_PEAK_DIST_SEC: float = 150.0


def _render_hilbert_trace(store, file_id: int,
                            channel: int | None,
                            blank_pre_ms: float = -5.0,
                            blank_post_ms: float = 15.0,
                            cutoff: float = _BHZ_CUTOFF,
                            min_peak_dist_sec: float = _BHZ_MIN_PEAK_DIST_SEC,
                            ) -> tuple:
    """Return (figure, status_text) for the BHZ Hilbert detector.

    Full pipeline matches behavioral_seizure_detection.m: load
    LFP -> tay_preprocess (smoothed Hilbert envelope, 20-200 Hz)
    -> peakseek (local-maxima + height + min-distance prune) ->
    overlay the candidate-event markers on the envelope.

    The envelope is drawn as a continuous green line. A dashed
    horizontal at ``cutoff`` marks the threshold. Each detected
    peak gets a magenta downward triangle + an annotation line
    so the reviewer can read off the candidate count at a
    glance.

    ``blank_pre_ms`` / ``blank_post_ms`` default to the same
    window the LFP path uses (-5 / 15 ms around each stim
    pulse); ``cutoff`` and ``min_peak_dist_sec`` default to the
    lab's BHZ_DETECTOR defaults (0.05 and 150 s).
    """
    assert isinstance(file_id, int), "file_id must be int"
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        return (_empty_lfp_fig("File not found in DB."), "")
    if channel is None:
        return (_empty_lfp_fig(
            "Pick a brain channel (Step 1) to see its envelope."),
                "")
    session_dir = _session_dir_for_file(store, file_id)
    stim_copy = _stim_copy_channels(store, session_dir)
    do_blank = _should_stim_blank(int(channel), stim_copy)
    stim_times = (_stim_times_for_file(store, file_id)
                   if do_blank
                   else np.asarray([], dtype=np.float64))
    try:
        series, fs, _ = _get_blanked_series(
            file_path, int(channel), stim_times,
            blank_pre_ms, blank_post_ms,
        )
    except Exception as e:
        logger.warning("Hilbert load failed file=%s ch=%s: %s",
                        file_id, channel, e)
        return (_empty_lfp_fig(
            f"Couldn't load LFP for Hilbert: {e}"), "")
    env = hilbert_envelope_20_200(series, fs)
    # BHZ peak detection runs on the FULL-resolution envelope
    # before decimation; sample indices then translate to seconds
    # the same way the LFP cursor does.
    min_peak_dist = max(1, int(round(fs * min_peak_dist_sec)))
    peak_locs, peak_heights = peakseek(
        env, minpeakdist=min_peak_dist, minpeakh=float(cutoff),
    )
    peak_times = peak_locs.astype(np.float64) / float(fs)

    target_bins = choose_target_bins(len(env)) or 4000
    t, display, _decim = envelope(env, fs, target_bins, t_start=0.0)
    fig = _build_lfp_figure(
        t, display,
        "Hilbert envelope (20-200 Hz)",
    )
    fig.data[0].line.color = "#30d158"
    fig.data[0].hovertemplate = (
        "t=%{x:.2f}s<br>env=%{y:.2f}<extra></extra>"
    )
    # Threshold line: horizontal dashed at y = cutoff. The
    # cursor lives in shapes[0]; the threshold takes shapes[1].
    fig.layout.shapes = list(fig.layout.shapes or []) + [dict(
        type="line", xref="paper", yref="y",
        x0=0, x1=1, y0=cutoff, y1=cutoff,
        line=dict(color="#ff9f0a", width=1, dash="dash"),
        opacity=0.8,
    )]
    # Threshold annotation in the right margin.
    fig.add_annotation(
        x=1.0, y=cutoff, xref="paper", yref="y",
        text=f"cutoff = {cutoff:.3g}",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=10, color="#ff9f0a"),
    )
    # Candidate-event markers: magenta downward triangles
    # plotted ABOVE the trace's local height so the reviewer
    # can scan for them. NASA Rule 3: cap at 256 candidates
    # per render (any more and the plot is unreadable anyway).
    n_candidates = int(min(len(peak_times), 256))
    if n_candidates:
        marker_y = float(np.max(display) * 1.05
                          if len(display) else cutoff)
        fig.add_trace(go.Scattergl(
            x=peak_times[:n_candidates],
            y=np.full(n_candidates, marker_y),
            mode="markers",
            marker=dict(symbol="triangle-down", size=10,
                         color="#ff2d92", line=dict(width=0)),
            hovertemplate="candidate @ t=%{x:.2f}s<extra></extra>",
            name="BHZ candidate",
            showlegend=False,
        ))
    status = (
        f"BHZ detection · cutoff={cutoff:.3g} · "
        f"min_dist={int(min_peak_dist_sec)}s · "
        f"{len(peak_times)} candidate"
        f"{'' if len(peak_times) == 1 else 's'} · "
        f"{len(env)/fs:.1f} s @ {int(fs)} Hz"
    )
    return (fig, status)


def _empty_lfp_fig(text: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=220,
        plot_bgcolor="#13131f", paper_bgcolor="#13131f",
        annotations=[dict(text=text, showarrow=False, x=0.5, y=0.5,
                          xref="paper", yref="paper",
                          font=dict(size=12, color="#a0a0b0"))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ===================================================================== #
#  Layout
# ===================================================================== #

def _step_header(number: str, title: str, sub: str | None = None):
    """Numbered step header used to break the tab into a 1-2-3 flow.

    Apple-style numbered badge + friendly subhead. Keeps an undergrad
    oriented without a wall of NASA-grade jargon.
    """
    children = [
        html.Div(number, style={
            "display": "inline-flex",
            "alignItems": "center", "justifyContent": "center",
            "width": "22px", "height": "22px",
            "borderRadius": "50%",
            "background": "linear-gradient(135deg, #5e7ce2, #4a64c4)",
            "color": "white", "fontSize": "12px",
            "fontWeight": "700", "marginRight": "10px",
            "flexShrink": "0",
        }),
        html.Span(title, style={
            "color": "#f0f0f5", "fontSize": "15px",
            "fontWeight": "600",
        }),
    ]
    head = html.Div(children, style={
        "display": "flex", "alignItems": "center",
        "marginTop": "8px",
    })
    if sub:
        return html.Div([
            head,
            html.Div(sub, style={
                "color": "#888", "fontSize": "12px",
                "marginLeft": "32px", "marginBottom": "8px",
                "marginTop": "2px",
            }),
        ], style={"marginBottom": "6px"})
    return html.Div([head], style={"marginBottom": "8px"})


def _details_card(summary_text: str, content,
                   summary_sub: str | None = None,
                   open_default: bool = False):
    """Lightweight ``html.Details`` wrapper used to hide advanced
    controls behind a click. Keeps the default view friendly.
    """
    summary_children: list = [
        html.Span("⚙  ", style={"color": "#888",
                                 "fontSize": "11px"}),
        html.Span(summary_text, style={
            "color": "#cfd0d6", "fontSize": "12px",
            "fontWeight": "600",
        }),
    ]
    if summary_sub:
        summary_children.append(html.Span(
            f"  · {summary_sub}",
            style={"color": "#888", "fontSize": "11px",
                    "marginLeft": "4px"},
        ))
    return html.Details([
        html.Summary(summary_children, style={
            "cursor": "pointer", "userSelect": "none",
            "padding": "8px 12px", "listStyle": "none",
            "borderRadius": "6px",
        }),
        html.Div(content, style={"padding": "0 6px 6px 6px"}),
    ], open=open_default, style={
        "background": "rgba(255,255,255,0.02)",
        "border": "1px solid rgba(255,255,255,0.06)",
        "borderRadius": "6px",
        "marginTop": "10px", "marginBottom": "10px",
    })


def _video_mass_analyze_panel() -> html.Details:
    """Undergrad bulk pre-screen via Hilbert envelope thresholding.

    Reads the animal from the Step 1 queue picker rather than
    keeping a separate dropdown -- one source of truth so the user
    can't get confused about which animal they're operating on.
    The scope counter ('92 pending files eligible') updates live
    as the queue picker / cutoff change. Scan opens a confirmation
    modal explaining the steps before any worker job is created."""
    return html.Details([
        html.Summary([
            html.Span("Mass Analyze pending files ",
                       style={"color": "#f0f0f5",
                               "fontWeight": "600"}),
            html.Span("(skip files with no events)",
                       style={"color": "#a0a0b0",
                               "fontSize": "11px",
                               "marginLeft": "6px"}),
        ], style={"cursor": "pointer",
                   "padding": "10px 14px",
                   "listStyle": "none",
                   "userSelect": "none"}),
        html.Div([
            html.Div(
                "Run BHZ detection over every pending file for "
                "the animal you picked in Step 1 at the threshold "
                "below. Files with zero peaks above the threshold "
                "are auto-submitted as 'no events seen' (the PI "
                "still verifies). Files with any peaks above the "
                "threshold stay in your queue for normal scoring. "
                "Tip: click anywhere on the Hilbert envelope plot "
                "to set the threshold to that line.",
                style={"color": "#a0a0b0",
                        "fontSize": "11px",
                        "padding": "8px 12px",
                        "background": "rgba(94,124,226,0.04)",
                        "border":
                            "1px solid rgba(94,124,226,0.18)",
                        "borderRadius": "6px",
                        "margin": "0 14px 10px"}),
            # Read-only animal display (mirrors Step 1 picker).
            html.Div(id="video-ma-animal-display",
                      style={"color": "#cfd0d6",
                              "fontSize": "12px",
                              "padding": "0 14px",
                              "marginBottom": "6px"}),
            # Live scope counter -- 'BCH061: 92 pending files'.
            html.Div(id="video-ma-scope",
                      style={"color": "#a0a0b0",
                              "fontSize": "11px",
                              "padding": "0 14px",
                              "marginBottom": "10px"}),
            html.Div([
                html.Label("Cutoff:",
                            style={"color": "#a0a0b0",
                                    "fontSize": "12px",
                                    "marginRight": "6px"}),
                dcc.Input(
                    id="video-ma-cutoff-input",
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
                    id="video-ma-scan-btn", n_clicks=0,
                    style={"background": "#5e7ce2",
                            "color": "white",
                            "border": "none",
                            "padding": "6px 14px",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                            "fontSize": "12px",
                            "fontWeight": "600"}),
                html.Button(
                    "Cancel scan",
                    id="video-ma-cancel-btn", n_clicks=0,
                    style={"background": "transparent",
                            "color": "#cfd0d6",
                            "border":
                                "1px solid rgba(255,255,255,0.15)",
                            "padding": "6px 14px",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                            "fontSize": "12px"}),
            ], style={"display": "flex",
                       "alignItems": "center",
                       "gap": "8px",
                       "flexWrap": "wrap",
                       "padding": "0 14px",
                       "marginBottom": "10px"}),
            html.Div(id="video-ma-progress",
                      style={"color": "#a0a0b0",
                              "fontSize": "12px",
                              "minHeight": "16px",
                              "padding": "0 14px",
                              "marginBottom": "8px"}),
            html.Div(id="video-ma-summary",
                      style={"padding": "0 14px",
                              "marginBottom": "10px"}),
            html.Div(id="video-ma-confirm-row",
                      style={"display": "flex", "gap": "8px",
                              "alignItems": "center",
                              "padding": "0 14px",
                              "marginBottom": "10px"}),
            # State stores + polling.
            dcc.Store(id="video-ma-job-id", data=None),
            dcc.Interval(id="video-ma-poll",
                           interval=1500, n_intervals=0,
                           disabled=True),
            # Confirmation modal (hidden by default; the scan
            # button opens it, Confirm here creates the job).
            html.Div(
                id="video-ma-confirm-modal",
                style={"position": "fixed", "inset": "0",
                        "display": "none",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "background": "rgba(10,10,20,0.65)",
                        "backdropFilter": "blur(6px)",
                        "zIndex": "10001"},
                children=html.Div(
                    style={"width": "min(640px, 92vw)",
                            "maxHeight": "82vh",
                            "overflowY": "auto",
                            "background": "#13131f",
                            "border":
                                "1px solid rgba(94,124,226,0.30)",
                            "borderRadius": "10px",
                            "padding": "24px",
                            "boxShadow":
                                "0 24px 48px rgba(0,0,0,0.55)"},
                    children=[
                        html.Div(
                            "Confirm Mass Analyze scan",
                            style={"color": "#f0f0f5",
                                    "fontWeight": "600",
                                    "fontSize": "16px",
                                    "marginBottom": "10px"}),
                        html.Div(
                            id="video-ma-modal-body",
                            style={"color": "#cfd0d6",
                                    "fontSize": "12px",
                                    "lineHeight": "1.55",
                                    "marginBottom": "16px"}),
                        html.Div([
                            html.Button(
                                "Cancel",
                                id="video-ma-modal-cancel-btn",
                                n_clicks=0,
                                style={"background": "transparent",
                                        "color": "#cfd0d6",
                                        "border":
                                            "1px solid rgba(255,255,255,0.15)",
                                        "padding": "8px 16px",
                                        "borderRadius": "5px",
                                        "cursor": "pointer",
                                        "fontSize": "12px",
                                        "marginRight": "8px"}),
                            html.Button(
                                "Confirm scan",
                                id="video-ma-modal-confirm-btn",
                                n_clicks=0,
                                style={"background": "#5e7ce2",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "8px 16px",
                                        "borderRadius": "5px",
                                        "cursor": "pointer",
                                        "fontSize": "12px",
                                        "fontWeight": "600"}),
                        ], style={"display": "flex",
                                   "justifyContent": "flex-end"}),
                    ],
                ),
            ),
        ]),
    ], open=False,
       style={"padding": "0",
               "background": "#13131f",
               "border":
                   "1px solid rgba(255,255,255,0.06)",
               "borderRadius": "8px",
               "marginTop": "10px",
               "marginBottom": "16px"})


def layout(store: Store):
    sessions = _sessions_with_video(store)
    session_options = [
        {"label": f"{s['session_name']} ({s['n_videos']} videos)",
         "value": s["session_dir"]}
        for s in sessions
    ]
    default_session = sessions[0]["session_dir"] if sessions else None

    if not sessions:
        return html.Div([
            html.Div("No videos to review yet",
                     style={"fontSize": "15px", "fontWeight": "600",
                            "color": "#f0f0f5", "marginBottom": "8px",
                            "textAlign": "center"}),
            html.Div(
                "Videos appear here once the dispatcher detects a "
                "_v1.mp4 file alongside a processed .mat file.",
                style={"color": "#a0a0b0", "fontSize": "13px",
                       "textAlign": "center", "maxWidth": "520px",
                       "margin": "0 auto"},
            ),
        ], style={"textAlign": "center", "padding": "32px 24px",
                  "marginTop": "32px"})

    return html.Div([
        # --- Quick Start ------------------------------------------------- #
        # First-visit help card. Native <details> so the user's collapsed
        # state survives navigation without us wiring a callback. Default
        # open so undergrads see the orientation on first visit.
        html.Details([
            html.Summary([
                html.Span("👋  Quick start", style={
                    "color": "#f0f0f5", "fontSize": "13px",
                    "fontWeight": "600",
                }),
                html.Span("  (click to hide)", style={
                    "color": "#888", "fontSize": "11px",
                    "marginLeft": "6px",
                }),
            ], style={
                "cursor": "pointer", "userSelect": "none",
                "padding": "10px 14px", "listStyle": "none",
            }),
            html.Div([
                html.Div(
                    "This tab pairs the cage video with the brain signal "
                    "(LFP) recorded at the same time. Use it to check "
                    "whether what's happening in the video matches what "
                    "the electrodes saw.",
                    style={"color": "#cfd0d6", "fontSize": "13px",
                            "marginBottom": "10px",
                            "lineHeight": "1.5"},
                ),
                html.Div([
                    html.Span("Three things to know:", style={
                        "color": "#a0a0b0", "fontSize": "11px",
                        "fontWeight": "600",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.5px",
                    }),
                ], style={"marginBottom": "6px"}),
                html.Ul([
                    html.Li([
                        html.B("Pick a recording"),
                        " above. The video and brain signal load together.",
                    ]),
                    html.Li([
                        html.B("Click on the brain trace"),
                        " to jump the video to that moment.",
                    ]),
                    html.Li([
                        html.B("Click a channel name in the legend"),
                        " to focus on just that channel "
                        "(double-click to isolate).",
                    ]),
                ], style={"color": "#cfd0d6", "fontSize": "13px",
                            "marginTop": "0", "paddingLeft": "20px",
                            "lineHeight": "1.7"}),
                html.Div(
                    "Everything else is optional — open the "
                    "\"Advanced\" sections if you need to filter the "
                    "signal or change what feature is plotted.",
                    style={"color": "#888", "fontSize": "12px",
                            "marginTop": "8px", "fontStyle": "italic"},
                ),
            ], style={"padding": "0 14px 14px 14px"}),
        ], open=True, style={
            "background": "rgba(94, 124, 226, 0.08)",
            "border": "1px solid rgba(94, 124, 226, 0.25)",
            "borderRadius": "8px",
            "marginBottom": "16px",
        }),

        # --- Loading pill (sticky at top of tab) ----------------------- #
        # Track B of the loading-feedback plan. Reads from
        # video-load-state; shown when any of {video, lfp,
        # hilbert} is "loading"; hidden when all done. Per-leg
        # chips colored by state so the reviewer can tell which
        # piece is still working.
        html.Div(id="video-load-pill",
                  className="video-load-pill",
                  style={"display": "none"}),
        dcc.Store(id="video-load-state", data={}),

        # --- Step 1: pick a recording ---------------------------------- #
        _step_header("1", "Pick a recording",
                      "Pick which animal you're reviewing. Recordings "
                      "show up oldest first (FIFO) so the backlog "
                      "clears from the front — work top to bottom."),

        # --- Mass Analyze (optional, collapsed by default) ----------- #
        # Hilbert envelope thresholding pre-screen. Lets the
        # undergrad sweep a cutoff and auto-clear zero-peak files
        # before they start manually scoring the rest. Same
        # underlying job/cache as the PI Mass Analyze in
        # Event Verification, just scoped to assigned animals.
        _video_mass_analyze_panel(),

        # --- My queue picker ----------------------------------------- #
        # Card-style row: animal picker on left, queue list on the
        # right. The queue items are buttons keyed by file_id; one
        # callback handles all of them via pattern-matching IDs and
        # writes the chosen file's session_dir + file_id back into
        # the existing dropdowns so every downstream callback keeps
        # working unchanged.
        html.Div([
            html.Div([
                html.Label("Animal you're reviewing",
                            style=LABEL_STYLE,
                            title="Pick one of your assigned animals, "
                                   "or pick from the unassigned pool. "
                                   "Ask the PI to add you to the "
                                   "Reviewer Assignments sheet."),
                dcc.Dropdown(
                    id="video-queue-animal",
                    options=[],
                    placeholder="Pick an animal",
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
                html.Div(id="video-queue-status",
                          style={"color": "#a0a0b0",
                                  "fontSize": "11px",
                                  "marginTop": "4px"}),
            ], style={"flex": "0 0 280px"}),
            html.Div([
                html.Label("Queue (oldest first · FIFO)",
                            style=LABEL_STYLE,
                            title="Recordings for the chosen animal "
                                   "that nobody has finished yet. "
                                   "Oldest at the top so you clear "
                                   "the backlog from the front. "
                                   "Click one to load it."),
                # Progress + motivation strip (P1-1). Rendered by
                # _render_queue_progress; mirrors Linear's sprint
                # progress + Duolingo's streak surface.
                html.Div(id="video-queue-progress",
                          style={"display": "flex",
                                  "alignItems": "center",
                                  "gap": "12px",
                                  "padding": "4px 6px 6px",
                                  "color": "#a0a0b0",
                                  "fontSize": "11px"}),
                # One-at-a-time browse card (Track F of the
                # UX polish plan). Replaces the scrollable
                # button list with prev/next arrows + Load.
                # The full list still lives below behind a
                # "Show all timestamps" expander so J/K cycle
                # + the active-file highlight have something
                # to attach to.
                html.Div([
                    html.Div(id="video-queue-card-title",
                              style={"color": "#a0a0b0",
                                      "fontSize": "11px",
                                      "marginBottom": "6px"}),
                    html.Div([
                        html.Button(
                            "◀",
                            id="video-queue-prev-btn",
                            n_clicks=0,
                            title="Browse to the next-newer "
                                   "recording (no load yet).",
                            style={"background": "transparent",
                                    "color": "#cfd0d6",
                                    "border": "1px solid "
                                               "rgba(255,255,255,0.15)",
                                    "borderRadius": "5px",
                                    "padding": "4px 10px",
                                    "cursor": "pointer",
                                    "fontSize": "12px"}),
                        html.Div(id="video-queue-card-body",
                                  style={"flex": "1",
                                          "padding": "0 10px",
                                          "textAlign": "center"}),
                        html.Button(
                            "▶",
                            id="video-queue-next-btn",
                            n_clicks=0,
                            title="Browse to the next-older "
                                   "recording (no load yet).",
                            style={"background": "transparent",
                                    "color": "#cfd0d6",
                                    "border": "1px solid "
                                               "rgba(255,255,255,0.15)",
                                    "borderRadius": "5px",
                                    "padding": "4px 10px",
                                    "cursor": "pointer",
                                    "fontSize": "12px"}),
                    ], style={"display": "flex",
                               "alignItems": "center",
                               "padding": "8px 6px",
                               "background": "#13131f",
                               "border":
                                   "1px solid rgba(255,255,255,0.06)",
                               "borderRadius": "6px"}),
                    html.Div([
                        html.Button(
                            "Load this recording",
                            id="video-queue-load-btn",
                            n_clicks=0,
                            style={"background": "#5e7ce2",
                                    "color": "white",
                                    "border": "none",
                                    "padding": "6px 16px",
                                    "borderRadius": "5px",
                                    "cursor": "pointer",
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                    "marginRight": "10px"}),
                        html.Span(id="video-queue-card-position",
                                   style={"color": "#a0a0b0",
                                           "fontSize": "11px"}),
                    ], style={"display": "flex",
                               "alignItems": "center",
                               "marginTop": "6px"}),
                    _details_card(
                        "Show all timestamps",
                        summary_sub="full FIFO list -- click "
                                     "any item to load it",
                        open_default=False,
                        content=html.Div(id="video-queue-list",
                                  style={"maxHeight": "220px",
                                          "overflowY": "auto",
                                          "background": "#13131f",
                                          "border":
                                              "1px solid rgba(255,255,255,0.06)",
                                          "borderRadius": "6px",
                                          "padding": "4px"}),
                    ),
                    dcc.Store(id="video-queue-position", data=0),
                ], id="video-queue-card"),
            ], style={"flex": "1", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "16px",
                   "marginBottom": "10px", "flexWrap": "wrap"}),

        # --- Advanced: free-form pickers (preserves legacy IDs) ----- #
        _details_card(
            "Pick any recording manually",
            summary_sub="bypass the queue and choose any session/file/"
                         "channel directly",
            open_default=False,
            content=html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE,
                            title="A session is a continuous run of "
                                   "recording with one animal."),
                dcc.Dropdown(
                    id="video-session-dropdown",
                    options=session_options,
                    value=default_session,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "2", "minWidth": "260px"}),
            html.Div([
                html.Label("File (chunk)", style=LABEL_STYLE,
                            title="Each chunk is roughly one hour of "
                                   "recording, named with its start "
                                   "timestamp."),
                dcc.Dropdown(
                    id="video-file-dropdown",
                    options=[],
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "3", "minWidth": "320px"}),
            html.Div([
                html.Label("Brain channel", style=LABEL_STYLE,
                            title="Which electrode contact to display "
                                   "the LFP trace for."),
                dcc.Dropdown(
                    id="video-channel-dropdown",
                    options=[],
                    value=0,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "180px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "8px",
                  "flexWrap": "wrap"}),
        ),

        # --- Step 2: watch ---------------------------------------------- #
        _step_header("2", "Watch",
                      "Video plays in sync with the LFP and Hilbert "
                      "envelope on the right. Scroll down to score "
                      "events; the video pips to the corner."),

        # --- Two-column Step 2: video LEFT | LFP+Hilbert RIGHT ----- #
        # CSS Grid: video spans both rows in column 1; the LFP block
        # lives in column 2 row 1, the Hilbert block in column 2 row
        # 2. On narrow viewports (<=900 px) the grid collapses to a
        # single column via the media query in theme.css.
        html.Div([
            dcc.Loading(
                id="video-player-loading",
                type="circle",
                color="#5e7ce2",
                # 180 ms before showing the spinner means warm-
                # cache renders don't flash; cold-cache reads
                # (where the reviewer needs the feedback) tip
                # over the threshold and show clearly.
                delay_show=180,
                parent_style={"gridArea": "video",
                               "minHeight": "360px"},
                children=html.Div(
                    id="video-player-container",
                    children=html.Div([
                        html.Div("📹", style={"fontSize": "32px",
                                              "marginBottom": "6px"}),
                        html.Div("Pick a recording above to load "
                                  "the video here.",
                                  style={"color": "#a0a0b0",
                                          "fontSize": "13px"}),
                    ], style={"textAlign": "center"}),
                    style={"backgroundColor": "#000",
                            "borderRadius": "10px",
                            "minHeight": "360px",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "overflow": "hidden"},
                ),
            ),

        # --- Time-locked LFP trace -------------------------------------- #
            html.Div([
            html.Div([
                html.Span("Brain signal (LFP)",
                          style={"color": "#cfd0d6", "fontSize": "13px",
                                 "fontWeight": "600"}),
                html.Span(id="video-lfp-status",
                          style={"color": "#888", "fontSize": "11px",
                                 "marginLeft": "12px"}),
                html.Span("💡  Click on the trace to jump the video here.",
                          style={"color": "#5e7ce2", "fontSize": "11px",
                                 "marginLeft": "auto",
                                 "fontStyle": "italic"}),
            ], style={"display": "flex", "alignItems": "baseline",
                      "marginBottom": "8px"}),
            # --- Filter strip (hidden by default, click to expand) ---
            _details_card(
                "Filter the brain signal",
                summary_sub="advanced — leave alone unless the trace "
                             "looks noisy",
                open_default=False,
                content=html.Div([
                html.Div([
                    html.Label("Remove slow drift (Hz)", style=LABEL_STYLE,
                                title="High-pass filter cutoff. Drops "
                                       "frequencies below this. 0 = off. "
                                       "Try 1 Hz if the trace drifts up "
                                       "and down."),
                    dcc.Input(id="video-filter-hp", type="number", min=0,
                              step=0.5, value=0,
                              style={"backgroundColor": "#262638",
                                     "color": "#f0f0f5", "width": "90px"}),
                ], style={"flex": "0 0 150px"}),
                html.Div([
                    html.Label("Remove fast noise (Hz)", style=LABEL_STYLE,
                                title="Low-pass filter cutoff. Drops "
                                       "frequencies above this. 0 = off. "
                                       "Try 300 Hz to keep only the LFP "
                                       "band."),
                    dcc.Input(id="video-filter-lp", type="number", min=0,
                              step=1, value=0,
                              style={"backgroundColor": "#262638",
                                     "color": "#f0f0f5", "width": "90px"}),
                ], style={"flex": "0 0 150px"}),
                html.Div([
                    html.Label("Power-line filter", style=LABEL_STYLE,
                                title="Removes 50 Hz or 60 Hz hum from "
                                       "nearby AC power. US lab = 60 Hz."),
                    dcc.Dropdown(
                        id="video-filter-notch",
                        options=[{"label": "Off", "value": 0},
                                  {"label": "50 Hz (EU)", "value": 50},
                                  {"label": "60 Hz (US)", "value": 60}],
                        value=0, clearable=False,
                        style={"backgroundColor": "#262638",
                               "color": "#f0f0f5"},
                        className="dark-dropdown",
                    ),
                ], style={"flex": "0 0 140px"}),
                html.Div([
                    html.Label("Smooth the LFP (ms)", style=LABEL_STYLE,
                                title="Gaussian smoothing window in "
                                       "milliseconds, applied to the LFP "
                                       "trace ONLY (Step 2). 0 = off. "
                                       "Distinct from Step 3's "
                                       "smooth-by-seconds, which smooths "
                                       "the per-event feature line. "
                                       "Helpful for "
                                       "spotting slow rhythms; bad for "
                                       "spotting fast spikes."),
                    dcc.Input(id="video-filter-smooth", type="number",
                              min=0, step=1, value=0,
                              style={"backgroundColor": "#262638",
                                     "color": "#f0f0f5", "width": "90px"}),
                ], style={"flex": "0 0 130px"}),
                html.Div([
                    html.Label(" ", style=LABEL_STYLE),
                    html.Button("Apply",
                                 id="video-apply-filter-btn", n_clicks=0,
                                 title="Apply the filter settings above.",
                                 style={"backgroundColor": "#5e7ce2",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "7px 18px",
                                        "borderRadius": "6px",
                                        "cursor": "pointer",
                                        "fontSize": "12px",
                                        "fontWeight": "600"}),
                ], style={"flex": "0 0 100px",
                          "display": "flex", "alignItems": "flex-end"}),
            ], style={"display": "flex", "gap": "12px",
                      "marginBottom": "4px", "flexWrap": "wrap",
                      "padding": "8px 10px", "backgroundColor": "#13131f",
                      "borderRadius": "6px",
                      "border": "1px solid rgba(255,255,255,0.06)"}),
            ),  # close _details_card("Filter the brain signal", ...)
            dcc.Store(id="video-filter-state",
                      data={"hp": 0, "lp": 0, "notch": 0, "smooth": 0}),
            dcc.Loading(
                id="video-lfp-loading",
                type="circle",
                color="#5e7ce2",
                delay_show=180,
                parent_style={"minHeight": "220px"},
                children=dcc.Graph(
                    id="video-lfp-trace",
                    figure=_empty_lfp_fig(
                        "Pick a recording above to see the brain "
                        "signal here."),
                    # Modebar on so the user has Zoom / Pan /
                    # Reset axes / download. doubleClick: "reset"
                    # snaps back to the default view (Plotly's
                    # hidden gem).
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
            ),
            # Step 4 used to live here -- it's been moved below
            # Step 3 so the reviewer sees the LFP + Hilbert
            # envelope before scoring. See the Step 4 block just
            # before "Past notes".

            # --- Step 3: analyze --------------------------------------- #
            # Time-locked analysis plot directly under the LFP. Same
            # x-axis (seconds since chunk start), each point = one stim
            # epoch, y = the chosen evoked feature. Cursor tracks the
            # video the same way the LFP does.
            _step_header(
                "3", "Look at a brain feature over time",
                "Pick a number that gets computed from the brain signal "
                "for every stim event in the recording. Click a point "
                "to jump the video to that moment.",
            ),
            # Top-level row: just the friendly Feature picker + a
            # live status pill. Everything else (smoothing, detrend,
            # etc.) is in the advanced collapsible below.
            html.Div([
                html.Div([
                    html.Label("Brain feature to plot",
                                style=LABEL_STYLE,
                                title="What number to compute and plot "
                                       "for each stim event. Line length "
                                       "is a good default — it's how "
                                       "wiggly the trace is around the "
                                       "event."),
                    dcc.Dropdown(
                        id="video-analysis-feature",
                        options=[
                            # Default for BHZ event scoring:
                            # a 1:1 port of tay_preprocess.m's
                            # 20-200 Hz Hilbert envelope. See
                            # src/utils/hilbert_envelope.py.
                            {"label": "Hilbert envelope "
                                       "(20-200 Hz, default)",
                                "value": "hilbert"},
                            {"label": "Line length (per-event)",
                                "value": "line_length"},
                            {"label": "Log(AUC) — area under the curve",
                                "value": "log_auc"},
                            {"label": "Peak amplitude",
                                "value": "peak_amplitude"},
                            {"label": "Trough amplitude",
                                "value": "trough_amplitude"},
                            {"label": "Peak-to-trough (size of swing)",
                                "value": "peak_to_trough"},
                            {"label": "RMS amplitude",
                                "value": "rms_amplitude"},
                            {"label": "Peak latency (ms)",
                                "value": "peak_latency_ms"},
                            {"label": "Trough latency (ms)",
                                "value": "trough_latency_ms"},
                            {"label": "Max slope",
                                "value": "max_slope"},
                            {"label": "Early area (0–50 ms)",
                                "value": "early_area"},
                            {"label": "Late area (50–200 ms)",
                                "value": "late_area"},
                            {"label": "Early/Late ratio",
                                "value": "early_late_ratio"},
                            {"label": "Recovery tau",
                                "value": "recovery_tau"},
                            {"label": "Template correlation",
                                "value": "template_correlation"},
                            {"label": "Variance",
                                "value": "variance"},
                            {"label": "Sum power (low freq.)",
                                "value": "sum_power_low"},
                            {"label": "Sum power (high freq.)",
                                "value": "sum_power_high"},
                        ],
                        value="hilbert", clearable=False,
                        style={"backgroundColor": "#262638",
                                "color": "#f0f0f5",
                                "minWidth": "260px"},
                        className="dark-dropdown",
                    ),
                ], style={"flex": "0 0 280px"}),
                html.Span(id="video-analysis-status",
                           style={"color": "#888", "fontSize": "11px",
                                   "marginLeft": "16px",
                                   "alignSelf": "center"}),
            ], style={"marginTop": "12px", "marginBottom": "4px",
                       "display": "flex", "gap": "10px",
                       "flexWrap": "wrap",
                       "alignItems": "flex-end"}),
            # Advanced post-processing: smooth, detrend, z-score, etc.
            # Default closed so undergrads aren't intimidated.
            _details_card(
                "Smooth, detrend, normalize",
                summary_sub="advanced — change how the feature trace is "
                             "cleaned up",
                open_default=False,
                content=html.Div([
                    html.Div([
                        html.Label("Smooth the feature line (s)",
                                    style=LABEL_STYLE,
                                    title="Gaussian smoothing window in "
                                           "seconds, applied to the "
                                           "per-event feature line ONLY "
                                           "(Step 3). 0 = off. Distinct "
                                           "from Step 2's smooth-the-LFP "
                                           "(ms), which smooths the raw "
                                           "trace."),
                        dcc.Input(id="video-analysis-smooth",
                                   type="number", min=0, step=0.5,
                                   value=0,
                                   style={"backgroundColor": "#262638",
                                           "color": "#f0f0f5",
                                           "width": "90px"}),
                    ], style={"flex": "0 0 150px"}),
                    html.Div([
                        html.Label("Median window (events)",
                                    style=LABEL_STYLE,
                                    title="Centered rolling median of N "
                                           "consecutive events. Good for "
                                           "rejecting single-event "
                                           "outliers. 0 = off."),
                        dcc.Input(id="video-analysis-rollwin",
                                   type="number", min=0, step=1,
                                   value=0,
                                   style={"backgroundColor": "#262638",
                                           "color": "#f0f0f5",
                                           "width": "90px"}),
                    ], style={"flex": "0 0 170px"}),
                    html.Div([
                        html.Label("Clean-up options",
                                    style=LABEL_STYLE,
                                    title="Optional transforms applied "
                                           "after the rolling median + "
                                           "smoothing."),
                        dcc.Checklist(
                            id="video-analysis-postproc",
                            options=[
                                {"label": " Remove linear drift",
                                    "value": "detrend"},
                                {"label": " Normalize (z-score)",
                                    "value": "zscore"},
                                {"label": " Hide bad epochs",
                                    "value": "hide_artifact"},
                                {"label": " Center on median",
                                    "value": "median"},
                            ],
                            value=["hide_artifact"], inline=True,
                            # labelStyle is what colors each
                            # individual option's text; without it
                            # the labels fall back to the browser's
                            # default (black on the dark theme).
                            labelStyle={"color": "#cfd0d6",
                                         "fontSize": "12px",
                                         "marginRight": "8px"},
                            inputStyle={"marginRight": "4px",
                                         "marginLeft": "8px"},
                        ),
                    ], style={"flex": "1 1 auto"}),
                    html.Div([
                        html.Label(" ", style=LABEL_STYLE),
                        html.Button(
                            "Apply",
                            id="video-analysis-apply-btn",
                            n_clicks=0,
                            title="Apply smoothing / detrend / "
                                   "normalize / clean-up choices.",
                            style={"backgroundColor": "#5e7ce2",
                                    "color": "white",
                                    "border": "none",
                                    "padding": "7px 18px",
                                    "borderRadius": "6px",
                                    "cursor": "pointer",
                                    "fontSize": "12px",
                                    "fontWeight": "600"}),
                    ], style={"flex": "0 0 100px",
                               "display": "flex",
                               "alignItems": "flex-end"}),
                ], style={"display": "flex", "gap": "12px",
                          "flexWrap": "wrap",
                          "alignItems": "flex-end",
                          "padding": "8px 10px",
                          "backgroundColor": "#13131f",
                          "borderRadius": "6px",
                          "border":
                              "1px solid rgba(255,255,255,0.06)"}),
            ),
            dcc.Loading(
                id="video-analysis-loading",
                type="circle",
                color="#5e7ce2",
                delay_show=180,
                parent_style={"minHeight": "220px"},
                children=dcc.Graph(
                    id="video-analysis-trace",
                    figure=_empty_lfp_fig(
                        "Pick a recording above to see the feature "
                        "trace."),
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
            ),
        ], style={"marginTop": "0", "gridArea": "lfp",
                   "minWidth": "0"}),
            # In-PiP "Dock" button. Hidden in normal grid view
            # via the CSS rule on .qc-pip-dock-btn; only appears
            # when the parent grid has the .qc-pip-grid class.
            # Click flips video-pip-state back to False (handled
            # by the same clientside callback that the refresh-
            # bar PiP button uses).
            html.Button(
                "✕ Dock",
                id="pip-dock-btn",
                n_clicks=0,
                title="Dock the PiP back into the page (P).",
                className="qc-pip-dock-btn",
            ),
        ], id="video-step2-grid",
           style={"display": "grid",
                   "gridTemplateColumns": "minmax(420px, 1.2fr) "
                                            "minmax(420px, 1fr)",
                   "gridTemplateAreas": "'video lfp'",
                   "gap": "16px",
                   "alignItems": "start",
                   "marginBottom": "12px"}),

        # --- PI-only: frame-variance quality threshold sampler ----- #
        # Hidden by default; _toggle_pi_quality_panel makes it visible
        # only for emails in config.review_queue.pi_emails. The
        # 'Sample current frame' button is a clientside JS handler
        # that draws the current paused frame to a hidden canvas,
        # computes grayscale pixel variance, and writes the result
        # to video-pi-variance-store; the server-side save persists
        # via store.set_video_quality_threshold.
        html.Div(
            id="video-pi-quality-panel",
            style={"display": "none",
                    "marginTop": "8px",
                    "padding": "10px 14px",
                    "background": "rgba(255,159,10,0.05)",
                    "border": "1px solid rgba(255,159,10,0.25)",
                    "borderRadius": "6px"},
            children=[
                html.Div(
                    "🛠️  Advanced: video quality threshold (PI)",
                    style={"color": "#ff9f0a", "fontSize": "12px",
                            "fontWeight": "600",
                            "marginBottom": "8px"}),
                html.Div([
                    html.Button(
                        "Sample current frame",
                        id="video-pi-sample-btn", n_clicks=0,
                        title="Pause the video first, then click. "
                               "Captures the current frame to a canvas "
                               "and computes grayscale pixel variance.",
                        style={"background": "#262638",
                                "color": "#f0f0f5",
                                "border":
                                    "1px solid rgba(255,255,255,0.15)",
                                "borderRadius": "5px",
                                "padding": "6px 12px",
                                "fontSize": "12px",
                                "fontWeight": "600",
                                "cursor": "pointer",
                                "marginRight": "10px"}),
                    html.Span(id="video-pi-variance-display",
                               style={"color": "#a0a0b0",
                                       "fontSize": "12px",
                                       "fontFamily":
                                           "ui-monospace, monospace"}),
                ], style={"display": "flex",
                           "alignItems": "center",
                           "marginBottom": "8px"}),
                html.Div([
                    html.Span("Threshold:",
                               style={"color": "#a0a0b0",
                                       "fontSize": "12px",
                                       "marginRight": "8px"}),
                    dcc.Input(
                        id="video-pi-threshold-input",
                        type="number", min=0, step=0.1,
                        placeholder="e.g. 200",
                        style={"backgroundColor": "#262638",
                                "color": "#f0f0f5",
                                "border":
                                    "1px solid rgba(255,255,255,0.1)",
                                "borderRadius": "4px",
                                "padding": "4px 8px",
                                "width": "110px",
                                "fontSize": "12px",
                                "marginRight": "8px"}),
                    html.Button(
                        "Save as quality threshold",
                        id="video-pi-set-threshold-btn", n_clicks=0,
                        style={"background": "#ff9f0a",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "5px",
                                "padding": "6px 14px",
                                "fontSize": "12px",
                                "fontWeight": "600",
                                "cursor": "pointer"}),
                    html.Span(id="video-pi-set-status",
                               style={"color": "#a0a0b0",
                                       "fontSize": "12px",
                                       "marginLeft": "10px"}),
                ], style={"display": "flex",
                           "alignItems": "center",
                           "marginBottom": "8px"}),
                html.Div(id="video-pi-current-threshold",
                          style={"color": "#a0a0b0",
                                  "fontSize": "11px",
                                  "marginTop": "6px"}),
                dcc.Store(id="video-pi-variance-store", data=None),
            ],
        ),

        # --- Step 2 reviewer note (optional, collapsed by default) ----- #
        # Replaces the always-visible 180-px-tall textarea that used to
        # sit beside the video. Most reviewers don't write a note per
        # file; the ones that do can expand this and write/save. The
        # ids stay (video-note-input, video-note-save-btn) so the
        # existing save callbacks keep working without changes.
        _details_card(
            "Add an optional note about this video",
            summary_sub="free-text reminder for yourself or the next "
                         "reviewer. Most files don't need one.",
            open_default=False,
            content=html.Div([
                dcc.Textarea(
                    id="video-note-input",
                    placeholder="Notes about this video / LFP chunk...",
                    style={"backgroundColor": "#262638",
                            "color": "#f0f0f5",
                            "border": "1px solid rgba(255,255,255,0.07)",
                            "borderRadius": "6px",
                            "padding": "10px 12px", "width": "100%",
                            "height": "80px", "fontFamily": "inherit",
                            "fontSize": "13px"}),
                html.Div([
                    html.Button(
                        "Save note",
                        id="video-note-save-btn", n_clicks=0,
                        style={"backgroundColor": "#5e7ce2",
                                "color": "white", "border": "none",
                                "padding": "6px 16px",
                                "borderRadius": "6px",
                                "cursor": "pointer",
                                "fontSize": "12px",
                                "fontWeight": "600",
                                "marginRight": "10px"}),
                    html.Span(id="video-note-status",
                               style={"fontSize": "12px",
                                       "color": "#a0a0b0"}),
                ], style={"display": "flex",
                           "alignItems": "center",
                           "marginTop": "6px"}),
            ]),
        ),

            # --- Step 4: score events + mark done --------------------- #
            # Lives below the LFP + Hilbert envelope so the reviewer can
            # see both the raw trace and the BHZ feature before
            # committing. BHZ_DETECTOR event taxonomy (LVF / HYP +
            # EO / LAS / BO / PID / BB + Racine 1-8) is built up by
            # the marker editor below; Mark-done writes the structured
            # event list to review_state AND appends rows to the
            # per-(animal, day) CSV at G:\BHZ\BHZ_CSV_Exports.
            _step_header(
                "4", "Score events + mark this recording done",
                "Add seizure events on the LFP and finish each one "
                "with a Racine score. This finishes the recording "
                "in your queue."),
            html.Div([
                dcc.RadioItems(
                    id="video-review-decision",
                    options=[
                        {"label": " No events seen",
                            "value": "no_events"},
                        {"label": " Events seen "
                                  "(add onset markers below)",
                            "value": "has_events"},
                    ],
                    value=None, inline=False,
                    labelStyle={"color": "#cfd0d6",
                                 "fontSize": "13px",
                                 "marginBottom": "6px",
                                 "display": "block"},
                    inputStyle={"marginRight": "8px"},
                ),
                # The BHZ-style event editor lives inside the
                # "Events seen" branch -- hidden until the
                # reviewer picks that radio. video-events-store
                # mirrors the structured event list; render_*
                # callbacks live in src/dashboard/tabs/video_events.py.
                html.Div(
                    _events.render_events_panel(),
                    id="video-events-wrapper",
                    style={"display": "none",
                            "marginTop": "12px"},
                ),
                html.Div([
                    html.Div(
                        "Click the brain trace above to drop onset "
                        "markers. Each click becomes a red dotted "
                        "vertical on the LFP at that time.",
                        style={"color": "#a0a0b0",
                                "fontSize": "12px",
                                "marginBottom": "6px"}),
                    html.Div([
                        html.Button(
                            "Clear markers",
                            id="video-review-clear-markers-btn",
                            n_clicks=0,
                            style={"backgroundColor": "transparent",
                                    "color": "#a0a0b0",
                                    "border": "1px solid "
                                               "rgba(255,255,255,0.15)",
                                    "padding": "4px 10px",
                                    "borderRadius": "5px",
                                    "cursor": "pointer",
                                    "fontSize": "11px",
                                    "marginRight": "8px"}),
                    ], style={"marginTop": "4px"}),
                    _details_card(
                        "Show timestamps as list",
                        html.Div(id="video-review-markers",
                                  style={"color": "#a0a0b0",
                                          "fontSize": "12px",
                                          "fontFamily": "ui-monospace, "
                                                         "SF Mono, monospace",
                                          "padding": "6px 10px",
                                          "minHeight": "32px",
                                          "background": "#13131f",
                                          "border": "1px solid "
                                                     "rgba(255,255,255,0.06)",
                                          "borderRadius": "6px"}),
                    ),
                ], id="video-review-marker-group",
                   style={"display": "none",
                           "marginTop": "10px",
                           "padding": "8px 10px",
                           "border": "1px solid "
                                      "rgba(255,255,255,0.08)",
                           "borderRadius": "6px",
                           "backgroundColor":
                               "rgba(255,255,255,0.02)"}),
                html.Div([
                    html.Label("Optional note (max 280 chars)",
                                style=LABEL_STYLE),
                    dcc.Textarea(
                        id="video-review-note",
                        maxLength=280,
                        placeholder="e.g. 'partial movement at "
                                     "0:32 — looks like grooming'",
                        style={"backgroundColor": "#262638",
                                "color": "#f0f0f5",
                                "border":
                                    "1px solid rgba(255,255,255,0.1)",
                                "borderRadius": "6px",
                                "padding": "8px 10px",
                                "width": "100%",
                                "height": "60px",
                                "fontFamily": "inherit",
                                "fontSize": "12px"}),
                ], style={"marginTop": "10px"}),
                # P1-3 decision preview. NN/g 'Visibility of system
                # status' + Gmail's compose-summary-before-send.
                # Read-only: shows the reviewer what's about to be
                # saved, so the Mark-done click is never a black box.
                html.Div(id="video-review-preview",
                          style={"display": "flex",
                                  "gap": "10px",
                                  "flexWrap": "wrap",
                                  "alignItems": "center",
                                  "padding": "6px 10px",
                                  "marginTop": "10px",
                                  "color": "#a0a0b0",
                                  "fontSize": "11px",
                                  "background": "#13131f",
                                  "border":
                                      "1px solid rgba(255,255,255,0.04)",
                                  "borderRadius": "6px"}),
                html.Div([
                    html.Button(
                        "Mark recording done",
                        id="video-review-save-btn",
                        n_clicks=0,
                        title="Saves your decision + any onset "
                               "markers. Removes the recording from "
                               "your queue.",
                        style={"backgroundColor": "#00CC96",
                                "color": "white",
                                "border": "none",
                                "padding": "8px 22px",
                                "borderRadius": "6px",
                                "cursor": "pointer",
                                "fontSize": "13px",
                                "fontWeight": "700",
                                "marginRight": "10px"}),
                    dcc.Loading(
                        id="video-review-status-loading",
                        type="dot",
                        color="#5e7ce2",
                        # 250 ms delay so a cache-hit save
                        # doesn't flash; the SQLite write +
                        # CSV append (the slow leg) tip the
                        # spinner clearly into view.
                        delay_show=250,
                        parent_style={"display": "inline-flex",
                                       "alignItems": "center"},
                        children=html.Span(
                            id="video-review-status",
                            style={"color": "#a0a0b0",
                                    "fontSize": "12px",
                                    "alignSelf": "center"}),
                    ),
                ], style={"marginTop": "10px",
                           "display": "flex",
                           "alignItems": "center"}),
                # Persists the running marker list across reviewer
                # actions; the click handler appends, the save
                # handler reads from this Store.
                dcc.Store(id="video-review-marker-store",
                           data=[]),
            ], style={"marginTop": "12px",
                       "padding": "12px 14px",
                       "background": "rgba(0, 204, 150, 0.05)",
                       "border": "1px solid rgba(0, 204, 150, 0.2)",
                       "borderRadius": "8px"}),

        # --- Existing video-review notes for this file ------------------ #
        html.Div([
            html.H4("Past notes for this file",
                    style={"color": "#a0a0b0", "fontSize": "13px",
                           "marginTop": "20px", "marginBottom": "8px",
                           "fontWeight": "600"}),
            html.Div(id="video-note-history",
                     style={"color": "#a0a0b0", "fontSize": "12px"}),
        ]),

        # --- Hidden state ----------------------------------------------- #
        # Polls every 100ms while the tab is open. The clientside callback
        # samples the <video> element's currentTime; if no video is loaded
        # the callback no-ops. Cheap.
        dcc.Interval(id="video-time-tick", interval=100, n_intervals=0),
        dcc.Store(id="video-current-time", data=0.0),
        # P1-4 predictive prefetch: a slow tick (2 s) drives a
        # background warm of the next-in-queue file's chunk so
        # pressing J / mark-done feels instant. State stores the
        # last file_id we prefetched against, to avoid redundant
        # work when nothing changed.
        dcc.Interval(id="video-prefetch-tick", interval=2000,
                       n_intervals=0),
        dcc.Store(id="video-prefetch-state", data=None),
        # Focused camera (1-based). Defaults to the master
        # (cam 1); each new file load resets to 1. The PiP
        # CSS hides every wrapper with data-focused="false"
        # so the floating column shows only the focused
        # camera + LFP.
        dcc.Store(id="video-focus-cam", data=1),
        # Sink for the multi-camera slave-sync clientside callback;
        # the callback returns the empty string and only side-effects
        # the slave <video> elements' currentTime / play / pause.
        html.Div(id="video-cam-sync-sink", style={"display": "none"}),
        # BHZ_DETECTOR-style time lock. We stash the LFP's true
        # duration (from chunk: n_samples / fs) here when the trace
        # loads. The clientside callbacks combine this with the
        # <video> element's own .duration to compute a per-chunk
        # scale factor (lfp_dur / video_dur) -- container FPS lies
        # can no longer accumulate drift over the chunk because
        # we're anchored to ground truth on both ends.
        dcc.Store(id="video-lfp-duration", data=0.0),
        # Dummy outputs for clientside callbacks that have side effects on
        # the DOM (setting video.currentTime) rather than returning data.
        html.Div(id="video-seek-sink", style={"display": "none"}),
    ])


# ===================================================================== #
#  Callbacks
# ===================================================================== #

def register_callbacks(app, store: Store, config: dict) -> None:
    fa = (config or {}).get("feature_analysis", {}) or {}
    blank_pre_ms = float(fa.get("stim_artifact_start_ms", -5.0))
    blank_post_ms = float(fa.get("stim_artifact_end_ms", 15.0))
    # ---- Bridge from LFP Browser "View video" buttons ----
    # Sets the session dropdown + filter inputs whenever the bridge
    # Store is freshly written. The cascade (session -> file ->
    # player -> channel) is handled by the existing callbacks
    # below, which both consult the bridge for their defaults.
    @app.callback(
        Output("video-session-dropdown", "value"),
        Output("video-filter-hp", "value"),
        Output("video-filter-lp", "value"),
        Output("video-filter-notch", "value"),
        Output("video-filter-smooth", "value"),
        Output("video-apply-filter-btn", "n_clicks",
                allow_duplicate=True),
        Input("lfp-to-video-bridge", "data"),
        State("video-apply-filter-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _consume_bridge(bridge, apply_clicks):
        if not bridge or not isinstance(bridge, dict):
            return (no_update, no_update, no_update, no_update,
                    no_update, no_update)
        session_dir = bridge.get("session_dir") or no_update
        # Bump the Apply button n_clicks so the existing
        # _update_lfp callback (Input on the button) re-fires once
        # the filter inputs are updated, applying the new settings
        # without the operator having to click anything.
        next_clicks = int(apply_clicks or 0) + 1
        return (
            session_dir,
            bridge.get("hp") or 0,
            bridge.get("lp") or 0,
            bridge.get("notch") or 0,
            bridge.get("smooth") or 0,
            next_clicks,
        )

    # ---- Step 1 queue: animal picker + auto-loaded queue ---- #
    review_cfg = (config or {}).get("review_queue", {}) or {}
    queue_cfg = review_cfg.get("queue", {}) or {}
    queue_limit = int(queue_cfg.get("max_per_view", 100))
    warn_age_days = int(queue_cfg.get("warn_age_days", 3))

    @app.callback(
        Output("video-queue-animal", "options"),
        Output("video-queue-animal", "value"),
        Output("video-queue-status", "children"),
        Input("manual-refresh-btn", "n_clicks"),
        Input("assignments-version", "data"),
        State("video-queue-animal", "value"),
    )
    def _populate_animal_picker(_clicks, _version, current_value):
        """Refresh the animal dropdown from the assignment sheet.

        Subscribes to two inputs:

          * ``manual-refresh-btn`` -- fires once on initial render
            (Dash convention) and on every user-clicked Refresh.
          * ``assignments-version`` -- bumped by the background
            warmer thread in ``src/utils/assignments.py`` when
            fresh data lands. Lets the picker auto-update without
            blocking on the Sheets API in this render path.

        Reads via ``cache_only=True`` so the callback is always
        instant: a ``None`` from ``animals_for_user`` means the
        warmer hasn't populated the cache yet and we render a
        "Loading..." placeholder; the version-bump will re-fire
        this callback once the cache is warm.
        """
        if not review_cfg.get("enabled", False):
            return [], None, "Queue feature is disabled in config."
        email = current_user_email() or ""
        my_animals = _assignments.animals_for_user(
            config, email, cache_only=True)
        # When the warmer has completed at least one iteration
        # (or the WARMER_TIMEOUT_SEC has elapsed without it
        # completing), treat None as 'sheet is empty / not
        # reachable' rather than spinning forever on
        # 'Loading...'. The Sheet tab could be genuinely empty,
        # the tab name could be wrong, or the API could be
        # unreachable; any of those should fall back to the
        # all-animals view, not deadlock the UI.
        if my_animals is None:
            if _assignments.warmer_should_have_settled():
                my_animals = []
            else:
                return ([], None,
                         "Loading assignments from Google "
                         "Sheets...")
        try:
            unassigned = _assignments.unassigned_animals(
                config, store, cache_only=True)
            if unassigned is None:
                unassigned = []
        except Exception as e:
            logger.debug("unassigned pool failed: %s", e)
            unassigned = []
        options: list[dict] = []
        if my_animals:
            # User HAS sheet rows -> mark them with a star and
            # offer the rest of the lab's animals below the
            # divider as a fallback.
            for a in my_animals:
                options.append({
                    "label": f"⭐  {a}  (assigned to you)",
                    "value": a,
                })
            if unassigned:
                options.append({"label": "── Other animals ──",
                                "value": "__UNASSIGNED__",
                                "disabled": True})
                for a in unassigned:
                    options.append({
                        "label": f"   {a}",
                        "value": f"_pool_{a}",
                    })
        else:
            # Sheet is empty for this user (or empty period).
            # Don't frame the lab's animals as an "Unassigned
            # pool" -- just list them as plain selectable
            # options. The queue still pretends they came from
            # the pool (the value stays _pool_<animal>) so
            # downstream callbacks don't need to change.
            for a in unassigned:
                options.append({
                    "label": a,
                    "value": f"_pool_{a}",
                })
        # Blank on first load. The reviewer explicitly picks an
        # animal each session -- no auto-default ever (caveat #6
        # of the queue-card plan). We only preserve the current
        # value if it's still a valid option (e.g. after the
        # warmer republishes the same animal list).
        new_value = current_value
        if new_value is not None and all(
                o["value"] != new_value for o in options):
            new_value = None
        # Friendlier status copy: don't shout "no animals
        # assigned" at the user when the lab simply hasn't
        # filled in the sheet yet.
        if not my_animals and not unassigned:
            status = "No animals in the DB yet."
        elif not my_animals:
            status = (f"Sheet has no assignments yet. "
                       f"Showing all {len(unassigned)} "
                       f"animal{'' if len(unassigned) == 1 else 's'} "
                       "in the DB.")
        else:
            status = (f"Assigned to you: "
                       f"{', '.join(my_animals)}. "
                       "Pick one above to start reviewing.")
        return options, new_value, status

    # Progress + streak strip. Pattern source: Linear sprint
    # progress + Duolingo streak. Fires on the same triggers as
    # _render_queue so the numbers move in lockstep with the
    # queue list. "Streak" is in *sessions* (2-day buckets) per
    # the plan's caveat #6 -- a bi-daily cadence means a calendar
    # off-day should NOT reset the streak.
    @app.callback(
        Output("video-queue-progress", "children"),
        Input("video-queue-animal", "value"),
        Input("refresh-trigger", "data"),
    )
    def _render_queue_progress(animal_value, _refresh):
        email = current_user_email() or ""
        if not email:
            return ""
        done_today = store.user_finishes_today(email)
        streak = store.user_streak_sessions(email)
        animal_ids = _animal_ids_from_picker(animal_value)
        n_waiting = 0
        oldest_label = ""
        if animal_ids:
            rows = store.get_review_queue(
                animal_ids, email,
                limit=queue_limit,
                since_iso=store.review_backlog_floor(),
            )
            n_waiting = len(rows)
            if rows:
                from datetime import datetime as _dt
                try:
                    dt = _dt.strptime(
                        rows[0]["chunk_datetime"] or "",
                        "%Y_%m_%d__%H_%M_%S")
                    age_days = ((_dt.now() - dt)
                                  .total_seconds() / 86400.0)
                    if age_days < 1:
                        oldest_label = (f"oldest "
                                         f"{age_days * 24:.0f} h")
                    else:
                        oldest_label = (f"oldest "
                                         f"{age_days:.1f} d")
                except ValueError:
                    oldest_label = ""
        # Color the warn / crit ages so the user sees backlog
        # heat without reading the number.
        heat_color = "#a0a0b0"
        if oldest_label and "d" in oldest_label:
            days = float(oldest_label.split()[1])
            if days >= 7:
                heat_color = "#ff453a"
            elif days >= warn_age_days:
                heat_color = "#ff9f0a"
        chips: list = [
            html.Span(f"Today: {done_today} reviewed",
                       style={"color": "#f0f0f5",
                               "fontWeight": "600"}),
            html.Span("·"),
            html.Span(
                (f"Streak: {streak} sessions"
                 if streak else "Streak: -- "),
                style={"color": ("#30d158" if streak >= 2
                                  else "#a0a0b0")}),
        ]
        if n_waiting:
            chips.append(html.Span("·"))
            chips.append(html.Span(f"{n_waiting} waiting",
                                     style={"color": "#a0a0b0"}))
        if oldest_label:
            chips.append(html.Span("·"))
            chips.append(html.Span(oldest_label,
                                     style={"color": heat_color}))
        return chips

    # ---- Track F: one-at-a-time queue card ---- #
    # The card renders the FIFO entry at video-queue-position;
    # prev/next adjust the position WITHOUT loading; the "Load
    # this recording" button is what actually opens the file.
    # Edge cases (empty queue, position out of range, no
    # animal picked) all surface inline in the card body.
    @app.callback(
        Output("video-queue-card-title", "children"),
        Output("video-queue-card-body", "children"),
        Output("video-queue-card-position", "children"),
        Output("video-queue-position", "data",
                allow_duplicate=True),
        Input("video-queue-animal", "value"),
        Input("video-queue-position", "data"),
        Input("refresh-trigger", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _render_queue_card(animal_value, position, _refresh):
        if not animal_value:
            return ("",
                    html.Div("Pick an animal above to see "
                              "your queue.",
                              style={"color": "#888",
                                      "fontSize": "12px"}),
                    "", no_update)
        animal_ids = _animal_ids_from_picker(animal_value)
        if not animal_ids:
            return ("",
                    html.Div("Pick an animal above to see "
                              "your queue.",
                              style={"color": "#888",
                                      "fontSize": "12px"}),
                    "", no_update)
        email = current_user_email() or ""
        floor = store.review_backlog_floor()
        rows = store.get_review_queue(animal_ids, email,
                                        limit=queue_limit,
                                        since_iso=floor)
        animal_label = animal_ids[0]
        if not rows:
            title = (f"Queue (FIFO · 0 unreviewed for "
                      f"{animal_label})")
            return (title,
                    html.Div([
                        html.Div("🎉  You're all caught up!",
                                  style={"color": "#00CC96",
                                          "fontWeight": "600",
                                          "fontSize": "13px"}),
                        html.Div("No unreviewed recordings.",
                                  style={"color": "#888",
                                          "fontSize": "11px",
                                          "marginTop": "2px"}),
                    ]),
                    "", 0)
        total = len(rows)
        clamped = max(0, min(int(position or 0), total - 1))
        row = rows[clamped]
        from datetime import datetime as _dt
        now = _dt.now()
        ts_raw = row.get("chunk_datetime") or ""
        try:
            ts = _dt.strptime(ts_raw, "%Y_%m_%d__%H_%M_%S")
            age_days = (now - ts).total_seconds() / 86400
            ts_label = ts.strftime("%Y-%m-%d  %H:%M")
        except ValueError:
            age_days, ts_label = 0, ts_raw
        dur = row.get("duration_sec") or 0
        dur_h = dur / 3600.0 if dur else 0
        age_label = (f"{age_days:.1f} d old"
                      if age_days >= 1
                      else f"{age_days * 24:.0f} h old")
        warn = age_days >= warn_age_days
        title = (f"Queue (FIFO · {total} unreviewed for "
                  f"{animal_label})")
        # If this file is back in the queue because the PI flagged
        # it for re-review, surface the PI's note so the
        # undergrad knows what to look for. Cheap query: one
        # SELECT against review_state by file_id.
        pi_flag_note = _pi_flag_note_for_file(store,
                                                 int(row["id"]))
        body_children: list = [
            html.Div(ts_label,
                      style={"color": "#f0f0f5",
                              "fontWeight": "600",
                              "fontSize": "14px"}),
            html.Div(
                f"{dur_h:.1f} h  ·  {age_label}",
                style={"color": ("#ff453a" if warn
                                  else "#888"),
                       "fontSize": "11px",
                       "marginTop": "2px"}),
        ]
        if pi_flag_note:
            body_children.append(html.Div(
                f"🚩 PI flagged: {pi_flag_note}",
                style={"color": "#ff9f0a",
                        "fontSize": "11px",
                        "marginTop": "4px",
                        "padding": "3px 8px",
                        "background": "rgba(255,159,10,0.12)",
                        "border":
                            "1px solid rgba(255,159,10,0.30)",
                        "borderRadius": "4px"}))
        body = html.Div(body_children)
        position_str = f"Position {clamped + 1} of {total}"
        # If we clamped, write the clamped position back so
        # the next prev/next click is from a valid base.
        if clamped != (position or 0):
            return (title, body, position_str, clamped)
        return (title, body, position_str, no_update)

    @app.callback(
        Output("video-queue-position", "data",
                allow_duplicate=True),
        Input("video-queue-prev-btn", "n_clicks"),
        Input("video-queue-next-btn", "n_clicks"),
        Input("video-queue-animal", "value"),
        State("video-queue-position", "data"),
        prevent_initial_call=True,
    )
    def _on_queue_arrow(_prev, _next, _animal, pos):
        trig = callback_context.triggered_id
        if trig == "video-queue-animal":
            # Reset position to head when the animal changes.
            return 0
        cur = int(pos or 0)
        if trig == "video-queue-prev-btn":
            return max(0, cur - 1)
        if trig == "video-queue-next-btn":
            # Render-side clamp catches overshoots; this lets
            # the next callback bump position by 1 without
            # re-querying the queue size from the click path.
            return cur + 1
        return no_update

    @app.callback(
        Output("video-session-dropdown", "value",
                allow_duplicate=True),
        Output("video-file-dropdown", "value",
                allow_duplicate=True),
        Input("video-queue-load-btn", "n_clicks"),
        State("video-queue-animal", "value"),
        State("video-queue-position", "data"),
        prevent_initial_call=True,
    )
    def _on_queue_load(n, animal_value, position):
        if not n or not animal_value:
            return no_update, no_update
        animal_ids = _animal_ids_from_picker(animal_value)
        if not animal_ids:
            return no_update, no_update
        email = current_user_email() or ""
        floor = store.review_backlog_floor()
        rows = store.get_review_queue(animal_ids, email,
                                        limit=queue_limit,
                                        since_iso=floor)
        if not rows:
            return no_update, no_update
        clamped = max(0, min(int(position or 0), len(rows) - 1))
        row = rows[clamped]
        # Log a claim event so the PI audit log knows the
        # reviewer touched this file (parity with the queue-
        # button click path).
        store.insert_review_event(int(row["id"]),
                                    email or "anon", "claim",
                                    {"source": "queue_card_load"})
        return row["session_dir"], int(row["id"])

    @app.callback(
        Output("video-queue-list", "children"),
        Input("video-queue-animal", "value"),
        Input("refresh-trigger", "data"),
        Input("video-file-dropdown", "value"),
    )
    def _render_queue(animal_value, _refresh, active_file_id):
        """Render the per-animal queue list. Each item is a Button
        with a pattern-matching id so one downstream callback
        handles all clicks. The button whose ``file_id`` matches
        ``active_file_id`` (the dropdown's current value) gets a
        left-border accent + a tinted background so the reviewer
        can always tell which recording is loaded."""
        if not animal_value:
            return html.Div(
                "Pick an animal above to see your queue.",
                style={"color": "#888", "fontSize": "12px",
                        "padding": "10px"})
        # Resolve animal_value to a list of animal ids (single
        # selection or the unassigned pool sentinel).
        if animal_value.startswith("_pool_"):
            animal_ids = [animal_value[len("_pool_"):]]
        else:
            animal_ids = [animal_value]
        email = current_user_email() or ""
        floor = store.review_backlog_floor()
        rows = store.get_review_queue(animal_ids, email,
                                        limit=queue_limit,
                                        since_iso=floor)
        if not rows:
            return html.Div([
                html.Div("🎉  You're all caught up!",
                          style={"color": "#00CC96",
                                  "fontWeight": "600",
                                  "fontSize": "13px",
                                  "marginBottom": "4px"}),
                html.Div(
                    "No unreviewed recordings for this animal.",
                    style={"color": "#888", "fontSize": "11px"}),
            ], style={"padding": "14px", "textAlign": "center"})
        from datetime import datetime as _dt
        now = _dt.now()
        items = []
        for r in rows[:queue_limit]:
            ts_raw = r.get("chunk_datetime") or ""
            try:
                ts = _dt.strptime(ts_raw,
                                    "%Y_%m_%d__%H_%M_%S")
                age_days = (now - ts).total_seconds() / 86400
                ts_label = ts.strftime("%Y-%m-%d  %H:%M")
            except ValueError:
                age_days = 0
                ts_label = ts_raw
            warn = age_days >= warn_age_days
            dur = r.get("duration_sec") or 0
            dur_h = dur / 3600.0 if dur else 0
            is_active = (active_file_id is not None
                          and int(r["id"]) == int(active_file_id))
            btn_style: dict = {
                "display": "flex", "alignItems": "center",
                "width": "100%", "border": "none",
                "background": ("rgba(94, 124, 226, 0.12)"
                                 if is_active else "transparent"),
                "color": "#cfd0d6",
                "padding": "6px 10px",
                "cursor": "pointer",
                "borderBottom":
                    "1px solid rgba(255,255,255,0.04)",
                "borderLeft": (
                    "3px solid #5e7ce2" if is_active
                    else "3px solid transparent"),
                "textAlign": "left",
            }
            items.append(html.Button([
                html.Span(ts_label, style={
                    "color": "#f0f0f5", "fontWeight": "600",
                    "fontSize": "12px"}),
                html.Span(f"  ({dur_h:.1f} h)", style={
                    "color": "#888", "fontSize": "11px",
                    "marginLeft": "4px"}),
                html.Span(
                    f"  · {age_days:.1f} d old"
                    if age_days >= 1 else
                    f"  · {age_days * 24:.0f} h old",
                    style={
                        "color": ("#EF553B" if warn
                                   else "#888"),
                        "fontSize": "10px",
                        "marginLeft": "auto"}),
            ], id={"type": "video-queue-item",
                    "file_id": int(r["id"])},
                n_clicks=0, style=btn_style))
        return items

    @app.callback(
        Output("video-session-dropdown", "value",
                allow_duplicate=True),
        Output("video-file-dropdown", "value",
                allow_duplicate=True),
        Input({"type": "video-queue-item", "file_id": ALL},
                "n_clicks"),
        prevent_initial_call=True,
    )
    def _load_from_queue(n_clicks_list):
        """Translate a queue button click into setting the existing
        session + file dropdowns, which cascades through every
        downstream callback unchanged."""
        if not n_clicks_list or not any(n_clicks_list):
            return no_update, no_update
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update, no_update
        file_id = trig.get("file_id")
        if file_id is None:
            return no_update, no_update
        # Get the session_dir + file_path of the chosen file.
        with store.connection() as conn:
            row = conn.execute(
                "SELECT session_dir, file_path "
                "FROM processed_files WHERE id = ?",
                (int(file_id),),
            ).fetchone()
        if not row:
            return no_update, no_update
        # The Video Review file-dropdown's value is the file_id
        # (see _update_files); session dropdown is session_dir.
        store.insert_review_event(int(file_id),
                                    current_user_email() or "anon",
                                    "claim",
                                    {"source": "queue_click"})
        return row["session_dir"], int(file_id)

    # ---- P1-4 predictive prefetch ---- #
    # 2 s after the current file loads (and every 2 s thereafter,
    # though debounced by the last-prefetched marker), warm
    # chunk_cache for the next-in-queue file in a daemon thread.
    # Pattern source: Superhuman "next email pre-rendered". Cost
    # is bounded -- chunk_cache LRU evicts beyond _MAX_ENTRIES.
    @app.callback(
        Output("video-prefetch-state", "data"),
        Input("video-prefetch-tick", "n_intervals"),
        State("video-file-dropdown", "value"),
        State("video-queue-animal", "value"),
        State("video-prefetch-state", "data"),
        prevent_initial_call=True,
    )
    def _prefetch_next_file(_n, current_file_id, animal_value,
                              state):
        if not current_file_id:
            return no_update
        last = (state or {}).get("last")
        if last == int(current_file_id):
            return no_update
        email = current_user_email() or ""
        _sd, next_id = _resolve_next_in_queue(
            store, animal_value, int(current_file_id), email,
            queue_limit=queue_limit,
        )
        if not next_id:
            return {"last": int(current_file_id)}
        next_path = _file_path_for_id(store, int(next_id))
        if not next_path:
            return {"last": int(current_file_id)}
        # Warm in a daemon thread so the request returns now.
        import threading as _th
        _th.Thread(
            target=_prefetch_chunk_safe,
            args=(next_path, int(next_id)),
            daemon=True,
            name=f"video-prefetch-{int(next_id)}",
        ).start()
        return {"last": int(current_file_id)}

    # ---- Hotkey: J / K (or arrow up / down) cycles the queue ---- #
    # Subscribes to the kbd-event bus from src/dashboard/keyboard.py
    # and translates next/prev actions into a session+file dropdown
    # update -- the same effect as clicking a queue button, so the
    # whole downstream cascade (video, LFP, analysis, review reset)
    # fires without any extra wiring.
    @app.callback(
        Output("video-session-dropdown", "value",
                allow_duplicate=True),
        Output("video-file-dropdown", "value",
                allow_duplicate=True),
        Input("kbd-event", "data"),
        State("video-file-dropdown", "value"),
        State("video-queue-animal", "value"),
        prevent_initial_call=True,
    )
    def _hotkey_queue_cycle(ev, current_file_id, animal_value):
        if not ev or ev.get("action") not in ("next", "prev"):
            return no_update, no_update
        direction = 1 if ev["action"] == "next" else -1
        email = current_user_email() or ""
        sd, new_file_id = _resolve_next_in_queue(
            store, animal_value,
            int(current_file_id) if current_file_id else 0,
            email, queue_limit=queue_limit, direction=direction,
        )
        if new_file_id is None:
            return no_update, no_update
        # Log a claim event so the PI audit log knows the
        # reviewer touched this file even if they hop past it.
        store.insert_review_event(int(new_file_id), email or "anon",
                                    "claim",
                                    {"source": f"hotkey_{ev['action']}"})
        return sd, new_file_id

    # ---- Hotkey: U / Undo button reopens the last decision ---- #
    # Reads kbd-undo State to find what to reopen, validates the
    # deadline, calls store.reopen_review, clears kbd-undo, and
    # jumps the file dropdown back to the reopened file. The R
    # hotkey lands here too -- it re-uses the same Undo payload
    # if one's pending, otherwise revert is a no-op (a separate
    # action targeting the *current* file is added in P0-4).
    @app.callback(
        Output("video-session-dropdown", "value",
                allow_duplicate=True),
        Output("video-file-dropdown", "value",
                allow_duplicate=True),
        Output("kbd-undo", "data", allow_duplicate=True),
        Output("video-review-status", "children",
                allow_duplicate=True),
        Input("kbd-event", "data"),
        Input("kbd-undo-button", "n_clicks"),
        State("kbd-undo", "data"),
        prevent_initial_call=True,
    )
    def _hotkey_undo(ev, _click, undo):
        trig = callback_context.triggered_id
        if trig == "kbd-event":
            if not ev or ev.get("action") not in ("undo", "revert"):
                return no_update, no_update, no_update, no_update
        if not undo or not undo.get("file_id"):
            return no_update, no_update, no_update, no_update
        # Honour the deadline: a stale U keystroke after the
        # toast already vanished should not reach this far, but
        # belt and braces against clock skew between server and
        # client.
        import time as _time
        if (undo.get("deadline_ms") or 0) < _time.time() * 1000:
            return no_update, no_update, None, no_update
        file_id = int(undo["file_id"])
        email = current_user_email() or "anon"
        ok = store.reopen_review(file_id, email)
        if not ok:
            return no_update, no_update, None, no_update
        sd = _session_dir_for_file(store, file_id)
        if sd is None:
            return no_update, no_update, None, no_update
        return sd, file_id, None, "↩ Reopened — review again."

    # ---- Step 4: reveal marker + events editor when "Events" picked ---- #
    @app.callback(
        Output("video-review-marker-group", "style"),
        Output("video-events-wrapper", "style"),
        Input("video-review-decision", "value"),
        State("video-review-marker-group", "style"),
        State("video-events-wrapper", "style"),
    )
    def _toggle_marker_editor(decision, marker_style,
                               events_style):
        m_base = dict(marker_style or {})
        e_base = dict(events_style or {})
        if decision == "has_events":
            m_base["display"] = "block"
            e_base["display"] = "block"
        else:
            m_base["display"] = "none"
            e_base["display"] = "none"
        return m_base, e_base

    # Register the BHZ events editor's own callbacks (Add /
    # Delete / type radio / Drop / Clear / Racine / comments).
    # These all read+write video-events-store with pattern-
    # matching ids so adding a new event field is a one-place
    # change in video_events.py.
    _events.register_callbacks(app, store)

    # Clear the events list when the file dropdown changes so
    # the next recording starts blank.
    @app.callback(
        Output("video-events-store", "data",
                allow_duplicate=True),
        Input("video-file-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _reset_events_on_file_change(_file_id):
        return []

    # ---- Track D: multi-camera focus selector ---- #
    # The focused-cam index (1-based) lives in video-focus-cam.
    # Defaults to 1 (master) and resets on file change. A
    # clientside callback walks the DOM and flips data-focused
    # on each camera card -- CSS targets [data-focused="true"]
    # for the accent border and [data-focused="false"] for the
    # in-PiP hide rule.
    @app.callback(
        Output("video-focus-cam", "data",
                allow_duplicate=True),
        Input({"type": "video-cam-focus-btn", "idx": ALL},
                "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_focus_cam_click(_clicks):
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update
        idx = trig.get("idx")
        if idx is None:
            return no_update
        return int(idx)

    @app.callback(
        Output("video-focus-cam", "data",
                allow_duplicate=True),
        Input("video-file-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _reset_focus_on_file_change(_file_id):
        return 1

    # ---- Track B: top-of-tab loading pill ---- #
    # On file-dropdown change, mark all three legs as
    # "loading"; each render callback flips its own leg to
    # "done" via a Patch when it returns. The pill renderer
    # watches the store, shows the pill while any leg is
    # loading, fades it ~700 ms after all three resolve.
    @app.callback(
        Output("video-load-state", "data",
                allow_duplicate=True),
        Input("video-file-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _start_load(file_id):
        if not file_id:
            # File cleared -> no load happening; the pill
            # disappears on the next renderer fire.
            return {}
        return {
            "video": "loading", "lfp": "loading",
            "hilbert": "loading",
            "for_file_id": int(file_id),
        }

    @app.callback(
        Output("video-load-state", "data",
                allow_duplicate=True),
        Input("video-channel-dropdown", "value"),
        State("video-load-state", "data"),
        prevent_initial_call=True,
    )
    def _start_channel_load(_ch, state):
        # Channel change re-fires LFP + Hilbert but not video.
        # Only mark those two; leave video chip whatever it was.
        if not state:
            return no_update
        next_state = dict(state)
        next_state["lfp"] = "loading"
        next_state["hilbert"] = "loading"
        return next_state

    @app.callback(
        Output("video-load-pill", "children"),
        Output("video-load-pill", "style"),
        Input("video-load-state", "data"),
        State("video-load-pill", "style"),
    )
    def _render_load_pill(state, current_style):
        legs = (("video", "Video"), ("lfp", "LFP"),
                 ("hilbert", "Hilbert"))
        s = state or {}
        any_loading = any(s.get(k) == "loading" for k, _ in legs)
        base_style = dict(current_style or {})
        if not s or not any_loading:
            base_style["display"] = "none"
            return [], base_style
        base_style.update({
            "display": "flex",
            "alignItems": "center",
            "gap": "8px",
            "padding": "6px 12px",
            "marginBottom": "10px",
            "borderRadius": "999px",
            "background": "rgba(94, 124, 226, 0.10)",
            "border": "1px solid rgba(94, 124, 226, 0.30)",
            "fontSize": "12px",
            "color": "#cfd0d6",
            "width": "fit-content",
        })
        chips = []
        for key, label in legs:
            status = s.get(key, "idle")
            if status == "done":
                chips.append(html.Span(
                    f"✓ {label}",
                    style={"color": "#30d158",
                            "fontSize": "11px",
                            "padding": "2px 8px",
                            "background":
                                "rgba(48, 209, 88, 0.12)",
                            "borderRadius": "4px"}))
            elif status == "loading":
                chips.append(html.Span(
                    label,
                    className="pulse-load-chip",
                    style={"color": "#a0a0b0",
                            "fontSize": "11px",
                            "padding": "2px 8px",
                            "background":
                                "rgba(255,255,255,0.06)",
                            "borderRadius": "4px"}))
            else:
                continue
        children = [
            html.Span("⟳", className="qc-spinner",
                       style={"fontSize": "14px",
                               "color": "#5e7ce2"}),
            html.Span("Loading new recording…",
                       style={"fontWeight": "600"}),
            *chips,
        ]
        return children, base_style

    # Three lightweight "leg done" watchers. Originally these
    # had Input("...trace", "figure") which sounded right --
    # the figure changes when the render callback returns. The
    # 10 Hz cursor clientside Patch on shapes[0].x0 ALSO
    # triggers Input on figure, and Dash serializes the entire
    # figure dict (120 k decimated points, several MB) on
    # every fire. That's ~10 MB/sec per plot of Dash transport
    # for the lifetime of the tab and the source of the
    # reviewer's "video takes forever to load + sometimes OOMs"
    # report. Fix: subscribe to the cheap STATUS TEXT outputs
    # that the same render callbacks write -- a tiny string
    # that only changes once per file/channel load. The
    # callback chain that should trip "done" still trips;
    # cursor ticks no longer fire these watchers at all.
    @app.callback(
        Output("video-load-state", "data",
                allow_duplicate=True),
        Input("video-player-container", "children"),
        State("video-load-state", "data"),
        prevent_initial_call=True,
    )
    def _mark_video_done(_children, state):
        if not state or state.get("video") != "loading":
            return no_update
        next_state = dict(state)
        next_state["video"] = "done"
        return next_state

    @app.callback(
        Output("video-load-state", "data",
                allow_duplicate=True),
        Input("video-lfp-status", "children"),
        State("video-load-state", "data"),
        prevent_initial_call=True,
    )
    def _mark_lfp_done(_status, state):
        if not state or state.get("lfp") != "loading":
            return no_update
        next_state = dict(state)
        next_state["lfp"] = "done"
        return next_state

    @app.callback(
        Output("video-load-state", "data",
                allow_duplicate=True),
        Input("video-analysis-status", "children"),
        State("video-load-state", "data"),
        prevent_initial_call=True,
    )
    def _mark_hilbert_done(_status, state):
        if not state or state.get("hilbert") != "loading":
            return no_update
        next_state = dict(state)
        next_state["hilbert"] = "done"
        return next_state

    # Clientside: paint data-focused="true"/"false" on each
    # .video-cam-wrapper based on the Store. Patch isn't useful
    # here because we're flipping DOM attributes, not Dash
    # component props.
    app.clientside_callback(
        """
        function (focus_idx) {
            const wrappers = document.querySelectorAll(
                '.video-cam-wrapper');
            const focused = Number(focus_idx || 1);
            wrappers.forEach(function (w) {
                const i = Number(w.dataset.camIdx || 0);
                if (i === focused) {
                    w.dataset.focused = 'true';
                    w.style.borderColor = '#5e7ce2';
                } else {
                    w.dataset.focused = 'false';
                    w.style.borderColor = 'transparent';
                }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output("video-focus-cam", "id"),  # write-only sink
        Input("video-focus-cam", "data"),
    )

    # ---- PI-only: video quality threshold panel + sampler ---- #
    pi_emails = (((config or {}).get("review_queue", {}) or {})
                  .get("pi_emails", []) or [])
    pi_email_set = {str(e).lower() for e in pi_emails}

    @app.callback(
        Output("video-pi-quality-panel", "style"),
        Output("video-pi-current-threshold", "children"),
        Input("video-file-dropdown", "value"),
        Input("refresh-trigger", "data"),
        State("video-pi-quality-panel", "style"),
    )
    def _toggle_pi_quality_panel(_file_id, _refresh, current):
        email = (current_user_email() or "").lower()
        is_pi = bool(email and email in pi_email_set)
        base = dict(current or {})
        base["display"] = "block" if is_pi else "none"
        if not is_pi:
            return base, ""
        rec = store.get_video_quality_threshold()
        if not rec:
            cur_label = ("No threshold set yet. Sample a frame "
                          "and save one to start.")
        else:
            cur_label = (
                f"Current threshold = "
                f"{rec.get('threshold', 0):.2f}  ·  "
                f"sampled variance = "
                f"{rec.get('sampled_variance', 0):.2f}  ·  "
                f"set by {rec.get('user_email', '?')} at "
                f"{(rec.get('at') or '')[:16]}"
                + (f"  ·  note: {rec['note']}"
                    if rec.get("note") else ""))
        return base, cur_label

    @app.callback(
        Output("video-pi-variance-display", "children"),
        Output("video-pi-threshold-input", "value",
                allow_duplicate=True),
        Input("video-pi-variance-store", "data"),
        prevent_initial_call=True,
    )
    def _show_pi_variance(data):
        if not data or data.get("variance") in (None, "null"):
            err = (data or {}).get("error")
            return (f"⚠️  {err}" if err else "(no sample yet)",
                    no_update)
        var = float(data["variance"])
        t_sec = float(data.get("time") or 0.0)
        paused = data.get("paused")
        return (f"variance = {var:.2f}  (t={t_sec:.2f}s · "
                 f"paused={bool(paused)} · {data.get('w')}x"
                 f"{data.get('h')} px)",
                round(var, 2))

    @app.callback(
        Output("video-pi-set-status", "children"),
        Output("video-pi-current-threshold", "children",
                allow_duplicate=True),
        Input("video-pi-set-threshold-btn", "n_clicks"),
        State("video-pi-threshold-input", "value"),
        State("video-pi-variance-store", "data"),
        State("video-file-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _save_pi_threshold(n, threshold, variance_data, file_id):
        if not n:
            return no_update, no_update
        email = (current_user_email() or "").lower()
        if not email or email not in pi_email_set:
            return "Not authorised.", no_update
        if threshold is None or threshold == "":
            return ("Enter a number first.", no_update)
        try:
            thr = float(threshold)
        except (TypeError, ValueError):
            return (f"Invalid threshold: {threshold!r}",
                    no_update)
        sampled = float((variance_data or {})
                          .get("variance") or 0.0)
        try:
            store.set_video_quality_threshold(
                threshold=thr,
                sampled_variance=sampled,
                set_by_email=email,
                file_id=int(file_id) if file_id else None,
                note="",
            )
        except Exception as e:
            logger.warning("set_video_quality_threshold failed: %s",
                            e)
            return (f"Save failed: {e}", no_update)
        rec = store.get_video_quality_threshold() or {}
        from datetime import datetime as _dt
        return (f"✓ Saved at "
                 f"{_dt.now().strftime('%H:%M')}.",
                 f"Current threshold = {rec.get('threshold', 0):.2f}"
                 f"  ·  sampled variance = "
                 f"{rec.get('sampled_variance', 0):.2f}  ·  "
                 f"set by {rec.get('user_email', '?')}")

    # ---- Step 4: reset marker store + decision when file changes ---- #
    @app.callback(
        Output("video-review-marker-store", "data",
                allow_duplicate=True),
        Output("video-review-decision", "value",
                allow_duplicate=True),
        Output("video-review-note", "value"),
        Output("video-review-status", "children"),
        Input("video-file-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _reset_review_panel(file_id):
        if not file_id:
            return [], None, "", ""
        email = current_user_email() or ""
        existing = store.get_review_state_by_user(int(file_id),
                                                     email)
        if existing and existing["status"] in (
                "no_events", "has_events"):
            badge = f"✓ You marked this " \
                     f"{existing['status'].replace('_', ' ')} " \
                     f"on {existing['updated_at'][:16]}."
            return ([], existing["status"], existing.get("note") or "",
                    badge)
        # Any other user already finalised it?
        other = store.get_review_state(int(file_id))
        if other and other["status"] in (
                "no_events", "has_events") and (
                other["user_email"] != email.lower()):
            badge = (f"Already reviewed by "
                      f"{other['user_email']} on "
                      f"{other['updated_at'][:16]}.")
            return [], None, "", badge
        return [], None, "", "Not reviewed yet."

    # ---- Step 4: click LFP -> append onset marker (when in events mode) ---- #
    @app.callback(
        Output("video-review-marker-store", "data",
                allow_duplicate=True),
        Input("video-lfp-trace", "clickData"),
        State("video-review-decision", "value"),
        State("video-review-marker-store", "data"),
        prevent_initial_call=True,
    )
    def _maybe_append_marker(click_data, decision, markers):
        # We don't want every click-to-seek to also drop a marker;
        # gate on the radio being in events mode.
        if decision != "has_events":
            return no_update
        if not click_data or not click_data.get("points"):
            return no_update
        try:
            t_sec = float(click_data["points"][0]["x"])
        except (KeyError, ValueError, TypeError):
            return no_update
        if t_sec < 0:
            return no_update
        new = list(markers or [])
        # De-dupe within 0.5 s so a double-click doesn't spam.
        for m in new:
            if abs(float(m.get("peak_time_sec", -1)) - t_sec) < 0.5:
                return no_update
        new.append({
            "type": "onset",
            "peak_time_sec": round(t_sec, 3),
        })
        new.sort(key=lambda m: m.get("peak_time_sec", 0))
        return new

    # The reviewer sees onset markers two ways: as red verticals
    # on the LFP trace (P0-3: recognition > recall) and -- when
    # they expand the details card -- as a comma-separated text
    # list. Two callbacks keep both views in sync from the same
    # video-review-marker-store Store.
    @app.callback(
        Output("video-review-markers", "children"),
        Input("video-review-marker-store", "data"),
    )
    def _render_markers(markers):
        if not markers:
            return (
                "Click on the brain trace above to add an onset "
                "marker here.")
        return ", ".join(
            f"{m['peak_time_sec']:.2f}s" for m in markers)

    # ---- Decision preview row (P1-3) ---- #
    # Live mirror of what the Mark-done button is about to save.
    # Pure read of three existing Stores plus the file dropdown
    # label; zero new state. Fires on any input change so the row
    # stays in sync as the reviewer edits.
    @app.callback(
        Output("video-review-preview", "children"),
        Input("video-review-decision", "value"),
        Input("video-review-marker-store", "data"),
        Input("video-review-note", "value"),
        Input("video-file-dropdown", "value"),
    )
    def _render_preview(decision, markers, note, file_id):
        if not file_id:
            return ""
        if decision == "no_events":
            pill_label = "No events"
            pill_color = "#30d158"
        elif decision == "has_events":
            pill_label = "Events"
            pill_color = "#ff9f0a"
        else:
            pill_label = "Not picked yet"
            pill_color = "#a0a0b0"
        n_markers = len(markers or [])
        note_label = (f"note: {len(note)} chars"
                       if note else "no note")
        # Lightweight file label (chunk id + datetime if we can
        # pull it cheaply).
        file_label = f"#{int(file_id)}"
        try:
            with store.connection() as conn:
                row = conn.execute(
                    "SELECT chunk_datetime FROM processed_files "
                    "WHERE id = ?", (int(file_id),),
                ).fetchone()
                if row and row["chunk_datetime"]:
                    file_label = (f"#{int(file_id)} · "
                                   f"{row['chunk_datetime'][:16]}")
        except Exception:
            pass  # cheap preview; don't crash on a transient DB read
        return [
            html.Span("Preview:",
                       style={"color": "#6c6c80",
                               "marginRight": "4px"}),
            html.Span([
                html.Span("●  ",
                           style={"color": pill_color,
                                   "fontSize": "12px"}),
                html.Span(pill_label,
                           style={"color": "#f0f0f5",
                                   "fontWeight": "600"}),
            ]),
            html.Span("·"),
            html.Span(f"{n_markers} marker"
                       f"{'' if n_markers == 1 else 's'}"),
            html.Span("·"),
            html.Span(note_label),
            html.Span("·"),
            html.Span(file_label,
                       style={"color": "#888"}),
        ]

    MAX_MARKER_SHAPES = 64  # NASA Rule 3 fixed bound

    # Per-field landmark colors. Distinct + semantic so the
    # reviewer can read EO / LAS / BO / PID / BB by hue alone.
    LANDMARK_COLORS: dict = {
        "EO":  "#ff453a",  # red    -- electrographic onset
        "LAS": "#ff9f0a",  # orange -- large-amp spiking
        "BO":  "#bf5af2",  # violet -- behavioral onset
        "PID": "#5e7ce2",  # blue   -- post-ictal depression
        "BB":  "#30d158",  # green  -- back to baseline
    }

    @app.callback(
        Output("video-lfp-trace", "figure", allow_duplicate=True),
        Input("video-review-marker-store", "data"),
        Input("video-events-store", "data"),
        State("video-lfp-trace", "figure"),
        prevent_initial_call=True,
    )
    def _render_marker_shapes(markers, events, fig):
        """Paint colored landmark verticals on the LFP.

        Two sources:
        * ``video-review-marker-store`` -- legacy flat onset
          markers (red dotted) from the pre-BHZ code path.
        * ``video-events-store`` -- the BHZ structured events;
          each filled EO/LAS/BO/PID/BB landmark renders as a
          line in its semantic color. Up to 5 per event.

        Preserves ``shapes[0]`` (the orange video cursor) by
        reading it out of the current figure State and re-using
        it untouched. Anything past index 0 is marker territory.
        Capped at ``MAX_MARKER_SHAPES`` per NASA Rule 3.
        """
        if not fig:
            return no_update
        existing = (fig.get("layout") or {}).get("shapes") or []
        cursor = existing[0] if existing else None
        new_shapes: list[dict] = []
        if cursor is not None:
            new_shapes.append(cursor)
        # Legacy flat markers (unchanged behavior).
        capped = (markers or [])[:MAX_MARKER_SHAPES]
        for m in capped:
            t = float(m.get("peak_time_sec", 0))
            new_shapes.append({
                "type": "line",
                "xref": "x", "yref": "paper",
                "x0": t, "x1": t, "y0": 0, "y1": 1,
                "line": {"color": "#ff453a", "width": 1.5,
                          "dash": "dot"},
                "opacity": 0.85,
            })
        # Structured BHZ landmarks. Per-field colors.
        for i, e in enumerate(events or []):
            if i >= 16:  # NASA Rule 3: bound events per file
                break
            for field, color in LANDMARK_COLORS.items():
                t_sec = e.get(f"{field}_sec")
                if t_sec is None:
                    continue
                try:
                    t = float(t_sec)
                except (TypeError, ValueError):
                    continue
                new_shapes.append({
                    "type": "line",
                    "xref": "x", "yref": "paper",
                    "x0": t, "x1": t, "y0": 0, "y1": 1,
                    "line": {"color": color, "width": 2},
                    "opacity": 0.85,
                })
        patch = Patch()
        patch["layout"]["shapes"] = new_shapes
        return patch

    @app.callback(
        Output("video-review-marker-store", "data",
                allow_duplicate=True),
        Input("video-review-clear-markers-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _clear_markers(n):
        if not n:
            return no_update
        return []

    # ---- Step 4: save the review (Mark done OR N hotkey) ---- #
    # On a successful save we ALSO push the next-in-queue file id
    # into the session+file dropdowns (auto-advance, Linear /
    # Gmail pattern) and emit an Undo toast payload that the
    # `U` hotkey + Undo button consume. The Undo window is 8 s.
    UNDO_WINDOW_MS = 8000

    @app.callback(
        Output("video-review-status", "children",
                allow_duplicate=True),
        Output("video-review-marker-store", "data",
                allow_duplicate=True),
        Output("video-review-decision", "value",
                allow_duplicate=True),
        Output("video-review-note", "value",
                allow_duplicate=True),
        Output("video-session-dropdown", "value",
                allow_duplicate=True),
        Output("video-file-dropdown", "value",
                allow_duplicate=True),
        Output("kbd-undo", "data", allow_duplicate=True),
        Input("video-review-save-btn", "n_clicks"),
        State("video-file-dropdown", "value"),
        State("video-review-decision", "value"),
        State("video-review-marker-store", "data"),
        State("video-review-note", "value"),
        State("video-queue-animal", "value"),
        State("video-events-store", "data"),
        State("video-channel-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _save_review(n_clicks, file_id, decision, markers, note,
                      animal_value, events, channel):
        nop7 = (no_update,) * 7
        if not n_clicks or not file_id:
            return ("Pick a recording first." if n_clicks
                    else no_update,
                    no_update, no_update, no_update,
                    no_update, no_update, no_update)
        if decision not in ("no_events", "has_events"):
            return ("Pick \"No events seen\" or \"Events seen\" "
                     "before saving.",
                    *nop7[1:])
        events = list(events or [])
        if decision == "has_events":
            if not events:
                return ("Add at least one event with + Add event "
                         "before saving.",
                        *nop7[1:])
            # Gating: every event must be complete (type +
            # required landmarks + Racine 1-8).
            incomplete = [i + 1 for i, e in enumerate(events)
                           if not _events.is_event_complete(e)]
            if incomplete:
                idxs = ", ".join(str(i) for i in incomplete)
                return (f"Event{'s' if len(incomplete) > 1 else ''} "
                         f"{idxs} still need landmarks + Racine "
                         "before saving.",
                        *nop7[1:])
        email = current_user_email()
        if not email:
            return ("Not signed in — can't record who reviewed "
                     "this.",
                    *nop7[1:])
        # PI verification pipeline (the BHZ event taxonomy plan):
        # every undergrad save lands in 'pending_pi_review'.
        # markers_json stores the canonical event list -- empty
        # for the no-events branch, populated for the events
        # branch. The CSV is NOT written here; that happens
        # exclusively when the PI hits Approve in the Event
        # Verification tab. This is the second-layer safety net
        # the lab asked for.
        markers_payload = events if decision == "has_events" else None
        try:
            store.mark_review(
                int(file_id), email, "pending_pi_review",
                markers=markers_payload,
                note=(note or None),
            )
        except Exception as e:
            logger.warning("mark_review failed: %s", e)
            return (f"Save failed: {e}", *nop7[1:])
        from datetime import datetime as _dt
        decision_label = ("0 events"
                           if decision == "no_events"
                           else f"{len(events)} event"
                                 f"{'' if len(events) == 1 else 's'}")
        badge = (f"✓ Saved for PI review at "
                  f"{_dt.now().strftime('%H:%M')}  ·  "
                  f"{decision_label}.  "
                  f"The PI will verify before final CSV export.")
        # Resolve the next file in the queue so we can auto-advance.
        next_session, next_file = _resolve_next_in_queue(
            store, animal_value, int(file_id), email,
            queue_limit=queue_limit,
        )
        # Undo toast payload: the JS reads label + deadline_ms and
        # the U-hotkey / Undo-button callback reads file_id.
        import time as _time
        undo_payload = {
            "file_id": int(file_id),
            "label": (f"Marked file #{int(file_id)} as "
                       f"{decision.replace('_', ' ')}"),
            "deadline_ms": int(_time.time() * 1000)
                             + UNDO_WINDOW_MS,
            "status": decision,
        }
        # If we found a next file, push it; otherwise leave the
        # dropdowns alone so the reviewer sees the queue empty
        # state via _render_queue.
        if next_file is None:
            return (badge, [], None, "",
                    no_update, no_update, undo_payload)
        return (badge, [], None, "",
                next_session, next_file, undo_payload)

    # ---- file/channel options ---- #
    @app.callback(
        Output("video-file-dropdown", "options"),
        Output("video-file-dropdown", "value"),
        Input("video-session-dropdown", "value"),
        State("lfp-to-video-bridge", "data"),
        State("video-file-dropdown", "value"),
    )
    def _update_files(session_dir, bridge, current_value):
        """Repopulate the file dropdown when the session changes.

        Priority for the default value:
          1. LFP-Browser bridge -- explicit hand-off.
          2. ``current_value`` if it's already in the new options.
             This is the path the queue button + J/K hotkey rely
             on: ``_load_from_queue`` writes (session, file_id) in
             one shot, and this callback fires from the session
             change a millisecond later. Without the preservation
             check, ``_update_files`` would overwrite the just-set
             file_id with ``options[0]`` (the newest file in the
             session) -- which is why the queue highlight tracked
             the wrong file AND the LFP loaded a different chunk
             than the one the reviewer clicked.
          3. ``options[0]`` (newest file) as the legacy default.
        """
        files = _files_with_video(store, session_dir)
        options = [
            {"label": f"{(f['chunk_datetime'] or '')[:16]} - {os.path.basename(f['file_path'])}",
             "value": f["id"]}
            for f in files
        ]
        default = None
        if (bridge and isinstance(bridge, dict)
                and bridge.get("session_dir") == session_dir):
            wanted = bridge.get("file_id")
            if any(o["value"] == wanted for o in options):
                default = wanted
        if default is None and current_value is not None:
            if any(o["value"] == current_value for o in options):
                default = current_value
        if default is None:
            default = options[0]["value"] if options else None
        return options, default

    # ---- video player + history + channel options ---- #
    @app.callback(
        Output("video-player-container", "children"),
        Output("video-note-history", "children"),
        Output("video-channel-dropdown", "options"),
        Output("video-channel-dropdown", "value"),
        Input("video-file-dropdown", "value"),
        State("video-channel-dropdown", "value"),
        State("lfp-to-video-bridge", "data"),
        State("video-queue-animal", "value"),
    )
    def _update_player(file_id, current_channel, bridge, queue_animal):
        if not file_id:
            return (
                html.P("Pick a file to load the video.",
                       style={"color": "#a0a0b0"}),
                "",
                [], 0,
            )

        # Build channel options from the .mat itself (truth) +
        # session_config (for friendly names). We need the actual shape.
        file_path = _file_path_for_id(store, file_id)
        ch_options: list[dict] = []
        ch_value = 0
        # Bridge override: when the LFP Browser sent a channel
        # AND it matches this file, use it as the default.
        if (bridge and isinstance(bridge, dict)
                and bridge.get("file_id") == file_id):
            current_channel = bridge.get("channel", current_channel)
        # Resolve the queue-picked animal (strip the "_pool_" prefix
        # so the unassigned-pool case behaves the same as an
        # assignment).
        target_animal = None
        if queue_animal:
            if queue_animal.startswith("_pool_"):
                target_animal = queue_animal[len("_pool_"):]
            else:
                target_animal = queue_animal
        if file_path:
            try:
                # Route the channel-count probe through the chunk cache
                # so the subsequent LFP callback hits a warm entry.
                chunk = get_chunk(file_path)
                n_ch = int(chunk.signal.shape[1])
                session_dir = _session_dir_for_file(store, file_id)
                ch_options = _channel_options(store, session_dir, n_ch)
                # Animal-aware default: when the reviewer picked
                # animal X from the queue, find that animal's
                # electrode contacts in this file and default to the
                # first one. Bridge overrides this. Free-form (no
                # queue selection) keeps the existing behaviour.
                animal_choice = None
                if target_animal and session_dir:
                    elecs = store.electrodes_for_animal_in_session(
                        session_dir, target_animal,
                    )
                    valid_ch_values = {o["value"] for o in ch_options}
                    for e in elecs:
                        if e["channel_index"] in valid_ch_values:
                            animal_choice = e["channel_index"]
                            break
                # Priority: explicit bridge > queue-picked animal
                # > previous user pick > 0.
                if (bridge and isinstance(bridge, dict)
                        and bridge.get("file_id") == file_id
                        and bridge.get("channel") is not None):
                    desired = bridge.get("channel")
                elif animal_choice is not None:
                    desired = animal_choice
                elif current_channel is not None:
                    desired = current_channel
                else:
                    desired = (ch_options[0]["value"]
                                if ch_options else 0)
                # Ensure desired ends up in the visible options set;
                # if not, fall back to the first option (which
                # excludes stim_copy by construction).
                valid = {o["value"] for o in ch_options}
                ch_value = (desired if desired in valid else
                             (ch_options[0]["value"]
                                if ch_options else 0))
            except Exception as e:
                logger.warning("Channel probe failed for file %d: %s", file_id, e)

        # Multi-camera support: enumerate every companion _vN.mp4
        # and render one <video> per camera side-by-side. The first
        # camera is the "master" (carries the controls, drives the
        # clientside cursor + currentTime store). Slaves are muted
        # and follow the master via a per-tick clientside sync.
        n_cams = 0
        if file_path:
            try:
                from src.utils.video import companion_video_paths
                n_cams = len(companion_video_paths(file_path))
            except Exception as e:
                logger.debug("companion enumeration failed: %s", e)
                n_cams = 0
        # Fall back to 1 so the legacy single-camera route still
        # renders something even if globbing fails on the SMB share.
        n_cams = max(1, n_cams)
        cam_videos = []
        for i in range(1, n_cams + 1):
            is_master = (i == 1)
            video_el = html.Video(
                id=(VIDEO_DOM_ID if is_master
                     else f"{VIDEO_DOM_ID}-cam{i}"),
                src=f"/media/video/{file_id}/{i}",
                controls=is_master,
                muted=(not is_master),
                preload="metadata",
                style={"width": "100%", "maxHeight": "520px",
                        "borderRadius": "10px",
                        "backgroundColor": "#000",
                        "minWidth": "0"},
            )
            # Wrap each camera in a card with a FOCUS badge.
            # Single-camera files skip the wrapper -- there's
            # nothing to focus against. Multi-cam wraps so the
            # reviewer can mark which camera shows the target
            # animal; PiP narrows to JUST the focused camera.
            if n_cams == 1:
                cam_videos.append(video_el)
            else:
                cam_videos.append(html.Div([
                    video_el,
                    html.Button(
                        "FOCUS",
                        id={"type": "video-cam-focus-btn",
                             "idx": i},
                        n_clicks=0,
                        title="Mark this camera as the one "
                               "showing the target animal. "
                               "Pip mode shows only the "
                               "focused camera.",
                        style={
                            "position": "absolute",
                            "top": "6px", "right": "6px",
                            "padding": "3px 8px",
                            "background":
                                "rgba(10, 10, 20, 0.7)",
                            "color": "#cfd0d6",
                            "border":
                                "1px solid rgba(255,255,255,0.2)",
                            "borderRadius": "4px",
                            "fontSize": "10px",
                            "fontWeight": "600",
                            "letterSpacing": "0.5px",
                            "cursor": "pointer",
                            "backdropFilter": "blur(6px)",
                        },
                    ),
                ], id={"type": "video-cam", "idx": i},
                   className="video-cam-wrapper",
                   style={"position": "relative",
                           "borderRadius": "10px",
                           # data-focused attribute is set by
                           # a clientside callback when the
                           # video-focus-cam Store changes.
                           "border": "2px solid transparent",
                           "transition": "border-color 0.15s "
                                          "ease-out"},
                   **{"data-cam-idx": i,
                       "data-focused":
                           "true" if is_master else "false"}))
        if n_cams == 1:
            player = cam_videos[0]
        elif n_cams == 2:
            # Two cameras: stack vertically. With the Step 2
            # grid taking a single column for video + LFP +
            # Hilbert, side-by-side at 2 wide compresses each
            # video to ~240 px which is too small to see
            # behavioral detail; vertical stack uses the full
            # column width.
            player = html.Div(cam_videos, style={
                "display": "grid",
                "gridTemplateColumns": "1fr",
                "gridTemplateRows": "1fr 1fr",
                "gap": "8px",
            })
        else:
            # 3+ cameras: side-by-side stays most space-
            # efficient (each video is at least 220 px wide
            # at 3 cams).
            player = html.Div(cam_videos, style={
                "display": "grid",
                "gridTemplateColumns": f"repeat({n_cams}, 1fr)",
                "gap": "8px",
            })
        history = _render_history(store, file_id)
        return player, history, ch_options, ch_value

    # ---- LFP trace ---- #
    @app.callback(
        Output("video-lfp-trace", "figure", allow_duplicate=True),
        Output("video-lfp-status", "children"),
        Output("video-filter-state", "data"),
        Output("video-lfp-duration", "data"),
        Input("video-file-dropdown", "value"),
        Input("video-channel-dropdown", "value"),
        Input("video-apply-filter-btn", "n_clicks"),
        State("video-filter-hp", "value"),
        State("video-filter-lp", "value"),
        State("video-filter-notch", "value"),
        State("video-filter-smooth", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _update_lfp(file_id, channel, _n_apply,
                     hp, lp, notch, smooth_ms):
        if not file_id:
            return (_empty_lfp_fig(
                "Pick a recording above to see the brain signal here."),
                    "", no_update, 0.0)
        if channel is None:
            return (_empty_lfp_fig(
                "Pick a brain channel to display."),
                    "", no_update, 0.0)
        file_path = _file_path_for_id(store, file_id)
        if not file_path:
            return (_empty_lfp_fig("File not found in DB."),
                    "", no_update, 0.0)

        # Stim-blank ONLY the LFP channel that sits directly after a
        # stim_copy channel in the channel list. The stim artifact
        # cross-talks into the immediately-following electrode via
        # capacitive coupling; channels further along read cleanly.
        # Stim-copy channels themselves are never blanked (you'd
        # erase the signal you want to see).
        session_dir = _session_dir_for_file(store, file_id)
        stim_copy = _stim_copy_channels(store, session_dir)
        is_stim_copy = channel in stim_copy
        do_blank = _should_stim_blank(channel, stim_copy)
        stim_times = (_stim_times_for_file(store, file_id)
                       if do_blank
                       else np.asarray([], dtype=np.float64))

        try:
            t, signal, duration, n_blanked = _decimated_lfp(
                file_path, channel,
                stim_times=stim_times if len(stim_times) else None,
                blank_pre_ms=blank_pre_ms,
                blank_post_ms=blank_post_ms,
            )
        except Exception as e:
            logger.warning("LFP load failed file=%s ch=%s: %s",
                           file_id, channel, e)
            return (_empty_lfp_fig(f"LFP load error: {e}"),
                    "", no_update, 0.0)

        # Apply live filter (HP/LP/notch/smooth) on the *decimated*
        # display series so we don't pay the cost of filtering the
        # full-resolution signal here -- the Video Review tab caps
        # display at INITIAL_TARGET_BINS regardless. For full-fidelity
        # filtered exploration, the operator hops to the LFP Browser.
        try:
            display_fs = float(len(t) / duration) if duration > 0 else 0
            filtered = apply_filter(
                np.asarray(signal, dtype=np.float32), display_fs,
                highpass=hp, lowpass=lp, notch=notch,
                smoothing_ms=smooth_ms,
            )
        except Exception as e:
            logger.warning("Video filter failed (using unfiltered): %s", e)
            filtered = signal

        fig = _build_lfp_figure(t, filtered, label=f"Ch{channel}")
        filt_bits = []
        if hp and hp > 0: filt_bits.append(f"HP={hp:g}")
        if lp and lp > 0: filt_bits.append(f"LP={lp:g}")
        if notch and notch > 0: filt_bits.append(f"Notch={notch}")
        if smooth_ms and smooth_ms > 0:
            filt_bits.append(f"Smooth={smooth_ms:g}ms")
        status_bits = [f"{duration:.1f}s", f"{len(t):,} display points"]
        if is_stim_copy:
            status_bits.append("stim-copy channel · not blanked")
        elif not do_blank:
            # Channel exists but isn't directly after a stim_copy
            # contact, so it stays raw even when stim events exist.
            status_bits.append(
                "two slots from stim-copy · not blanked")
        elif n_blanked:
            status_bits.append(
                f"stim-blanked: {n_blanked} pulses "
                f"({blank_pre_ms:+.0f}–{blank_post_ms:+.0f} ms)"
            )
        elif len(_stim_times_for_file(store, file_id)) == 0:
            status_bits.append("no stim events on file")
        if filt_bits:
            status_bits.append(" + ".join(filt_bits))
        new_state = {"hp": hp, "lp": lp, "notch": notch,
                     "smooth": smooth_ms}
        return (fig, " · ".join(status_bits), new_state,
                float(duration))

    # ---- Time-locked analysis trace (feature vs time) ----
    @app.callback(
        Output("video-analysis-trace", "figure"),
        Output("video-analysis-status", "children"),
        Input("video-file-dropdown", "value"),
        Input("video-analysis-feature", "value"),
        Input("video-analysis-apply-btn", "n_clicks"),
        # Channel must be an Input -- the file→channel cascade
        # sets channel AFTER _update_analysis first fires (with
        # channel=None). State would leave the Hilbert envelope
        # stuck on the empty 'Pick a brain channel' message.
        Input("video-channel-dropdown", "value"),
        # Mass Analyze cutoff drives the dashed-line threshold +
        # the BHZ peak detection on this view. Listening as an
        # Input means typing in the MA cutoff (or click-to-set
        # from the plot) immediately redraws the threshold.
        Input("video-ma-cutoff-input", "value"),
        State("video-analysis-smooth", "value"),
        State("video-analysis-rollwin", "value"),
        State("video-analysis-postproc", "value"),
    )
    def _update_analysis(file_id, feature, _n_apply,
                          channel, ma_cutoff,
                          smooth_sec, rollwin, postproc):
        if not file_id:
            return (_empty_lfp_fig(
                "Pick a recording above to see the feature trace."), "")
        if not feature:
            return (_empty_lfp_fig(
                "Pick a brain feature to plot."), "")
        # Hilbert envelope (BHZ default). Continuous-time trace
        # rather than per-epoch scatter; 1:1 with tay_preprocess.m.
        if feature == "hilbert":
            # Use the Mass Analyze cutoff so the dashed threshold
            # line tracks the MA input live. Falls back to the
            # lab default if the input is invalid/blank.
            try:
                cutoff = float(ma_cutoff) if ma_cutoff is not None \
                          else _BHZ_CUTOFF
                if cutoff <= 0:
                    cutoff = _BHZ_CUTOFF
            except (TypeError, ValueError):
                cutoff = _BHZ_CUTOFF
            return _render_hilbert_trace(
                store, int(file_id), channel,
                blank_pre_ms=blank_pre_ms,
                blank_post_ms=blank_post_ms,
                cutoff=cutoff,
            )
        try:
            rows = store.query_evoked_features(int(file_id))
        except Exception as e:
            logger.warning("Analysis load failed file=%s: %s",
                            file_id, e)
            return (_empty_lfp_fig(
                f"Couldn't load features: {e}"), "")
        if not rows:
            return (_empty_lfp_fig(
                "This recording hasn't been analyzed yet — "
                "features show up here once the pipeline finishes."), "")
        rows.sort(key=lambda r: (r.get("epoch_time_sec") or 0.0))
        ok_t, ok_y = [], []
        art_t, art_y = [], []
        for r in rows:
            v = r.get(feature)
            t = r.get("epoch_time_sec")
            if v is None or t is None:
                continue
            if r.get("is_artifact"):
                art_t.append(float(t))
                art_y.append(float(v))
            else:
                ok_t.append(float(t))
                ok_y.append(float(v))
        if not ok_t and not art_t:
            return (_empty_lfp_fig(
                f"No '{feature}' values on this file."), "")
        feature_label = feature.replace("_", " ")

        # ---- Post-processing on the OK series ----
        proc = set(postproc or [])
        ok_y_arr = np.asarray(ok_y, dtype=np.float64)
        ok_t_arr = np.asarray(ok_t, dtype=np.float64)
        applied_bits: list[str] = []

        # Rolling median window in EPOCHS (centered, odd-size).
        try:
            rw = int(rollwin or 0)
        except (TypeError, ValueError):
            rw = 0
        if rw >= 3 and ok_y_arr.size >= rw:
            from scipy.signal import medfilt as _medfilt
            k = rw if rw % 2 == 1 else rw + 1
            ok_y_arr = _medfilt(ok_y_arr, kernel_size=k)
            applied_bits.append(f"medfilt({k})")

        # Gaussian smoothing in SECONDS (sigma; converted to samples
        # via the median inter-epoch interval so the smoothing
        # behaves the same regardless of stim rate).
        try:
            sm = float(smooth_sec or 0)
        except (TypeError, ValueError):
            sm = 0.0
        if sm > 0 and ok_t_arr.size >= 3:
            dt_arr = np.diff(ok_t_arr)
            dt_arr = dt_arr[dt_arr > 0]
            median_dt = (float(np.median(dt_arr))
                         if dt_arr.size else 0.0)
            if median_dt > 0:
                from scipy.ndimage import gaussian_filter1d as _gf1
                sigma_samples = max(0.5, sm / median_dt)
                sigma_samples = min(sigma_samples,
                                     ok_y_arr.size / 4.0)
                ok_y_arr = _gf1(ok_y_arr.astype(np.float32),
                                 sigma=sigma_samples,
                                 mode="nearest").astype(np.float64)
                applied_bits.append(f"smooth {sm:g}s")

        if "detrend" in proc and ok_y_arr.size >= 3:
            from scipy.signal import detrend as _detrend
            ok_y_arr = _detrend(ok_y_arr, type="linear")
            applied_bits.append("detrend")

        if "zscore" in proc and ok_y_arr.size >= 2:
            std = float(np.std(ok_y_arr))
            if std > 0:
                ok_y_arr = (ok_y_arr - float(np.mean(ok_y_arr))) / std
                applied_bits.append("z-score")

        if "median" in proc and ok_y_arr.size >= 1:
            ok_y_arr = ok_y_arr - float(np.median(ok_y_arr))
            applied_bits.append("median-center")

        hide_art = "hide_artifact" in proc

        # ---- Build figure ----
        fig = go.Figure()
        ok_y = ok_y_arr.tolist()
        if ok_t:
            fig.add_trace(go.Scattergl(
                x=ok_t, y=ok_y, mode="lines+markers",
                line=dict(color="#5e7ce2", width=1.2),
                marker=dict(size=5, color="#5e7ce2"),
                name="ok",
                hovertemplate=("t=%{x:.2f}s<br>"
                                + feature_label
                                + "=%{y:.3f}<extra></extra>"),
            ))
        if art_t and not hide_art:
            fig.add_trace(go.Scattergl(
                x=art_t, y=art_y, mode="markers",
                marker=dict(size=6, color="#EF553B", symbol="x"),
                name="artifact",
                hovertemplate=("t=%{x:.2f}s<br>"
                                + feature_label
                                + "=%{y:.3f} (artifact)"
                                + "<extra></extra>"),
            ))
        fig.update_layout(
            plot_bgcolor="#13131f", paper_bgcolor="#13131f",
            height=180,
            margin=dict(l=60, r=20, t=10, b=40),
            xaxis=dict(title="Time (s)", showgrid=True,
                        gridcolor="rgba(255,255,255,0.05)",
                        zeroline=False, color="#cfd0d6"),
            yaxis=dict(title=feature_label, showgrid=True,
                        gridcolor="rgba(255,255,255,0.05)",
                        zeroline=False, color="#cfd0d6"),
            showlegend=bool(art_t and not hide_art),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                         xanchor="right", x=1, font_size=10,
                         bgcolor="rgba(0,0,0,0)"),
            shapes=[dict(
                type="line", xref="x", yref="paper",
                x0=0, x1=0, y0=0, y1=1,
                line=dict(color="#ff9f0a", width=2),
            )],
        )
        bits = [f"{len(ok_t) + len(art_t)} epochs",
                 f"{len(art_t)} artifact"]
        if applied_bits:
            bits.append(" + ".join(applied_bits))
        return fig, " · ".join(bits)

    # ---- X-axis sync: LFP zoom drives analysis zoom (and back) ----
    # Bidirectional clientside listeners on relayoutData so either
    # plot's pan/zoom/reset propagates to the other. We patch only
    # layout.xaxis.range / autorange -- the existing zoom-decimate
    # callback on the LFP still fires on its own relayoutData input
    # without seeing this clientside echo (Dash skips clientside
    # outputs when the input value is the same as last time).
    # Use Patch (not a full-figure return) so only layout.xaxis
    # is touched. Returning `{data: fig.data, ...}` from the
    # full-figure variant created a race on file/channel switch:
    # the State("...figure") snapshot was captured at chain
    # start, potentially BEFORE _update_analysis or _update_lfp
    # had written the new figure for the new file. Writing
    # back fig.data then clobbered the new envelope with the
    # previous file's data, producing the "Hilbert looks like
    # raw instantaneous amplitude" symptom the user reported.
    # Patch is merged atomically against whatever the figure
    # currently is, so the data array stays whatever the
    # latest _update_* callback wrote.
    _XAXIS_SYNC_JS = """
    function(rel) {
        if (!rel) { return window.dash_clientside.no_update; }
        var has_range = ('xaxis.range[0]' in rel
                          && 'xaxis.range[1]' in rel);
        var has_auto = !!rel['xaxis.autorange'];
        if (!has_range && !has_auto) {
            return window.dash_clientside.no_update;
        }
        var p = new window.dash_clientside.Patch();
        if (has_range) {
            p.layout.xaxis.range = [rel['xaxis.range[0]'],
                                     rel['xaxis.range[1]']];
            p.layout.xaxis.autorange = false;
        } else {
            p.layout.xaxis.autorange = true;
            delete p.layout.xaxis.range;
        }
        return p;
    }
    """
    app.clientside_callback(
        _XAXIS_SYNC_JS,
        Output("video-analysis-trace", "figure",
                allow_duplicate=True),
        Input("video-lfp-trace", "relayoutData"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        _XAXIS_SYNC_JS,
        Output("video-lfp-trace", "figure", allow_duplicate=True),
        Input("video-analysis-trace", "relayoutData"),
        prevent_initial_call=True,
    )

    # 4. Mirror the orange cursor on the analysis trace too.
    app.clientside_callback(
        """
        function(currentTime, fig, lfp_dur) {
            if (fig === undefined || fig === null) {
                return window.dash_clientside.no_update;
            }
            if (currentTime === null || currentTime === undefined) {
                return window.dash_clientside.no_update;
            }
            var t = currentTime;
            var v = document.getElementById('""" + VIDEO_DOM_ID + """');
            if (v && isFinite(v.duration) && v.duration > 0
                    && lfp_dur && lfp_dur > 0) {
                t = currentTime * (lfp_dur / v.duration);
            }
            const newFig = {
                data: fig.data,
                layout: Object.assign({}, fig.layout, {
                    shapes: [{
                        type: 'line', xref: 'x', yref: 'paper',
                        x0: t, x1: t, y0: 0, y1: 1,
                        line: {color: '#ff9f0a', width: 2}
                    }]
                })
            };
            return newFig;
        }
        """,
        Output("video-analysis-trace", "figure",
                allow_duplicate=True),
        Input("video-current-time", "data"),
        State("video-analysis-trace", "figure"),
        State("video-lfp-duration", "data"),
        prevent_initial_call=True,
    )

    # 5. Click an epoch on the analysis trace -> seek the video.
    app.clientside_callback(
        """
        function(clickData, lfp_dur) {
            if (!clickData || !clickData.points || !clickData.points.length) {
                return '';
            }
            const x = clickData.points[0].x;
            const v = document.getElementById('""" + VIDEO_DOM_ID + """');
            if (!v || !isFinite(x)) { return ''; }
            var vt = x;
            if (isFinite(v.duration) && v.duration > 0
                    && lfp_dur && lfp_dur > 0) {
                vt = x * (v.duration / lfp_dur);
            }
            if (vt < 0) { vt = 0; }
            if (isFinite(v.duration) && vt > v.duration) {
                vt = v.duration;
            }
            v.currentTime = vt;
            return '';
        }
        """,
        Output("video-seek-sink", "children", allow_duplicate=True),
        Input("video-analysis-trace", "clickData"),
        State("video-lfp-duration", "data"),
        prevent_initial_call=True,
    )

    # ---- Zoom-driven dynamic decimation ----
    # When the reviewer zooms in, re-decimate just the visible window so
    # narrow features (e.g. 150 us stim pulses) resolve to real samples.
    # The orange cursor lives in figure.layout.shapes[0] and the clientside
    # cursor callback writes layout only, so patching figure.data here is
    # safe -- the cursor and zoom callbacks edit disjoint paths.
    @app.callback(
        Output("video-lfp-trace", "figure", allow_duplicate=True),
        Input("video-lfp-trace", "relayoutData"),
        State("video-file-dropdown", "value"),
        State("video-channel-dropdown", "value"),
        State("video-filter-state", "data"),
        prevent_initial_call=True,
    )
    def _video_lfp_zoom(relayout, file_id, channel, filter_state):
        if not file_id or channel is None or not relayout:
            return no_update
        x0, x1, is_reset = parse_relayout(relayout)
        if x0 is None and x1 is None and not is_reset:
            return no_update

        file_path = _file_path_for_id(store, file_id)
        if not file_path:
            return no_update

        session_dir = _session_dir_for_file(store, file_id)
        stim_copy = _stim_copy_channels(store, session_dir)
        # Mirror the same "blank only the LFP directly after a
        # stim_copy contact" rule from _update_lfp so the
        # re-decimated zoom view shows the same NaN gaps the wider
        # view does. Anything else (stim-copy itself, or an LFP two
        # slots away) re-decimates raw.
        do_blank = _should_stim_blank(channel, stim_copy)
        stim_times = (_stim_times_for_file(store, file_id)
                       if do_blank
                       else np.asarray([], dtype=np.float64))

        try:
            series, fs, _ = _get_blanked_series(
                file_path, channel,
                stim_times if len(stim_times) else None,
                blank_pre_ms, blank_post_ms,
            )
        except Exception as e:
            logger.warning("zoom blanked-series load failed file=%s ch=%s: %s",
                           file_id, channel, e)
            return no_update

        # Apply the same filter the time-domain plot used. Filtering
        # is on the full-resolution series here (zoom needs raw fs
        # not the display rate) -- still cheap because get_chunk +
        # _get_blanked_series caches keep this hot.
        st = filter_state or {}
        try:
            series = apply_filter(
                np.asarray(series, dtype=np.float32), fs,
                highpass=st.get("hp"), lowpass=st.get("lp"),
                notch=st.get("notch"),
                smoothing_ms=st.get("smooth"),
            )
        except Exception as e:
            logger.debug("zoom filter skipped: %s", e)

        n = len(series)
        if is_reset:
            lo, hi = 0, n
        else:
            lo, hi = window_slice(n, fs, x0, x1)
        if hi <= lo:
            return no_update

        target_bins = choose_target_bins(hi - lo) or INITIAL_TARGET_BINS
        x_p, y_p, _decim = envelope(
            series[lo:hi], fs, target_bins, t_start=lo / fs,
        )

        p = Patch()
        p["data"][0]["x"] = x_p
        p["data"][0]["y"] = y_p
        return p

    # ---- Note save (unchanged) ---- #
    @app.callback(
        Output("video-note-status", "children"),
        Output("video-note-history", "children", allow_duplicate=True),
        Output("video-note-input", "value"),
        Input("video-note-save-btn", "n_clicks"),
        State("video-file-dropdown", "value"),
        State("video-note-input", "value"),
        State("video-current-time", "data"),
        prevent_initial_call=True,
    )
    def _save_note(n_clicks, file_id, note, current_time):
        if not n_clicks:
            return no_update, no_update, no_update
        if not file_id:
            return (html.Span("Select a file first.", style={"color": "#ff453a"}),
                    no_update, no_update)
        if not note or not note.strip():
            return (html.Span("Note cannot be empty.", style={"color": "#ff453a"}),
                    no_update, no_update)

        # Prefix the note with the playback timestamp so reviewers can
        # navigate back to the moment they were commenting on.
        ts_prefix = ""
        if isinstance(current_time, (int, float)) and current_time > 0:
            ts_prefix = f"[t={current_time:.2f}s] "
        body = ts_prefix + note.strip()

        email = current_user_email() or "unknown"
        session_dir = _session_dir_for_file(store, file_id)
        try:
            ann_id = store.add_annotation(
                timestamp=datetime.now().isoformat(),
                note=body,
                category="video_review",
                session_dir=session_dir,
                file_id=file_id,
                user_email=email,
            )
        except Exception as e:
            logger.error("video_review annotation save failed: %s", e, exc_info=True)
            return (html.Span(f"Error: {e}", style={"color": "#ff453a"}),
                    no_update, no_update)

        return (
            html.Span(f"Saved note #{ann_id} as {email}.",
                      style={"color": "#30d158"}),
            _render_history(store, file_id),
            "",
        )

    # ================================================================== #
    #  Clientside time-locking
    # ================================================================== #

    # 1. Poll the <video> element 10 Hz; push currentTime into a Store.
    app.clientside_callback(
        """
        function(_n) {
            const v = document.getElementById('""" + VIDEO_DOM_ID + """');
            if (!v || isNaN(v.currentTime)) {
                return window.dash_clientside.no_update;
            }
            return v.currentTime;
        }
        """,
        Output("video-current-time", "data"),
        Input("video-time-tick", "n_intervals"),
    )

    # 1a. Pending-seek consumer. When the LFP Browser's "View video"
    #     button fires, it writes pending-seek with the desired LFP
    #     time + the LFP's true duration. Every 100 ms we check
    #     whether the master <video> has loaded its metadata; once
    #     it has, we map LFP time -> video time using the same BHZ
    #     ratio (currentTime = lfp_t * video_dur / lfp_dur), seek,
    #     and clear the pending Store so we don't keep re-seeking.
    app.clientside_callback(
        """
        function(_n, pending) {
            if (!pending || pending.start_sec === null
                    || pending.start_sec === undefined) {
                return window.dash_clientside.no_update;
            }
            const v = document.getElementById('""" + VIDEO_DOM_ID + """');
            if (!v || !isFinite(v.duration) || v.duration <= 0
                    || v.readyState < 1) {
                return window.dash_clientside.no_update;
            }
            var target = pending.start_sec;
            if (pending.lfp_dur && pending.lfp_dur > 0) {
                target = pending.start_sec
                        * (v.duration / pending.lfp_dur);
            }
            if (target < 0) target = 0;
            if (target > v.duration) target = v.duration;
            v.currentTime = target;
            return null;
        }
        """,
        Output("pending-seek", "data", allow_duplicate=True),
        Input("video-time-tick", "n_intervals"),
        State("pending-seek", "data"),
        prevent_initial_call=True,
    )

    # 1b. Multi-camera sync. Every tick, walk the DOM for slave
    #     <video> elements (ids "lfp-video-camN") and keep their
    #     currentTime + play/pause state aligned with the master
    #     ("lfp-video"). querySelectorAll handles the variable
    #     slave count without a callback re-register on file change.
    app.clientside_callback(
        """
        function(_n) {
            const master = document.getElementById('""" + VIDEO_DOM_ID + """');
            if (!master) { return ''; }
            const slaves = document.querySelectorAll(
                'video[id^=\"""" + VIDEO_DOM_ID + """-cam\"]');
            slaves.forEach(function(s) {
                if (!isFinite(s.duration) || s.readyState < 1) return;
                if (Math.abs(s.currentTime - master.currentTime) > 0.12) {
                    s.currentTime = master.currentTime;
                }
                if (master.paused && !s.paused) { s.pause(); }
                else if (!master.paused && s.paused) {
                    s.play().catch(function(){});
                }
                if (Math.abs(s.playbackRate - master.playbackRate) > 0.01) {
                    s.playbackRate = master.playbackRate;
                }
            });
            return '';
        }
        """,
        Output("video-cam-sync-sink", "children"),
        Input("video-time-tick", "n_intervals"),
    )

    # 2. Move the LFP-trace cursor whenever the stored time changes.
    #    Translate video.currentTime -> LFP time using the BHZ
    #    method: cursor_x = currentTime * (lfp_duration /
    #    video_duration). Both endpoints are ground truth -- the
    #    .mat tells us how long the LFP runs, the <video> tag tells
    #    us how long the video runs -- so container FPS lies cannot
    #    accumulate drift. When either duration isn't known yet,
    #    fall back to the identity mapping (same as before).
    app.clientside_callback(
        """
        function(currentTime, fig, lfp_dur) {
            if (fig === undefined || fig === null) {
                return window.dash_clientside.no_update;
            }
            if (currentTime === null || currentTime === undefined) {
                return window.dash_clientside.no_update;
            }
            var t = currentTime;
            var v = document.getElementById('""" + VIDEO_DOM_ID + """');
            if (v && isFinite(v.duration) && v.duration > 0
                    && lfp_dur && lfp_dur > 0) {
                t = currentTime * (lfp_dur / v.duration);
            }
            const newFig = {
                data: fig.data,
                layout: Object.assign({}, fig.layout, {
                    shapes: [{
                        type: 'line', xref: 'x', yref: 'paper',
                        x0: t, x1: t,
                        y0: 0, y1: 1,
                        line: {color: '#ff9f0a', width: 2}
                    }]
                })
            };
            return newFig;
        }
        """,
        Output("video-lfp-trace", "figure", allow_duplicate=True),
        Input("video-current-time", "data"),
        State("video-lfp-trace", "figure"),
        State("video-lfp-duration", "data"),
        prevent_initial_call=True,
    )

    # 3. Click on the LFP trace -> seek the video to that x value.
    #    Inverse mapping: video_t = x * (video_duration / lfp_duration).
    app.clientside_callback(
        """
        function(clickData, lfp_dur) {
            if (!clickData || !clickData.points || !clickData.points.length) {
                return '';
            }
            const x = clickData.points[0].x;
            const v = document.getElementById('""" + VIDEO_DOM_ID + """');
            if (!v || !isFinite(x)) { return ''; }
            var vt = x;
            if (isFinite(v.duration) && v.duration > 0
                    && lfp_dur && lfp_dur > 0) {
                vt = x * (v.duration / lfp_dur);
            }
            // Clamp so a click near the end doesn't trip
            // <video>'s past-end guard and silently no-op.
            if (vt < 0) { vt = 0; }
            if (isFinite(v.duration) && vt > v.duration) {
                vt = v.duration;
            }
            v.currentTime = vt;
            return '';
        }
        """,
        Output("video-seek-sink", "children"),
        Input("video-lfp-trace", "clickData"),
        State("video-lfp-duration", "data"),
    )

    # =================================================================== #
    #  Mass Analyze (undergrad bulk pre-screen)
    # =================================================================== #
    # Reads the animal from the Step 1 queue picker (one source of
    # truth). The scope counter updates live so the user knows
    # whether there's anything to scan BEFORE they click. Scan opens
    # a confirmation modal explaining the steps + showing the count;
    # Confirm in the modal is the only path that actually creates a
    # worker job.

    @app.callback(
        Output("video-ma-animal-display", "children"),
        Input("video-queue-animal", "value"),
    )
    def _on_video_ma_animal_display(picker_value):
        animal = _ma_animal_from_picker(picker_value)
        if not animal:
            return html.Span([
                html.Span("Will scan: ",
                           style={"color": "#a0a0b0"}),
                html.Span("(pick an animal in Step 1)",
                           style={"color": "#ff9f0a",
                                   "fontStyle": "italic"}),
            ])
        return html.Span([
            html.Span("Will scan: ",
                       style={"color": "#a0a0b0"}),
            html.Span(str(animal),
                       style={"color": "#f0f0f5",
                               "fontWeight": "600"}),
        ])

    @app.callback(
        Output("video-ma-scope", "children"),
        Input("video-queue-animal", "value"),
        Input("video-ma-job-id", "data"),
    )
    def _on_video_ma_scope_count(picker_value, _job_id):
        animal = _ma_animal_from_picker(picker_value)
        if not animal:
            return ""
        try:
            n = _mass_analyze.count_pending_for_animal(
                store, str(animal))
        except Exception as e:
            logger.warning(
                "count_pending_for_animal failed: %s", e)
            return html.Span(f"(scope count failed: {e})",
                              style={"color": "#ff453a"})
        if n == 0:
            return html.Span([
                f"Nothing to scan -- {animal} has 0 pending "
                "files (you've cleared the backlog, or all "
                "files are currently claimed by other "
                "reviewers).",
            ], style={"color": "#ff9f0a"})
        return html.Span([
            f"{animal}: ",
            html.Span(f"{n} pending file"
                       + ("" if n == 1 else "s"),
                       style={"color": "#f0f0f5",
                               "fontWeight": "600"}),
            " eligible to scan (files this reviewer hasn't "
            "started or finalised).",
        ])

    @app.callback(
        Output("video-ma-confirm-modal", "style"),
        Output("video-ma-modal-body", "children"),
        Input("video-ma-scan-btn", "n_clicks"),
        Input("video-ma-modal-cancel-btn", "n_clicks"),
        Input("video-ma-modal-confirm-btn", "n_clicks"),
        State("video-queue-animal", "value"),
        State("video-ma-cutoff-input", "value"),
        State("video-ma-confirm-modal", "style"),
        prevent_initial_call=True,
    )
    def _on_video_ma_modal_toggle(_scan_n, _cancel_n,
                                     _confirm_n,
                                     picker_value, cutoff,
                                     current_style):
        animal = _ma_animal_from_picker(picker_value)
        trig = callback_context.triggered_id
        hidden = dict(current_style or {})
        hidden["display"] = "none"
        # Cancel + Confirm both close the modal. Confirm's
        # job-creation lives in _on_video_ma_modal_confirm
        # (separate callback so it can write its own outputs).
        if trig in ("video-ma-modal-cancel-btn",
                     "video-ma-modal-confirm-btn"):
            return hidden, no_update
        if trig != "video-ma-scan-btn":
            return no_update, no_update
        # Open path: build the preview body using the current
        # animal + cutoff + scope count.
        try:
            cutoff_f = float(cutoff)
        except (TypeError, ValueError):
            cutoff_f = 0.0
        if not animal:
            body = html.Div(
                "Pick an animal in Step 1 first -- Mass "
                "Analyze reads from the queue picker.",
                style={"color": "#ff9f0a"})
        elif cutoff_f <= 0:
            body = html.Div(
                f"Invalid cutoff {cutoff!r} -- must be > 0.",
                style={"color": "#ff453a"})
        else:
            try:
                n = _mass_analyze.count_pending_for_animal(
                    store, str(animal))
            except Exception as e:
                n = 0
                logger.warning(
                    "modal count_pending failed: %s", e)
            if n == 0:
                body = html.Div(
                    f"{animal} has 0 pending files. Nothing "
                    "to scan.", style={"color": "#ff9f0a"})
            else:
                # ~5 s per file cold, ~250 ms per file warm.
                # Typical zero-peak ratio at 0.05 is 70-85% on
                # the BHZ pipeline -- midpoint = 78%.
                est_total_s = int(n * 5)
                est_min = est_total_s // 60
                est_sec_rem = est_total_s % 60
                if est_min:
                    est_str = (f"~{est_min} min "
                                f"{est_sec_rem:02d} s")
                else:
                    est_str = f"~{est_total_s} s"
                est_zero = int(round(n * 0.78))
                est_with = n - est_zero
                body = [
                    html.Div([
                        html.Span(
                            f"Scan {animal} at cutoff "
                            f"{cutoff_f:g}?",
                            style={"color": "#f0f0f5",
                                    "fontWeight": "600",
                                    "fontSize": "13px"}),
                    ], style={"marginBottom": "10px"}),
                    html.Ul([
                        html.Li([
                            html.Span("Files in scope: ",
                                       style={"color": "#a0a0b0"}),
                            html.Span(f"{n}",
                                       style={"color": "#f0f0f5",
                                               "fontWeight": "600"}),
                        ]),
                        html.Li([
                            html.Span(
                                "Per-file: ~5 s of Hilbert + "
                                "peakseek (instant on cache hit)",
                                style={"color": "#a0a0b0"}),
                        ]),
                        html.Li([
                            html.Span(
                                "Estimated total time: ",
                                style={"color": "#a0a0b0"}),
                            html.Span(est_str,
                                       style={"color": "#f0f0f5",
                                               "fontWeight": "600"}),
                            html.Span(
                                " (cold cache; re-scans are "
                                "instant)",
                                style={"color": "#a0a0b0"}),
                        ]),
                    ], style={"margin": "0 0 14px 18px",
                               "padding": "0"}),
                    html.Div(
                        "What happens after you click Confirm",
                        style={"color": "#f0f0f5",
                                "fontWeight": "600",
                                "marginBottom": "4px"}),
                    html.Ol([
                        html.Li(
                            "Worker computes the 20-200 Hz "
                            "Hilbert envelope on each file's "
                            "first animal channel and counts "
                            f"peaks above {cutoff_f:g}.",
                            style={"color": "#a0a0b0"}),
                        html.Li(
                            "Results are cached so re-running "
                            "at the same cutoff is free.",
                            style={"color": "#a0a0b0"}),
                        html.Li(
                            "Each file is classified as 0-peak "
                            "(auto-clear candidate) or >=1-peak "
                            "(stays in your queue).",
                            style={"color": "#a0a0b0"}),
                    ], style={"margin": "0 0 14px 18px",
                               "padding": "0"}),
                    html.Div(
                        "Worked example",
                        style={"color": "#f0f0f5",
                                "fontWeight": "600",
                                "marginBottom": "4px"}),
                    html.Div([
                        f"Typical animal: ~78% of files have "
                        "0 peaks above this cutoff.",
                        html.Br(),
                        f"At {n} files, expect roughly:",
                    ], style={"color": "#a0a0b0",
                               "marginBottom": "4px"}),
                    html.Ul([
                        html.Li([
                            html.Span(f"~{est_zero}",
                                       style={"color": "#30d158",
                                               "fontWeight": "600"}),
                            html.Span(
                                " files marked as no-events, "
                                "queued for PI verification",
                                style={"color": "#a0a0b0"}),
                        ]),
                        html.Li([
                            html.Span(f"~{est_with}",
                                       style={"color": "#ff9f0a",
                                               "fontWeight": "600"}),
                            html.Span(
                                " files stay in your queue for "
                                "manual scoring",
                                style={"color": "#a0a0b0"}),
                        ]),
                    ], style={"margin": "0 0 8px 18px",
                               "padding": "0"}),
                    html.Div(
                        "(Estimates only -- the actual split "
                        "depends on the recording. The scan "
                        "doesn't commit anything; you'll get a "
                        "separate Confirm step after it finishes.)",
                        style={"color": "#6c6c80",
                                "fontSize": "11px",
                                "fontStyle": "italic"}),
                ]
        open_style = dict(current_style or {})
        open_style["display"] = "flex"
        return open_style, body

    @app.callback(
        Output("video-ma-job-id", "data"),
        Output("video-ma-poll", "disabled"),
        Output("video-ma-progress", "children",
                allow_duplicate=True),
        Input("video-ma-modal-confirm-btn", "n_clicks"),
        State("video-queue-animal", "value"),
        State("video-ma-cutoff-input", "value"),
        prevent_initial_call=True,
    )
    def _on_video_ma_modal_confirm(n_clicks, picker_value, cutoff):
        if not n_clicks:
            return no_update, no_update, no_update
        animal_id = _ma_animal_from_picker(picker_value)
        email = (current_user_email() or "").lower()
        if not email:
            return None, True, "Sign in first."
        if not animal_id:
            return None, True, "Pick an animal in Step 1 first."
        try:
            cutoff = float(cutoff)
        except (TypeError, ValueError):
            return None, True, f"Invalid cutoff: {cutoff!r}"
        if cutoff <= 0:
            return None, True, "Cutoff must be > 0."
        try:
            job_id = _mass_analyze.create_job(
                store, email, animal_id, cutoff)
        except Exception as e:
            logger.exception("Mass Analyze create_job failed")
            return None, True, f"Failed to start: {e}"
        return job_id, False, "Scan queued..."

    @app.callback(
        Output("video-ma-progress", "children"),
        Output("video-ma-summary", "children"),
        Output("video-ma-confirm-row", "children"),
        Output("video-ma-poll", "disabled",
                allow_duplicate=True),
        Input("video-ma-poll", "n_intervals"),
        Input("video-ma-job-id", "data"),
        prevent_initial_call=True,
    )
    def _on_video_ma_poll(_n, job_id):
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
              f">=1 peak: {n_with}"
        )
        summary = []
        confirm_row = []
        polling_disabled = status in ("done", "failed",
                                         "cancelled")
        if status == "done":
            summary = html.Div([
                html.Div([
                    html.Span("OK ",
                               style={"color": "#30d158",
                                       "marginRight": "6px",
                                       "fontWeight": "700"}),
                    html.Span(
                        f"{n_zero} files with 0 peaks above "
                        f"{job['cutoff']:g}  ·  will mark as "
                        "no-events pending PI review",
                        style={"color": "#30d158",
                                "fontWeight": "600"}),
                ], style={"fontSize": "12px",
                           "marginBottom": "4px"}),
                html.Div([
                    html.Span("- ",
                               style={"color": "#a0a0b0",
                                       "marginRight": "6px"}),
                    html.Span(
                        f"{n_with} files with >=1 peak  ·  "
                        "stay in your queue for normal review",
                        style={"color": "#a0a0b0"}),
                ], style={"fontSize": "12px"}),
            ])
            confirm_row = [
                html.Button(
                    f"Confirm threshold -> mark {n_zero} files "
                    "as no-events pending PI review",
                    id="video-ma-confirm-btn", n_clicks=0,
                    style={"background": ("#5e7ce2" if n_zero > 0
                                            else "transparent"),
                            "color": ("white" if n_zero > 0
                                       else "#cfd0d6"),
                            "border": ("none" if n_zero > 0
                                        else "1px solid "
                                              "rgba(255,255,255,0.15)"),
                            "padding": "6px 14px",
                            "borderRadius": "5px",
                            "cursor": ("pointer" if n_zero > 0
                                        else "not-allowed"),
                            "fontSize": "12px",
                            "fontWeight": "600"},
                    disabled=(n_zero == 0)),
                html.Button(
                    "Discard scan",
                    id="video-ma-discard-btn", n_clicks=0,
                    style={"background": "transparent",
                            "color": "#cfd0d6",
                            "border":
                                "1px solid rgba(255,255,255,0.15)",
                            "padding": "6px 14px",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                            "fontSize": "12px"}),
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
        Output("video-ma-job-id", "data",
                allow_duplicate=True),
        Output("video-ma-progress", "children",
                allow_duplicate=True),
        Output("video-ma-poll", "disabled",
                allow_duplicate=True),
        Input("video-ma-cancel-btn", "n_clicks"),
        State("video-ma-job-id", "data"),
        prevent_initial_call=True,
    )
    def _on_video_ma_cancel(n_clicks, job_id):
        if not n_clicks or not job_id:
            return no_update, no_update, no_update
        email = (current_user_email() or "").lower()
        if not email:
            return no_update, "Sign in first.", no_update
        _mass_analyze.cancel_job(store, int(job_id))
        return no_update, "Cancel requested.", no_update

    @app.callback(
        Output("video-ma-progress", "children",
                allow_duplicate=True),
        Output("video-ma-summary", "children",
                allow_duplicate=True),
        Output("video-ma-confirm-row", "children",
                allow_duplicate=True),
        Output("video-ma-job-id", "data",
                allow_duplicate=True),
        Output("video-queue-animal", "value",
                allow_duplicate=True),
        Input("video-ma-confirm-btn", "n_clicks"),
        Input("video-ma-discard-btn", "n_clicks"),
        State("video-ma-cutoff-input", "value"),
        State("video-queue-animal", "value"),
        prevent_initial_call=True,
    )
    def _on_video_ma_commit(_confirm_n, _discard_n,
                              cutoff, picker_value):
        trig = callback_context.triggered_id
        if trig == "video-ma-discard-btn":
            return ("", "", [], None, no_update)
        if trig != "video-ma-confirm-btn":
            return (no_update, no_update, no_update,
                     no_update, no_update)
        animal = _ma_animal_from_picker(picker_value)
        email = (current_user_email() or "").lower()
        if not email:
            return ("Sign in first.", "", [], None, no_update)
        if not animal:
            return ("Pick an animal in Step 1 first.",
                     "", [], None, no_update)
        try:
            cutoff = float(cutoff)
        except (TypeError, ValueError):
            return (f"Invalid cutoff: {cutoff!r}",
                     "", [], None, no_update)
        try:
            result = _mass_analyze.commit_threshold(
                store, animal, cutoff, email)
        except Exception as e:
            logger.exception(
                "Mass Analyze commit_threshold failed")
            return (f"Commit failed: {e}", "", [], None,
                     no_update)
        msg = (f"OK  {result['n_cleared']} files moved to "
                "pending_pi_review. The PI will verify them.")
        # Round-trip the queue picker's raw value (NOT the
        # resolved animal id -- the dropdown speaks _pool_/
        # sentinel) to force the queue list to re-render so
        # zero-peak files drop out of the FIFO immediately.
        return (msg, "", [], None, picker_value)

    # Click-to-set on the Hilbert envelope plot. Clientside so
    # the dashed threshold line snaps to the click without a
    # server round-trip. Writes the click's y value (rounded to
    # 4 decimals) into the MA cutoff input -- the existing
    # Input("video-ma-cutoff-input", "value") on _update_analysis
    # then triggers a redraw of the trace with the new threshold.
    app.clientside_callback(
        """
        function(clickData) {
            if (!clickData || !clickData.points
                    || !clickData.points.length) {
                return window.dash_clientside.no_update;
            }
            var y = clickData.points[0].y;
            if (typeof y !== 'number' || !isFinite(y)
                    || y <= 0) {
                return window.dash_clientside.no_update;
            }
            return Math.round(y * 10000) / 10000;
        }
        """,
        Output("video-ma-cutoff-input", "value"),
        Input("video-analysis-trace", "clickData"),
        prevent_initial_call=True,
    )


def _render_history(store: Store, file_id: int):
    try:
        with store.connection() as conn:
            rows = conn.execute(
                """SELECT timestamp, user_email, note
                   FROM annotations
                   WHERE file_id = ? AND category = 'video_review'
                   ORDER BY timestamp DESC LIMIT 50""",
                (file_id,),
            ).fetchall()
    except sqlite3.OperationalError:
        rows = []

    if not rows:
        return html.Span("No video-review notes yet for this file.")
    return html.Ul([
        html.Li([
            html.Span(f"{(r['timestamp'] or '')[:19]} ",
                      style={"color": "#6c6c80"}),
            html.Span(f"{r['user_email'] or 'unknown'}: ",
                      style={"color": "#5e7ce2"}),
            html.Span(r["note"], style={"color": "#a0a0b0"}),
        ], style={"marginBottom": "4px"})
        for r in rows
    ], style={"paddingLeft": "16px"})
