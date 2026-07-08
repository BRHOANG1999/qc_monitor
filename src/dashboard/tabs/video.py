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
from plotly.colors import sample_colorscale
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
from src.utils import stim_blank as _stim_blank
from src.utils.hilbert_envelope import (
    hilbert_envelope_20_200, windowed_auc, band_envelope)
from src.utils.peakseek import peakseek
from src.utils import bhz_csv as _bhz_csv
from src.utils.animal import split_animal_electrode, is_animal_channel
from src.dashboard.tabs import video_events as _events
from src.dashboard.components import (
    button, empty_state, loading_icon, DROPDOWN_STYLE, LABEL_STYLE)
from src.dashboard.icons import icon
from src.dashboard.design import (
    COLOR_ACCENT, COLOR_SUCCESS, COLOR_SURFACE_1, COLOR_DIVIDER,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_TERTIARY,
    RADIUS_MD, SPACE_2, SPACE_3, SPACE_4)
from src.utils import mass_analyze as _mass_analyze
from src.utils.decimate import (
    envelope, window_slice, choose_target_bins, parse_relayout,
)
from src.utils.filters import (
    apply_filter, compute_psd, band_power, SLOW_GAMMA_BAND,
    epoch_band_power, epoch_band_ratio_db, epoch_line_length)
from src.utils import wavelet as _wav

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

# LABEL_STYLE / DROPDOWN_STYLE now come from src.dashboard.components
# (the shared design system) -- imported above. The local copies used to
# duplicate those token values inline and drifted (the old LABEL_STYLE
# failed WCAG AA); consuming the shared ones keeps this tab consistent
# with the rest of the dashboard.

# DOM id assigned to the <video> element so clientside JS can find it.
VIDEO_DOM_ID = "lfp-video"


# The EEG-bar spinner now lives in components.loading_icon (shared across
# tabs). Keep the private alias so existing video.py call sites are untouched.
_loading_icon = loading_icon


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


def _landmark_shapes(events) -> list[dict]:
    """Colored vertical lines for every filled BHZ onset/landmark in
    *events* (EO/LAS/BO/PID/BB), so they can be drawn on the LFP and
    Hilbert. Bounded per NASA Rule 3. Shared by the build-time injection
    (always-on, survives figure rebuilds) and the events-store painters."""
    shapes: list[dict] = []
    for i, e in enumerate(events or []):
        if i >= 16:  # bound events per file
            break
        for field, color in _events.LANDMARK_COLORS.items():
            t_sec = (e or {}).get(f"{field}_sec")
            if t_sec is None:
                continue
            try:
                t = float(t_sec)
            except (TypeError, ValueError):
                continue
            shapes.append({
                "type": "line", "xref": "x", "yref": "paper",
                "x0": t, "x1": t, "y0": 0, "y1": 1,
                "line": {"color": color, "width": 2}, "opacity": 0.85,
            })
    return shapes


def _inject_landmarks(fig, events):
    """Append landmark verticals to a freshly-built figure so flagged
    onsets are visible the moment the LFP/Hilbert renders -- not only
    after the events-store changes. No-op when there are no onsets."""
    lm = _landmark_shapes(events)
    if not lm or fig is None:
        return fig
    try:
        existing = list(fig.layout.shapes or ())
        fig.update_layout(shapes=existing + lm)
    except Exception:  # noqa: BLE001 -- never block the plot on shape merge
        pass
    return fig


# Stim-blanking lives in src/utils/stim_blank.py so the Mass Analyze
# scan computes the envelope on the exact same blanked signal. These
# thin wrappers keep the existing call sites.
def _stim_times_for_file(store: Store, file_id: int) -> np.ndarray:
    return _stim_blank.stim_times_for_file(store, file_id)


def _stim_copy_channels(store: Store, session_dir: str | None) -> set[int]:
    return _stim_blank.stim_copy_channels(store, session_dir)


def _animal_for_session(store: Store, session_dir: str,
                          prefer: str | None = None) -> str:
    """Animal id to label a session by. Sessions can hold several
    animals (e.g. BCH062SR + BCH061SLM in one cage); when *prefer* (the
    animal the reviewer selected) is one of them, use it, so the banner
    matches who they're reviewing. Otherwise the first animal, or '?'."""
    if not session_dir:
        return "?"
    names = store._channel_names_for_session(session_dir)
    animals: list[str] = []
    for n in names:
        if isinstance(n, str) and is_animal_channel(n):
            a, _ = split_animal_electrode(n)
            if a and a not in animals:
                animals.append(a)
    if prefer and prefer in animals:
        return prefer
    return animals[0] if animals else "?"


def _now_viewing_info(store: Store, file_id: int,
                        prefer_animal: str | None = None) -> dict | None:
    """Orientation for the loaded recording: animal, date/time, its
    position among the session's hour-chunks, and the neighbouring
    file ids for prev/next-hour stepping. None if not resolvable.
    *prefer_animal* (the reviewer's selected animal) wins the label in
    multi-animal sessions."""
    session_dir = _session_dir_for_file(store, file_id)
    if not session_dir:
        return None
    files = _files_with_video(store, session_dir)  # chunk_datetime asc
    ids = [int(f["id"]) for f in files]
    try:
        i = ids.index(int(file_id))
    except ValueError:
        return None
    cdt = files[i].get("chunk_datetime") or ""
    try:
        ts = datetime.strptime(cdt, "%Y_%m_%d__%H_%M_%S")
        date_label = ts.strftime("%Y-%m-%d")
        time_label = ts.strftime("%H:%M")
    except ValueError:
        date_label, time_label = cdt, ""
    return {
        "animal": _animal_for_session(store, session_dir, prefer_animal),
        "date": date_label,
        "time": time_label,
        "index": i + 1,
        "total": len(ids),
        "prev_id": ids[i - 1] if i > 0 else None,
        "next_id": ids[i + 1] if i < len(ids) - 1 else None,
    }


def _session_dir_for_file(store: Store, file_id: int) -> str | None:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT session_dir FROM processed_files WHERE id = ?", (file_id,),
        ).fetchone()
        return row["session_dir"] if row else None


def _next_in_pool(store: Store, view, cursor,
                    current_file_id: int):
    """If the reviewer is browsing a pool AND the loaded file is the
    pool's current entry, return the next step ``(session_dir, file_id,
    new_cursor)``; else None. Lets Save / Quick-flag advance WITHIN the
    pool (the primary source once pool navigation is in use) instead of
    falling back to the FIFO queue."""
    if not view or not cursor:
        return None
    active = cursor.get("active")
    if active not in ("1", "2", "3"):
        return None
    key = {"1": "pool1", "2": "pool2", "3": "pool3"}[active]
    files = view.get(key) or []
    idx = int(cursor.get("idx") or 0)

    def _fid(e):
        return int(e["file_id"]) if isinstance(e, dict) else int(e)

    # The loaded file must BE the pool's current entry -- otherwise the
    # reviewer got here some other way and the pool isn't primary.
    if idx >= len(files) or _fid(files[idx]) != int(current_file_id):
        return None
    nxt = idx + 1
    if nxt >= len(files):
        return None  # end of the pool
    fid = _fid(files[nxt])
    return _session_dir_for_file(store, fid), fid, \
        {"active": active, "idx": nxt}


# review_state statuses for the pool-progress readout: "done" = the file
# was submitted and left the pool; "needs_scoring" = flagged for later;
# anything else (claimed / abandoned / none) = not started.
_POOL_DONE_STATUSES = {"pending_pi_review", "pi_approved", "pi_flagged",
                        "no_events", "has_events"}


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


# The three carousel navigation modes: normal unreviewed queue, the auto-filter
# "has events" flag pool (still-reviewable), and the quick-flag needs-scoring
# pool. Labels drive the RadioItems + the mode-aware card title/empty text.
_QUEUE_MODES = [
    {"value": "queue", "label": "Queue",
     "title": "Queue (FIFO · {n} unreviewed for {a})",
     "empty_ok": "🎉  All caught up — no unreviewed recordings for {a}.",
     "empty": "No unreviewed recordings for {a}."},
    {"value": "flagged", "label": "🚩 Flag (has events)",
     "title": "🚩 Flag (FIFO · {n} auto-flagged has-events for {a})",
     "empty_ok": "No auto-flagged (has-events) recordings for {a}.",
     "empty": "No auto-flagged (has-events) recordings for {a}."},
    {"value": "needs_scoring", "label": "⚠️ Needs scoring",
     "title": "⚠️ Needs scoring ({n} awaiting scoring for {a})",
     "empty_ok": "No recordings awaiting scoring for {a}.",
     "empty": "No recordings awaiting scoring for {a}."},
]
_QUEUE_MODE_BY_VALUE = {m["value"]: m for m in _QUEUE_MODES}


def _fetch_queue_by_mode(store, mode, animal_ids, email, floor, limit):
    """Return the carousel rows for the chosen navigation *mode*, normalized so
    every dict carries ``id`` / ``session_dir`` / ``chunk_datetime`` /
    ``duration_sec`` (the fields the carousel reads). Modes:

    * ``queue`` — normal unreviewed FIFO queue.
    * ``flagged`` — queue ∩ auto-filter has-events flag (still reviewable).
    * ``needs_scoring`` — the reviewer quick-flag pool (single animal).
    """
    if not animal_ids:
        return []
    if mode == "needs_scoring":
        rows = store.files_needing_scoring_for_animal(animal_ids[0])
        # Normalize file_id -> id so the carousel's row["id"] works uniformly.
        for r in rows:
            r.setdefault("id", r.get("file_id"))
        return rows
    return store.get_review_queue(
        animal_ids, email, limit=limit, since_iso=floor,
        flagged_only=(mode == "flagged"))


def _ma_animal_from_picker(animal_value: str | None
                             ) -> str | None:
    """Single plain animal id from the Step-1 picker value, or
    None. Mass Analyze operates on one animal, so it takes the
    first id ``_animal_ids_from_picker`` resolves (strips the
    ``_pool_`` prefix / drops the ``__UNASSIGNED__`` sentinel)."""
    ids = _animal_ids_from_picker(animal_value)
    return ids[0] if ids else None


def _animal_for_channel(store, file_id, channel) -> str | None:
    """Animal id for the selected LFP channel's electrode (e.g.
    'BCH062SLM' -> 'BCH062'), or None when the channel isn't an animal
    channel. This is the PHYSICAL ground truth for which animal an onset
    belongs to -- the onset is dropped on this channel's trace -- so review
    writes + the review panel scope by this rather than the queue picker.
    Delegates to the shared store helper (single source of truth)."""
    return store.animal_for_file_channel(file_id, channel)


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


def _as_float(v, default):
    """Parse *v* to float, returning *default* on blank/invalid/non-positive
    sentinels handled by the caller."""
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _build_csv_file_meta(store, file_id: int, channel: int,
                         cutoff: float | None = None,
                         auc_threshold: float | None = None,
                         auc_window: float | None = None,
                         ) -> tuple[dict, float, str, "date"]:
    """Assemble the file-level meta dict + (fs, animal_id,
    chunk_date) for a BHZ CSV row.

    *cutoff* is the raw-envelope threshold used; *auc_threshold* /
    *auc_window* the AUC-screen settings -- all recorded per row so the
    export documents how each event was detected.

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
        fs=fs, cutoff=(cutoff if (cutoff and cutoff > 0) else _BHZ_CUTOFF),
        channel=int(channel),
        # Peak_Index/Peak_Stamp are the automated detector's envelope-peak
        # output; manual review has no detected peak, so they stay None (the
        # onset lives in EventEO / EventEO_WallClock / EO_HourOfDay instead).
        peak_index=None, peak_stamp=None, peak_dt=chunk_dt,
        auc_threshold=auc_threshold, auc_window=auc_window,
    )
    chunk_date = (chunk_dt.date() if chunk_dt
                   else date.today())
    return meta, fs, animal, chunk_date


def _export_partial_csv(store, config: dict, file_id: int,
                          channel, events: list,
                          cutoff=None, auc_threshold=None,
                          auc_window=None) -> tuple[int, str]:
    """Append the dropped onsets to the official BHZ day CSV right now.

    ``bhz_csv.write_event_rows`` tolerates partial events (unset landmarks
    render as ``NaN``), so an EEG-onset-only event writes its ``EventEO``
    and leaves the rest blank -- usable immediately. *cutoff* (raw envelope)
    and *auc_threshold* / *auc_window* (AUC screen) are the live detector
    settings, recorded on every row. Returns ``(n_rows, csv_name)``; raises
    ``ValueError`` on misconfig / missing data so the caller can surface a
    clean message. A later full score is replaced when the PI finalizes the
    day CSV in overwrite mode.
    """
    assert events, "events required for partial export"
    bhz_cfg = (config or {}).get("bhz_csv", {}) or {}
    if not bhz_cfg.get("enabled"):
        raise ValueError("BHZ CSV export is disabled in config.")
    if channel is None or channel == "":
        raise ValueError("Pick a channel before exporting.")
    cut = _as_float(cutoff, _BHZ_CUTOFF)
    meta, fs, animal, chunk_date = _build_csv_file_meta(
        store, int(file_id), int(channel),
        cutoff=cut if cut and cut > 0 else _BHZ_CUTOFF,
        auc_threshold=_as_float(auc_threshold, None),
        auc_window=_as_float(auc_window, None))
    csv_path = _bhz_csv.resolve_csv_path(
        bhz_cfg.get("base_dir", ""),
        bhz_cfg.get("filename_template", "{date}_{animal}.csv"),
        chunk_date, animal)
    n = _bhz_csv.write_event_rows(csv_path, meta, events, fs)
    return n, csv_path.name


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
                    "label": f"Ch{i} · {raw_names[i]}", "value": i,
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
        # Animal-electrode first ("Ch2 · BCH062SR") so the reviewer sees WHICH
        # animal each channel belongs to -- onsets file under the scored
        # channel's animal.
        options.append({"label": f"Ch{i} · {name}", "value": i})
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

    n_blanked = (0 if stim_times is None else int(len(stim_times)))
    series = _stim_blank.blank_series_with_stim_times(
        series, fs, stim_times, blank_pre_ms, blank_post_ms)

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


def _mmss(sec) -> str:
    """Seconds -> ``mm:ss`` (or ``h:mm:ss`` past an hour). Guards
    None/negative/non-finite to ``00:00`` so callers never crash."""
    try:
        s = float(sec)
    except (TypeError, ValueError):
        return "00:00"
    if not np.isfinite(s) or s < 0:
        return "00:00"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, ss = divmod(rem, 60)
    return f"{h}:{m:02d}:{ss:02d}" if h else f"{m:02d}:{ss:02d}"


def _fmt_time(sec) -> str:
    """Readable time: raw seconds AND mm:ss side by side, e.g.
    ``"123.4 s · 02:03"`` -- so long recordings stay interpretable."""
    try:
        s = float(sec)
    except (TypeError, ValueError):
        s = 0.0
    if not np.isfinite(s):
        s = 0.0
    return f"{s:.1f} s · {_mmss(s)}"


def _apply_mmss_xaxis(fig: go.Figure, t_max: float) -> None:
    """Relabel a seconds x-axis with ~8 nice mm:ss ticks across
    ``[0, t_max]``. Data stays in seconds (cursor/seek math untouched);
    only the tick *labels* change. No-op for tiny/invalid spans."""
    assert fig is not None, "fig required"
    try:
        span = float(t_max)
    except (TypeError, ValueError):
        return
    if not np.isfinite(span) or span <= 0:
        return
    # "Nice" step: 1/2/5 x 10^k closest to span/8, min 1 s.
    raw = span / 8.0
    mag = 10.0 ** np.floor(np.log10(max(raw, 1e-9)))
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = mult * mag
        if step >= raw:
            break
    step = max(1.0, step)
    n_ticks = int(span // step) + 1
    vals = [i * step for i in range(n_ticks + 1) if i * step <= span + step]
    fig.update_xaxes(tickmode="array", tickvals=vals,
                      ticktext=[_mmss(v) for v in vals])


def _build_lfp_figure(t: np.ndarray, signal: np.ndarray, label: str,
                       uirevision: str | None = None,
                       title: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t, y=signal,
        mode="lines",
        line=dict(color="#5e7ce2", width=0.8),
        hovertemplate="t=%{x:.2f}s<br>%{y:.1f} μV<extra></extra>",
        name=label,
    ))
    # Cursor placeholder — clientside callback updates the x position.
    # dragmode='pan' (Seek mode default): a brief click still registers as
    # a click (clickData) and seeks the video, while a slightly-longer
    # press-drag PANS a hair instead of drawing a zoom box. We deliberately
    # avoid dragmode=False here: Plotly.js does not reliably honor it (it
    # silently falls back to 'zoom', so a tiny drag would zoom -- the exact
    # bug reviewers hit). 'pan' is a positively-supported value, so the
    # gesture can never zoom by accident. The "Mode" toggle flips this to
    # 'zoom' (drag draws a zoom box) via a clientside enforcer that
    # re-applies it after every rebuild. Scroll-zoom + the modebar zoom/pan
    # buttons stay available in both modes; double-click resets.
    fig.update_layout(
        plot_bgcolor="#13131f", paper_bgcolor="#13131f",
        height=220,
        margin=dict(l=60, r=20, t=10, b=40),
        dragmode="pan",
        # uirevision (stable per recording) tells Plotly to preserve
        # the user's zoom/pan across figure updates -- the cursor
        # mirror + re-decimate redraws no longer snap the view back.
        # It changes when the file/channel changes, so a new recording
        # resets to full view.
        uirevision=uirevision,
        xaxis=dict(title="Time (mm:ss)", showgrid=True,
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
    # Optional title (e.g. the animal we're looking at) -- needs a little
    # top margin so it doesn't collide with the trace.
    if title:
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", y=0.98,
                       yanchor="top", font=dict(size=12, color="#cfd0d6")),
            margin=dict(l=60, r=20, t=26, b=40))
    # mm:ss tick labels so long recordings stay readable; hover still
    # shows exact seconds via the per-trace hovertemplate.
    if t is not None and len(t):
        _apply_mmss_xaxis(fig, float(t[-1]))
    return fig


def add_click_catcher(fig) -> None:
    """Make the signal a FAT click target so "Set on plot" reliably registers.

    Plotly emits clickData only when the pointer lands on the ~1px WebGL line,
    so clicks kept missing it. Overlay a transparent, WIDE (~16px) Scattergl
    line on the SAME points as the signal: a click anywhere along the trace path
    now returns that exact sample time -- and the red onset line lands where you
    clicked. Critically this is ONE lightweight WebGL trace, not a bar per sample
    (thousands of SVG rects, which froze the tab under the 10 Hz cursor redraw).
    hoverinfo='skip' keeps it out of the unified hover. Mutates *fig* in place;
    a no-op if the primary trace has no usable data."""
    try:
        base = fig.data[0]
        xs, ys = base.x, base.y
    except (IndexError, AttributeError):
        return
    if xs is None or ys is None or len(xs) < 2:
        return
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=16),
        hoverinfo="skip", showlegend=False, name="_click_catcher"))


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
                            show_raw: bool = False,
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
    # Blank by default (match _update_lfp) so the envelope reflects the
    # EEG, not the stim artifact; never blank a stim_copy channel.
    do_blank = (int(channel) not in stim_copy) and (not show_raw)
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
    # Drop candidates the reviewer hand-rejected as false positives
    # (within 0.5 s of a saved rejection time). Curated set persists.
    try:
        rejected = store.get_rejected_peaks(int(file_id), int(channel))
    except Exception:
        rejected = []
    if rejected and len(peak_times):
        rej = np.asarray(rejected, dtype=np.float64)
        keep = np.array(
            [np.all(np.abs(rej - t) > 0.5) for t in peak_times],
            dtype=bool)
        peak_times = peak_times[keep]

    target_bins = choose_target_bins(len(env)) or 4000
    t, display, _decim = envelope(env, fs, target_bins, t_start=0.0)
    fig = _build_lfp_figure(
        t, display,
        "Hilbert envelope (20-200 Hz)",
        uirevision=f"{file_id}:{channel}",
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
            hovertemplate=(
                "kept peak @ t=%{x:.2f}s<br>"
                f"local max ≥ {cutoff:.3g}, ≥ "
                f"{int(min_peak_dist_sec)} s from any larger peak"
                "<extra></extra>"),
            name="BHZ candidate",
            showlegend=False,
        ))
    # How many times the envelope rose above the threshold at all --
    # usually far more than the kept peaks, because peakseek keeps only
    # one local maximum per min-distance window. Spelling both out
    # makes the "many crossings, few triangles" behaviour legible.
    crossings = (int(np.sum((env[:-1] <= cutoff) & (env[1:] > cutoff)))
                  if len(env) > 1 else 0)
    n_kept = len(peak_times)
    status = (
        f"BHZ detection · cutoff={cutoff:.3g} · "
        f"{n_kept} peak{'' if n_kept == 1 else 's'} kept of "
        f"{crossings} threshold crossing"
        f"{'' if crossings == 1 else 's'} "
        f"(kept = a local maximum ≥ cutoff AND ≥ "
        f"{int(min_peak_dist_sec)} s from any larger peak) · "
        f"{len(env)/fs:.1f} s @ {int(fs)} Hz"
    )
    return (fig, status)


# Evoked per-stim window for the slow-gamma band-power feature. 200 ms
# post-stim resolves the 30-50 Hz band (>= ~6 cycles of 30 Hz).
_EVOKED_GAMMA_WIN_SEC: float = 0.200


def _render_band_power_trace(store, file_id: int, channel: int | None,
                              lo: float, hi: float,
                              blank_pre_ms: float = -5.0,
                              blank_post_ms: float = 15.0,
                              show_raw: bool = False) -> tuple:
    """(figure, status) for continuous band power vs time.

    The lab's band-limited analytic envelope (``band_envelope``, the same
    FFT-zero-mask as the 20-200 Hz default) over ``[lo, hi]`` Hz, squared to
    instantaneous power, then decimated for display. Stim-blanked like the
    LFP by default. Used for the slow-gamma (30-50 Hz) "LFP analysis" trace.
    """
    assert isinstance(file_id, int), "file_id must be int"
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        return (_empty_lfp_fig("File not found in DB."), "")
    if channel is None:
        return (_empty_lfp_fig(
            "Pick a brain channel (Step 1) to see band power."), "")
    session_dir = _session_dir_for_file(store, file_id)
    stim_copy = _stim_copy_channels(store, session_dir)
    do_blank = (int(channel) not in stim_copy) and (not show_raw)
    stim_times = (_stim_times_for_file(store, file_id)
                   if do_blank else np.asarray([], dtype=np.float64))
    try:
        series, fs, _ = _get_blanked_series(
            file_path, int(channel), stim_times,
            blank_pre_ms, blank_post_ms)
    except Exception as e:
        logger.warning("Band-power load failed file=%s ch=%s: %s",
                        file_id, channel, e)
        return (_empty_lfp_fig(f"Couldn't load LFP: {e}"), "")
    env = band_envelope(np.asarray(series, dtype=np.float64), fs, lo, hi)
    power = env * env  # instantaneous band power
    target_bins = choose_target_bins(len(power)) or 4000
    t, display, _decim = envelope(power, fs, target_bins, t_start=0.0)
    fig = _build_lfp_figure(
        t, display, f"Slow gamma power ({lo:g}-{hi:g} Hz)",
        uirevision=f"{file_id}:{channel}")
    fig.data[0].line.color = "#bf5af2"
    fig.data[0].hovertemplate = (
        "t=%{x:.2f}s<br>power=%{y:.3g}<extra></extra>")
    fig.layout.yaxis.title = "band power (μV²)"
    status = (f"Slow gamma {lo:g}-{hi:g} Hz · continuous band power · "
              f"{len(power) / fs:.1f} s @ {int(fs)} Hz")
    return (fig, status)


def _render_evoked_band_power(store, file_id: int, channel: int | None,
                               lo: float, hi: float,
                               win_sec: float = _EVOKED_GAMMA_WIN_SEC
                               ) -> tuple:
    """(figure, status) for per-stim-epoch slow-gamma power.

    For each stim onset, integrate the PSD over ``[lo, hi]`` Hz across a
    short post-stim window -- the stim-locked ("evoked") band power. One
    point per epoch, plotted vs stim time (same shape as the DB-backed
    per-epoch features). Computed live, so no pipeline reprocess is needed.
    """
    assert isinstance(file_id, int), "file_id must be int"
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        return (_empty_lfp_fig("File not found in DB."), "")
    if channel is None:
        return (_empty_lfp_fig(
            "Pick a brain channel (Step 1) first."), "")
    stim_times = _stim_times_for_file(store, file_id)
    if stim_times is None or len(stim_times) == 0:
        return (_empty_lfp_fig(
            "No stim epochs on this file — evoked band power needs "
            "stimulation onsets."), "")
    try:
        chunk = get_chunk(file_path)
        sig = np.asarray(chunk.signal[:, int(channel)], dtype=np.float64)
        fs = float(chunk.fs)
    except Exception as e:
        logger.warning("Evoked band-power load failed file=%s ch=%s: %s",
                        file_id, channel, e)
        return (_empty_lfp_fig(f"Couldn't load LFP: {e}"), "")
    times, vals = epoch_band_power(
        sig, fs, stim_times, lo, hi, (0.0, float(win_sec)))
    if len(times) == 0:
        return (_empty_lfp_fig(
            "No usable stim epochs (windows fell off the recording)."),
                "")
    fig = _build_lfp_figure(
        np.asarray(times), np.asarray(vals),
        f"Slow gamma power ({lo:g}-{hi:g} Hz, per stim)",
        uirevision=f"{file_id}:{channel}")
    fig.data[0].mode = "markers+lines"
    fig.data[0].line.color = "#bf5af2"
    fig.data[0].marker = dict(size=5, color="#bf5af2")
    fig.data[0].hovertemplate = (
        "stim @ t=%{x:.2f}s<br>power=%{y:.3g}<extra></extra>")
    fig.layout.yaxis.title = "evoked band power (μV²)"
    status = (f"Slow gamma {lo:g}-{hi:g} Hz · per stim epoch "
              f"({win_sec * 1000:.0f} ms window) · {len(times)} epochs")
    return (fig, status)


def _render_wavelet_band_power(store, file_id: int, channel: int | None,
                                lo: float, hi: float,
                                blank_pre_ms: float = -5.0,
                                blank_post_ms: float = 15.0,
                                show_raw: bool = False) -> tuple:
    """(figure, status): continuous Morlet-WAVELET band power vs time --
    the transient-aware sibling of ``_render_band_power_trace`` (which uses
    the analytic envelope). Same stim-blanking + display decimation."""
    assert isinstance(file_id, int), "file_id must be int"
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        return (_empty_lfp_fig("File not found in DB."), "")
    if channel is None:
        return (_empty_lfp_fig(
            "Pick a brain channel (Step 1) to see wavelet power."), "")
    session_dir = _session_dir_for_file(store, file_id)
    stim_copy = _stim_copy_channels(store, session_dir)
    do_blank = (int(channel) not in stim_copy) and (not show_raw)
    stim_times = (_stim_times_for_file(store, file_id)
                   if do_blank else np.asarray([], dtype=np.float64))
    try:
        series, fs, _ = _get_blanked_series(
            file_path, int(channel), stim_times, blank_pre_ms, blank_post_ms)
    except Exception as e:
        logger.warning("Wavelet power load failed file=%s ch=%s: %s",
                        file_id, channel, e)
        return (_empty_lfp_fig(f"Couldn't load LFP: {e}"), "")
    power, work_fs = _wav.wavelet_band_power(
        np.asarray(series, dtype=np.float64), fs, lo, hi)
    if power.size == 0:
        return (_empty_lfp_fig("Window too short for a wavelet transform."),
                "")
    target_bins = choose_target_bins(len(power)) or 4000
    t, display, _decim = envelope(power, work_fs, target_bins, t_start=0.0)
    fig = _build_lfp_figure(
        t, display, f"Wavelet power ({lo:g}-{hi:g} Hz)",
        uirevision=f"{file_id}:{channel}")
    fig.data[0].line.color = "#ffd60a"
    fig.data[0].hovertemplate = (
        "t=%{x:.2f}s<br>power=%{y:.3g}<extra></extra>")
    fig.layout.yaxis.title = "wavelet power (μV²)"
    status = (f"Wavelet {lo:g}-{hi:g} Hz · continuous · "
              f"{len(power) / work_fs:.1f} s @ {int(work_fs)} Hz (decimated)")
    return (fig, status)


def _render_wavelet_evoked_band_power(store, file_id: int,
                                       channel: int | None,
                                       lo: float, hi: float,
                                       win_sec: float = _EVOKED_GAMMA_WIN_SEC
                                       ) -> tuple:
    """(figure, status): per-stim wavelet band power -- the wavelet sibling
    of ``_render_evoked_band_power`` (PSD-based). One point per epoch."""
    assert isinstance(file_id, int), "file_id must be int"
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        return (_empty_lfp_fig("File not found in DB."), "")
    if channel is None:
        return (_empty_lfp_fig("Pick a brain channel (Step 1) first."), "")
    stim_times = _stim_times_for_file(store, file_id)
    if stim_times is None or len(stim_times) == 0:
        return (_empty_lfp_fig(
            "No stim epochs on this file — wavelet evoked power needs "
            "stimulation onsets."), "")
    try:
        chunk = get_chunk(file_path)
        sig = np.asarray(chunk.signal[:, int(channel)], dtype=np.float64)
        fs = float(chunk.fs)
    except Exception as e:
        logger.warning("Wavelet evoked load failed file=%s ch=%s: %s",
                        file_id, channel, e)
        return (_empty_lfp_fig(f"Couldn't load LFP: {e}"), "")
    times, vals = _wav.epoch_wavelet_band_power(
        sig, fs, stim_times, lo, hi, (0.0, float(win_sec)))
    if len(times) == 0:
        return (_empty_lfp_fig(
            "No usable stim epochs (windows fell off the recording)."), "")
    fig = _build_lfp_figure(
        np.asarray(times), np.asarray(vals),
        f"Wavelet power ({lo:g}-{hi:g} Hz, per stim)",
        uirevision=f"{file_id}:{channel}")
    fig.data[0].mode = "markers+lines"
    fig.data[0].line.color = "#ffd60a"
    fig.data[0].marker = dict(size=5, color="#ffd60a")
    fig.data[0].hovertemplate = (
        "stim @ t=%{x:.2f}s<br>power=%{y:.3g}<extra></extra>")
    fig.layout.yaxis.title = "wavelet evoked power (μV²)"
    status = (f"Wavelet {lo:g}-{hi:g} Hz · per stim epoch "
              f"({win_sec * 1000:.0f} ms window) · {len(times)} epochs")
    return (fig, status)


def _render_wavelet_scalogram(store, file_id: int, channel: int | None,
                               win: tuple[float, float] = (-0.025, 0.200)
                               ) -> tuple:
    """(figure, status): mean stim-locked scalogram -- the average Morlet
    time-frequency response across stim epochs (an ERSP-style heatmap,
    freq x ms). Shows what frequencies the stim evokes and when."""
    assert isinstance(file_id, int), "file_id must be int"
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        return (_empty_lfp_fig("File not found in DB."), "")
    if channel is None:
        return (_empty_lfp_fig("Pick a brain channel (Step 1) first."), "")
    stim_times = _stim_times_for_file(store, file_id)
    if stim_times is None or len(stim_times) == 0:
        return (_empty_lfp_fig(
            "No stim epochs — the scalogram is stim-locked."), "")
    try:
        chunk = get_chunk(file_path)
        sig = np.asarray(chunk.signal[:, int(channel)], dtype=np.float64)
        fs = float(chunk.fs)
    except Exception as e:
        logger.warning("Wavelet scalogram load failed file=%s ch=%s: %s",
                        file_id, channel, e)
        return (_empty_lfp_fig(f"Couldn't load LFP: {e}"), "")
    n = sig.shape[0]
    length = int(round((win[1] - win[0]) * fs))
    fmax = min(300.0, fs / 2.0)
    acc = None
    freqs = times = None
    n_used = 0
    for s in np.asarray(stim_times, dtype=np.float64)[:300]:
        i0 = int(round((float(s) + win[0]) * fs))
        i1 = i0 + length
        if i0 < 0 or i1 > n:
            continue
        freqs, times, power = _wav.scalogram(
            sig[i0:i1], fs, fmin=10.0, fmax=fmax, n_freqs=48)
        if freqs.size == 0:
            continue
        if acc is None or acc.shape == power.shape:
            acc = power if acc is None else acc + power
            n_used += 1
    if acc is None or n_used == 0:
        return (_empty_lfp_fig(
            "No usable stim epochs for the scalogram."), "")
    mean_power = acc / n_used
    z = 10.0 * np.log10(mean_power + 1e-12)
    t_ms = (times - times[0]) * 1000.0 + win[0] * 1000.0
    fig = go.Figure(go.Heatmap(
        x=t_ms, y=freqs, z=z, colorscale="Turbo",
        colorbar=dict(title="dB", thickness=10),
        hovertemplate="%{x:.1f}ms · %{y:.0f}Hz · %{z:.1f}dB<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#ffffff", width=1.5, dash="dot"))
    fig.update_layout(
        plot_bgcolor="#13131f", paper_bgcolor="#13131f", height=300,
        margin=dict(l=60, r=20, t=24, b=40),
        title=dict(text=f"Mean stim-locked scalogram · {n_used} epochs",
                    font=dict(size=11, color="#ffffff"), x=0.01, y=0.97),
        xaxis=dict(title="ms from stim", color="#ffffff"),
        yaxis=dict(title="Hz", type="log", color="#ffffff"),
        font=dict(color="#ffffff"))
    status = (f"Mean scalogram · {n_used} epochs · "
              f"{win[0] * 1000:.0f}..{win[1] * 1000:.0f} ms")
    return (fig, status)


def _ll_window_ms(start_ms, end_ms) -> tuple[float, float]:
    """Coerce the two line-length-window inputs to ``(start, end)`` ms,
    defaulting to the -1..1 ms stim-onset window when blank/invalid
    (``dcc.Input`` hands us ``None`` while the user is mid-type)."""
    try:
        s = float(start_ms) if start_ms is not None else -1.0
    except (TypeError, ValueError):
        s = -1.0
    try:
        e = float(end_ms) if end_ms is not None else 1.0
    except (TypeError, ValueError):
        e = 1.0
    return s, e


def _render_evoked_line_length(store, file_id: int, channel: int | None,
                                win_ms: tuple[float, float]) -> tuple:
    """(figure, status) for per-stim line length over a user window (ms).

    Line length = sum of |sample-to-sample diff| inside ``[onset+start,
    onset+end]`` ms. Computed live from the raw epoch, so the window is
    fully adjustable -- a tight stim-locked window (e.g. -1..1 ms) tracks
    stim-onset instability. Replaces the old full-epoch DB ``line_length``.
    """
    assert isinstance(file_id, int), "file_id must be int"
    assert len(win_ms) == 2, "win_ms must be (start, end)"
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        return (_empty_lfp_fig("File not found in DB."), "")
    if channel is None:
        return (_empty_lfp_fig(
            "Pick a brain channel (Step 1) first."), "")
    start_ms, end_ms = float(win_ms[0]), float(win_ms[1])
    if end_ms <= start_ms:
        return (_empty_lfp_fig(
            "Line-length window end must be greater than start."), "")
    stim_times = _stim_times_for_file(store, file_id)
    if stim_times is None or len(stim_times) == 0:
        return (_empty_lfp_fig(
            "No stim epochs on this file — line length needs "
            "stimulation onsets."), "")
    try:
        chunk = get_chunk(file_path)
        sig = np.asarray(chunk.signal[:, int(channel)], dtype=np.float64)
        fs = float(chunk.fs)
    except Exception as e:
        logger.warning("Line-length load failed file=%s ch=%s: %s",
                        file_id, channel, e)
        return (_empty_lfp_fig(f"Couldn't load LFP: {e}"), "")
    win = (start_ms / 1000.0, end_ms / 1000.0)
    times, vals = epoch_line_length(sig, fs, stim_times, win)
    if len(times) == 0:
        return (_empty_lfp_fig(
            "No usable stim epochs — the window is too short for this "
            "sample rate, or fell off the recording. Try a wider window."),
                "")
    fig = _build_lfp_figure(
        np.asarray(times), np.asarray(vals),
        f"Line length ({start_ms:g}..{end_ms:g} ms, per stim)",
        uirevision=f"{file_id}:{channel}")
    fig.data[0].mode = "markers+lines"
    fig.data[0].line.color = "#ffd60a"
    fig.data[0].marker = dict(size=5, color="#ffd60a")
    fig.data[0].hovertemplate = (
        "stim @ t=%{x:.2f}s<br>line length=%{y:.3g}<extra></extra>")
    fig.layout.yaxis.title = "line length (μV)"
    status = (f"Line length · {start_ms:g}..{end_ms:g} ms window · "
              f"{len(times)} epochs @ {int(fs)} Hz")
    return (fig, status)


def _overlay_segments(sig, fs, stim_times, idx, win, n_prev):
    """``(kept_idx, t_ms, segs)``: the current stim + up to ``n_prev`` prior
    windowed segments, all the same fixed length (so they stack into a
    heatmap). Out-of-bounds windows are skipped. Oldest -> newest order."""
    members = list(range(max(0, idx - n_prev), idx + 1))
    n = sig.shape[0]
    length = int(round((win[1] - win[0]) * fs))
    if length < 2:
        return [], None, []
    t_ms = (np.arange(length) / fs * 1000.0) + win[0] * 1000.0
    kept, segs = [], []
    for i in members:
        i0 = int(round((float(stim_times[i]) + win[0]) * fs))
        i1 = i0 + length
        if i0 < 0 or i1 > n:
            continue
        kept.append(i)
        segs.append(sig[i0:i1])
    return kept, t_ms, segs


def _add_overlay_lines(fig, kept, t_ms, segs, idx) -> None:
    """Overlay each segment as a line, oldest->newest coloured by a Turbo
    hue ramp (newest brightest + thicker) -- distinguishable where the old
    alpha ramp wasn't."""
    assert len(kept) == len(segs), "kept/segs length mismatch"
    k = len(kept)
    colors = (sample_colorscale("Turbo", [j / (k - 1) for j in range(k)])
              if k > 1 else ["#ffd60a"])
    for j, i in enumerate(kept):
        newest = (i == idx)
        fig.add_trace(go.Scatter(
            x=t_ms, y=segs[j], mode="lines",
            line=dict(color=colors[j], width=2.6 if newest else 1.1),
            name=(f"current (stim #{i + 1})" if newest
                  else f"stim #{i + 1}"),
            showlegend=False,
            hoverinfo="x+y" if newest else "skip",
        ))


def _add_overlay_heatmap(fig, kept, t_ms, segs) -> None:
    """ERP-image: one row per stim (newest on top), x = ms, colour = µV.
    Scales to many epochs where overlaid lines would clutter."""
    assert len(kept) == len(segs), "kept/segs length mismatch"
    z = np.asarray(segs, dtype=np.float64)  # rows oldest..newest (bottom..top)
    fig.add_trace(go.Heatmap(
        x=t_ms, y=[f"#{i + 1}" for i in kept], z=z,
        colorscale="RdBu_r", zmid=0.0,
        colorbar=dict(title="µV", thickness=10),
        hovertemplate=("%{x:.2f} ms · stim %{y}<br>"
                        "%{z:.1f} µV<extra></extra>"),
    ))


def _style_overlay(fig, start_ms, end_ms, n_added, mode="lines") -> None:
    """Dark-theme the stim-overlay figure and mark the stim onset (0 ms)."""
    assert n_added >= 1, "need at least one overlay trace"
    assert mode in ("lines", "heatmap"), "bad overlay mode"
    label = "heatmap" if mode == "heatmap" else "overlay"
    y_title = "stim #" if mode == "heatmap" else "μV"
    fig.update_layout(
        plot_bgcolor="#13131f", paper_bgcolor="#13131f",
        # autosize (no fixed height): the dcc.Graph is responsive and
        # fills its container, so the pop-out can be resized freely.
        autosize=True, margin=dict(l=60, r=20, t=24, b=40),
        title=dict(
            text=(f"Stim {label} · last {n_added} epochs · "
                  f"{start_ms:g}..{end_ms:g} ms"),
            font=dict(size=11, color="#ffffff"), x=0.01, y=0.97),
        font=dict(color="#ffffff"),
        xaxis=dict(title="ms from stim", showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                    zeroline=False, color="#ffffff"),
        yaxis=dict(title=y_title, showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                    zeroline=False, color="#ffffff"),
        showlegend=False,
    )
    if start_ms <= 0 <= end_ms:
        fig.add_vline(x=0, line=dict(color="#ff9f0a", width=1.5,
                                      dash="dot"))


def _render_overlay(store, file_id: int, channel: int | None,
                     stim_idx, win_ms: tuple[float, float],
                     n_prev: int = 10, mode: str = "lines"):
    """Stim overlay: the current stim's windowed trace + the previous
    ``n_prev``. ``mode='lines'`` colours them by a recency hue ramp;
    ``mode='heatmap'`` stacks them as an ERP-image. ``stim_idx`` is the
    nearest stim at/before the playhead (clientside-computed)."""
    assert isinstance(file_id, int), "file_id must be int"
    assert n_prev >= 1, "n_prev must be positive"
    if channel is None:
        return _empty_lfp_fig("Pick a brain channel (Step 1) first.")
    if stim_idx is None or int(stim_idx) < 0:
        return _empty_lfp_fig("Waiting for the first stim…")
    start_ms, end_ms = float(win_ms[0]), float(win_ms[1])
    if end_ms <= start_ms:
        return _empty_lfp_fig(
            "Line-length window end must be greater than start.")
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        return _empty_lfp_fig("File not found in DB.")
    stim_times = _stim_times_for_file(store, file_id)
    if stim_times is None or len(stim_times) == 0:
        return _empty_lfp_fig("No stim epochs on this file.")
    idx = min(int(stim_idx), len(stim_times) - 1)
    try:
        chunk = get_chunk(file_path)
        sig = np.asarray(chunk.signal[:, int(channel)], dtype=np.float64)
        fs = float(chunk.fs)
    except Exception as e:
        logger.warning("Overlay load failed file=%s ch=%s: %s",
                        file_id, channel, e)
        return _empty_lfp_fig(f"Couldn't load LFP: {e}")
    win = (start_ms / 1000.0, end_ms / 1000.0)
    kept, t_ms, segs = _overlay_segments(sig, fs, stim_times, idx, win, n_prev)
    if not kept:
        return _empty_lfp_fig(
            "Window too short for this sample rate — widen it.")
    fig = go.Figure()
    if mode == "heatmap":
        _add_overlay_heatmap(fig, kept, t_ms, segs)
    else:
        _add_overlay_lines(fig, kept, t_ms, segs, idx)
    _style_overlay(fig, start_ms, end_ms, len(kept), mode)
    return fig


def _render_evoked_band_ratio(store, file_id: int, channel: int | None,
                               lo: float, hi: float,
                               pre_win: tuple[float, float],
                               post_win: tuple[float, float]) -> tuple:
    """(figure, status) for baseline-normalized ("induced") slow gamma:
    per stim, ``10*log10(post/pre)`` band power in dB.

    Post window starts after the evoked transient (default 50-200 ms) so
    it reflects an oscillation rather than the spectral leakage of the
    evoked deflection; the pre window is the ongoing pre-stim baseline. A
    point above 0 dB means the stim *increased* slow gamma. Live -- no
    pipeline reprocess.
    """
    assert isinstance(file_id, int), "file_id must be int"
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        return (_empty_lfp_fig("File not found in DB."), "")
    if channel is None:
        return (_empty_lfp_fig("Pick a brain channel (Step 1) first."), "")
    stim_times = _stim_times_for_file(store, file_id)
    if stim_times is None or len(stim_times) == 0:
        return (_empty_lfp_fig(
            "No stim epochs on this file — induced slow gamma needs "
            "stimulation onsets."), "")
    try:
        chunk = get_chunk(file_path)
        sig = np.asarray(chunk.signal[:, int(channel)], dtype=np.float64)
        fs = float(chunk.fs)
    except Exception as e:
        logger.warning("Induced band-power load failed file=%s ch=%s: %s",
                        file_id, channel, e)
        return (_empty_lfp_fig(f"Couldn't load LFP: {e}"), "")
    times, db = epoch_band_ratio_db(
        sig, fs, stim_times, lo, hi, pre_win, post_win)
    if len(times) == 0:
        return (_empty_lfp_fig(
            "No usable stim epochs (a baseline or post window fell off "
            "the recording)."), "")
    fig = _build_lfp_figure(
        np.asarray(times), np.asarray(db),
        f"Induced slow gamma ({lo:g}-{hi:g} Hz, dB)",
        uirevision=f"{file_id}:{channel}")
    fig.data[0].mode = "markers+lines"
    fig.data[0].line.color = "#bf5af2"
    fig.data[0].marker = dict(size=5, color="#bf5af2")
    fig.data[0].hovertemplate = (
        "stim @ t=%{x:.2f}s<br>%{y:+.1f} dB vs baseline<extra></extra>")
    fig.layout.yaxis.title = "post / pre slow gamma (dB)"
    # 0 dB reference (no change). Cursor lives in shapes[0]; this takes [1].
    fig.layout.shapes = list(fig.layout.shapes or []) + [dict(
        type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0,
        line=dict(color="#888", width=1, dash="dash"), opacity=0.8)]
    mean_db = float(np.mean(db))
    frac_up = float(np.mean(np.asarray(db) > 0))
    status = (
        f"Induced slow gamma {lo:g}-{hi:g} Hz · post "
        f"[{post_win[0] * 1000:.0f},{post_win[1] * 1000:.0f}] ms vs pre "
        f"[{pre_win[0] * 1000:.0f},{pre_win[1] * 1000:.0f}] ms · "
        f"mean {mean_db:+.1f} dB · {frac_up * 100:.0f}% of "
        f"{len(times)} epochs increased")
    return (fig, status)


def _render_auc_trace(store, file_id: int, channel: int | None,
                       window_sec: float,
                       blank_pre_ms: float = -5.0,
                       blank_post_ms: float = 15.0,
                       show_raw: bool = False) -> tuple:
    """Sliding-window AUC (moving integral) of the Hilbert envelope.

    Sustained seizure events show as plateaus; transient noise spikes
    stay near zero -- the visual the PI uses to rescue low-amplitude
    events the peak detector misses."""
    assert isinstance(file_id, int), "file_id must be int"
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        return (_empty_lfp_fig("File not found in DB."), "")
    if channel is None:
        return (_empty_lfp_fig(
            "Pick a brain channel (Step 1) to see the AUC."), "")
    session_dir = _session_dir_for_file(store, file_id)
    stim_copy = _stim_copy_channels(store, session_dir)
    do_blank = (int(channel) not in stim_copy) and (not show_raw)
    stim_times = (_stim_times_for_file(store, file_id)
                   if do_blank
                   else np.asarray([], dtype=np.float64))
    try:
        series, fs, _ = _get_blanked_series(
            file_path, int(channel), stim_times,
            blank_pre_ms, blank_post_ms)
    except Exception as e:
        logger.warning("AUC load failed file=%s ch=%s: %s",
                        file_id, channel, e)
        return (_empty_lfp_fig(f"Couldn't load LFP for AUC: {e}"), "")
    env = hilbert_envelope_20_200(series, fs)
    try:
        win = float(window_sec) if window_sec else 5.0
        if win <= 0:
            win = 5.0
    except (TypeError, ValueError):
        win = 5.0
    auc = windowed_auc(env, fs, win)
    target_bins = choose_target_bins(len(auc)) or 4000
    t, display, _decim = envelope(auc, fs, target_bins, t_start=0.0)
    fig = _build_lfp_figure(
        t, display, f"Hilbert AUC ({win:g} s window)",
        uirevision=f"{file_id}:{channel}:auc")
    fig.data[0].line.color = "#30d158"
    fig.data[0].hovertemplate = (
        "t=%{x:.1f}s<br>AUC=%{y:.3g}<extra></extra>")
    peak_auc = float(np.max(auc)) if len(auc) else 0.0
    status = (
        f"Sliding-window AUC · window={win:g} s · "
        f"peak AUC={peak_auc:.3g} · "
        f"sustained events plateau, spikes stay low · "
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


def _panel_header(title: str, sub: str | None = None):
    """Un-numbered sub-section header for panels WITHIN a stage (e.g. the
    brain-feature trace or the scoring controls inside 'Review & score').

    Deliberately subordinate to the numbered ``_step_header`` -- a small
    accent bar instead of a number badge -- so the IA reads as two stages
    (pick -> review&score), not a false 1-2-3-4 pipeline. The watch /
    analyze / score work happens concurrently, so those panels aren't
    sequential steps.
    """
    head = html.Div([
        html.Span(style={"display": "inline-block", "width": "3px",
                          "height": "14px", "borderRadius": "2px",
                          "background": COLOR_ACCENT, "marginRight": "9px",
                          "flexShrink": "0"}),
        html.Span(title, style={"color": "#cfd0d6", "fontSize": "13px",
                                 "fontWeight": "600"}),
    ], style={"display": "flex", "alignItems": "center",
               "marginTop": "10px"})
    if sub:
        return html.Div([
            head,
            html.Div(sub, style={"color": "#888", "fontSize": "11px",
                                  "marginLeft": "12px", "marginTop": "2px",
                                  "marginBottom": "6px"}),
        ], style={"marginBottom": "6px"})
    return html.Div([head], style={"marginBottom": "8px"})


def _details_card(summary_text: str, content,
                   summary_sub: str | None = None,
                   open_default: bool = False):
    """Lightweight ``html.Details`` wrapper used to hide advanced
    controls behind a click. Keeps the default view friendly.
    """
    # No leading icon -- the +/- marker (theme.css) already signals that
    # the card expands, so an extra "sliders" glyph just adds clutter.
    summary_children: list = [
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
            "padding": "8px 8px", "listStyle": "none",
            "borderRadius": "6px",
        }),
        html.Div(content, style={"padding": "0 8px 8px 8px"}),
    ], open=open_default, className="qc-expandable", style={
        # Borderless + no fill (NN/g: borders should be functional, not
        # decorative; whitespace-first). The +/- marker + the summary
        # hover carry the affordance; whitespace separates stacked cards.
        # A thin divider keeps adjacent expanders distinguishable.
        "borderBottom": "1px solid rgba(255,255,255,0.06)",
        "marginTop": "8px", "marginBottom": "8px",
    })


_ELECTRODE_OPTIONS = [
    {"label": "First electrode", "value": 0},
    {"label": "Second", "value": 1},
    {"label": "Third", "value": 2},
    {"label": "Fourth", "value": 3},
]

# "Show all timestamps" lists the whole per-animal queue. High ceiling
# (vs the prev/next browse cap) so every recording appears; large enough
# for any real single-animal backlog.
_QUEUE_LIST_MAX = 100000


_POOL_BTN_STYLE = {
    "flex": "1",
    "background": "rgba(94,124,226,0.18)",
    "color": "#cfd0d6",
    "border": "1px solid rgba(94,124,226,0.4)",
    "padding": "6px 12px",
    "borderRadius": "5px",
    "cursor": "pointer",
    "fontSize": "12px",
    "fontWeight": "600",
}

_POOL_NAV_STYLE = {
    "background": "transparent",
    "color": "#cfd0d6",
    "border": "1px solid rgba(255,255,255,0.15)",
    "padding": "5px 12px",
    "borderRadius": "5px",
    "cursor": "pointer",
    "fontSize": "12px",
}


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
            # Simple row: the essentials -- threshold + Scan/Cancel.
            html.Div([
                html.Label("Cutoff:",
                            style={"color": "#a0a0b0",
                                    "fontSize": "12px",
                                    "marginRight": "6px"}),
                dcc.Input(
                    id="video-ma-cutoff-input",
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
                button(
                    "Scan for events",
                    "video-ma-scan-btn", variant="primary"),
                button(
                    "Cancel scan",
                    "video-ma-cancel-btn", variant="ghost"),
            ], style={"display": "flex",
                       "alignItems": "center",
                       "gap": "8px",
                       "flexWrap": "wrap",
                       "padding": "0 14px",
                       "marginBottom": "8px"}),
            # Advanced detection: the AUC screen + electrode, hidden by
            # default so the common path is just Cutoff + Scan.
            html.Div(_details_card(
                "Advanced detection",
                summary_sub="AUC screen + which electrode to scan",
                open_default=False,
                content=html.Div([
                    html.Label("AUC thresh:",
                                style={"color": "#a0a0b0",
                                        "fontSize": "12px",
                                        "marginRight": "6px"}),
                    dcc.Input(
                        id="video-ma-auc-threshold-input",
                        type="number", min=0, step="any",
                        placeholder="e.g. 0.5",
                        style={"flex": "0 0 100px",
                                "padding": "6px 10px",
                                "background": "#262638",
                                "color": "#f0f0f5",
                                "border":
                                    "1px solid rgba(255,255,255,0.10)",
                                "borderRadius": "4px",
                                "fontSize": "12px"}),
                    html.Label("AUC win (s):",
                                style={"color": "#a0a0b0",
                                        "fontSize": "12px",
                                        "marginRight": "6px"}),
                    dcc.Input(
                        id="video-ma-auc-window-input",
                        type="number", min=0, step="any",
                        value=5,
                        style={"flex": "0 0 80px",
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
                                        "marginRight": "6px"}),
                    dcc.Dropdown(
                        id="video-ma-electrode",
                        options=_ELECTRODE_OPTIONS,
                        value=0, clearable=False,
                        style={"flex": "0 0 130px",
                                "minWidth": "110px"},
                        className="dark-dropdown"),
                ], style={"display": "flex", "alignItems": "center",
                           "gap": "8px", "flexWrap": "wrap"})),
                style={"padding": "0 14px", "marginBottom": "10px"}),
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
            # Two-pool browser (revealed once a scan completes).
            # Pool 1 = peak hits; Pool 2 = peak misses to rescue
            # by eyeballing the AUC trace.
            html.Div(
                id="video-ma-browse",
                style={"display": "none",
                        "padding": "0 14px",
                        "marginBottom": "10px"},
                children=dcc.Loading(
                  id="video-ma-browse-loading",
                  custom_spinner=_loading_icon("Building pools…"),
                  delay_show=180,
                  children=[
                    html.Div(id="video-ma-pool-counts",
                              style={"fontSize": "12px",
                                      "color": "#cfd0d6",
                                      "marginBottom": "6px"}),
                    # Filter: narrow the pools to one session before
                    # browsing. (Stim is implicit in the session.)
                    html.Div([
                        html.Label("Session:",
                                    style={"color": "#a0a0b0",
                                            "fontSize": "11px",
                                            "marginRight": "6px"}),
                        dcc.Dropdown(
                            id="video-ma-filter-session",
                            options=[{"label": "All sessions",
                                       "value": "__all__"}],
                            value="__all__", clearable=False,
                            style={"flex": "1 1 260px",
                                    "minWidth": "180px"},
                            className="dark-dropdown"),
                    ], style={"display": "flex",
                               "alignItems": "center",
                               "gap": "4px", "marginBottom": "4px"}),
                    html.Div(id="video-ma-filter-counts",
                              style={"fontSize": "11px",
                                      "color": "#a0a0b0",
                                      "marginBottom": "6px"}),
                    # Per-session statistics (Phase 3); filled when a
                    # specific session is selected above.
                    html.Div(id="video-ma-session-stats",
                              style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("🔬 Threshold detections",
                                  style={"color": "#5e7ce2",
                                          "fontWeight": "700",
                                          "fontSize": "12px"}),
                        html.Span(
                            " — the detector flagged these at your cutoff. "
                            "Browse a pool and REVIEW each: score the real "
                            "events, or mark “No events seen.” "
                            "(Different from the \U0001f6a9 Needs-scoring "
                            "pool, which is your own onsets to finish.)",
                            style={"color": "#a0a0b0", "fontSize": "11px"}),
                    ], style={"marginBottom": "6px", "padding": "0 2px"}),
                    html.Div([
                        html.Button(
                            "Browse Pool 1 (envelope)",
                            id="video-ma-browse-p1", n_clicks=0,
                            style=_POOL_BTN_STYLE),
                        html.Button(
                            "Browse Pool 2 (AUC)",
                            id="video-ma-browse-p2", n_clicks=0,
                            style=_POOL_BTN_STYLE),
                        html.Button(
                            "Browse Pool 3 (disagreement)",
                            id="video-ma-browse-p3", n_clicks=0,
                            style=_POOL_BTN_STYLE),
                    ], style={"display": "flex", "gap": "8px",
                               "marginBottom": "6px"}),
                    html.Div([
                        html.Button(
                            "< Prev", id="video-ma-pool-prev",
                            n_clicks=0, style=_POOL_NAV_STYLE),
                        html.Span(
                            id="video-ma-pool-status",
                            style={"fontSize": "12px",
                                    "color": "#a0a0b0",
                                    "minWidth": "150px",
                                    "textAlign": "center"}),
                        html.Button(
                            "Next >", id="video-ma-pool-next",
                            n_clicks=0, style=_POOL_NAV_STYLE),
                    ], style={"display": "flex", "gap": "8px",
                               "alignItems": "center"}),
                  ])),
            # State stores + polling.
            dcc.Store(id="video-ma-pools", data=None),
            # Per-file meta {file_id: {session_dir, session_name,
            # has_stim}} so the filters work without re-querying.
            dcc.Store(id="video-ma-pool-meta", data=None),
            # The filtered view _on_pool_nav actually browses.
            dcc.Store(id="video-ma-pools-view", data=None),
            dcc.Store(id="video-ma-pool-cursor", data=None),
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
    ], open=False, className="qc-expandable",
       style={"padding": "0",
               # Matches the borderless _details_card treatment.
               "borderBottom": "1px solid rgba(255,255,255,0.06)",
               "marginTop": "0",
               "marginBottom": "16px"})


def layout(store: Store, bridge: dict | None = None):
    sessions = _sessions_with_video(store)
    session_options = [
        {"label": f"{s['session_name']} ({s['n_videos']} videos)",
         "value": s["session_dir"]}
        for s in sessions
    ]
    # Nothing auto-loads: the reviewer picks an animal (queue) or a
    # session (manual picker) first. Defaulting to the newest session
    # used to silently load a recording the reviewer never chose.
    default_session = None
    # Deep-link hand-off (LFP Browser / Event Verification): pre-select
    # the requested session AT BUILD TIME so the file/channel cascade
    # (_update_files / _update_player, which read the bridge as State)
    # loads the exact recording on mount. Honor the bridge ONLY when
    # it's fresh (set within the last 30 s) -- the bridge persists in
    # memory for the page session, so without this a stale hand-off
    # would re-load that recording every time the tab is reopened.
    if isinstance(bridge, dict):
        want = bridge.get("session_dir")
        try:
            age_ms = datetime.now().timestamp() * 1000 - float(
                bridge.get("seq") or 0)
        except (TypeError, ValueError):
            age_ms = 1e12
        fresh = age_ms < 30_000
        if (fresh and want
                and any(o["value"] == want for o in session_options)):
            default_session = want

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
                icon("info", size=14, color=COLOR_ACCENT,
                     style={"marginRight": SPACE_2}),
                html.Span("Quick start", style={
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
                "display": "flex", "alignItems": "center",
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
        ], open=False, className="qc-expandable", style={
            # Quiet chrome: neutral card surface + hairline, with a thin
            # accent left edge for identity (no saturated blue wash).
            "background": COLOR_SURFACE_1,
            "border": f"1px solid {COLOR_DIVIDER}",
            "borderLeft": f"3px solid {COLOR_ACCENT}",
            "borderRadius": RADIUS_MD,
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
        # Per-leg "render completed" tokens. The LFP / Hilbert
        # status strings are a pure function of duration + point
        # count + stim/candidate count, so consecutive fixed-
        # length recordings produce an IDENTICAL status -- the
        # done-watchers (which fire on a status *change*) would
        # never see the second load complete and the pill would
        # hang on "loading". These tokens carry a fresh timestamp
        # on every render so the watchers always fire.
        dcc.Store(id="video-lfp-loadtoken", data=0),
        dcc.Store(id="video-hilbert-loadtoken", data=0),
        # Write-only sink for the soft-claim callback (claiming a file
        # has no visible output; this just gives the callback an Output
        # to satisfy Dash).
        dcc.Store(id="video-claim-sink", data=0),
        # Write-only sink for the x-axis sync callbacks: they drive the
        # partner graph via Plotly.relayout and return no_update, so
        # they just need a harmless Output.
        dcc.Store(id="video-xsync-sink", data=0),

        # --- Step 1: pick a recording ---------------------------------- #
        _step_header("1", "Pick a recording",
                      "Pick which animal you're reviewing. Recordings "
                      "show up oldest first (FIFO) so the backlog "
                      "clears from the front — work top to bottom."),

        # Step 1 is a COMPACT full-width strip (animal · queue chips ·
        # browse · Load) so it stays short and the video (Step 2) rises
        # up. The secondary actions sit in a responsive grid BELOW --
        # which also moves Mass Analyze out of the lead pick position
        # (it's a batch pre-screen tool, not part of picking *this*
        # recording). NN/g grounding: one-level progressive disclosure,
        # a grid that fills the width, and common-region grouping that
        # separates "browse the queue" from "other tools".
        html.Div([
            # Animal picker
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
            ], style={"flex": "0 0 240px"}),
            # Queue: title/chips + current-recording browse strip
            html.Div([
                html.Div([
                    html.Span("Queue (oldest first · FIFO)",
                               style=LABEL_STYLE,
                               title="Recordings for the chosen animal "
                                      "that nobody has finished yet. "
                                      "Oldest first."),
                    # Progress + streak chips (rendered by
                    # _render_queue_progress).
                    html.Div(id="video-queue-progress",
                              style={"display": "flex",
                                      "alignItems": "center",
                                      "gap": "8px"}),
                ], style={"display": "flex", "alignItems": "center",
                           "gap": "10px", "flexWrap": "wrap",
                           "marginBottom": "6px"}),
                # Navigation mode: browse the normal queue, the auto-filter
                # has-events flag pool, or the needs-scoring pool. Same
                # carousel + Load below, different source list.
                dcc.RadioItems(
                    id="video-queue-mode",
                    options=[{"label": f" {m['label']}", "value": m["value"]}
                             for m in _QUEUE_MODES],
                    value="queue", inline=True,
                    labelStyle={"color": "#ddd", "fontSize": "12px",
                                "marginRight": "14px", "cursor": "pointer"},
                    inputStyle={"marginRight": "4px"},
                    style={"marginBottom": "6px"}),
                html.Div(id="video-queue-card-title",
                          style={"color": "#a0a0b0", "fontSize": "11px",
                                  "marginBottom": "6px"}),
                # Slim hint when no animal / empty queue (navigator hidden).
                html.Div(id="video-queue-empty",
                          style={"color": "#888", "fontSize": "12px"}),
                # Compact browse strip: ◀ [recording] ▶ [Load].
                html.Div([
                    html.Div([
                        html.Button(
                            "◀", id="video-queue-prev-btn", n_clicks=0,
                            title="Browse to the next-newer recording "
                                   "(no load yet).",
                            className="video-queue-arrow"),
                        html.Div(id="video-queue-card-body",
                                  style={"flex": "1", "padding": "0 10px",
                                          "textAlign": "center"}),
                        html.Button(
                            "▶", id="video-queue-next-btn", n_clicks=0,
                            title="Browse to the next-older recording "
                                   "(no load yet).",
                            className="video-queue-arrow"),
                        button("Load this recording",
                               "video-queue-load-btn",
                               variant="primary", icon_name="video"),
                    ], style={"display": "flex", "alignItems": "center",
                               "gap": "8px", "padding": "8px",
                               "background": "#13131f",
                               "border": "1px solid rgba(255,255,255,0.06)",
                               "borderRadius": "6px"}),
                    html.Span(id="video-queue-card-position",
                               style={"color": "#a0a0b0", "fontSize": "11px",
                                       "marginTop": "4px",
                                       "display": "inline-block"}),
                ], id="video-queue-nav-wrap"),
            ], style={"flex": "1", "minWidth": "320px",
                       "maxWidth": "640px"}),
        ], style={"display": "flex", "gap": "24px",
                   "alignItems": "flex-start", "flexWrap": "wrap",
                   "marginBottom": "16px"}),

        # Secondary tools -- a responsive grid that fills the width.
        # Mass Analyze lives here (a batch tool), grouped with the full
        # list, the needs-scoring pool, and the manual picker.
        html.Div([
            _details_card(
                "Show all timestamps",
                summary_sub="full FIFO list -- click any item to load it",
                open_default=False,
                content=html.Div(id="video-queue-list",
                          style={"maxHeight": "220px", "overflowY": "auto",
                                  "background": "#13131f",
                                  "border":
                                      "1px solid rgba(255,255,255,0.06)",
                                  "borderRadius": "6px", "padding": "4px"}),
            ),
            _details_card(
                "🚩 Needs scoring — finish your onsets",
                summary_sub="Files where YOU dropped an EEG onset but haven't "
                            "fully scored it yet (type + Racine + all "
                            "landmarks). Open one to complete it.",
                open_default=False,
                content=html.Div(
                    id="video-needs-scoring-list",
                    style={"maxHeight": "220px", "overflowY": "auto",
                            "background": "#13131f",
                            "border": "1px solid rgba(240,180,41,0.25)",
                            "borderRadius": "6px", "padding": "4px"}),
            ),
            _details_card(
                "Pick any recording manually",
                summary_sub="bypass the queue -- choose session/file/channel",
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
                            className="dark-dropdown"),
                    ], style={"flex": "2", "minWidth": "200px"}),
                    html.Div([
                        html.Label("File (chunk)", style=LABEL_STYLE,
                                    title="Each chunk is roughly one hour of "
                                           "recording, named with its start "
                                           "timestamp."),
                        dcc.Dropdown(
                            id="video-file-dropdown", options=[],
                            style=DROPDOWN_STYLE,
                            className="dark-dropdown"),
                    ], style={"flex": "3", "minWidth": "220px"}),
                    html.Div([
                        html.Label("Brain channel", style=LABEL_STYLE,
                                    title="Which electrode contact to "
                                           "display the LFP trace for."),
                        dcc.Dropdown(
                            id="video-channel-dropdown", options=[],
                            value=0, style=DROPDOWN_STYLE,
                            className="dark-dropdown"),
                    ], style={"flex": "1", "minWidth": "160px"}),
                ], style={"display": "flex", "gap": "16px",
                           "flexWrap": "wrap"}),
            ),
            # Bulk pre-screen (Hilbert thresholding) -- a batch tool
            # scoped to the animal picked above, NOT part of picking one
            # recording. Same job/cache as the PI Mass Analyze.
            _video_mass_analyze_panel(),
        ], style={"display": "grid",
                   "gridTemplateColumns":
                       "repeat(auto-fit, minmax(320px, 1fr))",
                   "gap": "4px 20px", "alignItems": "start"}),
        dcc.Store(id="video-queue-position", data=0),

        # --- Step 2: watch ---------------------------------------------- #
        _step_header("2", "Review & score",
                      "Watch the video in sync with the LFP/feature traces "
                      "and score events — you do these together, not in "
                      "order. The video pips to the corner as you scroll."),

        # --- Now-viewing banner + hour stepping ----------------------- #
        # Tells the reviewer exactly which animal / day / hour is loaded
        # (a recurring "which file am I on?" confusion) and lets them
        # walk the recording hour by hour without reopening the picker.
        html.Div([
            html.Div(id="video-now-viewing",
                      style={"flex": "1 1 auto",
                              "color": "#f0f0f5",
                              "fontSize": "13px",
                              "fontWeight": "600"}),
            html.Button("◀ Prev hour",
                         id="video-prev-hour-btn", n_clicks=0,
                         title="Load the previous hour-chunk in this "
                               "session.",
                         style={"background": "transparent",
                                 "color": "#cfd0d6",
                                 "border": "1px solid "
                                            "rgba(255,255,255,0.15)",
                                 "borderRadius": "5px",
                                 "padding": "5px 12px",
                                 "cursor": "pointer",
                                 "fontSize": "12px",
                                 "marginRight": "6px"}),
            html.Button("Next hour ▶",
                         id="video-next-hour-btn", n_clicks=0,
                         title="Load the next hour-chunk in this "
                               "session.",
                         style={"background": "transparent",
                                 "color": "#cfd0d6",
                                 "border": "1px solid "
                                            "rgba(255,255,255,0.15)",
                                 "borderRadius": "5px",
                                 "padding": "5px 12px",
                                 "cursor": "pointer",
                                 "fontSize": "12px"}),
        ], style={"display": "flex", "alignItems": "center",
                   "gap": "8px", "flexWrap": "wrap",
                   "padding": "8px 12px", "marginBottom": "10px",
                   # Sticky so it stays visible while scrolling down to
                   # score (Nielsen #5/#6: recognition over recall; you
                   # never lose track of which file you're marking done).
                   "position": "sticky", "top": "0", "zIndex": 60,
                   # Quiet chrome: neutral surface + accent left edge.
                   "background": COLOR_SURFACE_1,
                   "border": f"1px solid {COLOR_DIVIDER}",
                   "borderLeft": f"3px solid {COLOR_ACCENT}",
                   "borderRadius": RADIUS_MD,
                   "boxShadow": "0 2px 8px rgba(0,0,0,0.35)"}),

        # --- Layout presets: resize the video / LFP split ---------- #
        html.Div([
            html.Span("Layout:",
                       style={"color": "#a0a0b0", "fontSize": "11px",
                               "marginRight": "8px"}),
            html.Button("Big video", id="video-layout-bigvideo",
                         n_clicks=0, className="video-layout-btn"),
            html.Button("50 / 50", id="video-layout-5050",
                         n_clicks=0, className="video-layout-btn"),
            html.Button("Big LFP", id="video-layout-biglfp",
                         n_clicks=0, className="video-layout-btn"),
            html.Button([icon("maximize", size=12,
                               color=COLOR_TEXT_SECONDARY,
                               style={"marginRight": "5px"}),
                         "Fullscreen"],
                         id="video-fullscreen-btn",
                         n_clicks=0, className="video-layout-btn",
                         title="Fill the screen with the video; the "
                               "LFP/Hilbert stays as a card bottom-right "
                               "(Esc to exit)."),
            html.Div(id="video-fs-sink",
                      style={"display": "none"}),
        ], style={"display": "flex", "alignItems": "center",
                   "gap": "6px", "marginBottom": "8px"}),

        # --- Two-column Step 2: video LEFT | LFP+Hilbert RIGHT ----- #
        # CSS Grid: video spans both rows in column 1; the LFP block
        # lives in column 2 row 1, the Hilbert block in column 2 row
        # 2. On narrow viewports (<=900 px) the grid collapses to a
        # single column via the media query in theme.css.
        html.Div([
            dcc.Loading(
                id="video-player-loading",
                custom_spinner=_loading_icon("Loading video…"),
                # 180 ms before showing the spinner means warm-
                # cache renders don't flash; cold-cache reads
                # (where the reviewer needs the feedback) tip
                # over the threshold and show clearly.
                delay_show=180,
                parent_className="video-player-col",
                parent_style={"gridArea": "video",
                               "minHeight": "360px"},
                children=html.Div(
                    id="video-player-container",
                    children=empty_state(
                        "No recording loaded",
                        hint="Pick a recording above to load the video "
                             "here.",
                        icon_name="video"),
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
                # Live playhead: shows BOTH clocks -- the video player's
                # container time and the LFP-estimated time. They differ
                # because the 5 fps video drops frames, so the video clock
                # runs short; the LFP time is an endpoint-stretch ESTIMATE
                # (not frame-exact). Updated clientside off the 10 Hz poll.
                html.Span(id="video-playhead-readout",
                          title="The video player shows its container "
                                "clock. 'LFP≈' is the dropped-frame-"
                                "adjusted estimate (linear endpoint "
                                "stretch -- not frame-exact). Recording "
                                "length is LFP ground truth.",
                          style={"color": "#ff9f0a", "fontSize": "11px",
                                 "marginLeft": "12px",
                                 "fontVariantNumeric": "tabular-nums"}),
                # Modeless interaction (no Seek/Zoom toggle): click the
                # trace to seek the video, scroll to zoom, drag to pan,
                # double-click to reset. Removing the mode kills the classic
                # "I'm in the wrong mode" error (Raskin: modes cause
                # mistakes).
                html.Span("Click to seek · scroll to zoom · drag to pan",
                          style={"color": "#9a9aa8", "fontSize": "11px",
                                 "marginLeft": "auto"}),
            ], style={"display": "flex", "alignItems": "center",
                      "flexWrap": "wrap",
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
                                       "milliseconds, applied to the raw "
                                       "LFP trace ONLY. 0 = off. "
                                       "Distinct from the 'Brain feature "
                                       "over time' smooth-by-seconds, which "
                                       "smooths the per-event feature line. "
                                       "Helpful for "
                                       "spotting slow rhythms; bad for "
                                       "spotting fast spikes."),
                    dcc.Input(id="video-filter-smooth", type="number",
                              min=0, step=1, value=0,
                              style={"backgroundColor": "#262638",
                                     "color": "#f0f0f5", "width": "90px"}),
                ], style={"flex": "0 0 130px"}),
                html.Div([
                    html.Label("Stim artifact", style=LABEL_STYLE,
                                title="By default the brief stim pulses "
                                       "are blanked out (gaps in the "
                                       "trace) so the EEG underneath is "
                                       "visible. Tick this to show the "
                                       "raw signal WITH the stim artifact "
                                       "instead."),
                    dcc.Checklist(
                        id="video-blank-raw",
                        options=[{"label": " Show raw",
                                   "value": "raw"}],
                        value=[],
                        labelStyle={"color": "#cfd0d6",
                                     "fontSize": "12px"},
                        style={"paddingTop": "6px"},
                    ),
                ], style={"flex": "0 0 120px"}),
                html.Div([
                    html.Label(" ", style=LABEL_STYLE),
                    button("Apply filters",
                           "video-apply-filter-btn", variant="secondary",
                           title="Apply the filter settings above."),
                ], style={"flex": "0 0 120px",
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
                custom_spinner=_loading_icon("Filtering & re-decimating…"),
                # 0 ms: show the loading overlay the instant a re-decimate
                # starts. Fast zoom-ins (few points in the window) used to
                # finish under the old 80 ms threshold and never showed the
                # spinner, so a zoom felt like it had no loading feedback.
                delay_show=0,
                # Keep the trace visible (dimmed) under the spinner so a
                # zoom re-decimate shows progress without the graph
                # vanishing.
                overlay_style={"visibility": "visible",
                                "opacity": 0.45},
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
            _panel_header(
                "Brain feature over time",
                "Pick a number computed from the brain signal for every "
                "stim event. Click a point to jump the video there.",
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
                            {"label": "Hilbert AUC (sliding window) "
                                       "-- catches sustained events",
                                "value": "hilbert_auc"},
                            # Slow gamma (30-50 Hz) band power -- live,
                            # no pipeline reprocess. Continuous (LFP) +
                            # per-stim (evoked) views.
                            {"label": "Slow gamma power "
                                       "(30-50 Hz, continuous)",
                                "value": "slow_gamma_power"},
                            {"label": "Slow gamma power "
                                       "(30-50 Hz, per stim)",
                                "value": "slow_gamma_evoked"},
                            {"label": "Slow gamma induced "
                                       "(30-50 Hz, post vs baseline dB)",
                                "value": "slow_gamma_induced"},
                            {"label": "Wavelet power "
                                       "(slow γ 30-50 Hz, continuous)",
                                "value": "wavelet_sg_power"},
                            {"label": "Wavelet power "
                                       "(slow γ 30-50 Hz, per stim)",
                                "value": "wavelet_sg_evoked"},
                            {"label": "Wavelet scalogram "
                                       "(mean stim-locked heatmap)",
                                "value": "wavelet_scalogram"},
                            {"label": "Line length (live windowed)",
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
                # Detection threshold (Hilbert feature only). This is
                # the dashed line on the envelope: BHZ flags a candidate
                # when the 20-200 Hz envelope crosses it. Typing here or
                # clicking the plot both move the line. Synced to the
                # Mass Analyze cutoff so one threshold drives both.
                html.Div([
                    html.Label("Detection threshold",
                                style=LABEL_STYLE,
                                title="The dashed line on the Hilbert "
                                       "envelope. A candidate event is "
                                       "flagged when the envelope crosses "
                                       "it. Type a value or click the plot "
                                       "to move it. Lower = more sensitive."),
                    dcc.Input(
                        id="video-threshold-input",
                        type="number", min=0, step="any",
                        value=0.05,
                        style={"backgroundColor": "#262638",
                                "color": "#f0f0f5", "width": "90px"}),
                ], style={"flex": "0 0 150px"}),
                # AUC window (only meaningful for the Hilbert AUC
                # feature): how many seconds the sliding integral spans.
                html.Div([
                    html.Label("AUC window (s)", style=LABEL_STYLE,
                                title="For the 'Hilbert AUC' feature: "
                                       "the sliding integral spans this "
                                       "many seconds. Longer favours "
                                       "sustained events over brief "
                                       "spikes."),
                    dcc.Input(
                        id="video-auc-window-input",
                        type="number", min=0.5, step="any",
                        value=5,
                        style={"backgroundColor": "#262638",
                                "color": "#f0f0f5", "width": "80px"}),
                ], style={"flex": "0 0 130px"}),
                # Line-length window (only meaningful for the Line length
                # feature, and drives the stim-overlay below): the start
                # and end (ms, relative to stim onset) of the window the
                # line length is measured over. A tight window like
                # -1..1 ms tracks stim-onset instability. NB: this is a
                # live computation and intentionally differs from the old
                # full-epoch DB line_length column.
                html.Div([
                    html.Label("Line-length window (ms)", style=LABEL_STYLE,
                                title="For the 'Line length' feature (and "
                                       "the stim overlay): the start/end in "
                                       "ms, relative to each stim onset, "
                                       "that line length is measured over. "
                                       "A tight -1..1 ms window tracks "
                                       "stim-onset instability."),
                    html.Div([
                        dcc.Input(
                            id="video-llwin-start-input",
                            type="number", step="any", value=-1,
                            style={"backgroundColor": "#262638",
                                    "color": "#f0f0f5", "width": "70px"}),
                        html.Span("to", style={"color": "#9a9aa8",
                                                "margin": "0 6px"}),
                        dcc.Input(
                            id="video-llwin-end-input",
                            type="number", step="any", value=1,
                            style={"backgroundColor": "#262638",
                                    "color": "#f0f0f5", "width": "70px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], id="video-llwin-group",
                   # Hidden unless the Line length feature is selected
                   # (toggled by _toggle_llwin_group). Default feature is
                   # 'hilbert', so start hidden.
                   style={"flex": "0 0 200px", "display": "none"}),
                # Remove false-positive candidate peaks (the pink
                # triangles). When the mode is on, clicking a triangle
                # rejects it (persisted per file); Undo restores the
                # last one.
                html.Div([
                    html.Label("Curate peaks", style=LABEL_STYLE,
                                title="The detector flags candidates "
                                       "generously. Turn this on, then "
                                       "click a pink triangle that isn't "
                                       "a real seizure to remove it. "
                                       "Removals are saved per recording."),
                    html.Div([
                        dcc.Checklist(
                            id="video-remove-peaks-mode",
                            options=[{"label": " Remove-peaks mode",
                                       "value": "on"}],
                            value=[],
                            labelStyle={"color": "#cfd0d6",
                                         "fontSize": "12px"},
                            style={"display": "inline-block",
                                    "marginRight": "8px"}),
                        html.Button(
                            "Undo removal",
                            id="video-undo-reject-btn", n_clicks=0,
                            style={"background": "transparent",
                                    "color": "#cfd0d6",
                                    "border": "1px solid "
                                               "rgba(255,255,255,0.15)",
                                    "borderRadius": "5px",
                                    "padding": "3px 10px",
                                    "cursor": "pointer",
                                    "fontSize": "11px"}),
                    ], style={"display": "flex",
                               "alignItems": "center"}),
                ], style={"flex": "0 0 220px"}),
                dcc.Store(id="video-rejected-version", data=0),
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
                                           "per-event feature line ONLY. "
                                           "0 = off. Distinct from the "
                                           "'Smooth the LFP (ms)' control, "
                                           "which smooths the raw trace."),
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
                        button(
                            "Update plot",
                            "video-analysis-apply-btn",
                            variant="secondary",
                            title="Apply smoothing / detrend / "
                                  "normalize / clean-up choices."),
                    ], style={"flex": "0 0 120px",
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
                custom_spinner=_loading_icon("Filtering & re-decimating…"),
                # 0 ms: immediate loading overlay on zoom / feature redraws
                # (matches video-lfp-loading above).
                delay_show=0,
                overlay_style={"visibility": "visible",
                                "opacity": 0.45},
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
            # Stim overlay (oscilloscope persistence view): during
            # playback shows the raw trace in the line-length window for
            # the current stim + the previous 10, faded by recency so you
            # can watch the stim-locked shape drift. Driven by the
            # clientside throttle below (rebuilds only on stim crossing).
            html.Div([
                # Header toolbar: title + trace-count + collapse + pop-out.
                html.Div([
                    html.Span("Stim overlay",
                              style={"color": "#ffffff", "fontSize": "12px",
                                     "fontWeight": "600"}),
                    dcc.RadioItems(
                        id="video-overlay-mode",
                        options=[{"label": " Lines", "value": "lines"},
                                  {"label": " Heatmap", "value": "heatmap"}],
                        value="lines", inline=True,
                        style={"color": "#ffffff", "fontSize": "11px",
                               "marginLeft": "12px"},
                        # labelStyle colours the actual option text (the
                        # container `style` doesn't reach the <label>s).
                        labelStyle={"color": "#ffffff",
                                     "marginRight": "10px"},
                        inputStyle={"marginRight": "3px",
                                     "marginLeft": "8px"}),
                    html.Span([
                        html.Label("Traces",
                                    style={"color": "#ffffff",
                                           "fontSize": "11px",
                                           "marginRight": "5px"}),
                        dcc.Input(
                            id="video-overlay-ntraces-input",
                            type="number", min=1, max=100, step=1, value=10,
                            style={"backgroundColor": "#262638",
                                    "color": "#f0f0f5", "width": "60px"}),
                    ], style={"display": "flex", "alignItems": "center",
                               "marginLeft": "auto", "marginRight": "8px"}),
                    # "+" = collapsed (click to expand), "-" = expanded
                    # (click to collapse). Set by applyCollapsed in JS;
                    # starts "+" since the overlay is collapsed by default.
                    html.Button("+", id="video-overlay-collapse-btn",
                                 className="qc-overlay-iconbtn",
                                 title="Expand / collapse the overlay"),
                    html.Button("⤢", id="video-overlay-pip-btn",
                                 className="qc-overlay-iconbtn",
                                 title="Pop out (drag the header to move, "
                                       "drag a corner to resize)"),
                ], className="qc-overlay-header"),
                html.Div(
                    dcc.Graph(
                        id="video-overlay-graph",
                        figure=_empty_lfp_fig(
                            "Play the video to see the stim overlay — the "
                            "current stim's trace plus the last 10, faded "
                            "by recency."),
                        responsive=True,
                        style={"height": "100%", "width": "100%"},
                        config={"displayModeBar": False,
                                 "displaylogo": False},
                    ),
                    id="video-overlay-body",
                    style={"height": "190px"},
                ),
            ], id="video-overlay-panel",
               # Collapsed by default (class also set so there's no flash
               # before the clientside mirror runs).
               className="qc-overlay-panel qc-overlay-collapsed"),
        ], className="video-lfp-col",
           style={"marginTop": "0", "gridArea": "lfp",
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
        ], id="video-step2-grid", className="layout-5050",
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
                    "background": "rgba(255,159,10,0.05)",
                    "border": "1px solid rgba(255,159,10,0.25)",
                    "borderRadius": "6px"},
            children=[
              # Collapsible (collapsed by default): even PIs only open it
              # when they're actually re-tuning the quality threshold.
              html.Details([
                html.Summary([
                    icon("alert-triangle", size=13, color="#ff9f0a",
                         style={"marginRight": "6px"}),
                    html.Span("Advanced: video quality threshold (PI)"),
                    html.Span("  (click to expand)",
                              style={"color": "#a0a0b0",
                                      "fontSize": "11px",
                                      "marginLeft": "6px",
                                      "fontWeight": "400"}),
                ], style={"color": "#ff9f0a", "fontSize": "12px",
                           "fontWeight": "600", "cursor": "pointer",
                           "userSelect": "none", "listStyle": "none",
                           "display": "flex", "alignItems": "center",
                           "padding": "10px 14px"}),
                html.Div([
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
                ], style={"padding": "0 14px 12px"}),
              ], open=False, style={"background": "transparent",
                                     "border": "none"}),
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
            "Timestamped note — jump back here later",
            summary_sub="saved with the playback time + full review "
                         "context; click it in the history to return to "
                         "this exact spot. Separate from your decision note.",
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
            _panel_header(
                "Score events & finish",
                "Add seizure events on the LFP and finish each one "
                "with a Racine score. This finishes the recording "
                "in your queue."),
            # Which animal these onsets file under (= the SCORED channel's
            # animal) + the recording's channel->animal legend, so a
            # multi-animal mis-attribution can't slip through unseen.
            html.Div(id="video-filed-under-banner",
                     style={"marginBottom": "8px"}),
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
                        "To record an onset that saves, use an event "
                        "card above: \"+ Add event\", then \"Set on "
                        "plot\" and click the LFP or Hilbert trace. "
                        "(The quick marks below are a scratch aid only "
                        "-- they are not saved.)",
                        style={"color": "#a0a0b0",
                                "fontSize": "12px",
                                "marginBottom": "6px"}),
                    html.Div([
                        button("Clear quick marks",
                               "video-review-clear-markers-btn",
                               variant="ghost",
                               style={"fontSize": "11px",
                                      "padding": "4px 10px"}),
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
                    html.Label("Note saved with your decision "
                                "(max 280 chars)",
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
                    # The single loud action on this step: solid success
                    # fill + larger footprint so it's unmistakably the
                    # next thing to do. Quick-flag recedes to secondary.
                    button(
                        "Mark recording done",
                        "video-review-save-btn",
                        variant="primary", tone="success",
                        icon_name="check-circle",
                        title="Saves your decision + any onset markers. "
                              "Removes the recording from your queue.",
                        style={"padding": "10px 26px",
                               "fontSize": "14px", "fontWeight": "700",
                               "marginRight": "10px"}),
                    button(
                        "EEG onset → CSV + flag",
                        "video-review-partial-btn",
                        variant="secondary",
                        icon_name="file-text",
                        title="Write the onsets you've dropped to the "
                              "BHZ day CSV right now (EEG/EO onset only is "
                              "fine -- other landmarks save as NaN) so you "
                              "can use it immediately, AND keep the file in "
                              "the 'Needs scoring' pool for full scoring "
                              "later. The PI's finalize (overwrite) replaces "
                              "these preliminary rows. Your onset is in the "
                              "EventEO / EventEO_WallClock / EO_HourOfDay "
                              "columns; Peak_Index/Peak_Stamp stay blank "
                              "(they're set only by the automated detector).",
                        style={"marginRight": "10px"}),
                    dcc.Loading(
                        id="video-review-status-loading",
                        custom_spinner=_loading_icon(small=True),
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
                       # Quiet chrome: neutral surface + success left edge
                       # for step identity. The saturated green now lives
                       # only on the "Mark recording done" primary button.
                       "background": COLOR_SURFACE_1,
                       "border": f"1px solid {COLOR_DIVIDER}",
                       "borderLeft": f"3px solid {COLOR_SUCCESS}",
                       "borderRadius": RADIUS_MD}),

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
        # Stim overlay support. `video-stim-times` holds this file's stim
        # onsets (LFP seconds) so the clientside throttle can find the
        # nearest stim at/before the playhead without a server round-trip.
        # `video-overlay-stim-idx` is that nearest-stim index, pushed only
        # when it changes -- so the server rebuilds the overlay figure at
        # most once per stim crossing, not at the 10 Hz playback cadence.
        dcc.Store(id="video-stim-times", data=[]),
        dcc.Store(id="video-overlay-stim-idx", data=-1),
        # Overlay panel UI state: collapsed body, and popped-out (floating)
        # toggled by the header buttons, mirrored to DOM classes by JS.
        dcc.Store(id="video-overlay-collapsed", data=True),
        dcc.Store(id="video-overlay-pip", data=False),
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
        def _chip(text, color="#cfd0d6", bg="rgba(255,255,255,0.06)"):
            return html.Span(text, style={
                "background": bg, "color": color,
                "padding": "2px 9px", "borderRadius": "11px",
                "fontSize": "11px", "fontWeight": "600",
                "whiteSpace": "nowrap"})

        chips: list = [
            _chip(f"Today · {done_today} reviewed", color="#f0f0f5"),
            _chip((f"Streak · {streak} sessions" if streak
                   else "Streak · --"),
                  color=("#30d158" if streak >= 2 else "#a0a0b0"),
                  bg=("rgba(48,209,88,0.12)" if streak >= 2
                      else "rgba(255,255,255,0.06)")),
        ]
        if n_waiting:
            chips.append(_chip(f"{n_waiting} waiting"))
        if oldest_label:
            chips.append(_chip(
                oldest_label, color=heat_color,
                bg=("rgba(255,69,58,0.12)" if heat_color == "#ff453a"
                    else "rgba(255,159,10,0.12)"
                    if heat_color == "#ff9f0a"
                    else "rgba(255,255,255,0.06)")))
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
        Output("video-queue-nav-wrap", "style"),
        Output("video-queue-empty", "children"),
        Input("video-queue-animal", "value"),
        Input("video-queue-position", "data"),
        Input("video-queue-mode", "value"),
        Input("refresh-trigger", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _render_queue_card(animal_value, position, mode, _refresh):
        # nav-wrap visibility: hidden (slim hint instead) until there's a
        # queue to browse; shown once we have rows.
        _HIDE = {"display": "none"}
        _SHOW = {}
        mode = mode or "queue"
        spec = _QUEUE_MODE_BY_VALUE.get(mode, _QUEUE_MODE_BY_VALUE["queue"])
        hint = "Pick an animal above to load your queue."
        if not animal_value:
            return ("", "", "", no_update, _HIDE, hint)
        animal_ids = _animal_ids_from_picker(animal_value)
        if not animal_ids:
            return ("", "", "", no_update, _HIDE, hint)
        email = current_user_email() or ""
        floor = store.review_backlog_floor()
        rows = _fetch_queue_by_mode(store, mode, animal_ids, email, floor,
                                    queue_limit)
        animal_label = animal_ids[0]
        if not rows:
            caught_up = html.Div([
                html.Span("🎉  All caught up — "
                          if mode == "queue" else "",
                          style={"color": "#00CC96", "fontWeight": "600"}),
                html.Span(spec["empty_ok"].format(a=animal_label),
                          style={"color": "#888"}),
            ], style={"fontSize": "12px"})
            return ("", "", "", 0, _HIDE, caught_up)
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
        title = spec["title"].format(n=total, a=animal_label)
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
            return (title, body, position_str, clamped, _SHOW, "")
        return (title, body, position_str, no_update, _SHOW, "")

    @app.callback(
        Output("video-queue-position", "data",
                allow_duplicate=True),
        Input("video-queue-prev-btn", "n_clicks"),
        Input("video-queue-next-btn", "n_clicks"),
        Input("video-queue-animal", "value"),
        Input("video-queue-mode", "value"),
        State("video-queue-position", "data"),
        prevent_initial_call=True,
    )
    def _on_queue_arrow(_prev, _next, _animal, _mode, pos):
        trig = callback_context.triggered_id
        if trig in ("video-queue-animal", "video-queue-mode"):
            # Reset position to head when the animal OR the mode changes.
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
        State("video-queue-mode", "value"),
        prevent_initial_call=True,
    )
    def _on_queue_load(n, animal_value, position, mode):
        if not n or not animal_value:
            return no_update, no_update
        animal_ids = _animal_ids_from_picker(animal_value)
        if not animal_ids:
            return no_update, no_update
        email = current_user_email() or ""
        floor = store.review_backlog_floor()
        rows = _fetch_queue_by_mode(store, mode or "queue", animal_ids, email,
                                    floor, queue_limit)
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

    # The "Needs scoring" pool: quick-flagged files awaiting full
    # scoring. Reuses the video-queue-item id type so a click loads the
    # file through the existing _load_from_queue handler (which prefills
    # the draft events on file change).
    @app.callback(
        Output("video-needs-scoring-list", "children"),
        Input("video-queue-animal", "value"),
        Input("refresh-trigger", "data"),
        Input("video-file-dropdown", "value"),
    )
    def _render_needs_scoring(animal_value, _refresh, active_file_id):
        animal = _ma_animal_from_picker(animal_value)
        if not animal:
            return html.Div(
                "Pick an animal to see its quick-flag pool.",
                style={"color": "#888", "fontSize": "11px",
                        "padding": "8px"})
        try:
            rows = store.files_needing_scoring_for_animal(str(animal))
        except Exception as e:
            logger.warning("files_needing_scoring failed: %s", e)
            rows = []
        if not rows:
            return html.Div(
                "Nothing flagged for scoring.",
                style={"color": "#888", "fontSize": "11px",
                        "padding": "8px"})
        from datetime import datetime as _dt
        items = []
        for r in rows:
            ts_raw = r.get("chunk_datetime") or ""
            try:
                ts_label = _dt.strptime(
                    ts_raw, "%Y_%m_%d__%H_%M_%S").strftime(
                        "%Y-%m-%d  %H:%M")
            except ValueError:
                ts_label = ts_raw
            try:
                n_ev = len(json.loads(r.get("markers_json") or "[]"))
            except (json.JSONDecodeError, TypeError):
                n_ev = 0
            fname = os.path.basename(r.get("file_path") or "")
            is_active = (active_file_id is not None
                          and int(r["file_id"]) == int(active_file_id))
            items.append(html.Button([
                html.Div([
                    html.Span(ts_label, style={
                        "color": "#f0f0f5", "fontWeight": "600",
                        "fontSize": "12px"}),
                    html.Span(
                        f"🚩 {n_ev} onset{'' if n_ev == 1 else 's'} · "
                        f"finish scoring",
                        title="You flagged an EEG onset here but haven't "
                              "scored type + Racine + landmarks. Open to "
                              "complete it.",
                        style={"color": "#f0b429", "fontSize": "11px",
                               "fontWeight": "600", "marginLeft": "auto"}),
                ], style={"display": "flex", "width": "100%"}),
                html.Span(fname, title=fname, style={
                    "color": "#6f7080", "fontSize": "10px",
                    "fontFamily": "ui-monospace, monospace",
                    "maxWidth": "100%", "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap"}),
            ], id={"type": "video-needs-item",
                    "file_id": int(r["file_id"])},
                n_clicks=0,
                style={"display": "flex", "flexDirection": "column",
                        "alignItems": "flex-start", "width": "100%",
                        "border": "none",
                        "background": ("rgba(240,180,41,0.10)"
                                        if is_active else "transparent"),
                        "color": "#cfd0d6", "padding": "6px 10px",
                        "cursor": "pointer",
                        "borderBottom":
                            "1px solid rgba(255,255,255,0.04)",
                        "borderLeft": ("3px solid #f0b429" if is_active
                                        else "3px solid transparent"),
                        "textAlign": "left"}))
        return items

    @app.callback(
        Output("video-queue-list", "children"),
        Input("video-queue-animal", "value"),
        Input("video-queue-mode", "value"),
        Input("refresh-trigger", "data"),
        Input("video-file-dropdown", "value"),
    )
    def _render_queue(animal_value, mode, _refresh, active_file_id):
        """Render the per-animal queue list for the active mode. Each item is a
        Button with a pattern-matching id so one downstream callback
        handles all clicks. The button whose ``file_id`` matches
        ``active_file_id`` (the dropdown's current value) gets a
        left-border accent + a tinted background so the reviewer
        can always tell which recording is loaded."""
        if not animal_value:
            return html.Div(
                "Pick an animal above to see your queue.",
                style={"color": "#888", "fontSize": "12px",
                        "padding": "10px"})
        mode = mode or "queue"
        spec = _QUEUE_MODE_BY_VALUE.get(mode, _QUEUE_MODE_BY_VALUE["queue"])
        # Resolve animal_value to a list of animal ids (single
        # selection or the unassigned pool sentinel).
        animal_ids = _animal_ids_from_picker(animal_value)
        if not animal_ids:
            return html.Div(
                "Pick an animal above to see your queue.",
                style={"color": "#888", "fontSize": "12px",
                        "padding": "10px"})
        email = current_user_email() or ""
        floor = store.review_backlog_floor()
        # The "Show all timestamps" list shows the WHOLE pool, not
        # just the first queue_limit (that cap is for the prev/next
        # browse card). High ceiling so every recording is listed.
        rows = _fetch_queue_by_mode(store, mode, animal_ids, email, floor,
                                    _QUEUE_LIST_MAX)
        if not rows:
            return html.Div([
                html.Div("🎉  You're all caught up!"
                         if mode == "queue" else "Nothing here",
                         style={"color": "#00CC96", "fontWeight": "600",
                                 "fontSize": "13px", "marginBottom": "4px"}),
                html.Div(
                    spec["empty"].format(a=animal_ids[0]),
                    style={"color": "#888", "fontSize": "11px"}),
            ], style={"padding": "14px", "textAlign": "center"})
        from datetime import datetime as _dt
        now = _dt.now()
        # Two-level grouping: day (expandable) -> hours within it.
        day_groups: "OrderedDict[str, dict]" = OrderedDict()
        for r in rows:
            ts_raw = r.get("chunk_datetime") or ""
            try:
                ts = _dt.strptime(ts_raw,
                                    "%Y_%m_%d__%H_%M_%S")
                age_days = (now - ts).total_seconds() / 86400
                day_key = ts.strftime("%Y-%m-%d")
                hour_label = ts.strftime("%H:%M")
            except ValueError:
                age_days = 0
                day_key = "undated"
                hour_label = ts_raw
            warn = age_days >= warn_age_days
            dur = r.get("duration_sec") or 0
            dur_h = dur / 3600.0 if dur else 0
            fname = os.path.basename(r.get("file_path") or "")
            is_active = (active_file_id is not None
                          and int(r["id"]) == int(active_file_id))
            btn_style: dict = {
                "display": "flex", "flexDirection": "column",
                "alignItems": "flex-start",
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
            btn = html.Button([
                html.Div([
                    html.Span(hour_label, style={
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
                ], style={"display": "flex", "alignItems": "center",
                           "width": "100%"}),
                html.Span(
                    fname, title=fname,
                    style={"color": "#6f7080", "fontSize": "10px",
                            "fontFamily": "ui-monospace, SF Mono, "
                                           "monospace",
                            "maxWidth": "100%",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                            "whiteSpace": "nowrap"}),
            ], id={"type": "video-queue-item",
                    "file_id": int(r["id"])},
                n_clicks=0, style=btn_style)
            g = day_groups.setdefault(
                day_key, {"items": [], "active": False})
            g["items"].append(btn)
            if is_active:
                g["active"] = True

        # One collapsible section per day; auto-open the day holding
        # the loaded recording, collapse the rest.
        out = []
        for day_key, g in day_groups.items():
            n_rec = len(g["items"])
            out.append(html.Details([
                html.Summary([
                    html.Span(day_key, style={
                        "color": "#f0f0f5", "fontWeight": "600",
                        "fontSize": "12px"}),
                    html.Span(
                        f"  ·  {n_rec} recording"
                        f"{'' if n_rec == 1 else 's'}",
                        style={"color": "#888", "fontSize": "11px"}),
                ], style={"cursor": "pointer", "padding": "6px 8px",
                           "userSelect": "none"}),
                html.Div(g["items"]),
            ], open=g["active"],
               style={"borderBottom":
                          "1px solid rgba(255,255,255,0.06)"}))
        return out

    @app.callback(
        Output("video-session-dropdown", "value",
                allow_duplicate=True),
        Output("video-file-dropdown", "value",
                allow_duplicate=True),
        Input({"type": "video-queue-item", "file_id": ALL},
                "n_clicks"),
        Input({"type": "video-needs-item", "file_id": ALL},
                "n_clicks"),
        prevent_initial_call=True,
    )
    def _load_from_queue(_queue_clicks, _needs_clicks):
        """Translate a queue OR needs-scoring button click into setting the
        existing session + file dropdowns, which cascades through every
        downstream callback unchanged.

        The queue "Show all timestamps" list and the dedicated "Needs scoring"
        card use DIFFERENT id types (video-queue-item vs video-needs-item) on
        purpose: in needs_scoring mode both lists render the SAME files, and a
        shared id type produced DUPLICATE component ids in the DOM, which breaks
        Dash's pattern-match click routing -- clicking a needs-scoring file did
        nothing. Distinct types keep every button uniquely addressable.
        """
        # Act ONLY on a genuine click: SOME triggered input's n_clicks must be
        # >= 1. Both lists re-render on refresh-trigger / file-dropdown change,
        # re-creating buttons with n_clicks=0; the old `any(n_clicks)` guard
        # fired on those re-renders too (reloading the current file). Gate on
        # the triggered VALUE -- and scan ALL triggered entries, not just
        # triggered[0], since a coincident re-render can order a value=0 entry
        # first and mask the real click.
        if not any((t.get("value") or 0)
                   for t in (callback_context.triggered or [])):
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
        logger.info("queue/needs-scoring click -> load file_id=%s session=%s",
                    int(file_id), row["session_dir"])
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
        State("video-queue-animal", "value"),
        prevent_initial_call=True,
    )
    def _hotkey_undo(ev, _click, undo, picker_value):
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
        ok = store.reopen_review(
            file_id, email, animal_id=_ma_animal_from_picker(picker_value))
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

    # Re-scope the events draft when the FILE or the scored CHANNEL
    # (=> animal) changes. Each (file, animal) keeps its own draft: the
    # outgoing animal's onsets are stashed and the incoming animal's are
    # restored (from the in-session stash, else prefilled from a saved
    # 'needs_scoring' draft). So animal A's onsets never bleed into B on a
    # multi-animal recording. Driven by the channel, matching how reviews
    # are filed (_animal_for_channel), not the queue picker.
    @app.callback(
        Output("video-events-store", "data", allow_duplicate=True),
        Output("video-events-by-animal", "data"),
        Output("video-events-current-key", "data"),
        Input("video-file-dropdown", "value"),
        Input("video-channel-dropdown", "value"),
        State("video-events-store", "data"),
        State("video-events-by-animal", "data"),
        State("video-events-current-key", "data"),
        prevent_initial_call=True,
    )
    def _rescope_events(file_id, channel, cur_events, by_animal, cur_key):
        by_animal = dict(by_animal or {})
        if not file_id:
            return [], by_animal, None
        animal = _animal_for_channel(store, file_id, channel)
        if not animal:
            # Non-animal channel (e.g. stim copy) -> don't swap; the draft
            # stays put (it can't be saved without an animal channel).
            return no_update, no_update, no_update
        new_key = f"{int(file_id)}:{animal}"
        if new_key == cur_key:
            return no_update, no_update, no_update
        # Stash the outgoing draft under its key.
        if cur_key:
            by_animal[cur_key] = list(cur_events or [])
        # Restore: in-session stash first, else the saved needs_scoring draft.
        if new_key in by_animal:
            new_events = by_animal[new_key]
        else:
            new_events = []
            try:
                latest = store.get_review_state(int(file_id),
                                                 animal_id=animal)
            except Exception:
                latest = None
            if latest and latest.get("status") == "needs_scoring":
                try:
                    drafts = json.loads(latest.get("markers_json") or "[]")
                except (json.JSONDecodeError, TypeError):
                    drafts = []
                if isinstance(drafts, list):
                    new_events = drafts
        return new_events, by_animal, new_key

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
        # Patch (not dict(state) write-back) so a near-simultaneous
        # _mark_legs_done can't clobber this with a stale whole-dict snapshot.
        if not state:
            return no_update
        patched = Patch()
        patched["lfp"] = "loading"
        patched["hilbert"] = "loading"
        return patched

    # ---- Instant blank-on-load (clientside) ---- #
    # The moment the reviewer prompts a new file, wipe BOTH plot figures
    # to a "Loading…" placeholder straight from the browser -- before the
    # server even starts. Previously the stale trace from the PREVIOUS
    # recording lingered: all three render callbacks share the file
    # dropdown as their Input, but the heavy video render runs first, so
    # the LFP/Hilbert dcc.Loading overlays (and their real figures) only
    # arrived AFTER the video finished. A clientside callback runs in the
    # browser the instant the value changes, so the transition to a
    # loading state is immediate and unmistakable. The server
    # _update_lfp / _update_analysis overwrite this placeholder with the
    # real trace when they finish; the dcc.Loading spinner animates on top
    # meanwhile.
    app.clientside_callback(
        """
        function(file_id) {
            var NU = window.dash_clientside.no_update;
            if (!file_id) { return [NU, NU]; }
            function mk() {
                return {
                    data: [],
                    layout: {
                        height: 220,
                        plot_bgcolor: '#13131f', paper_bgcolor: '#13131f',
                        xaxis: {visible: false}, yaxis: {visible: false},
                        margin: {l: 20, r: 20, t: 20, b: 20},
                        annotations: [{
                            text: 'Loading…', showarrow: false,
                            x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
                            font: {size: 12, color: '#a0a0b0'}
                        }]
                    }
                };
            }
            return [mk(), mk()];
        }
        """,
        Output("video-lfp-trace", "figure", allow_duplicate=True),
        Output("video-analysis-trace", "figure", allow_duplicate=True),
        Input("video-file-dropdown", "value"),
        prevent_initial_call=True,
    )

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
            "padding": "8px 14px",
            "borderRadius": "999px",
            "background": "rgba(19, 19, 31, 0.92)",
            "border": "1px solid rgba(94, 124, 226, 0.45)",
            "fontSize": "12px",
            "color": "#cfd0d6",
            "width": "fit-content",
            # Float bottom-right so it follows the reviewer as they
            # scroll, instead of sitting inline at the top.
            "position": "fixed",
            "right": "20px",
            "bottom": "20px",
            "zIndex": "10000",
            "boxShadow": "0 8px 24px rgba(0, 0, 0, 0.45)",
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
    # Single owner of the leg "done" transitions. Three separate
    # done-watchers each did read-modify-write on video-load-state
    # with allow_duplicate; when the LFP and Hilbert render tokens
    # both landed in the same Dash batch (the normal case once the
    # tokens made the marks reliable), the two callbacks read the
    # SAME state snapshot and each wrote its own copy back -- the
    # last write clobbered the other, leaving one leg stuck on
    # "loading" forever (LFP usually won; Hilbert hung). Collapsing
    # them into one callback makes the multi-leg transition a single
    # atomic write keyed off callback_context.triggered, so both
    # legs flip together. Watches the cheap per-render TOKENS (not
    # the figures) to keep the 10 Hz cursor Patch from re-firing
    # this -- the original OOM guard.
    @app.callback(
        Output("video-load-state", "data",
                allow_duplicate=True),
        Input("video-player-container", "children"),
        Input("video-lfp-loadtoken", "data"),
        Input("video-hilbert-loadtoken", "data"),
        State("video-load-state", "data"),
        prevent_initial_call=True,
    )
    def _mark_legs_done(_children, _lfp_token, _hil_token, state):
        if not state:
            return no_update
        triggered = {
            t["prop_id"].split(".")[0]
            for t in callback_context.triggered
        }
        # Patch only the leg(s) that actually flipped (guarding on the CURRENT
        # state value read from State), so a concurrent _start_channel_load
        # writing the other legs isn't clobbered by a whole-dict write-back.
        patched = Patch()
        changed = False
        if ("video-player-container" in triggered
                and state.get("video") == "loading"):
            patched["video"] = "done"
            changed = True
        if ("video-lfp-loadtoken" in triggered
                and state.get("lfp") == "loading"):
            patched["lfp"] = "done"
            changed = True
        if ("video-hilbert-loadtoken" in triggered
                and state.get("hilbert") == "loading"):
            patched["hilbert"] = "done"
            changed = True
        return patched if changed else no_update

    # Soft-claim: when a reviewer loads a recording, claim it so it
    # drops out of OTHER reviewers' queues for the TTL window (see
    # Store.claim_file / CLAIM_TTL_MINUTES). Claim on load only -- no
    # heartbeat, so a reviewer who leaves a tab open and walks away
    # releases the file once the TTL lapses rather than holding it
    # indefinitely. The matching release happens on submit.
    @app.callback(
        Output("video-claim-sink", "data"),
        Input("video-file-dropdown", "value"),
        State("video-queue-animal", "value"),
        prevent_initial_call=True,
    )
    def _claim_on_load(file_id, picker_value):
        if not file_id:
            return no_update
        email = current_user_email()
        if email:
            try:
                store.claim_file(
                    int(file_id), email,
                    animal_id=_ma_animal_from_picker(picker_value))
            except Exception as e:
                logger.warning("claim_file failed: %s", e)
        return no_update

    # ---- Now-viewing banner + hour stepping (orientation) ---- #
    @app.callback(
        Output("video-now-viewing", "children"),
        Output("video-prev-hour-btn", "disabled"),
        Output("video-next-hour-btn", "disabled"),
        Input("video-file-dropdown", "value"),
        Input("video-queue-animal", "value"),
        Input("video-channel-dropdown", "value"),
    )
    def _render_now_viewing(file_id, picker_value, channel):
        if not file_id:
            return ("No recording loaded -- pick one above.",
                    True, True)
        prefer = _ma_animal_from_picker(picker_value)
        info = _now_viewing_info(store, int(file_id), prefer)
        if not info:
            return ("Recording loaded.", True, True)
        # The animal an onset is FILED UNDER = the scored channel's animal (the
        # electrode is ground truth), NOT the queue picker. Show that, so a
        # multi-animal mismatch is obvious.
        scored = (_animal_for_channel(store, int(file_id), channel)
                  if channel is not None else None) or info["animal"]
        parts = [
            html.Span("Now viewing (scoring): ",
                      style={"color": "#a0a0b0", "fontWeight": "400"}),
            html.Span(scored, style={"color": "#5e7ce2",
                                     "fontWeight": "700"}),
            html.Span(f"  ·  {info['date']}  ·  {info['time']}"),
            html.Span(f"   (hour {info['index']} of {info['total']})",
                      style={"color": "#a0a0b0", "fontWeight": "400",
                             "fontSize": "11px"}),
        ]
        # If the reviewer entered via a different animal's queue, flag it.
        if prefer and prefer != scored:
            parts.append(html.Span(f"   · queue: {prefer}",
                                   style={"color": "#ff9f0a",
                                          "fontWeight": "400",
                                          "fontSize": "11px"}))
        return (html.Span(parts), info["prev_id"] is None,
                info["next_id"] is None)

    # ---- "Filed under" banner: which animal onsets go to + legend ---- #
    @app.callback(
        Output("video-filed-under-banner", "children"),
        Input("video-file-dropdown", "value"),
        Input("video-channel-dropdown", "value"),
        Input("video-review-decision", "value"),
    )
    def _render_filed_under(file_id, channel, decision):
        if not file_id or channel is None:
            return ""
        from src.utils.animal import (
            split_animal_electrode, is_animal_channel)
        scored = _animal_for_channel(store, int(file_id), channel)
        session_dir = _session_dir_for_file(store, int(file_id))
        names = (store._channel_names_for_session(session_dir)
                 if session_dir else [])
        try:
            ci = int(channel)
            ch_name = names[ci] if 0 <= ci < len(names) else f"Ch{ci}"
        except (ValueError, TypeError):
            ch_name = f"Ch{channel}"
        # Channel -> animal legend for the (possibly multi-animal) recording.
        groups: dict = {}
        order: list = []
        for i, n in enumerate(names):
            if isinstance(n, str) and is_animal_channel(n):
                a, _ = split_animal_electrode(n)
                if a not in groups:
                    groups[a] = []
                    order.append(a)
                groups[a].append(f"Ch{i}")
        multi = len(order) > 1
        legend = " · ".join(f"{a} ({', '.join(groups[a])})" for a in order)
        rows: list = []
        if not scored:
            rows.append(html.Span(
                f"Scored channel Ch{channel} · {ch_name} isn't an animal "
                "electrode — pick an animal's brain channel above.",
                style={"color": "#ff453a", "fontSize": "12px"}))
        else:
            rows.append(html.Div([
                html.Span("⚠ " if multi else "✓ ",
                          style={"color": "#ff9f0a" if multi else "#30d158"}),
                html.Span("These onsets file under ",
                          style={"color": "#a0a0b0", "fontSize": "12px"}),
                html.Span(scored, style={
                    "color": "#ffd60a" if multi else "#30d158",
                    "fontWeight": "700", "fontSize": "13px"}),
                html.Span(
                    f"  — scored channel Ch{channel} · {ch_name}."
                    + ("  Switch the brain channel above to file under a "
                       "different animal in this recording." if multi else ""),
                    style={"color": "#a0a0b0", "fontSize": "12px"}),
            ]))
        if multi:
            rows.append(html.Div(
                f"Recording animals: {legend}",
                style={"color": "#8a8a99", "fontSize": "11px",
                       "marginTop": "3px"}))
        border = ("1px solid rgba(255,159,10,0.30)" if multi
                  else "1px solid rgba(48,209,88,0.22)")
        bg = ("rgba(255,159,10,0.10)" if multi else "rgba(48,209,88,0.06)")
        return html.Div(rows, style={
            "padding": "6px 10px", "borderRadius": "6px",
            "background": bg, "border": border})

    # Multi-animal sessions: switching the reviewed animal on an
    # already-loaded file re-targets the LFP/Hilbert to THAT animal's
    # electrode (the per-file channel default only re-runs on a file
    # change, so without this the trace stayed on the prior animal).
    @app.callback(
        Output("video-channel-dropdown", "value",
                allow_duplicate=True),
        Input("video-queue-animal", "value"),
        State("video-file-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _retarget_channel_on_animal(picker_value, file_id):
        if not file_id:
            return no_update
        animal = _ma_animal_from_picker(picker_value)
        if not animal:
            return no_update
        sd = _session_dir_for_file(store, int(file_id))
        if not sd:
            return no_update
        try:
            elecs = store.electrodes_for_animal_in_session(sd, animal)
        except Exception:
            return no_update
        if elecs:
            return int(elecs[0]["channel_index"])
        return no_update

    @app.callback(
        Output("video-file-dropdown", "value",
                allow_duplicate=True),
        Input("video-prev-hour-btn", "n_clicks"),
        Input("video-next-hour-btn", "n_clicks"),
        State("video-file-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _step_hour(_prev, _next, file_id):
        if not file_id:
            return no_update
        # Only act on a real click (pattern: button recreation /
        # initial fires carry n_clicks 0/None).
        if not any((t.get("value") or 0)
                    for t in (callback_context.triggered or [])):
            return no_update
        info = _now_viewing_info(store, int(file_id))
        if not info:
            return no_update
        trig = callback_context.triggered_id
        target = (info["prev_id"] if trig == "video-prev-hour-btn"
                   else info["next_id"])
        return target if target is not None else no_update

    # ---- Detection threshold <-> Mass Analyze cutoff sync ---- #
    # The scoring-view "Detection threshold" input and the Mass Analyze
    # cutoff are the same logical value; keep them mirrored so typing in
    # one (or click-to-set on the plot, which writes the MA cutoff) moves
    # the Hilbert threshold line everywhere. Echo-guarded (skip when the
    # values already match) to break the A->B->A loop.
    def _near(a, b) -> bool:
        try:
            return a is not None and b is not None \
                and abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False

    @app.callback(
        Output("video-ma-cutoff-input", "value",
                allow_duplicate=True),
        Input("video-threshold-input", "value"),
        State("video-ma-cutoff-input", "value"),
        prevent_initial_call=True,
    )
    def _sync_threshold_to_ma(v, cur):
        if v is None or _near(v, cur):
            return no_update
        return v

    @app.callback(
        Output("video-threshold-input", "value",
                allow_duplicate=True),
        Input("video-ma-cutoff-input", "value"),
        State("video-threshold-input", "value"),
        prevent_initial_call=True,
    )
    def _sync_ma_to_threshold(v, cur):
        if v is None or _near(v, cur):
            return no_update
        return v

    # ---- Curate candidate peaks: click a triangle to reject ---- #
    @app.callback(
        Output("video-rejected-version", "data",
                allow_duplicate=True),
        Input("video-analysis-trace", "clickData"),
        State("video-remove-peaks-mode", "value"),
        State("video-file-dropdown", "value"),
        State("video-channel-dropdown", "value"),
        State("video-rejected-version", "data"),
        prevent_initial_call=True,
    )
    def _reject_peak(click_data, mode, file_id, channel, ver):
        if "on" not in (mode or []):
            return no_update
        if not file_id or channel is None or not click_data:
            return no_update
        pts = click_data.get("points") or []
        if not pts:
            return no_update
        p = pts[0]
        # Only the candidate-marker trace (curveNumber 1) is a peak;
        # curveNumber 0 is the envelope line itself.
        if p.get("curveNumber") != 1:
            return no_update
        x = p.get("x")
        if x is None:
            return no_update
        try:
            store.reject_peak(int(file_id), int(channel), float(x),
                               current_user_email())
        except Exception as e:
            logger.warning("reject_peak failed: %s", e)
            return no_update
        return int(ver or 0) + 1

    @app.callback(
        Output("video-rejected-version", "data",
                allow_duplicate=True),
        Input("video-undo-reject-btn", "n_clicks"),
        State("video-file-dropdown", "value"),
        State("video-channel-dropdown", "value"),
        State("video-rejected-version", "data"),
        prevent_initial_call=True,
    )
    def _undo_reject(n_clicks, file_id, channel, ver):
        if not n_clicks or not file_id or channel is None:
            return no_update
        try:
            removed = store.unreject_last_peak(
                int(file_id), int(channel))
        except Exception as e:
            logger.warning("unreject_last_peak failed: %s", e)
            return no_update
        return int(ver or 0) + 1 if removed else no_update

    # Layout presets: switch the Step-2 grid class clientside (the CSS
    # classes carry !important grid-template-columns that override the
    # inline default). No server round-trip.
    app.clientside_callback(
        """
        function(_a, _b, _c) {
            var ctx = window.dash_clientside.callback_context;
            var nu = window.dash_clientside.no_update;
            if (!ctx || !ctx.triggered || !ctx.triggered.length) {
                return [nu, nu, nu, nu];
            }
            var id = ctx.triggered[0].prop_id.split('.')[0];
            var grid = 'layout-5050', active = 'video-layout-5050';
            if (id === 'video-layout-bigvideo') {
                grid = 'layout-bigvideo'; active = 'video-layout-bigvideo';
            } else if (id === 'video-layout-biglfp') {
                grid = 'layout-biglfp'; active = 'video-layout-biglfp';
            }
            // Reflect the active preset on the buttons (Nielsen #1:
            // visibility of system status) -- the clicked one gets .active.
            var cls = function(b){
                return 'video-layout-btn' + (b === active ? ' active' : '');
            };
            return [grid, cls('video-layout-bigvideo'),
                    cls('video-layout-5050'), cls('video-layout-biglfp')];
        }
        """,
        Output("video-step2-grid", "className"),
        Output("video-layout-bigvideo", "className"),
        Output("video-layout-5050", "className"),
        Output("video-layout-biglfp", "className"),
        Input("video-layout-bigvideo", "n_clicks"),
        Input("video-layout-5050", "n_clicks"),
        Input("video-layout-biglfp", "n_clicks"),
        prevent_initial_call=True,
    )

    # Fullscreen: fill the screen with the Step-2 grid (video). CSS in
    # theme.css (#video-step2-grid:fullscreen) makes the video fill the
    # viewport and floats the LFP/Hilbert column as a bottom-right card.
    # A one-time fullscreenchange listener dispatches a window resize so
    # the Plotly traces redraw at the card size (and on exit).
    app.clientside_callback(
        """
        function(n) {
            if (!window.__qcFsInit) {
                window.__qcFsInit = true;
                document.addEventListener('fullscreenchange', function() {
                    setTimeout(function() {
                        window.dispatchEvent(new Event('resize'));
                    }, 220);
                });
            }
            if (!n) return window.dash_clientside.no_update;
            var el = document.getElementById('video-step2-grid');
            if (!el) return window.dash_clientside.no_update;
            if (document.fullscreenElement) {
                document.exitFullscreen();
            } else if (el.requestFullscreen) {
                el.requestFullscreen();
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("video-fs-sink", "children"),
        Input("video-fullscreen-btn", "n_clicks"),
        prevent_initial_call=True,
    )

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

    # ---- Step 4: reset marker store + decision when the file OR the
    # scored channel (=> animal) changes ---- #
    # Scoped by the channel's animal (not the picker) and re-fires on
    # channel change, so a multi-animal recording shows each animal's own
    # saved decision/note/markers -- consistent with _rescope_events.
    @app.callback(
        Output("video-review-marker-store", "data",
                allow_duplicate=True),
        Output("video-review-decision", "value",
                allow_duplicate=True),
        Output("video-review-note", "value"),
        Output("video-review-status", "children"),
        Input("video-file-dropdown", "value"),
        Input("video-channel-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _reset_review_panel(file_id, channel):
        if not file_id:
            return [], None, "", ""
        email = current_user_email() or ""
        animal = _animal_for_channel(store, file_id, channel)
        if not animal:
            # Non-animal channel (e.g. stim copy): nothing to scope to.
            return [], None, "", "Pick an animal's brain channel."
        # Quick-flagged file: resume in events mode so the draft
        # events (prefilled by _reset_events_on_file_change) show.
        latest = store.get_review_state(int(file_id), animal_id=animal)
        if latest and latest.get("status") == "needs_scoring":
            return ([], "has_events", latest.get("note") or "",
                    "Resuming quick-flagged events -- finish "
                    "scoring each, then Mark recording done.")
        existing = store.get_review_state_by_user(int(file_id),
                                                     email, animal_id=animal)
        if existing and existing["status"] in (
                "no_events", "has_events"):
            badge = f"✓ You marked this " \
                     f"{existing['status'].replace('_', ' ')} " \
                     f"on {existing['updated_at'][:16]}."
            return ([], existing["status"], existing.get("note") or "",
                    badge)
        # Any other user already finalised it (for this animal)?
        other = store.get_review_state(int(file_id), animal_id=animal)
        if other and other["status"] in (
                "no_events", "has_events") and (
                other["user_email"] != email.lower()):
            badge = (f"Already reviewed by "
                      f"{other['user_email']} on "
                      f"{other['updated_at'][:16]}.")
            return [], None, "", badge
        return [], None, "", "Not reviewed yet."

    # (Generic click-to-onset removed: in Seek mode a click only seeks
    # the video. Onset placement is the structured-events flow -- the
    # events panel + "Set on LFP" landmark drop in video_events.py.)

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
                "No quick marks. To record an onset that saves, use "
                "an event card above: \"+ Add event\" -> \"Set on "
                "plot\", then click the LFP or Hilbert trace.")
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

    # Per-field landmark colors -- the single source of truth lives in
    # video_events.py so the plot verticals and the on-screen color
    # legend (events panel) can never drift apart.
    LANDMARK_COLORS: dict = _events.LANDMARK_COLORS

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
        Output("video-analysis-trace", "figure",
                allow_duplicate=True),
        Input("video-events-store", "data"),
        State("video-analysis-trace", "figure"),
        prevent_initial_call=True,
    )
    def _render_marker_shapes_hilbert(events, fig):
        """Same landmark verticals on the Hilbert envelope. The
        Hilbert keeps two reserved shapes -- shapes[0] cursor,
        shapes[1] the dashed detection threshold -- so preserve both
        and append landmarks past them."""
        if not fig:
            return no_update
        existing = (fig.get("layout") or {}).get("shapes") or []
        new_shapes: list[dict] = list(existing[:2])  # cursor+threshold
        for i, e in enumerate(events or []):
            if i >= 16:  # NASA Rule 3
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
                    "type": "line", "xref": "x", "yref": "paper",
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
        Output("video-ma-pool-cursor", "data",
                allow_duplicate=True),
        Input("video-review-save-btn", "n_clicks"),
        State("video-file-dropdown", "value"),
        State("video-review-decision", "value"),
        State("video-review-marker-store", "data"),
        State("video-review-note", "value"),
        State("video-queue-animal", "value"),
        State("video-events-store", "data"),
        State("video-channel-dropdown", "value"),
        State("video-ma-pools-view", "data"),
        State("video-ma-pool-cursor", "data"),
        prevent_initial_call=True,
    )
    def _save_review(n_clicks, file_id, decision, markers, note,
                      animal_value, events, channel,
                      pools_view, pool_cursor):
        nop = (no_update,) * 8
        if not n_clicks or not file_id:
            return ("Pick a recording first." if n_clicks
                    else no_update, *nop[1:])
        if decision not in ("no_events", "has_events"):
            return ("Pick \"No events seen\" or \"Events seen\" "
                     "before saving.",
                    *nop[1:])
        events = list(events or [])
        if decision == "has_events":
            if not events:
                return ("Add at least one event with + Add event "
                         "before saving.",
                        *nop[1:])
            # Gating: every event must be complete (type +
            # required landmarks + Racine 1-8).
            incomplete = [i + 1 for i, e in enumerate(events)
                           if not _events.is_event_complete(e)]
            if incomplete:
                idxs = ", ".join(str(i) for i in incomplete)
                return (f"Event{'s' if len(incomplete) > 1 else ''} "
                         f"{idxs} still need landmarks + Racine "
                         "before saving.",
                        *nop[1:])
        email = current_user_email()
        if not email:
            return ("Not signed in — can't record who reviewed "
                     "this.",
                    *nop[1:])
        # PI verification pipeline (the BHZ event taxonomy plan):
        # every undergrad save lands in 'pending_pi_review'.
        # markers_json stores the canonical event list -- empty
        # for the no-events branch, populated for the events
        # branch. The CSV is NOT written here; that happens
        # exclusively when the PI hits Approve in the Event
        # Verification tab. This is the second-layer safety net
        # the lab asked for.
        markers_payload = events if decision == "has_events" else None
        # The review is filed against the animal whose channel was scored
        # (the onsets are on that electrode), NOT the queue picker -- and we
        # never write a file-wide (animal-less) row.
        animal = _animal_for_channel(store, file_id, channel)
        if not animal:
            return ("Pick the animal's brain channel (Step 1) before "
                     "saving — the review is filed per animal, not "
                     "per file.", *nop[1:])
        try:
            store.mark_review(
                int(file_id), email, "pending_pi_review",
                markers=markers_payload,
                note=(note or None),
                animal_id=animal,
            )
        except Exception as e:
            logger.warning("mark_review failed: %s", e)
            return (f"Save failed: {e}", *nop[1:])
        # Submitted -> drop this animal's soft-claim so the row doesn't linger
        # (the file is now excluded from this animal's queue by its
        # pending_pi_review status anyway; this just keeps file_claim tidy).
        try:
            store.release_claim(int(file_id), animal_id=animal)
        except Exception as e:
            logger.warning("release_claim failed: %s", e)
        from datetime import datetime as _dt
        decision_label = ("0 events"
                           if decision == "no_events"
                           else f"{len(events)} event"
                                 f"{'' if len(events) == 1 else 's'}")
        badge = (f"✓ Saved for PI review at "
                  f"{_dt.now().strftime('%H:%M')}  ·  "
                  f"{decision_label}.  "
                  f"The PI will verify before final CSV export.")
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
        # Auto-advance: if the reviewer is stepping through a pool, the
        # pool is the primary source -- advance to the next POOL file and
        # keep the pool cursor in sync. Otherwise fall back to the FIFO
        # queue.
        pooled = _next_in_pool(store, pools_view, pool_cursor,
                                 int(file_id))
        if pooled is not None:
            p_session, p_file, new_cursor = pooled
            return (badge, [], None, "",
                    p_session, p_file, undo_payload, new_cursor)
        next_session, next_file = _resolve_next_in_queue(
            store, animal_value, int(file_id), email,
            queue_limit=queue_limit,
        )
        # If we found a next file, push it; otherwise leave the
        # dropdowns alone so the reviewer sees the queue empty
        # state via _render_queue.
        if next_file is None:
            return (badge, [], None, "",
                    no_update, no_update, undo_payload, no_update)
        return (badge, [], None, "",
                next_session, next_file, undo_payload, no_update)

    # "EEG onset -> CSV + flag": write a preliminary CSV row now (partial
    # events tolerated) AND keep the file in the Needs-scoring pool, so an
    # early EEG-onset read is usable immediately without shelving the file
    # as complete.
    @app.callback(
        Output("video-review-status", "children",
                allow_duplicate=True),
        Output("video-events-store", "data",
                allow_duplicate=True),
        Output("video-session-dropdown", "value",
                allow_duplicate=True),
        Output("video-file-dropdown", "value",
                allow_duplicate=True),
        Output("video-ma-pool-cursor", "data",
                allow_duplicate=True),
        Input("video-review-partial-btn", "n_clicks"),
        State("video-file-dropdown", "value"),
        State("video-events-store", "data"),
        State("video-channel-dropdown", "value"),
        State("video-queue-animal", "value"),
        State("video-ma-pools-view", "data"),
        State("video-ma-pool-cursor", "data"),
        State("video-ma-cutoff-input", "value"),
        State("video-ma-auc-threshold-input", "value"),
        State("video-auc-window-input", "value"),
        prevent_initial_call=True,
    )
    def _partial_export(n_clicks, file_id, events, channel,
                         animal_value, pools_view, pool_cursor,
                         cutoff, auc_threshold, auc_window):
        if not n_clicks or not file_id:
            return (no_update,) * 5
        email = current_user_email()
        if not email:
            return ("Not signed in -- can't export.",
                    no_update, no_update, no_update, no_update)
        events = list(events or [])
        has_eo = any(e.get("EO_sec") not in (None, "") for e in events)
        if not has_eo:
            return ("Drop at least one EEG onset (EO) before exporting.",
                    no_update, no_update, no_update, no_update)
        # Both the CSV row AND the needs-scoring flag are scoped to the
        # scored channel's animal -- one source, never file-wide.
        animal = _animal_for_channel(store, file_id, channel)
        if not animal:
            return ("Pick the animal's brain channel (Step 1) before "
                     "exporting — onsets are filed per animal.",
                    no_update, no_update, no_update, no_update)
        try:
            n_rows, csv_name = _export_partial_csv(
                store, config, int(file_id), channel, events,
                cutoff=cutoff, auc_threshold=auc_threshold,
                auc_window=auc_window)
        except Exception as e:  # noqa: BLE001 -- surface, never crash UI
            logger.warning("partial CSV export failed: %s", e)
            return (f"Partial export failed: {e}",
                    no_update, no_update, no_update, no_update)
        # Keep the file flagged: drafts + needs_scoring (same pool as the
        # quick-flag path), with a note recording the preliminary export.
        drafts = [{**e, "draft": True} for e in events]
        try:
            store.mark_review(
                int(file_id), email, "needs_scoring", markers=drafts,
                note=f"Partial CSV exported (EEG onset) to {csv_name}; "
                     "needs full scoring.", animal_id=animal)
            store.release_claim(int(file_id), animal_id=animal)
        except Exception as e:
            logger.warning("partial-export mark_review failed: %s", e)
            return (f"Exported to {csv_name} but flagging failed: {e}",
                    no_update, no_update, no_update, no_update)
        from datetime import datetime as _dt
        badge = (f"Exported {n_rows} row{'' if n_rows == 1 else 's'} to "
                 f"{csv_name} at {_dt.now().strftime('%H:%M')}  ·  kept in "
                 "'Needs scoring' for full scoring.")
        pooled = _next_in_pool(store, pools_view, pool_cursor, int(file_id))
        if pooled is not None:
            p_session, p_file, new_cursor = pooled
            return (badge, [], p_session, p_file, new_cursor)
        next_session, next_file = _resolve_next_in_queue(
            store, animal_value, int(file_id), email,
            queue_limit=queue_limit)
        if next_file is None:
            # No file to advance to -> STAY on this one. Don't clear the events
            # store (the [] paths above are safe only because they hand off to a
            # new file whose drafts _rescope_events reloads); clearing here would
            # blank the onsets the reviewer just exported and can still edit.
            return (badge, no_update, no_update, no_update, no_update)
        return (badge, [], next_session, next_file, no_update)

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
        Output("video-lfp-loadtoken", "data"),
        Input("video-file-dropdown", "value"),
        Input("video-channel-dropdown", "value"),
        Input("video-apply-filter-btn", "n_clicks"),
        State("video-filter-hp", "value"),
        State("video-filter-lp", "value"),
        State("video-filter-notch", "value"),
        State("video-filter-smooth", "value"),
        Input("video-blank-raw", "value"),
        State("video-events-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _update_lfp(file_id, channel, _n_apply,
                     hp, lp, notch, smooth_ms, blank_raw, events):
        # Fresh token each fire so the load-pill done-watcher
        # triggers even when the human-readable status string is
        # identical to the previous recording (fixed-length rig).
        tok = datetime.now().timestamp()
        if not file_id:
            return (_empty_lfp_fig(
                "Pick a recording above to see the brain signal here."),
                    "", no_update, 0.0, tok)
        if channel is None:
            return (_empty_lfp_fig(
                "Pick a brain channel to display."),
                    "", no_update, 0.0, tok)
        file_path = _file_path_for_id(store, file_id)
        if not file_path:
            return (_empty_lfp_fig("File not found in DB."),
                    "", no_update, 0.0, tok)

        # Stim-blank by DEFAULT on every real LFP channel so the EEG is
        # visible through stimulation (the artifact otherwise dwarfs the
        # signal). The only channel never blanked is a stim_copy channel
        # itself -- that IS the stim signal. "Show raw" lets the
        # reviewer deliberately inspect the unblanked artifact.
        session_dir = _session_dir_for_file(store, file_id)
        stim_copy = _stim_copy_channels(store, session_dir)
        is_stim_copy = channel in stim_copy
        show_raw = "raw" in (blank_raw or [])
        do_blank = (not is_stim_copy) and (not show_raw)
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
                    "", no_update, 0.0, tok)

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

        fig = _build_lfp_figure(t, filtered, label=f"Ch{channel}",
                                 uirevision=f"{file_id}:{channel}")
        # Always draw any flagged onsets so they survive this rebuild
        # (channel/filter/blank changes), not just events-store updates.
        _inject_landmarks(fig, events)
        filt_bits = []
        if hp and hp > 0: filt_bits.append(f"HP={hp:g}")
        if lp and lp > 0: filt_bits.append(f"LP={lp:g}")
        if notch and notch > 0: filt_bits.append(f"Notch={notch}")
        if smooth_ms and smooth_ms > 0:
            filt_bits.append(f"Smooth={smooth_ms:g}ms")
        status_bits = [_fmt_time(duration), f"{len(t):,} display points"]
        if is_stim_copy:
            status_bits.append("stim-copy channel · not blanked")
        elif show_raw:
            status_bits.append("RAW · stim artifact shown")
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
                float(duration), tok)

    # ---- Show the line-length window only when that feature is plotted ----
    @app.callback(
        Output("video-llwin-group", "style"),
        Input("video-analysis-feature", "value"),
    )
    def _toggle_llwin_group(feature):
        base = {"flex": "0 0 200px"}
        if feature == "line_length":
            return base
        return {**base, "display": "none"}

    # ---- Time-locked analysis trace (feature vs time) ----
    @app.callback(
        Output("video-analysis-trace", "figure"),
        Output("video-analysis-status", "children"),
        Output("video-hilbert-loadtoken", "data"),
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
        Input("video-blank-raw", "value"),
        Input("video-rejected-version", "data"),
        Input("video-auc-window-input", "value"),
        Input("video-llwin-start-input", "value"),
        Input("video-llwin-end-input", "value"),
        State("video-analysis-smooth", "value"),
        State("video-analysis-rollwin", "value"),
        State("video-analysis-postproc", "value"),
        State("video-events-store", "data"),
    )
    def _update_analysis(file_id, feature, _n_apply,
                          channel, ma_cutoff, blank_raw, _rej_ver,
                          auc_window, ll_start_ms, ll_end_ms,
                          smooth_sec, rollwin, postproc, events):
        # Thin wrapper: delegate, then stamp a fresh token so the
        # load-pill done-watcher fires even when the status string
        # repeats (0-candidate fixed-length recordings all render
        # an identical hilbert status).
        fig, status = _compute_analysis(
            file_id, feature, channel, ma_cutoff, blank_raw,
            auc_window, ll_start_ms, ll_end_ms,
            smooth_sec, rollwin, postproc)
        # Draw flagged onsets on every Hilbert rebuild, not only on
        # events-store changes.
        _inject_landmarks(fig, events)
        return fig, status, datetime.now().timestamp()

    def _compute_analysis(file_id, feature,
                           channel, ma_cutoff, blank_raw,
                           auc_window, ll_start_ms, ll_end_ms,
                           smooth_sec, rollwin, postproc):
        if not file_id:
            return (_empty_lfp_fig(
                "Pick a recording above to see the feature trace."), "")
        if not feature:
            return (_empty_lfp_fig(
                "Pick a brain feature to plot."), "")
        # Sliding-window AUC of the Hilbert envelope -- the
        # sustained-vs-transient discriminator.
        if feature == "hilbert_auc":
            return _render_auc_trace(
                store, int(file_id), channel, auc_window,
                blank_pre_ms=blank_pre_ms,
                blank_post_ms=blank_post_ms,
                show_raw="raw" in (blank_raw or []))
        # Slow gamma (30-50 Hz) band power -- computed live, no DB column.
        if feature == "slow_gamma_power":
            lo, hi = SLOW_GAMMA_BAND
            return _render_band_power_trace(
                store, int(file_id), channel, lo, hi,
                blank_pre_ms=blank_pre_ms,
                blank_post_ms=blank_post_ms,
                show_raw="raw" in (blank_raw or []))
        if feature == "slow_gamma_evoked":
            lo, hi = SLOW_GAMMA_BAND
            return _render_evoked_band_power(
                store, int(file_id), channel, lo, hi)
        if feature == "slow_gamma_induced":
            lo, hi = SLOW_GAMMA_BAND
            sg = (config or {}).get("slow_gamma", {}) or {}
            pre_ms = sg.get("pre_ms", [-200, -2])
            post_ms = sg.get("post_ms", [50, 200])
            pre_win = (float(pre_ms[0]) / 1000.0,
                       float(pre_ms[1]) / 1000.0)
            post_win = (float(post_ms[0]) / 1000.0,
                        float(post_ms[1]) / 1000.0)
            return _render_evoked_band_ratio(
                store, int(file_id), channel, lo, hi, pre_win, post_win)
        # Wavelet (Morlet CWT) variants -- transient-aware siblings of the
        # slow-gamma options above. All live-computed (no DB column).
        if feature == "wavelet_sg_power":
            lo, hi = SLOW_GAMMA_BAND
            return _render_wavelet_band_power(
                store, int(file_id), channel, lo, hi,
                blank_pre_ms=blank_pre_ms, blank_post_ms=blank_post_ms,
                show_raw="raw" in (blank_raw or []))
        if feature == "wavelet_sg_evoked":
            lo, hi = SLOW_GAMMA_BAND
            return _render_wavelet_evoked_band_power(
                store, int(file_id), channel, lo, hi)
        if feature == "wavelet_scalogram":
            return _render_wavelet_scalogram(store, int(file_id), channel)
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
                show_raw="raw" in (blank_raw or []),
            )
        # Line length -- computed live over a user window (ms), so it no
        # longer reads the precomputed DB column. Drives the stim overlay.
        if feature == "line_length":
            s, e = _ll_window_ms(ll_start_ms, ll_end_ms)
            return _render_evoked_line_length(
                store, int(file_id), channel, (s, e))
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
            xaxis=dict(title="Time (mm:ss)", showgrid=True,
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
        # mm:ss tick labels (data stays in seconds for the cursor).
        _t_all = ok_t + art_t
        if _t_all:
            _apply_mmss_xaxis(fig, max(_t_all))
        bits = [f"{len(ok_t) + len(art_t)} epochs",
                 f"{len(art_t)} artifact"]
        if applied_bits:
            bits.append(" + ".join(applied_bits))
        return fig, " · ".join(bits)

    # ---- X-axis sync: zoom/pan one plot drives the other ----
    # The LFP and Hilbert share a time axis, so a zoom on one should
    # move the other. Plotly's `matches` only links axes WITHIN one
    # figure, so for two separate dcc.Graphs we drive the partner
    # directly: on a relayout we call Plotly.relayout on the other
    # graph's DOM node. (The previous version returned a clientside
    # Patch to the partner's `figure` prop; it silently no-op'd --
    # the plots never moved together.) An echo guard -- skip if the
    # target already shows this range -- breaks the A->B->A relayout
    # loop. We side-effect via Plotly only and return no_update, so
    # the Dash figure prop is never rewritten; combined with the
    # uirevision set per recording in _build_lfp_figure, the zoom
    # also survives the cursor + re-decimate figure updates.
    def _xsync_js(target_id: str) -> str:
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

    app.clientside_callback(
        _xsync_js("video-analysis-trace"),
        Output("video-xsync-sink", "data", allow_duplicate=True),
        Input("video-lfp-trace", "relayoutData"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        _xsync_js("video-lfp-trace"),
        Output("video-xsync-sink", "data", allow_duplicate=True),
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
            // Keep shapes[1:] (threshold line + landmark verticals);
            // only shapes[0] (the cursor) moves. Replacing the whole
            // array here used to wipe the markers 10x/sec.
            var keep = ((fig.layout && fig.layout.shapes)
                         || []).slice(1);
            var cursor = {
                type: 'line', xref: 'x', yref: 'paper',
                x0: t, x1: t, y0: 0, y1: 1,
                line: {color: '#ff9f0a', width: 2}
            };
            const newFig = {
                data: fig.data,
                layout: Object.assign({}, fig.layout, {
                    shapes: [cursor].concat(keep)
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
    #    Modeless: a click always seeks (drag pans, so it can't double as
    #    a zoom-drag that also seeks).
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

    # Enforce dragmode on both traces to match the Mode toggle, re-applied
    # after every figure rebuild so it survives channel/feature/zoom
    # redraws. Only relayouts when the live dragmode differs -> cheap, no
    # loop (a dragmode-only relayout changes no axis range, so it can't
    # re-trigger the zoom re-decimate).
    app.clientside_callback(
        """
        function(_figA, _figB) {
            // Modeless: always 'pan' so a drag pans (click still seeks,
            // scroll still zooms). Re-applied after every figure rebuild
            // so channel/feature/zoom redraws keep the pan dragmode.
            var want = 'pan';
            ['video-lfp-trace', 'video-analysis-trace'].forEach(function(id) {
                var gd = document.getElementById(id);
                if (!gd || !gd._fullLayout) { return; }
                if (gd._fullLayout.dragmode !== want) {
                    try { Plotly.relayout(gd, {dragmode: want}); } catch (e) {}
                }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output("video-fs-sink", "children", allow_duplicate=True),
        Input("video-lfp-trace", "figure"),
        Input("video-analysis-trace", "figure"),
        prevent_initial_call=True,
    )

    # Armed-landmark cursor: when a "Set on plot" landmark is armed, mark
    # the Step-2 grid so CSS swaps the plot cursor to a crosshair -- a
    # visible "click the trace to place this onset" affordance that
    # reinforces the armed banner. Cleared the moment the landmark drops
    # or is disarmed (video-armed-landmark -> null).
    app.clientside_callback(
        """
        function(armed) {
            var grid = document.getElementById('video-step2-grid');
            if (grid) {
                if (armed) { grid.classList.add('qc-armed'); }
                else { grid.classList.remove('qc-armed'); }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("video-fs-sink", "children", allow_duplicate=True),
        Input("video-armed-landmark", "data"),
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
        State("video-blank-raw", "value"),
        prevent_initial_call=True,
    )
    def _video_lfp_zoom(relayout, file_id, channel, filter_state,
                         blank_raw):
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
        # Match _update_lfp: blank by default on every real LFP
        # channel (so the zoomed view shows the same NaN gaps), never
        # blank a stim_copy channel, and honor the "Show raw" toggle.
        do_blank = (channel not in stim_copy) and (
            "raw" not in (blank_raw or []))
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

        series = np.asarray(series, dtype=np.float32)
        n = len(series)
        if is_reset:
            lo, hi = 0, n
        else:
            lo, hi = window_slice(n, fs, x0, x1)
        if hi <= lo:
            return no_update

        # Apply the same filter the time-domain plot used -- but only on
        # the VISIBLE window plus a settling margin, not the whole
        # recording. The old code filtfilt'd the entire full-rate chunk
        # on every zoom (millions of samples), which is the lingering
        # post-decimate wait the reviewer noticed. The margin keeps
        # filtfilt's edge transients out of the displayed window; it
        # scales with the high-pass cutoff (low cutoffs settle slowly).
        # No filter set -> skip the pass entirely.
        st = filter_state or {}
        has_filter = any(st.get(k) for k in ("hp", "lp", "notch",
                                               "smooth"))
        if has_filter:
            hp = float(st.get("hp") or 0)
            margin = int(min(n, max(2.0, 5.0 / max(hp, 0.5)) * fs))
            a = max(0, lo - margin)
            b = min(n, hi + margin)
            try:
                filt = apply_filter(
                    series[a:b], fs,
                    highpass=st.get("hp"), lowpass=st.get("lp"),
                    notch=st.get("notch"),
                    smoothing_ms=st.get("smooth"),
                )
                seg = filt[lo - a:hi - a]
            except Exception as e:
                logger.debug("zoom filter skipped: %s", e)
                seg = series[lo:hi]
        else:
            seg = series[lo:hi]

        # choose_target_bins returns 0 when the window is small enough
        # to render RAW. The old `... or INITIAL_TARGET_BINS` turned
        # that 0 into a decimate (0 or N == N), so zooming in never
        # showed real samples. When raw, pass target_bins = window
        # length so envelope() takes its decim==1 (raw) path.
        tb = choose_target_bins(hi - lo)
        target_bins = tb if tb else max(1, hi - lo)
        x_p, y_p, _decim = envelope(
            seg, fs, target_bins, t_start=lo / fs,
        )

        p = Patch()
        p["data"][0]["x"] = x_p
        p["data"][0]["y"] = y_p
        return p

    # ---- Note save: capture the full review context for restore ---- #
    @app.callback(
        Output("video-note-status", "children"),
        Output("video-note-history", "children", allow_duplicate=True),
        Output("video-note-input", "value"),
        Input("video-note-save-btn", "n_clicks"),
        State("video-file-dropdown", "value"),
        State("video-note-input", "value"),
        State("video-current-time", "data"),
        # Review context snapshot -- restored when the note is clicked.
        State("video-channel-dropdown", "value"),
        State("video-analysis-feature", "value"),
        State("video-threshold-input", "value"),
        State("video-llwin-start-input", "value"),
        State("video-llwin-end-input", "value"),
        State("video-queue-animal", "value"),
        State("video-filter-hp", "value"),
        State("video-filter-lp", "value"),
        State("video-filter-notch", "value"),
        State("video-filter-smooth", "value"),
        State("video-lfp-duration", "data"),
        prevent_initial_call=True,
    )
    def _save_note(n_clicks, file_id, note, current_time,
                    channel, feature, threshold, ll_start, ll_end,
                    animal, hp, lp, notch, smooth, lfp_dur):
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
            ts_prefix = f"[t={current_time:.2f}s / {_mmss(current_time)}] "
        body = ts_prefix + note.strip()

        email = current_user_email() or "unknown"
        session_dir = _session_dir_for_file(store, file_id)
        context = json.dumps({
            "session_dir": session_dir,
            "file_id": file_id,
            "channel": channel,
            "feature": feature,
            "threshold": threshold,
            "ll_start": ll_start,
            "ll_end": ll_end,
            "animal": animal,
            "hp": hp, "lp": lp, "notch": notch, "smooth": smooth,
            "video_time": (float(current_time)
                            if isinstance(current_time, (int, float))
                            else 0.0),
            "lfp_dur": (float(lfp_dur)
                         if isinstance(lfp_dur, (int, float)) else 0.0),
        })
        try:
            ann_id = store.add_annotation(
                timestamp=datetime.now().isoformat(),
                note=body,
                category="video_review",
                session_dir=session_dir,
                file_id=file_id,
                user_email=email,
                context=context,
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

    # ---- Note click -> restore the saved review context ---- #
    # Reuses the LFP-Browser bridge (session->file->channel cascade) +
    # pending-seek (now supports a direct video_time) to jump back to the
    # exact state, and sets the feature / threshold / window / animal.
    @app.callback(
        Output("lfp-to-video-bridge", "data", allow_duplicate=True),
        Output("pending-seek", "data", allow_duplicate=True),
        Output("video-analysis-feature", "value", allow_duplicate=True),
        Output("video-threshold-input", "value", allow_duplicate=True),
        Output("video-llwin-start-input", "value", allow_duplicate=True),
        Output("video-llwin-end-input", "value", allow_duplicate=True),
        Output("video-queue-animal", "value", allow_duplicate=True),
        Output("video-note-status", "children", allow_duplicate=True),
        Input({"type": "video-note-item", "ann_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _restore_note_context(n_list):
        nope = (no_update,) * 8
        # Act only on a genuine click (triggered value >= 1), not a re-render
        # re-fire that any(n_list) would let through with a stale triggered_id.
        triggered = callback_context.triggered or []
        trig = callback_context.triggered_id
        if (not isinstance(trig, dict) or not triggered
                or not (triggered[0].get("value") or 0)):
            return nope
        ann_id = trig.get("ann_id")
        try:
            with store.connection() as conn:
                row = conn.execute(
                    "SELECT context FROM annotations WHERE id = ?",
                    (ann_id,)).fetchone()
            ctx = json.loads(row["context"]) if row and row["context"] else None
        except Exception as e:
            logger.warning("Note restore load failed id=%s: %s", ann_id, e)
            ctx = None
        if not ctx:
            return nope
        seq = int(datetime.now().timestamp() * 1000)
        bridge = {
            "session_dir": ctx.get("session_dir"),
            "file_id": ctx.get("file_id"),
            "channel": ctx.get("channel"),
            "hp": ctx.get("hp") or 0, "lp": ctx.get("lp") or 0,
            "notch": ctx.get("notch") or 0, "smooth": ctx.get("smooth") or 0,
            "start_sec": 0.0, "lfp_dur": ctx.get("lfp_dur") or 0.0,
            "seq": seq,
        }
        pending = {"video_time": ctx.get("video_time") or 0.0, "seq": seq}

        def _or(v):
            return v if v is not None else no_update
        msg = html.Span("↩ Restored the note's review context.",
                         style={"color": "#30d158"})
        return (bridge, pending, _or(ctx.get("feature")),
                _or(ctx.get("threshold")), _or(ctx.get("ll_start")),
                _or(ctx.get("ll_end")), _or(ctx.get("animal")), msg)

    # ================================================================== #
    #  Stim overlay (oscilloscope persistence view)
    # ================================================================== #

    # Publish this file's stim onsets (LFP seconds) so the clientside
    # throttle can find the nearest stim without a server round-trip.
    @app.callback(
        Output("video-stim-times", "data"),
        Input("video-file-dropdown", "value"),
    )
    def _publish_stim_times(file_id):
        if not file_id:
            return []
        try:
            st = _stim_times_for_file(store, int(file_id))
        except Exception as e:
            logger.warning("Stim-times publish failed file=%s: %s",
                            file_id, e)
            return []
        return [] if st is None else np.asarray(st).tolist()

    # Rebuild the overlay only when the nearest-stim index changes (≤ once
    # per stim crossing) or when the line-length window is edited.
    @app.callback(
        Output("video-overlay-graph", "figure"),
        Input("video-overlay-stim-idx", "data"),
        Input("video-llwin-start-input", "value"),
        Input("video-llwin-end-input", "value"),
        Input("video-overlay-ntraces-input", "value"),
        Input("video-overlay-mode", "value"),
        State("video-file-dropdown", "value"),
        State("video-channel-dropdown", "value"),
    )
    def _update_overlay(stim_idx, ll_start_ms, ll_end_ms, n_traces,
                         mode, file_id, channel):
        if not file_id:
            return _empty_lfp_fig(
                "Pick a recording above to see the stim overlay.")
        try:
            n_prev = int(n_traces) if n_traces is not None else 10
        except (TypeError, ValueError):
            n_prev = 10
        n_prev = max(1, min(100, n_prev))
        mode = mode if mode in ("lines", "heatmap") else "lines"
        s, e = _ll_window_ms(ll_start_ms, ll_end_ms)
        return _render_overlay(store, int(file_id), channel,
                                stim_idx, (s, e), n_prev=n_prev, mode=mode)

    # Overlay collapse: button flips the Store; the Store mirrors to a DOM
    # class via the qcOverlay helper (assets/overlay_panel.js).
    app.clientside_callback(
        """
        function(n, state) {
            if (!n) { return window.dash_clientside.no_update; }
            return !state;
        }
        """,
        Output("video-overlay-collapsed", "data"),
        Input("video-overlay-collapse-btn", "n_clicks"),
        State("video-overlay-collapsed", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        """
        function(on) {
            if (window.qcOverlay && window.qcOverlay.applyCollapsed) {
                window.qcOverlay.applyCollapsed(!!on);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("video-overlay-collapsed", "id"),  # write-only sink
        Input("video-overlay-collapsed", "data"),
    )

    # Overlay pop-out: same flip + DOM-mirror pattern. applyPip toggles the
    # floating class and lazily wires header drag.
    app.clientside_callback(
        """
        function(n, state) {
            if (!n) { return window.dash_clientside.no_update; }
            return !state;
        }
        """,
        Output("video-overlay-pip", "data"),
        Input("video-overlay-pip-btn", "n_clicks"),
        State("video-overlay-pip", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        """
        function(on) {
            if (window.qcOverlay && window.qcOverlay.applyPip) {
                window.qcOverlay.applyPip(!!on);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("video-overlay-pip", "id"),  # write-only sink
        Input("video-overlay-pip", "data"),
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

    # 1-time. Live playhead readout. Shows BOTH clocks honestly: the
    #     video's container time AND the LFP estimate (endpoint stretch
    #     lfp_t = currentTime * lfp_dur / video_dur). They diverge because
    #     the 5 fps video drops frames, so 'LFP≈' is flagged as an estimate
    #     -- not frame-exact.
    app.clientside_callback(
        """
        function(currentTime, lfpDur) {
            function mmss(s) {
                if (s === null || s === undefined || !isFinite(s)
                        || s < 0) { s = 0; }
                s = Math.round(s);
                var h = Math.floor(s / 3600);
                var m = Math.floor((s % 3600) / 60);
                var ss = s % 60;
                var pad = function(n){ return (n < 10 ? '0' : '') + n; };
                return (h ? h + ':' + pad(m) : pad(m)) + ':' + pad(ss);
            }
            var vt = (currentTime && currentTime > 0) ? currentTime : 0;
            var lfpT = vt;
            var v = document.getElementById('""" + VIDEO_DOM_ID + """');
            var stretched = false;
            if (v && isFinite(v.duration) && v.duration > 0
                    && lfpDur && lfpDur > 0) {
                lfpT = vt * (lfpDur / v.duration);
                stretched = (Math.abs(lfpT - vt) > 0.5);
            }
            var dur = (lfpDur && lfpDur > 0) ? lfpDur : 0;
            var lfpLabel = stretched
                ? ('LFP≈' + mmss(lfpT) + ' (est)')
                : ('LFP ' + mmss(lfpT));
            return 'video ' + mmss(vt) + ' · ' + lfpLabel
                   + ' / ' + mmss(dur);
        }
        """,
        Output("video-playhead-readout", "children"),
        Input("video-current-time", "data"),
        State("video-lfp-duration", "data"),
    )

    # 1-stim. Stim-overlay throttle. Map the playhead to LFP time with the
    #     same BHZ ratio (lfp_t = currentTime * lfp_dur / video_dur), find
    #     the nearest stim at/before it, and push that index ONLY when it
    #     changes -- so the server rebuilds the overlay at most once per
    #     stim crossing, not at the 10 Hz playback cadence. Returns
    #     no_update before the first stim (idx stays -1).
    app.clientside_callback(
        """
        function(currentTime, stimTimes, lfpDur, lastIdx) {
            if (!stimTimes || !stimTimes.length) {
                return window.dash_clientside.no_update;
            }
            if (currentTime === null || currentTime === undefined) {
                return window.dash_clientside.no_update;
            }
            var lfpT = currentTime;
            var v = document.getElementById('""" + VIDEO_DOM_ID + """');
            if (v && isFinite(v.duration) && v.duration > 0
                    && lfpDur && lfpDur > 0) {
                lfpT = currentTime * (lfpDur / v.duration);
            }
            var idx = -1;
            for (var i = 0; i < stimTimes.length; i++) {
                if (stimTimes[i] <= lfpT) { idx = i; } else { break; }
            }
            if (idx === lastIdx) {
                return window.dash_clientside.no_update;
            }
            return idx;
        }
        """,
        Output("video-overlay-stim-idx", "data"),
        Input("video-current-time", "data"),
        State("video-stim-times", "data"),
        State("video-lfp-duration", "data"),
        State("video-overlay-stim-idx", "data"),
        prevent_initial_call=True,
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
            var hasVideoT = pending && pending.video_time !== null
                    && pending.video_time !== undefined;
            var hasStart = pending && pending.start_sec !== null
                    && pending.start_sec !== undefined;
            if (!pending || (!hasVideoT && !hasStart)) {
                return window.dash_clientside.no_update;
            }
            const v = document.getElementById('""" + VIDEO_DOM_ID + """');
            if (!v || !isFinite(v.duration) || v.duration <= 0
                    || v.readyState < 1) {
                return window.dash_clientside.no_update;
            }
            var target;
            if (hasVideoT) {
                // Note-restore: seek the video to the exact container time
                // it was at when the note was saved (frame-exact -- no
                // stretch, since it's the same file/clock).
                target = pending.video_time;
            } else {
                // LFP Browser hand-off: LFP seconds -> video via BHZ ratio.
                target = pending.start_sec;
                if (pending.lfp_dur && pending.lfp_dur > 0) {
                    target = pending.start_sec
                            * (v.duration / pending.lfp_dur);
                }
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
            // Preserve shapes[1:] (landmark verticals) -- only the
            // cursor (shapes[0]) moves. A full replace here wiped the
            // markers _render_marker_shapes painted.
            var keep = ((fig.layout && fig.layout.shapes)
                         || []).slice(1);
            var cursor = {
                type: 'line', xref: 'x', yref: 'paper',
                x0: t, x1: t, y0: 0, y1: 1,
                line: {color: '#ff9f0a', width: 2}
            };
            const newFig = {
                data: fig.data,
                layout: Object.assign({}, fig.layout, {
                    shapes: [cursor].concat(keep)
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
    #    Modeless: a click always seeks (drag pans rather than zoom-box).
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

    # Reattach the progress poll after a tab reload. The scan
    # worker is a server-side daemon, so a scan started before
    # the reload keeps running -- but the job-id Store and the
    # poll Interval both reset to their defaults on reload, so
    # the panel would otherwise come back blank with no hint
    # that work is still happening. On animal pick, look for an
    # in-progress job for that animal and re-arm the poll.
    @app.callback(
        Output("video-ma-job-id", "data", allow_duplicate=True),
        Output("video-ma-poll", "disabled",
                allow_duplicate=True),
        Output("video-ma-progress", "children",
                allow_duplicate=True),
        Input("video-queue-animal", "value"),
        prevent_initial_call=True,
    )
    def _on_video_ma_reattach(picker_value):
        animal = _ma_animal_from_picker(picker_value)
        if not animal:
            return None, True, ""
        try:
            job = _mass_analyze.active_job_for_animal(
                store, str(animal))
        except Exception as e:
            logger.warning(
                "active_job_for_animal failed: %s", e)
            return no_update, no_update, no_update
        if not job:
            # Animal with no running scan: drop any stale poll
            # attachment from a previously-viewed animal.
            return None, True, no_update
        return (job["id"], False,
                html.Span(
                    f"Reconnected to a scan already running for "
                    f"{animal} (started before this reload). "
                    "Progress will update below.",
                    style={"color": "#5e7ce2"}))

    # Restore the pool browser from this animal's most recent COMPLETED
    # scan when the animal is (re)selected -- so the pools survive a tab
    # close. The scan's cache rows persist, so pool_files rebuilds the
    # same pools with no recompute. To re-analyze at new thresholds, the
    # PI just runs a fresh scan.
    @app.callback(
        Output("video-ma-pools", "data", allow_duplicate=True),
        Output("video-ma-browse", "style", allow_duplicate=True),
        Output("video-ma-pool-counts", "children",
                allow_duplicate=True),
        Output("video-ma-pool-cursor", "data",
                allow_duplicate=True),
        Output("video-ma-filter-session", "value",
                allow_duplicate=True),
        Input("video-queue-animal", "value"),
        prevent_initial_call=True,
    )
    def _on_video_ma_restore_pools(picker_value):
        hidden = {"display": "none", "padding": "0 14px",
                   "marginBottom": "10px"}
        animal = _ma_animal_from_picker(picker_value)
        if not animal:
            return None, hidden, "", None, "__all__"
        try:
            job = _mass_analyze.last_done_job_for_animal(
                store, str(animal))
        except Exception as e:
            logger.warning("last_done_job_for_animal failed: %s", e)
            return (no_update, no_update, no_update, no_update,
                    "__all__")
        if not job:
            return None, hidden, "", None, "__all__"
        auc_t = job.get("auc_threshold")
        auc_w = job.get("auc_window_sec")
        try:
            pools = _mass_analyze.pool_files(
                store, job["animal_id"], float(job["cutoff"]),
                float(auc_t) if auc_t else None,
                float(auc_w) if auc_w else None,
                electrode=int(job.get("electrode") or 0),
                # Keep flagged / done files in the restored browse pool
                # so progress (flagged / done) survives a tab reopen.
                include_reviewed=True)
        except Exception as e:
            logger.warning("restore pool_files failed: %s", e)
            return no_update, no_update, no_update, no_update
        n1, n2, n3 = (len(pools["pool1"]), len(pools["pool2"]),
                       len(pools["pool3"]))
        if n1 + n2 + n3 == 0:
            return None, hidden, "", None
        prefix = (f"Restored from last scan (cutoff "
                   f"{float(job['cutoff']):g}")
        if auc_t and auc_w:
            prefix += (f", AUC {float(auc_t):g}, win "
                        f"{float(auc_w):g}s)")
            counts = (f"{prefix}  ·  Pool 1 -- envelope: {n1}  ·  "
                       f"Pool 2 -- AUC: {n2}  ·  Pool 3 -- "
                       f"disagreement: {n3}")
        else:
            counts = f"{prefix})  ·  Pool 1 -- envelope: {n1}"
        browse = {"display": "block", "padding": "0 14px",
                   "marginBottom": "10px"}
        return pools, browse, counts, None, "__all__"

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
        State("video-ma-auc-threshold-input", "value"),
        State("video-ma-auc-window-input", "value"),
        State("video-ma-electrode", "value"),
        prevent_initial_call=True,
    )
    def _on_video_ma_modal_confirm(n_clicks, picker_value, cutoff,
                                     auc_threshold, auc_window,
                                     electrode):
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
        # AUC screen is optional: only runs if a valid threshold +
        # window are supplied. A blank threshold scans the
        # envelope screen alone (Pool 1 only).
        auc_t = None
        auc_w = None
        try:
            if auc_threshold not in (None, "") and float(auc_threshold) > 0:
                auc_t = float(auc_threshold)
                auc_w = float(auc_window) if auc_window else 5.0
        except (TypeError, ValueError):
            auc_t, auc_w = None, None
        try:
            job_id = _mass_analyze.create_job(
                store, email, animal_id, cutoff,
                auc_threshold=auc_t, auc_window_sec=auc_w,
                electrode=int(electrode or 0))
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
        Output("video-ma-pools", "data"),
        Output("video-ma-browse", "style"),
        Output("video-ma-pool-counts", "children"),
        Input("video-ma-poll", "n_intervals"),
        Input("video-ma-job-id", "data"),
        prevent_initial_call=True,
    )
    def _on_video_ma_poll(_n, job_id):
        _browse_hidden = {"display": "none", "padding": "0 14px",
                           "marginBottom": "10px"}
        if not job_id:
            return ("", "", [], True, None, _browse_hidden, "")
        job = _mass_analyze.get_job(store, int(job_id))
        if not job:
            return ("Job vanished.", "", [], True,
                    None, _browse_hidden, "")
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
        pools_data = no_update
        browse_style = no_update
        pool_counts = no_update
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
            # Build the screen pools off the caches the scan just
            # populated (read-only; no recompute). Pool 2/3 only
            # exist when the AUC screen ran (threshold supplied).
            auc_t = job.get("auc_threshold")
            auc_w = job.get("auc_window_sec")
            pools = {"pool1": [], "pool2": [], "pool3": []}
            try:
                pools = _mass_analyze.pool_files(
                    store, job["animal_id"], float(job["cutoff"]),
                    float(auc_t) if auc_t else None,
                    float(auc_w) if auc_w else None,
                    electrode=int(job.get("electrode") or 0),
                    # Keep flagged / done files in the browse pool so the
                    # progress tracker can show them (they leave the
                    # pending set the moment they're flagged).
                    include_reviewed=True)
            except Exception as e:
                logger.warning("pool_files failed: %s", e)
            pools_data = pools
            browse_style = {"display": "block",
                             "padding": "0 14px",
                             "marginBottom": "10px"}
            if auc_t and auc_w:
                pool_counts = (
                    f"Pool 1 -- envelope screen: "
                    f"{len(pools['pool1'])}  ·  "
                    f"Pool 2 -- AUC screen: {len(pools['pool2'])}"
                    f"  ·  Pool 3 -- disagreement: "
                    f"{len(pools['pool3'])}")
            else:
                pool_counts = (
                    f"Pool 1 -- envelope screen: "
                    f"{len(pools['pool1'])}  (add an AUC threshold "
                    "to also run the AUC screen)")
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
                polling_disabled, pools_data, browse_style,
                pool_counts)

    # Per-session statistics: shown when a specific session is picked
    # in the pool-filter dropdown.
    @app.callback(
        Output("video-ma-session-stats", "children"),
        Input("video-ma-filter-session", "value"),
        prevent_initial_call=True,
    )
    def _on_video_ma_session_stats(session_dir):
        if not session_dir or session_dir == "__all__":
            return ""
        try:
            s = store.session_review_stats(session_dir)
        except Exception as e:
            logger.warning("session_review_stats failed: %s", e)
            return ""

        def _cell(label, val):
            return html.Div([
                html.Div(str(val), style={"color": "#f0f0f5",
                                           "fontWeight": "700",
                                           "fontSize": "14px"}),
                html.Div(label, style={"color": "#a0a0b0",
                                        "fontSize": "10px"}),
            ], style={"padding": "4px 10px",
                       "borderRight": "1px solid rgba(255,255,255,0.06)"})
        return html.Div([
            _cell("files", s["n_files"]),
            _cell("reviewed", s["n_reviewed"]),
            _cell("with events", s["n_files_with_events"]),
            _cell("events", s["n_events"]),
            _cell("max Racine", s["max_racine"]),
            _cell("needs scoring", s["n_needs_scoring"]),
            _cell("stim files", s["n_stim_files"]),
            _cell("during-stim events", s["n_during_stim_events"]),
        ], style={"display": "flex", "flexWrap": "wrap",
                   "background": "#13131f",
                   "border": "1px solid rgba(94,124,226,0.18)",
                   "borderRadius": "6px", "padding": "4px"})

    # Build the per-file meta map whenever the pools change, so the
    # session + stim filters resolve without re-querying.
    @app.callback(
        Output("video-ma-pool-meta", "data"),
        Input("video-ma-pools", "data"),
        State("video-queue-animal", "value"),
        prevent_initial_call=True,
    )
    def _on_video_ma_build_meta(pools, picker_value):
        animal = _ma_animal_from_picker(picker_value)
        if not pools or not animal:
            return None
        try:
            return _mass_analyze.pool_meta(store, str(animal))
        except Exception as e:
            logger.warning("pool_meta failed: %s", e)
            return None

    # Apply the session filter -> the view _on_pool_nav browses. Also
    # populates the session dropdown options + the filtered counts.
    @app.callback(
        Output("video-ma-pools-view", "data"),
        Output("video-ma-filter-session", "options"),
        Output("video-ma-filter-counts", "children"),
        Output("video-ma-pool-cursor", "data", allow_duplicate=True),
        Input("video-ma-pools", "data"),
        Input("video-ma-pool-meta", "data"),
        Input("video-ma-filter-session", "value"),
        prevent_initial_call=True,
    )
    def _on_video_ma_filter(pools, meta, sess_val):
        pools = pools or {"pool1": [], "pool2": [], "pool3": []}
        meta = meta or {}
        # Session dropdown options from the meta map (stable order).
        seen: "OrderedDict[str, str]" = OrderedDict()
        for m in meta.values():
            sd = m.get("session_dir")
            if sd and sd not in seen:
                seen[sd] = m.get("session_name") or sd
        opts = [{"label": "All sessions", "value": "__all__"}]
        opts += [{"label": name, "value": sd}
                  for sd, name in seen.items()]
        # Defensive: a session value left over from a previously-viewed
        # animal isn't in THIS animal's options -- treat it as "all" so
        # it never nukes the whole view (the Pool-2-empty bug).
        valid_sessions = set(seen.keys())
        if sess_val not in (None, "__all__") \
                and sess_val not in valid_sessions:
            sess_val = "__all__"

        def _keep(fid: int) -> bool:
            if sess_val in (None, "__all__"):
                return True
            m = meta.get(str(fid))
            if m is None:
                return True  # meta not built yet -> don't hide
            return m.get("session_dir") == sess_val

        def _fid(entry):
            return int(entry["file_id"]) if isinstance(entry, dict) \
                else int(entry)

        def _sess_key(entry):
            m = meta.get(str(_fid(entry))) or {}
            return m.get("session_dir") or ""

        def _filtered_sorted(pool_list):
            # Keep the session filter, then GROUP BY SESSION. The scan
            # orders pools by time, so an unfiltered pool interleaves
            # sessions -- browsing / Flag / Save would then flip the
            # session on every step. A stable sort by session_dir keeps
            # each session's files contiguous (and the original order
            # within a session), so you finish one session's pool before
            # the next instead of the session changing "randomly".
            kept = [f for f in (pool_list or []) if _keep(_fid(f))]
            return sorted(kept, key=_sess_key)

        view = {
            "pool1": _filtered_sorted(pools.get("pool1", [])),
            "pool2": _filtered_sorted(pools.get("pool2", [])),
            "pool3": _filtered_sorted(pools.get("pool3", [])),
        }
        if sess_val not in (None, "__all__"):
            counts = (f"Filtered  ·  Pool 1: {len(view['pool1'])}  ·  "
                       f"Pool 2: {len(view['pool2'])}  ·  Pool 3: "
                       f"{len(view['pool3'])}")
        else:
            counts = ""
        return view, opts, counts, None

    # Loading feedback: grey out the Browse / Prev / Next controls until
    # the filtered view is ready, so the reviewer never clicks into a
    # half-built pool (and gets a clear "not yet" signal).
    @app.callback(
        Output("video-ma-browse-p1", "disabled"),
        Output("video-ma-browse-p2", "disabled"),
        Output("video-ma-browse-p3", "disabled"),
        Output("video-ma-pool-prev", "disabled"),
        Output("video-ma-pool-next", "disabled"),
        Input("video-ma-pools-view", "data"),
    )
    def _toggle_browse_enabled(view):
        ready = bool(view) and any(
            view.get(k) for k in ("pool1", "pool2", "pool3"))
        d = not ready
        return d, d, d, d, d

    # Scan button reflects the job state: disabled + "Scanning..." while
    # a scan is running, so it's obvious work is in progress.
    @app.callback(
        Output("video-ma-scan-btn", "disabled"),
        Output("video-ma-scan-btn", "children"),
        Input("video-ma-job-id", "data"),
        Input("video-ma-poll", "disabled"),
    )
    def _scan_btn_state(job_id, poll_disabled):
        if bool(job_id) and not poll_disabled:
            return True, "Scanning..."
        return False, "Scan files"

    @app.callback(
        Output("video-ma-pool-cursor", "data"),
        Output("video-session-dropdown", "value",
                allow_duplicate=True),
        Output("video-file-dropdown", "value",
                allow_duplicate=True),
        Output("video-analysis-feature", "value",
                allow_duplicate=True),
        Output("video-ma-pool-status", "children"),
        Input("video-ma-browse-p1", "n_clicks"),
        Input("video-ma-browse-p2", "n_clicks"),
        Input("video-ma-browse-p3", "n_clicks"),
        Input("video-ma-pool-prev", "n_clicks"),
        Input("video-ma-pool-next", "n_clicks"),
        State("video-ma-pools-view", "data"),
        State("video-ma-pool-cursor", "data"),
        prevent_initial_call=True,
    )
    def _on_pool_nav(_p1, _p2, _p3, _prev, _next, pools, cursor):
        """Walk Pool 1 / 2 / 3 file-by-file in the player.

        Browse-pool-N picks a screen's pool and jumps to its first
        file; Prev/Next step within the active pool. Each move
        resolves the file's session + id and drives the existing
        session/file dropdowns (same hand-off as the queue button).
        The brain-feature trace follows the screen that flagged the
        file: Pool 2 -> Hilbert AUC; Pool 3 mixes both, so it sets
        the trace per entry's ``only`` tag (envelope-only ->
        Hilbert, AUC-only -> Hilbert AUC).
        """
        # Real-click guard: pattern recreation / initial fires
        # carry n_clicks 0/None across all buttons.
        if not any((t.get("value") or 0)
                    for t in (callback_context.triggered or [])):
            return (no_update, no_update, no_update,
                    no_update, no_update)
        if not pools:
            return (no_update, no_update, no_update,
                    no_update, "Run a scan first.")
        trig = callback_context.triggered_id
        cursor = cursor or {"active": None, "idx": 0}
        active = cursor.get("active")
        idx = int(cursor.get("idx") or 0)
        feature_out = no_update
        if trig == "video-ma-browse-p1":
            active, idx = "1", 0
        elif trig == "video-ma-browse-p2":
            active, idx = "2", 0
        elif trig == "video-ma-browse-p3":
            active, idx = "3", 0
        elif trig == "video-ma-pool-prev":
            idx = max(0, idx - 1)
        elif trig == "video-ma-pool-next":
            idx = idx + 1
        if active not in ("1", "2", "3"):
            return (no_update, no_update, no_update,
                    no_update, "Pick a pool to browse.")
        key = {"1": "pool1", "2": "pool2", "3": "pool3"}[active]
        files = pools.get(key, [])
        n = len(files)
        if n == 0:
            return ({"active": active, "idx": 0},
                    no_update, no_update, no_update,
                    f"Pool {active} is empty.")
        idx = min(idx, n - 1)
        entry = files[idx]
        # pool3 entries are {file_id, only}; pool1/2 are bare ids.
        if isinstance(entry, dict):
            file_id = int(entry["file_id"])
            feature_out = ("hilbert_auc" if entry.get("only") == "auc"
                            else "hilbert")
        else:
            file_id = int(entry)
            feature_out = "hilbert_auc" if active == "2" else "hilbert"
        with store.connection() as conn:
            row = conn.execute(
                "SELECT session_dir FROM processed_files "
                "WHERE id = ?", (file_id,)).fetchone()
        if not row:
            return ({"active": active, "idx": idx},
                    no_update, no_update, no_update,
                    f"Pool {active} · file {idx + 1} of {n} "
                    "(missing)")
        badge = ""
        if isinstance(entry, dict):
            badge = f" ({entry.get('only')}-only)"
        status = f"Pool {active} · file {idx + 1} of {n}{badge}"
        return ({"active": active, "idx": idx},
                row["session_dir"], file_id, feature_out, status)

    # Keep the "Pool N · file x of y" counter in sync with the cursor on
    # EVERY move -- including the auto-advance after Mark-done / Flag for
    # scoring, which bump the cursor but not the status text. Without
    # this the counter only refreshed on the Prev/Next/Browse buttons,
    # so progress through a pool looked frozen while auto-advancing.
    @app.callback(
        Output("video-ma-pool-status", "children",
                allow_duplicate=True),
        Input("video-ma-pool-cursor", "data"),
        State("video-ma-pools-view", "data"),
        State("video-queue-animal", "value"),
        prevent_initial_call=True,
    )
    def _render_pool_status(cursor, pools, picker_value):
        if not cursor or not isinstance(cursor, dict):
            return no_update
        active = cursor.get("active")
        if active not in ("1", "2", "3"):
            return no_update
        key = {"1": "pool1", "2": "pool2", "3": "pool3"}[active]
        files = (pools or {}).get(key, []) or []
        n = len(files)
        if n == 0:
            return no_update
        idx = min(int(cursor.get("idx") or 0), n - 1)

        def _fid(e):
            return int(e["file_id"]) if isinstance(e, dict) else int(e)

        file_ids = [_fid(e) for e in files]
        try:
            statuses = store.review_statuses_for_files(
                file_ids,
                animal_id=_ma_animal_from_picker(picker_value))
        except Exception:
            statuses = {}
        flagged = sum(1 for f in file_ids
                      if statuses.get(f) == "needs_scoring")
        done = sum(1 for f in file_ids
                   if statuses.get(f) in _POOL_DONE_STATUSES)
        todo = n - flagged - done
        entry = files[idx]
        badge = (f" ({entry.get('only')}-only)"
                 if isinstance(entry, dict) else "")
        cur_st = statuses.get(file_ids[idx])
        if cur_st == "needs_scoring":
            cur_lbl, cur_col = "⚑ flagged", "#f0b429"
        elif cur_st in _POOL_DONE_STATUSES:
            cur_lbl, cur_col = "✓ done", "#30d158"
        else:
            cur_lbl, cur_col = "• not started", "#a0a0b0"
        # The list is grouped by session, so when a session is selected in
        # the filter this tally IS that (session, pool)'s completion -- the
        # answer to "did I finish this session's pool?" (to-start = 0).
        return [
            html.Span(f"Pool {active} · file {idx + 1} of {n}{badge} · "),
            html.Span(cur_lbl, style={"color": cur_col,
                                       "fontWeight": "600"}),
            html.Span("   |   ", style={"color": "#555"}),
            html.Span(f"✓ {done} done", style={"color": "#30d158"}),
            html.Span(" · "),
            html.Span(f"⚑ {flagged} flagged", style={"color": "#f0b429"}),
            html.Span(" · "),
            html.Span(f"{todo} to start", style={"color": "#a0a0b0"}),
        ]

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
        State("video-ma-electrode", "value"),
        prevent_initial_call=True,
    )
    def _on_video_ma_commit(_confirm_n, _discard_n,
                              cutoff, picker_value, electrode):
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
                store, animal, cutoff, email,
                electrode=int(electrode or 0))
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
            var pt = clickData.points[0];
            // Only the envelope line (curveNumber 0) sets the cutoff.
            // Clicking a candidate triangle (curveNumber 1, drawn near
            // the top) must NOT yank the threshold up to the marker.
            if (pt.curveNumber !== 0) {
                return window.dash_clientside.no_update;
            }
            var y = pt.y;
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
                """SELECT id, timestamp, user_email, note, context, category
                   FROM annotations
                   WHERE file_id = ?
                     AND category IN ('video_review', 'operator_log')
                   ORDER BY timestamp DESC LIMIT 80""",
                (file_id,),
            ).fetchall()
    except sqlite3.OperationalError:
        rows = []

    if not rows:
        return html.Span("No notes yet for this file.")
    return html.Ul([_history_item(r) for r in rows],
                    style={"paddingLeft": "16px"})


def _history_item(r):
    """One note row. Review notes with a saved context render as a
    clickable button that restores that exact state; operator-log notes
    (synced from the KMrecorder sheet) render read-only with a badge."""
    is_oplog = r["category"] == "operator_log"
    meta = [
        html.Span(f"{(r['timestamp'] or '')[:19]} ",
                  style={"color": "#6c6c80"}),
        html.Span(f"{r['user_email'] or 'unknown'}: ",
                  style={"color": "#5e7ce2"}),
    ]
    if is_oplog:
        badge = html.Span("operator log ", style={
            "color": "#ff9f0a", "fontSize": "10px",
            "border": "1px solid rgba(255,159,10,0.4)",
            "borderRadius": "4px", "padding": "0 4px",
            "marginRight": "4px"})
        return html.Li(meta + [badge, html.Span(
            r["note"], style={"color": "#a0a0b0"})],
            style={"marginBottom": "4px"})
    if r["context"]:
        note_el = html.Button(
            ["↩ ", r["note"]],
            id={"type": "video-note-item", "ann_id": int(r["id"])},
            className="qc-note-restore",
            title="Jump back to the review state saved with this note "
                  "(animal, recording, channel, feature, window, video "
                  "position).",
            n_clicks=0)
    else:
        note_el = html.Span(r["note"], style={"color": "#a0a0b0"})
    return html.Li(meta + [note_el], style={"marginBottom": "4px"})
