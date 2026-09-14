"""Overview tab -- the dashboard home page.

The single richest tab: a top KPI strip, a "today at the lab" home
grid (surgery / maintenance / schedule / incidents / data-log /
quick-links blocks), the live processing queue, a latest-evoked
thumbnail with a raw/mean trace toggle, a KM-recorder log summary, a
recording-arrivals timeline (24h + 7d), a last-frame video snapshot
with click-to-expand, and a per-animal behavioral-seizure status
card.

Most of the module is figure/section builders. The six callbacks at
the bottom keep the volatile pieces fresh on the refresh-trigger tick
without re-rendering the whole tab (that fine-grained refresh is what
killed the white-flash the tab used to have), plus the home-grid
"Open >" buttons that jump to the relevant Lab sub-tab.

Three app-shell callbacks that *also* fire on refresh-trigger stay in
app.py rather than move here: the header status dot, the
"last refresh Ns ago" label, and the Sessions-row jump -- they write
header/nav chrome, not Overview-owned outputs.

Lifted out of ``src/dashboard/app.py`` (the last and largest tab to
be modularized). The internal builder + widget helpers kept their
original underscore names; the only behavioral change in the move was
dropping the unused ``_status_card`` widget.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date, datetime, timedelta

import plotly.graph_objects as go
from dash import (MATCH, Input, Output, Patch, State, callback_context,
                   dash_table, dcc, html, no_update)
from plotly.subplots import make_subplots

from src.dashboard.components import (
    DARK_TABLE_STYLE, DROPDOWN_STYLE, ZEBRA_STRIPE,
    card as _card, loading_icon, pill as _pill,
    section_header as _section_header,
)
from src.dashboard.data_helpers import (
    channel_map as _get_channel_map,
    color_for_role as _color_for_role,
    default_session as _default_session,
    empty_fig as _empty_fig,
    processed_files_for_session as _get_processed_files_for_session,
    session_dropdown_options as _session_dropdown_options,
)
from src.dashboard.design import (
    COLOR_ACCENT, COLOR_DANGER, COLOR_DIVIDER, COLOR_SUCCESS, COLOR_SURFACE_1,
    COLOR_SURFACE_2, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY,
    COLOR_TEXT_TERTIARY, COLOR_WARNING, FONT_SIZE_BODY, FONT_SIZE_CAPTION,
    RADIUS_MD, RADIUS_SM, ROLE_COLORS, SPACE_1, SPACE_2, SPACE_3,
    SPACE_5,
)
from src.db.store import Store, _current_fidelity_pct
from src.dashboard import perf as _perf
from src.dashboard.single_flight import SingleFlight, STALL_SEC
from src.utils.deadline import call_with_deadline

logger = logging.getLogger("qc_monitor.dashboard.overview")

# Guards the auto-refresh callback so a slow refresh tick can't overlap itself
# and pile heavy builds onto a fixed server thread pool. SingleFlight = the same
# non-blocking acquire, plus a start-time + a loud log if it's ever held too long
# (so a wedged build is visible instead of silently skipping every later tick).
_REFRESH_LOCK = SingleFlight("overview-cards")

# Separate guard for the 4 stim/electrode cards. They were bundled into
# refresh_overview_dynamic with the status pills + review queue, so a slow
# impedance build (the heaviest Overview card) held the pills on the placeholder
# for its whole duration. Splitting them onto their own callback + lock lets the
# high-priority pills paint immediately.
_STIM_LOCK = SingleFlight("overview-stim-cards")


class _CachedBuilder:
    """Single-flight + stale-while-revalidate cache for an expensive card build.

    Fixes the Overview pile-up: a slow builder (the behavioral-seizure card;
    the Google-Sheets home grid / KM log) fires on every ~10s refresh tick, and
    with no guard the calls STACK concurrently, thrashing disk / GIL / the Sheets
    network so each one balloons from ~seconds to tens of minutes and the queue
    never drains. Here only ONE thread ever rebuilds; every other caller serves
    the last good result (or no_update if nothing built yet). Results are also
    reused for *ttl* seconds so a fresh tick doesn't recompute at all. The cache
    is a process global (shared across users/threads) -- correct, the card is the
    same for everyone.

    The single-flight guard is a ``SingleFlight`` so a wedged rebuild is
    observable: ``built_at`` / ``age`` expose freshness (so a card can show
    "updated 3 min ago" instead of pretending stale data is live), and a rebuild
    held past ``STALL_SEC`` logs loudly and can be surfaced as a "stalled" state
    (see ``_guarded_card``) rather than an eternal spinner.
    """

    __slots__ = ("_ttl", "_sf", "_t", "_val", "_persist_key")

    def __init__(self, ttl_sec: float, label: str = "cached-build",
                 persist_key: str | None = None):
        self._ttl = ttl_sec
        self._sf = SingleFlight(label)
        self._t = 0.0                        # time.time() of last SUCCESSFUL build
        self._val = None
        # When set, the last-good value is pickled to the ui_snapshot table on
        # every successful build and restored on boot (stale-while-revalidate
        # across restarts). None -> in-memory only. See _persist / restore.
        self._persist_key = persist_key

    def built_at(self) -> float | None:
        """Epoch seconds of the last successful build, or None if never built."""
        return self._t or None

    def age(self) -> float | None:
        """Seconds since the last successful build, or None if never built."""
        return (time.time() - self._t) if self._t else None

    def inflight_for(self) -> float:
        """Seconds the current rebuild has been running (0.0 if none)."""
        return self._sf.held_for()

    def get(self, build, *, label: str | None = None):
        now = time.time()
        val = self._val
        if val is not None and (now - self._t) < self._ttl:
            return val                       # fresh -> serve cached, no rebuild
        if not self._sf.try_begin():
            # A rebuild is already in flight -> don't pile on; serve stale. If it
            # has been wedged too long, say so loudly in the log.
            self._sf.warn_if_stalled()
            return val if val is not None else no_update
        try:
            if self._val is not None and (time.time() - self._t) < self._ttl:
                return self._val             # re-check under the lock
            if label:
                with _perf.Timer(label):
                    built = build()
            else:
                built = build()
            self._val = built
            self._t = time.time()
            self._persist()
            return built
        finally:
            self._sf.end()

    def _persist(self) -> None:
        """Pickle the last-good value to SQLite off the request thread. Best
        effort: any failure (unpicklable tree, DB busy) is swallowed -- the
        snapshot is a pure optimization, never load-bearing. Fire-and-forget so
        the callback return is never delayed by disk I/O / the write lock."""
        store = _SNAPSHOT_STORE
        key = self._persist_key
        if store is None or not key:
            return
        val, built_at = self._val, self._t
        if val is None:
            return

        def _write():
            try:
                import pickle
                blob = pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL)
                store.set_ui_snapshot(key, _SNAPSHOT_VERSION, built_at, blob)
            except Exception as e:  # noqa: BLE001
                logger.debug("snapshot persist failed (%s): %s", key, e)

        threading.Thread(target=_write, daemon=True, name=f"snap-{key}").start()

    def restore(self) -> bool:
        """Load the persisted last-good value into the in-memory cache so the
        first request after a restart paints it immediately. No-op (False) if
        snapshots are off, nothing is stored, the version mismatches, it's older
        than the max age, or it fails to unpickle. Called once at boot, before
        any request can hit the cache."""
        store = _SNAPSHOT_STORE
        key = self._persist_key
        if store is None or not key or self._val is not None:
            return False
        try:
            snap = store.get_ui_snapshot(key)
        except Exception as e:  # noqa: BLE001
            logger.debug("snapshot load failed (%s): %s", key, e)
            return False
        if not snap or snap["version"] != _SNAPSHOT_VERSION:
            return False
        age = time.time() - snap["built_at"]
        if age < 0 or age > _SNAPSHOT_MAX_AGE_SEC:
            return False                      # too stale to paint; rebuild cold
        try:
            import pickle
            val = pickle.loads(snap["blob"])
        except Exception as e:  # noqa: BLE001 -- a shape change across a deploy
            logger.debug("snapshot unpickle failed (%s): %s", key, e)
            return False
        self._val, self._t = val, snap["built_at"]
        logger.info("overview snapshot restored: %s (age %.0f min)",
                    key, age / 60.0)
        return True


# ------------------------------------------------------------------ #
#  Persistent last-good snapshots (stale-while-revalidate across restarts).
#  A _CachedBuilder with a persist_key pickles its last-good card into the
#  ui_snapshot table on every successful build and restores it on boot, so the
#  FIRST request after a daemon restart paints the previous result INSTANTLY
#  instead of everyone racing a cold rebuild. Pure cache: any miss / version
#  mismatch / over-age row just falls back to a fresh build.
# ------------------------------------------------------------------ #
# Bump when a persisted card's component SHAPE changes incompatibly (e.g. a
# rename that would make an old pickle deserialize into a wrong/legacy tree). A
# version mismatch is ignored on restore, forcing a fresh build -- so an old
# schema can never render.
_SNAPSHOT_VERSION = 1
_SNAPSHOT_STORE = None                # bound once in register_callbacks
_SNAPSHOT_MAX_AGE_SEC = 24 * 3600     # don't paint a card older than this on boot
_SNAPSHOT_ENABLED = True


def _bind_snapshot_store(store, config) -> None:
    """Wire the Store used for snapshot persistence + apply config overrides
    (overview.snapshot_enabled / snapshot_max_age_min). Called once at boot,
    when a Store first exists."""
    global _SNAPSHOT_STORE, _SNAPSHOT_MAX_AGE_SEC, _SNAPSHOT_ENABLED
    ov = (config or {}).get("overview", {}) or {}
    _SNAPSHOT_ENABLED = bool(ov.get("snapshot_enabled", True))
    try:
        _SNAPSHOT_MAX_AGE_SEC = max(
            60.0, float(ov.get("snapshot_max_age_min", 24 * 60)) * 60.0)
    except (TypeError, ValueError):
        _SNAPSHOT_MAX_AGE_SEC = 24 * 3600
    _SNAPSHOT_STORE = store if _SNAPSHOT_ENABLED else None


# Behavioral-seizure status card: computed from ~3k files + ~7k review rows and
# fired every 10s -> single-flight + 45s cache so it can't pile up (was ~36 min
# per call under the pile-up; a single build is ~1-3s). Persisted so a restart
# doesn't drop the user back to a cold ~seconds rebuild.
_BSZ_CACHE = _CachedBuilder(45.0, "overview-bsz-status",
                            persist_key="overview:bsz-status")

# Home grid + KM-log cards read GOOGLE SHEETS (surgery / maintenance / schedule /
# data-log). These live in their OWN callbacks (see _fill_home_grid/_fill_km_log)
# so a slow/unreachable Sheets read can't block the DB cards. Each build still
# costs ~10-20s per unreachable sheet (socket timeout), so a 300s single-flight
# cache keeps that stall to at most once / 5 min. Sheets data changes on the
# order of hours, so 5 min is plenty fresh. Persisted: the Sheets round-trip is
# the single worst cold-open cost, so painting yesterday's grid instantly (then
# revalidating) is the biggest win of the snapshot layer.
_HOME_GRID_CACHE = _CachedBuilder(300.0, "overview-home-grid",
                                  persist_key="overview:home-grid")
_KM_LOG_CACHE = _CachedBuilder(300.0, "overview-km-log",
                               persist_key="overview:km-log")

# Single-flight for the heavy (mean/sem/overlay) evoked-thumbnail build so
# concurrent ticks (on load / when the newest-recording signature changes)
# don't stack .mat reads + figure builds. The hist24 path is a cheap poll over
# a background worker and is intentionally NOT gated. SingleFlight adds the
# stall watchdog (the thumbnail's .mat fallback read is the historical wedge).
_THUMB_LOCK = SingleFlight("overview-thumbnail")

# Wall-clock ceiling on any single evokedOutput .mat/share read on a REQUEST or
# worker thread. A healthy share returns a ~100s-of-MB file in a few seconds; a
# hung share used to block forever under _THUMB_LOCK and freeze the whole page.
_THUMB_READ_TIMEOUT = 8.0


def _pending_placeholder(label: str = "Loading…"):
    """Lightweight stand-in for a not-yet-built Overview card. The tab shell
    renders these INSTANTLY; the fill callbacks (mount + refresh-trigger)
    replace them with the real card. Progressive render: the page appears at
    once and each card populates when its callback lands, instead of the whole
    tab blocking on ~12 synchronous DB-backed builds in render_tab."""
    return html.Div(
        f"⏳ {label}",
        style={"color": "#6c6c80", "fontSize": "12px",
               "padding": "14px 6px", "letterSpacing": "0.2px"},
    )


def _freshness_footer(built_at_epoch: float | None):
    """Small themed 'updated HH:MM:SS · N ago' row for a cached card, so stale
    data can never masquerade as live. Green when fresh (<90 min), amber (<24 h),
    red older -- same bands as the KM-log stamp (_fmt_age). Hidden if never
    built."""
    if not built_at_epoch:
        return html.Div()
    age_min = max(0.0, (time.time() - built_at_epoch) / 60.0)
    text, color = _fmt_age(age_min)
    hhmm = datetime.fromtimestamp(built_at_epoch).strftime("%H:%M:%S")
    return html.Div(
        f"updated {hhmm} · {text}",
        style={"color": color, "fontSize": FONT_SIZE_CAPTION,
               "textAlign": "right", "padding": f"{SPACE_1} {SPACE_2} 0",
               "opacity": "0.85"},
    )


def _stalled_banner(title: str, seconds: float, *, compact: bool = False):
    """Themed 'this card's background build looks wedged' banner -- replaces the
    silent, eternal ⏳ so a hang is honest and actionable. Points at the existing
    global Refresh control to retry (a per-card button would collide on id if two
    cards stalled at once)."""
    mins = seconds / 60.0
    msg = (f"⚠ {title} hasn't updated in {mins:.0f} min — the background build "
           f"looks stalled. Click Refresh (top bar) to retry; if it persists the "
           f"daemon may need attention.")
    return html.Div(
        msg,
        style={"color": COLOR_DANGER, "fontSize": FONT_SIZE_CAPTION,
               "background": COLOR_SURFACE_2,
               "border": f"1px solid {COLOR_DANGER}",
               "borderRadius": RADIUS_SM,
               "padding": f"{SPACE_2} {SPACE_3}",
               "marginBottom": (SPACE_1 if compact else SPACE_2)},
    )


def _guarded_card(cache: "_CachedBuilder", build, *, label: str, title: str):
    """Render a _CachedBuilder-backed card with a freshness footer, or a themed
    'stalled' banner when the background build has wedged -- never a silent
    no_update-forever behind an eternal spinner.

    - fresh / just built -> card + 'updated …' footer
    - rebuild in flight, not yet stalled -> last-good card + footer (or the ⏳
      placeholder via no_update if nothing has ever built)
    - rebuild wedged past STALL_SEC -> stalled banner (prepended to any stale
      card so the user still sees the last-good data, clearly marked)
    """
    inflight = cache.inflight_for()          # measure BEFORE get() may build here
    children = cache.get(build, label=label)
    stalled = inflight > STALL_SEC
    if children is no_update:
        # Nothing cached yet AND a rebuild is in flight elsewhere.
        if stalled:
            return _stalled_banner(title, inflight)
        return no_update                      # keep the ⏳ placeholder
    footer = _freshness_footer(cache.built_at())
    # A builder may return a LIST (home-grid, KM-log tiles) or a single
    # component (bsz card). html.Div([list, footer]) nests a list inside
    # children, which Dash 4.1.0 fails to render (raw component dicts reach
    # React -> error #31 -> blank card). Spread lists so children is always a
    # FLAT sequence of components.
    kids = children if isinstance(children, list) else [children]
    if stalled:
        return html.Div([_stalled_banner(title, inflight, compact=True),
                         *kids, footer])
    return html.Div([*kids, footer])


# Shared row style for the "today at the lab" home-grid block rows.
_HOME_ROW_STYLE = {
    "display": "flex", "alignItems": "center",
    "padding": f"{SPACE_1} 0",
}


# ===================================================================== #
#  Widget helpers (Overview-only)
# ===================================================================== #

def _collapsible(title: str, content, *, open_default: bool = True,
                  badge: str | None = None,
                  badge_color: str | None = None,
                  full_width: bool = False,
                  title_color: str = "#ddd"):
    """Section wrapped in a native <details> element so the user can
    collapse anything they don't want to see.

    Apple HIG: one container per section, not nested chrome. The
    collapsible IS the card. Inner content should NOT carry its own
    border/background/padding -- it'd duplicate the visual frame.
    The summary row is the title strip; the body just gets a thin
    horizontal padding so the inner figure / table edges align.
    """
    summary_children = [
        html.Span(title, style={
            "color": title_color, "fontSize": "12px", "fontWeight": "600",
            "letterSpacing": "0.2px",
        }),
    ]
    if badge:
        summary_children.append(html.Span(
            badge,
            style={
                "marginLeft": "6px",
                "padding": "0 6px",
                "borderRadius": "8px",
                "background": (badge_color or "rgba(255,255,255,0.08)"),
                "color": "#fff", "fontSize": "10px",
                "fontWeight": "600", "lineHeight": "16px",
            }))
    base_style = {
        "background": COLOR_SURFACE_1,
        "border": f"1px solid {COLOR_DIVIDER}",
        "borderRadius": "6px",
        "marginBottom": "6px",
    }
    if full_width:
        base_style["gridColumn"] = "1 / -1"
    return html.Details([
        html.Summary(summary_children, style={
            "cursor": "pointer", "userSelect": "none",
            "padding": "5px 10px",
            "borderBottom":
                "1px solid rgba(255,255,255,0.04)",
            "listStyle": "none",
        }),
        html.Div(content, style={"padding": "6px 10px"}),
    ], open=open_default, style=base_style)



def _fmt_disk_space(gb: float) -> str:
    """Format a disk-free value with auto-scaling unit.

    Anything 1024 GB and up reads as TB with one decimal (1.2 TB);
    smaller values stay as GB without a decimal (543 GB). Matches
    the "round up to the natural unit" pattern macOS Finder uses.
    """
    if gb is None:
        return "—"
    try:
        gb_f = float(gb)
    except (TypeError, ValueError):
        return "—"
    if gb_f >= 1024:
        return f"{gb_f / 1024:.1f} TB"
    return f"{gb_f:.0f} GB"


def _status_pill(title: str, value: str, color: str = "#636EFA"):
    """Compact single-line pill. ~28px tall instead of _status_card's
    ~72px. Reference info doesn't need to read as a headline; the
    operator only consults these when something looks wrong, so they
    earn their visual weight from color (not size)."""
    return html.Div([
        html.Span(title, style={
            "color": "#888", "fontSize": "10px",
            "textTransform": "uppercase",
            "letterSpacing": "0.5px",
            "marginRight": "6px",
        }),
        html.Span(value, style={
            "color": color, "fontSize": "13px",
            "fontWeight": "600",
        }),
    ], style={
        "padding": "4px 10px",
        "background": COLOR_SURFACE_1,
        "border": f"1px solid {COLOR_DIVIDER}",
        "borderRadius": "999px",
        "fontFamily": "ui-monospace, SF Mono, monospace",
        "whiteSpace": "nowrap",
    })


# _empty_fig moved to src/dashboard/data_helpers.empty_fig. The
# alias below keeps the 19 in-module callsites working unchanged.
from src.dashboard.data_helpers import empty_fig as _empty_fig  # noqa: E402,F401


# _empty_state moved to src/dashboard/components.tab_empty_state.
# Aliased here so the in-module callsites keep working.
from src.dashboard.components import (  # noqa: E402,F401
    tab_empty_state as _empty_state,
)


# _session_dropdown_options moved to src/dashboard/data_helpers.
from src.dashboard.data_helpers import (  # noqa: E402,F401
    session_dropdown_options as _session_dropdown_options,
)


# _default_session moved to src/dashboard/data_helpers.default_session.
from src.dashboard.data_helpers import (  # noqa: E402,F401
    default_session as _default_session,
)


# _get_processed_files_for_session moved to data_helpers (re-imported above).




def _home_surgery_block(store: Store, config: dict | None,
                         today: date) -> html.Div:
    """Today's surgery checklist in compact form."""
    try:
        from src.notifications.surgery_digest import _build_today
        grouped, _upcoming, _ = _build_today(today, config or {})
    except Exception as e:
        logger.warning("Home: surgery block failed: %s", e)
        grouped = {}

    rows = []
    bucket_labels = [
        ("Pre-op (Motrin)", "Pre-op (Motrin)"),
        ("Surgery day", "Surgery day"),
        ("Post-op day 1", "Post-op day 1"),
        ("Post-op day 2–3", ["Post-op day 2", "Post-op day 3"]),
    ]
    for label, key in bucket_labels:
        if isinstance(key, list):
            items = []
            for k in key:
                items.extend(grouped.get(k, []))
        else:
            items = grouped.get(key, [])
        n = len(items)
        animals = ", ".join(t["animal"] for t in items[:6])
        color = COLOR_TEXT_PRIMARY if n else COLOR_TEXT_TERTIARY
        rows.append(html.Div([
            html.Span(label, style={"flex": "0 0 200px", "color": color}),
            html.Span(str(n), style={"flex": "0 0 32px",
                                     "color": color, "fontWeight": "600"}),
            html.Span(animals, style={"flex": "1",
                                       "color": COLOR_TEXT_SECONDARY,
                                       "fontSize": FONT_SIZE_CAPTION}),
        ], style=_HOME_ROW_STYLE))

    return _card(
        _section_header("Surgeries today",
                         on_open_id="home-open-surgeries"),
        html.Div(rows),
    )


def _home_maintenance_block(config: dict | None) -> html.Div:
    """Today's maintenance status, mirroring the Apps Script truth."""
    try:
        from src.dashboard.tabs.maintenance import rig_status
        rows = rig_status(config or {})
    except Exception as e:
        logger.warning("Home: maintenance block failed: %s", e)
        rows = []

    if not rows:
        body = html.Div("Maintenance tracker disabled or no data.",
                        style={"color": COLOR_TEXT_TERTIARY,
                               "fontSize": FONT_SIZE_BODY})
    else:
        attention = [r for r in rows
                     if r.status in ("pending", "overdue")]
        nothing_active = all(r.status == "not_active" for r in rows)
        if nothing_active:
            body = html.Div(
                "No active rigs today.",
                style={"color": COLOR_TEXT_TERTIARY,
                       "fontSize": FONT_SIZE_BODY},
            )
        elif not attention:
            done_count = sum(1 for r in rows if r.status == "done")
            if done_count:
                msg = f"All {done_count} scheduled task(s) done today."
                color = COLOR_SUCCESS
            else:
                msg = "Nothing scheduled today."
                color = COLOR_ACCENT
            body = html.Div(msg, style={
                "color": color, "fontSize": FONT_SIZE_BODY,
                "fontWeight": "600",
            })
        else:
            by_rig: dict[str, list] = {}
            for r in attention:
                by_rig.setdefault(r.rig, []).append(r)
            body_rows = []
            for rig in sorted(by_rig.keys()):
                chunks = []
                for r in by_rig[rig]:
                    short = ("battery" if r.task == "battery" else "cage")
                    label = short
                    if r.task == "battery" and r.pending_shelves:
                        label += (" shelf "
                                  + ",".join(map(str, r.pending_shelves)))
                    chunks.append(html.Span(
                        _pill(r.status, label=label),
                        style={"marginRight": SPACE_2},
                    ))
                body_rows.append(html.Div([
                    html.Span(f"Rig {rig}", style={
                        "flex": "0 0 80px", "fontWeight": "600",
                        "color": COLOR_TEXT_PRIMARY,
                    }),
                    html.Span(chunks, style={"flex": "1",
                                              "fontSize": FONT_SIZE_CAPTION}),
                ], style=_HOME_ROW_STYLE))
            body = html.Div(body_rows)

    return _card(
        _section_header("Maintenance",
                         on_open_id="home-open-maintenance"),
        body,
    )


def _home_schedule_block(config: dict | None) -> html.Div:
    """Today's scheduled tasks + assignees, sourced from the
    Maintenance Tracker '📅 Schedule' tab."""
    try:
        from src.dashboard.tabs.maintenance import todays_schedule
        rows = todays_schedule(config or {})
    except Exception as e:
        logger.warning("Home: schedule block failed: %s", e)
        rows = []

    if not rows:
        body = html.Div(
            "Nothing scheduled today.",
            style={"color": COLOR_TEXT_TERTIARY,
                   "fontSize": FONT_SIZE_BODY},
        )
    else:
        by_task: dict[str, list] = {}
        for r in rows:
            by_task.setdefault(r["task"], []).append(r)
        items = []
        for task in sorted(by_task.keys()):
            names = ", ".join(r["name"] or r["email"]
                              for r in by_task[task]
                              if (r["name"] or r["email"]))
            items.append(html.Div([
                html.Span(task, style={"flex": "0 0 200px",
                                        "color": COLOR_TEXT_PRIMARY,
                                        "fontWeight": "600"}),
                html.Span(names or "-",
                          style={"flex": "1",
                                 "color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION}),
            ], style=_HOME_ROW_STYLE))
        body = html.Div(items)

    return _card(
        _section_header("Today's schedule",
                         on_open_id="home-open-schedule"),
        body,
    )


import re as _re_incident

# Software is checked FIRST so "dashboard crashed" wins over the
# substring "board" in the hardware bucket.
_INCIDENT_TAGS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("software", "#AB63FA",
        ("software", "code", "crash", "freeze", "hang", "deploy",
         "bug", "version", "update", "script", "matlab", "python",
         "dashboard", "kmrecord")),
    ("signal", "#19D3F3",
        ("noise", "drift", "artifact", "spike", "baseline",
         "channel", "filter", "60hz", "stim", "amplifier",
         "amp")),
    ("hardware", "#FFA15A",
        ("battery", "batteries", "electrode", "headstage", "cable",
         "connector", "board", "circuit", "broken", "charge",
         "charging", "wire", "led", "solder", "pin", "shaft")),
)
# Word-boundary regex per word so "dashboard" stops matching "board",
# "line" stops matching "online", etc. Compiled once.
_INCIDENT_PATTERNS: list[tuple[str, str, _re_incident.Pattern]] = [
    (label, color,
     _re_incident.compile(r"\b(" + "|".join(_re_incident.escape(w)
                                              for w in words) + r")",
                           _re_incident.IGNORECASE))
    for label, color, words in _INCIDENT_TAGS
]


def _classify_incident(text: str) -> tuple[str, str] | None:
    """First matching (label, color) for a free-form incident report,
    or None when no keyword family fires. Word-boundary regex --
    'Dashboard crashed' must not classify as hardware via 'board'."""
    if not text:
        return None
    for label, color, pattern in _INCIDENT_PATTERNS:
        if pattern.search(text):
            return (label, color)
    return None


def _incident_age_style(days_old: float | None) -> dict:
    """Visual aging for incidents based on how long ago they happened.

    Fresh (0-2 d): full opacity, normal weight.
    Recent (3-14 d): slight fade.
    Stale (15+ d): heavy fade so old open items don't read identical
    to fresh ones at a glance.
    """
    if days_old is None or days_old < 3:
        return {"opacity": "1.0"}
    if days_old < 15:
        return {"opacity": "0.75"}
    return {"opacity": "0.5"}


def _fmt_days_ago(days_old: float | None) -> tuple[str, str]:
    """(label, color) for a 'N days ago' badge."""
    if days_old is None:
        return "", "#888"
    if days_old < 1:
        return "today", "#00CC96"
    if days_old < 2:
        return "yesterday", "#00CC96"
    if days_old < 7:
        return f"{int(days_old)}d ago", "#FFA15A"
    if days_old < 30:
        return f"{int(days_old)}d ago", "#EF553B"
    return f"{int(days_old)}d ago", "#EF553B"


def _home_incidents_block(config: dict | None) -> html.Div:
    """Latest 3 incident reports from the Maintenance Tracker.

    Stale incidents fade visually so an open battery report from
    three weeks ago doesn't claim the same attention as a fresh
    one. Each item gets a keyword-derived severity badge (hardware
    / signal / software) so the operator can triage at a glance.
    """
    try:
        from src.dashboard.tabs.maintenance import recent_incidents
        rows = recent_incidents(config or {}, limit=3)
    except Exception as e:
        logger.warning("Home: incidents block failed: %s", e)
        rows = []

    if not rows:
        body = html.Div(
            "No incident reports.",
            style={"color": COLOR_TEXT_TERTIARY,
                   "fontSize": FONT_SIZE_BODY},
        )
    else:
        now = datetime.now()
        items = []
        for r in rows:
            when_dt = r["when"]
            when = when_dt.strftime("%Y-%m-%d %H:%M")
            who = r.get("by") or "?"
            report = r.get("report") or ""
            if len(report) > 140:
                report = report[:137] + "..."
            try:
                days_old = max(0.0,
                                (now - when_dt).total_seconds() / 86400.0)
            except (TypeError, ValueError):
                days_old = None
            age_label, age_color = _fmt_days_ago(days_old)
            tag = _classify_incident(report)
            age_style = _incident_age_style(days_old)
            header_children = [
                html.Span(when, style={
                    "color": COLOR_TEXT_SECONDARY,
                    "fontSize": FONT_SIZE_CAPTION,
                    "marginRight": SPACE_3,
                }),
                html.Span(who, style={
                    "color": COLOR_ACCENT,
                    "fontSize": FONT_SIZE_CAPTION,
                    "fontWeight": "600",
                    "marginRight": SPACE_3,
                }),
            ]
            if age_label:
                header_children.append(html.Span(
                    age_label,
                    style={
                        "color": age_color, "fontSize": "10px",
                        "fontWeight": "600",
                        "marginRight": SPACE_3,
                    }))
            if tag is not None:
                tag_label, tag_color = tag
                header_children.append(html.Span(
                    tag_label,
                    style={
                        "color": "#fff", "fontSize": "9px",
                        "fontWeight": "600",
                        "background": tag_color,
                        "padding": "1px 6px",
                        "borderRadius": "8px",
                        "letterSpacing": "0.3px",
                        "textTransform": "uppercase",
                    }))
            items.append(html.Div([
                html.Div(header_children, style={
                    "display": "flex", "alignItems": "center",
                    "flexWrap": "wrap", "rowGap": "2px",
                }),
                html.Div(report, style={
                    "color": COLOR_TEXT_PRIMARY,
                    "fontSize": FONT_SIZE_BODY,
                    "marginTop": "2px",
                }),
            ], style={
                "padding": f"{SPACE_2} 0",
                "borderBottom": f"1px solid {COLOR_DIVIDER}",
                **age_style,
            }))
        body = html.Div(items)

    return _card(
        _section_header("Recent incidents",
                         on_open_id="home-open-incidents"),
        body,
    )


def _home_data_log_block(store: Store, config: dict | None) -> html.Div:
    try:
        from src.dashboard.tabs.data_log_xref import diff_counts
        counts = diff_counts(store, config or {})
    except Exception as e:
        logger.warning("Home: data log block failed: %s", e)
        counts = {"enabled": False}

    if not counts.get("enabled"):
        body = html.Div("Data log cross-reference disabled.",
                        style={"color": COLOR_TEXT_TERTIARY,
                               "fontSize": FONT_SIZE_BODY})
    else:
        ln = counts.get("logged_not_qcd", 0)
        qn = counts.get("qcd_not_logged", 0)
        if ln == 0 and qn == 0:
            body = html.Div("Sheet and DB are in sync.",
                            style={"color": COLOR_SUCCESS,
                                   "fontSize": FONT_SIZE_BODY,
                                   "fontWeight": "600"})
        else:
            body = html.Div([
                html.Span(f"{ln} ", style={"color": COLOR_WARNING,
                                            "fontWeight": "700",
                                            "fontSize": "16px"}),
                html.Span("logged, not QC'd",
                         style={"color": COLOR_TEXT_SECONDARY,
                                "marginRight": SPACE_5,
                                "fontSize": FONT_SIZE_BODY}),
                html.Span(f"{qn} ", style={"color": COLOR_ACCENT,
                                            "fontWeight": "700",
                                            "fontSize": "16px"}),
                html.Span("QC'd, not logged",
                         style={"color": COLOR_TEXT_SECONDARY,
                                "fontSize": FONT_SIZE_BODY}),
            ])

    return _card(
        _section_header("Data log diff",
                         on_open_id="home-open-datalog"),
        body,
    )


def _home_quick_links_block(config: dict | None) -> html.Div:
    """One tile holding every external resource the Overview cards
    point at (Google Sheets, drives, dashboards) so the operator
    doesn't have to dig into config or a separate doc to find a URL.

    Apple HIG: grouped list of related entry points. Each row is a
    plain anchor opening in a new tab; the sheet_id comes from
    config (same source the digest / data-log tabs already use).
    """
    cfg = config or {}
    links: list[tuple[str, str, str]] = []

    def _sheet_url(sid: str) -> str:
        return f"https://docs.google.com/spreadsheets/d/{sid}/edit"

    # Maintenance + incidents (one sheet, multiple tabs)
    maint_id = (cfg.get("maintenance", {}) or {}).get("sheet_id")
    if maint_id:
        links.append(("Maintenance tracker", "sheet",
                      _sheet_url(maint_id)))

    # Surgery / implant sheets
    for s in (cfg.get("surgeries", {}) or {}).get("sheets", []) or []:
        sid = s.get("sheet_id")
        label = s.get("label") or "Surgery log"
        if sid:
            links.append((label, "sheet", _sheet_url(sid)))

    # JAX cage index
    cage_cfg = ((cfg.get("surgeries", {}) or {})
                 .get("cage_index", {}) or {})
    cage_id = cage_cfg.get("sheet_id")
    if cage_id and cage_cfg.get("enabled"):
        links.append(("JAX cage index", "sheet", _sheet_url(cage_id)))

    # KMrecorder Data Log
    xref_id = (cfg.get("data_log_xref", {}) or {}).get("sheet_id")
    if xref_id:
        links.append(("KMrecorder Data Log", "sheet",
                      _sheet_url(xref_id)))

    if not links:
        body = html.Div(
            "No external resources configured.",
            style={"color": COLOR_TEXT_TERTIARY,
                    "fontSize": FONT_SIZE_BODY},
        )
    else:
        items = []
        for label, kind, href in links:
            tag_color = ("#5e7ce2" if kind == "sheet"
                          else COLOR_TEXT_TERTIARY)
            items.append(html.A([
                html.Span(label, style={
                    "color": COLOR_TEXT_PRIMARY,
                    "fontSize": FONT_SIZE_BODY,
                    "flex": "1",
                }),
                html.Span(kind.upper(), style={
                    "color": tag_color,
                    "fontSize": "10px", "letterSpacing": "0.5px",
                    "fontWeight": "600",
                    "marginLeft": SPACE_3,
                }),
            ], href=href, target="_blank", rel="noopener",
               style={
                   "display": "flex", "alignItems": "center",
                   "padding": f"{SPACE_2} 0",
                   "borderBottom": f"1px solid {COLOR_DIVIDER}",
                   "textDecoration": "none",
               }))
        body = html.Div(items)

    return _card(
        _section_header("Quick links"),
        body,
    )


def _build_home_grid_children(store: Store, config: dict | None,
                                today: date) -> list:
    """Lab-side home tiles. Extracted so the refresh callback can
    rebuild them in place without re-rendering everything else."""
    return [
        _home_surgery_block(store, config, today),
        _home_schedule_block(config),
        _home_maintenance_block(config),
        _home_incidents_block(config),
        _home_data_log_block(store, config),
        _home_quick_links_block(config),
    ]


def _build_overview_cards(store: Store):
    """Inner content for the system-health card strip. Returned without
    a wrapping Div so a callback can swap it via Patch / children."""
    health = store.get_health_history(hours=1)
    latest = health[-1] if health else {}
    net_ok = latest.get("network_share_accessible")
    cpu = latest.get("cpu_pct", 0)
    mem = latest.get("memory_pct", 0)
    disk_pc = latest.get("disk_free_gb", 0)       # analysis PC's own drive
    disk_data = latest.get("data_disk_free_gb")   # recording share (where data lands)
    fph = latest.get("files_processed_last_hour", 0)
    # Video QC: 24-hour status mix. Color reflects worst severity in
    # the window (critical > warning > ok). Empty window stays gray.
    try:
        videos = store.get_recent_video_qc(hours=24)
    except Exception:
        videos = []
    n_ok = sum(1 for v in videos if v.get("status") == "ok")
    n_warn = sum(1 for v in videos if v.get("status") == "warning")
    n_crit = sum(1 for v in videos if v.get("status") == "critical")
    if videos:
        vid_label = f"{n_ok}/{n_warn}/{n_crit}"
    else:
        vid_label = "—"
    # Muted palette: a healthy value should read as CALM, not shout. Only
    # attention states (warning amber / critical red) get a loud color; "good"
    # is a quiet slate-gray close to the surface so the strip stops reading as a
    # wall of green noise.
    calm = "#8890a4"
    vid_color = ("#EF553B" if n_crit
                  else "#FFA15A" if n_warn
                  else calm if n_ok
                  else "#666")

    def _disk_color(gb):
        # Low free space is the only disk state worth a loud color.
        return "#EF553B" if (gb is not None and gb <= 50) else calm

    return [
        _status_pill("Network", "OK" if net_ok else "DOWN",
                     calm if net_ok else "#EF553B"),
        _status_pill("CPU", f"{cpu:.0f}%",
                     calm if cpu < 80 else "#FFA15A"),
        _status_pill("Memory", f"{mem:.0f}%",
                     calm if mem < 85 else "#FFA15A"),
        # Two disks, delineated: where recordings actually land vs this PC.
        _status_pill("Data Free",
                     _fmt_disk_space(disk_data) if disk_data is not None else "—",
                     _disk_color(disk_data)),
        _status_pill("PC Free", _fmt_disk_space(disk_pc), _disk_color(disk_pc)),
        _status_pill("Files/Hour", str(fph), calm),
        _status_pill("Videos (24h)", vid_label, vid_color),
    ]


def _yrange_pad(time_ms, trace, x0: float, x1: float,
                  fill_frac: float = 0.8) -> tuple[float, float] | None:
    """Y-axis range such that data within [x0, x1] spans *fill_frac*
    of the plot's vertical extent (default 80%, i.e. 10% padding each
    side). Returns None when there are fewer than 3 finite samples in
    the window so the caller can fall back to Plotly's autorange.
    """
    if not time_ms or not trace or len(time_ms) != len(trace):
        return None
    vals = [v for t, v in zip(time_ms, trace)
            if t is not None and v is not None
            and x0 <= t <= x1 and isinstance(v, (int, float)) and v == v]
    if len(vals) < 3:
        return None
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    if span <= 0:
        # Flat trace; give the axis a tiny window so the line draws.
        eps = abs(hi) * 0.05 if hi != 0 else 1.0
        return lo - eps, hi + eps
    # We want data span = fill_frac * total span ⇒ total span =
    # data_span / fill_frac. Pad = (total - data) / 2 on each side.
    pad = span * (1.0 - fill_frac) / (2.0 * fill_frac)
    return lo - pad, hi + pad


def _decimate_trace(t, m, s, target: int = 1400, fine_ms: float = 2.0):
    """Decimate three aligned numpy arrays to ~*target* points for the
    figure, but keep FULL resolution within +/- *fine_ms* of t=0.

    The evoked file's time axis is ~20 k samples over +/-500 ms (20 kHz);
    a uniform stride to *target* points lands ~1 ms apart and strides
    right past the sub-millisecond stimulus artifact at t=0 -- which is
    exactly what the +/-1 ms stim-window panel needs. So we preserve
    every sample near t=0 and only stride-decimate the slow far field.
    """
    import numpy as np
    assert len(t) == len(m) == len(s), "trace arrays must align"
    assert target > 0, "target must be positive"
    n = len(t)
    if n <= target:
        return t.tolist(), m.tolist(), s.tolist()
    fine = np.abs(np.asarray(t)) <= fine_ms
    coarse_idx = np.where(~fine)[0]
    budget = max(1, target - int(fine.sum()))
    step = max(1, coarse_idx.size // budget)
    keep = fine.copy()
    keep[coarse_idx[::step]] = True
    return t[keep].tolist(), m[keep].tolist(), s[keep].tolist()


def _evoked_channels_to_waveforms(chans: dict, latest_dt) -> list[dict]:
    """Shape ``read_file_evoked`` output into the per-channel waveform
    dicts the thumbnail expects (mean/SEM across this file's epochs).

    Only real animal channels are kept (stimCopy/ref dropped). SEM =
    std/sqrt(n). Channel index is positional since evokedOutput keys on
    channel name, not the DB's integer index.
    """
    import numpy as np
    from src.utils.animal import is_animal_channel
    assert isinstance(chans, dict), "chans must be a dict"
    assert latest_dt is not None, "latest_dt required"
    chunk_dt = latest_dt.strftime(_CHUNK_DT_FMT)
    out: list[dict] = []
    for ch_name in sorted(chans.keys()):
        if not is_animal_channel(ch_name):
            continue
        rec = chans[ch_name]
        traces = rec.get("traces")
        time_ms = rec.get("time_ms")
        if traces is None or time_ms is None or len(traces) == 0:
            continue
        arr = np.asarray(traces, dtype=np.float64)
        n_ep = int(arr.shape[0])
        mean_tr = arr.mean(axis=0)
        sem_tr = (arr.std(axis=0, ddof=1) / np.sqrt(n_ep)
                  if n_ep > 1 else np.zeros_like(mean_tr))
        t, m, s = _decimate_trace(np.asarray(time_ms, dtype=np.float64),
                                  mean_tr, sem_tr)
        out.append({
            "channel": len(out), "channel_name": ch_name,
            "n_epochs": n_ep, "time_axis_ms": t,
            "mean_trace": m, "sem_trace": s, "chunk_datetime": chunk_dt,
        })
    return out


# Memoize the evokedOutput fallback by (path, mtime) so the Overview
# refresh tick doesn't re-read the (potentially ~100s-of-MB) latest
# *_evoked.mat every few seconds. Invalidates when the file changes.
_EVOKED_FALLBACK_CACHE: dict = {"key": None, "value": ([], "")}

# Memoize the snapshot caption's camera count by file_path -- companion_video_
# paths globs the SMB share (GIL-holding), and the caption callback fires every
# refresh tick. Keyed by the newest-video file_path (changes ~hourly).
_SNAP_CAM_CACHE: dict = {}


def _latest_evoked_from_output(config: dict | None) -> tuple[list[dict], str]:
    """Per-channel mean traces from the most recent ``*_evoked.mat`` in
    the toolkit's evokedOutput folder.

    Fallback source for the Latest-Evoked thumbnail when QC's own
    ``evoked_waveforms`` DB table has no rows for the newest session
    (the dispatcher's MATLAB Tier-2 hasn't repopulated it, but the
    toolkit already processed the recording into evokedOutput). Returns
    ``(waveforms, latest_chunk_dt)`` or ``([], "")`` when nothing's there.
    """
    ce = (config or {}).get("chronic_evoked", {}) or {}
    try:
        from src.utils.evoked_output import (
            DEFAULT_EVOKED_DIR, list_evoked_files, parse_recording_dt,
            read_file_evoked)
    except Exception as e:
        logger.debug("evokedOutput fallback import failed: %s", e)
        return [], ""
    evoked_dir = ce.get("evoked_output_dir") or DEFAULT_EVOKED_DIR
    dated = [(parse_recording_dt(f), f)
             for f in list_evoked_files(evoked_dir)]
    dated = [(d, f) for d, f in dated if d is not None]
    if not dated:
        return [], ""
    latest_dt, latest_path = max(dated, key=lambda t: t[0])
    try:
        key = (latest_path, os.path.getmtime(latest_path))
    except OSError:
        key = (latest_path, 0.0)
    cache = _EVOKED_FALLBACK_CACHE
    if cache["key"] == key:
        return cache["value"]
    try:
        chans = read_file_evoked(latest_path)
    except Exception as e:
        logger.debug("evokedOutput fallback read failed: %s", e)
        return [], ""
    waveforms = _evoked_channels_to_waveforms(chans, latest_dt)
    value = (waveforms,
             latest_dt.strftime(_CHUNK_DT_FMT) if waveforms else "")
    cache["key"], cache["value"] = key, value
    return value


# ------------------------------------------------------------------ #
#  "Last 24 recordings" overlay -- background-computed + cached.
#  Reading 24 *_evoked.mat (h5py-decoding ~20k-sample evokedData each) is too
#  heavy for the refresh thread, so a daemon worker builds the per-channel mean
#  traces off-thread and publishes them under a lock; a poll Interval drives a
#  "Computing... (n/24)" state. Mirrors chronic_evoked's _kick_warm /
#  _warm_worker / chronic-warm-poll pattern.
# ------------------------------------------------------------------ #
_HIST24_N = 24               # default; overridable via UI + config
_HIST24_MAX = 500            # hard cap (NASA Rule 2 loop bound)
_HIST24_LOCK = threading.Lock()
# Overlay thumbnail: newest N historical files drawn under the mean (each
# decimated) so "overlay" mode doesn't serialize hundreds of raw traces.
_OVERLAY_MAX_FILES = 30


def _hist24_default_n(config: dict | None) -> int:
    """Default recording count for the recent-recordings overlay --
    config.overview.hist_recordings, else _HIST24_N."""
    ov = (config or {}).get("overview", {}) or {}
    try:
        return max(1, min(int(ov.get("hist_recordings", _HIST24_N)),
                          _HIST24_MAX))
    except (TypeError, ValueError):
        return _HIST24_N


def _clamp_hist_n(n) -> int:
    try:
        return max(1, min(int(n), _HIST24_MAX))
    except (TypeError, ValueError):
        return _HIST24_N
# Published atomically by the worker; readers compare sig before use.
_HIST24_CACHE: dict = {"sig": None, "per_ch": None}
_HIST24_PROGRESS: dict = {"done": 0, "total": 0, "phase": "idle", "sig": None}
_HIST24_THREAD = None        # threading.Thread | None
# Per-file memo: (path, mtime) -> that file's decimated per-channel waveforms.
# So a window shift (one new recording in, oldest out) only re-reads the ONE
# new file -- a cold read of all 24 is ~2 min; a single file is a few seconds.
_HIST24_FILE_CACHE: dict = {}
_HIST24_FILE_CACHE_MAX = 48


def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _hist24_file_set(config: dict | None, n: int | None = None):
    """The last *n* evoked recordings (time-ordered) + a (path, mtime)
    signature. Cheap enough for the callback thread. Returns ``(sig, files)``
    with ``files = [(path, dt), ...]`` oldest->newest, or ``(None, [])``.
    *n* defaults to the config value (clamped to [1, _HIST24_MAX])."""
    assert config is None or isinstance(config, dict), "config dict|None"
    n = _clamp_hist_n(n if n is not None else _hist24_default_n(config))
    ce = (config or {}).get("chronic_evoked", {}) or {}
    try:
        from src.utils.evoked_output import (
            DEFAULT_EVOKED_DIR, list_evoked_files, parse_recording_dt)
    except Exception as e:  # noqa: BLE001
        logger.debug("hist24 import failed: %s", e)
        return None, []
    evoked_dir = ce.get("evoked_output_dir") or DEFAULT_EVOKED_DIR
    # This runs on the callback (request) thread, so bound the share glob: a hung
    # evokedOutput mount would otherwise block a Dash worker here.
    listed = call_with_deadline(
        lambda: list_evoked_files(evoked_dir), _THUMB_READ_TIMEOUT, default=None)
    if listed is None:
        logger.warning("hist24 file listing timed out (evoked_dir=%s)", evoked_dir)
        return None, []
    dated = [(parse_recording_dt(f), f) for f in listed]
    dated = [(d, f) for d, f in dated if d is not None]
    if not dated:
        return None, []
    dated.sort(key=lambda t: t[0])                       # oldest -> newest
    files = [(f, d) for d, f in dated[-n:]]
    sig = tuple((p, _safe_mtime(p)) for p, _ in files)
    return sig, files


def _set_hist24_progress(done, total, phase, sig) -> None:
    with _HIST24_LOCK:
        _HIST24_PROGRESS.update(done=int(done), total=int(total),
                                phase=phase, sig=sig)


def _hist24_file_waveforms(path: str, dt) -> list:
    """One file's decimated per-channel mean waveforms, memoized by
    (path, mtime). The heavy h5py decode happens once per recording; a bad
    file yields []."""
    assert path, "path required"
    fkey = (path, _safe_mtime(path))
    with _HIST24_LOCK:
        cached = _HIST24_FILE_CACHE.get(fkey)
    if cached is not None:
        return cached
    try:
        from src.utils.evoked_output import read_file_evoked
        # Bound the h5py decode: a hung share read on the worker thread would
        # otherwise block hist24 forever (and _kick_hist24 gates re-kicks on
        # is_alive(), so one wedged read stalls ALL future overlays).
        chans = call_with_deadline(
            lambda: read_file_evoked(path), _THUMB_READ_TIMEOUT, default=None)
        if chans is None:
            logger.warning("hist24 read timed out: %s", path)
            wfs = []
        else:
            wfs = _evoked_channels_to_waveforms(chans, dt)
    except Exception as e:  # noqa: BLE001 -- skip a bad file, keep going
        logger.debug("hist24 read failed %s: %s", path, e)
        wfs = []
    with _HIST24_LOCK:
        _HIST24_FILE_CACHE[fkey] = wfs
        while len(_HIST24_FILE_CACHE) > _HIST24_FILE_CACHE_MAX:
            _HIST24_FILE_CACHE.pop(next(iter(_HIST24_FILE_CACHE)))
    return wfs


def _hist24_worker(sig, files) -> None:
    """Daemon: read each (<=24) file (memoized), group decimated per-channel
    means by channel name, publish atomically. Never raises (mirrors chronic
    ``_warm_worker``)."""
    try:
        n = len(files)
        per_ch: dict[str, list[dict]] = {}
        for i, (path, dt) in enumerate(files):
            assert i < _HIST24_MAX, "hist24 file loop runaway"
            _set_hist24_progress(i, n, "reading", sig)
            wfs = _hist24_file_waveforms(path, dt)
            for wf in wfs:
                per_ch.setdefault(wf["channel_name"], []).append({
                    "time_axis_ms": wf["time_axis_ms"],
                    "mean_trace": wf["mean_trace"],
                    "chunk_datetime": wf["chunk_datetime"],
                    "n_epochs": wf["n_epochs"],
                })
        with _HIST24_LOCK:
            _HIST24_CACHE["sig"], _HIST24_CACHE["per_ch"] = sig, per_ch
            _HIST24_PROGRESS.update(done=n, total=n, phase="done", sig=sig)
        logger.info("hist24 overlay built: %d files, %d channels",
                    n, len(per_ch))
    except Exception as e:  # noqa: BLE001 -- never crash the daemon
        logger.warning("hist24 worker failed: %s", e)


def _kick_hist24(sig, files) -> bool:
    """Start the background build unless one is already running (mirrors
    ``_kick_warm``). A stale in-flight run finishes; the next poll re-kicks."""
    global _HIST24_THREAD
    with _HIST24_LOCK:
        t = _HIST24_THREAD
        if t is not None and t.is_alive():
            return False
        _HIST24_PROGRESS.update(done=0, total=len(files), phase="reading",
                                sig=sig)
        th = threading.Thread(target=_hist24_worker, args=(sig, files),
                              daemon=True, name="overview-hist24")
        _HIST24_THREAD = th
        th.start()
        return True


def _hist24_ready(sig):
    """The cached per-channel data iff it matches *sig*, else None."""
    with _HIST24_LOCK:
        if (_HIST24_CACHE["sig"] == sig
                and _HIST24_CACHE["per_ch"] is not None):
            return _HIST24_CACHE["per_ch"]
    return None


def _hist24_progress() -> dict:
    with _HIST24_LOCK:
        return dict(_HIST24_PROGRESS)


def _turbo_ramp(n: int) -> list:
    """Turbo colors oldest->newest. Local copy of chronic_evoked._time_colors
    to avoid importing that heavy module for 4 lines."""
    from plotly.colors import sample_colorscale
    if n <= 1:
        return ["#5e7ce2"] * max(1, n)
    return sample_colorscale("Turbo", [i / (n - 1) for i in range(n)])


def _yrange_pad_multi(traces, x0: float, x1: float,
                      fill_frac: float = 0.8):
    """Combined y-range (data fills *fill_frac* of the panel) across several
    ``(time_ms, trace)`` pairs within [x0, x1]; None when <3 finite samples.
    Same padding math as ``_yrange_pad`` but over the overlay set."""
    assert traces is not None, "traces required"
    assert x0 < x1, "x0 < x1"
    vals: list = []
    for i, (t, m) in enumerate(traces):
        assert i < _HIST24_MAX, "hist24 yrange loop runaway"
        if not t or not m or len(t) != len(m):
            continue
        vals.extend(
            v for tv, v in zip(t, m)
            if tv is not None and v is not None and x0 <= tv <= x1
            and isinstance(v, (int, float)) and v == v)
    if len(vals) < 3:
        return None
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        eps = abs(hi) * 0.05 if hi != 0 else 1.0
        return lo - eps, hi + eps
    pad = span * (1.0 - fill_frac) / (2.0 * fill_frac)
    return lo - pad, hi + pad


def _hist24_add_channel(fig, ri, ch, recs, stim_x0, stim_x1, ex0, ex1,
                        n_ch) -> None:
    """One channel row: overlay each recording's mean (Turbo oldest->newest)
    in both the stim (-1..1 ms) and evoked windows; combined tight y-range."""
    assert ri - 1 < n_ch, "row index out of range"
    assert recs, "recs required"
    colors = _turbo_ramp(len(recs))
    for j, rec in enumerate(recs):
        assert j < _HIST24_MAX, "hist24 overlay loop runaway"
        t, m = rec["time_axis_ms"], rec["mean_trace"]
        if not t or not m:
            continue
        for col in (1, 2):
            fig.add_trace(go.Scatter(
                x=t, y=m, mode="lines",
                line=dict(color=colors[j], width=0.6),
                hoverinfo="skip", showlegend=False), row=ri, col=col)
    for col in (1, 2):
        fig.add_vline(x=0, line=dict(color="white", width=0.5, dash="dash"),
                      row=ri, col=col)
    fig.update_xaxes(range=[stim_x0, stim_x1], row=ri, col=1)
    fig.update_xaxes(range=[ex0, ex1], row=ri, col=2)
    fig.update_yaxes(title_text=ch, title_font_size=10, row=ri, col=1)
    pairs = [(r["time_axis_ms"], r["mean_trace"]) for r in recs]
    yl = _yrange_pad_multi(pairs, stim_x0, stim_x1)
    if yl is not None:
        fig.update_yaxes(range=list(yl), row=ri, col=1)
    yr = _yrange_pad_multi(pairs, ex0, ex1)
    if yr is not None:
        fig.update_yaxes(range=list(yr), row=ri, col=2)
    if ri < n_ch:
        fig.update_xaxes(showticklabels=False, row=ri, col=1)
        fig.update_xaxes(showticklabels=False, row=ri, col=2)


def _hist24_colorbar(fig) -> None:
    """A Turbo colorbar (oldest->newest) as a trace-less time legend."""
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale="Turbo", cmin=0, cmax=1, color=[0],
                    showscale=True,
                    colorbar=dict(title=dict(text="rec", font=dict(size=9)),
                                  tickvals=[0, 1], ticktext=["old", "new"],
                                  thickness=10, len=0.6, x=1.005))),
        row=1, col=2)


def _hist24_annotations(fig, stim_x0, stim_x1, ex0, ex1, span_lbl) -> None:
    fig.add_annotation(xref="paper", yref="paper", x=0.04, y=1.01,
                       xanchor="left", yanchor="bottom",
                       text=f"Stim window  {stim_x0:g} to {stim_x1:g} ms",
                       showarrow=False, font=dict(size=10, color="#888"))
    fig.add_annotation(xref="paper", yref="paper", x=0.22, y=1.01,
                       xanchor="left", yanchor="bottom",
                       text=f"Evoked  {ex0:g}–{ex1:g} ms",
                       showarrow=False, font=dict(size=10, color="#888"))
    if span_lbl:
        fig.add_annotation(xref="paper", yref="paper", x=0.985, y=1.01,
                           xanchor="right", yanchor="bottom", text=span_lbl,
                           showarrow=False, font=dict(size=9, color="#888"))


def _build_hist24_overlay(per_ch: dict, config: dict | None) -> list:
    """Per-channel overlay of the last 24 recordings' mean traces (stim window
    + evoked window), Turbo-colored oldest->newest. Returns the Graph children
    (reuses the latest-evoked grid geometry)."""
    assert isinstance(per_ch, dict), "per_ch must be a dict"
    if not per_ch:
        return [html.Div()]
    fa_cfg = (config or {}).get("feature_analysis", {}) or {}
    ex0 = float(fa_cfg.get("analysis_start_ms", 2.0))
    ex1 = float(fa_cfg.get("analysis_end_ms", 200.0))
    stim_x0, stim_x1 = -1.0, 1.0
    channels = sorted(per_ch.keys())
    n_ch = len(channels)
    fig = make_subplots(
        rows=n_ch, cols=2, column_widths=[0.16, 0.84], shared_xaxes=False,
        subplot_titles=None, vertical_spacing=0.08, horizontal_spacing=0.035)
    span_lbl = ""
    for ri, ch in enumerate(channels, 1):
        recs = per_ch[ch]
        _hist24_add_channel(fig, ri, ch, recs, stim_x0, stim_x1, ex0, ex1,
                            n_ch)
        if recs and not span_lbl:
            span_lbl = (f"{recs[0]['chunk_datetime'][:16]} → "
                        f"{recs[-1]['chunk_datetime'][:16]}")
    fig.update_xaxes(title_text="Time (ms)", row=n_ch, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=n_ch, col=2)
    _hist24_colorbar(fig)
    n_recs = max((len(v) for v in per_ch.values()), default=0)
    fig.update_layout(
        title=dict(text=f"Latest Evoked — last {n_recs} recordings "
                        "(oldest→newest)",
                   font=dict(size=11), x=0.5, xanchor="center", y=0.985,
                   yanchor="top"),
        autosize=True, margin=dict(l=42, r=130, t=44, b=32), showlegend=False)
    _hist24_annotations(fig, stim_x0, stim_x1, ex0, ex1, span_lbl)
    return [dcc.Graph(
        figure=fig, responsive=True,
        style={"marginTop": "4px", "height": "58vh",
               "minHeight": f"{max(420, 108 * n_ch)}px",
               "maxHeight": "780px"})]


def _hist24_placeholder(prog: dict):
    """Spinner + 'Computing... (done/total)' loading state for the overlay."""
    done = int(prog.get("done") or 0)
    total = int(prog.get("total") or _HIST24_N)
    return html.Div(
        loading_icon(f"Computing last-24 overlay… ({done}/{total})"),
        style={"display": "flex", "alignItems": "center",
               "justifyContent": "center", "minHeight": "300px"})


def _hist24_render(config: dict | None, n: int | None = None):
    """``(children, poll_disabled)`` for the recent-recordings mode. Cache hit
    -> overlay + poll off; miss -> kick the worker, show 'Computing...' + poll
    on. *n* = how many recent recordings to overlay (defaults to config)."""
    sig, files = _hist24_file_set(config, n)
    if sig is None:
        return [html.Div("No evoked recordings found.",
                         style={"color": "#888", "padding": "12px"})], True
    per_ch = _hist24_ready(sig)
    if per_ch is not None:
        return _build_hist24_overlay(per_ch, config), True
    _kick_hist24(sig, files)
    return [_hist24_placeholder(_hist24_progress())], False


def _thumb_unavailable(msg: str) -> list:
    """Themed 'the evoked source is slow/unreachable' node for the thumbnail --
    shown instead of a bare, eternal spinner when the .mat fallback read times
    out, so the user knows it's an infra stall, not a frozen app."""
    return [html.Div(
        f"⚠ {msg}",
        style={"color": COLOR_WARNING, "fontSize": FONT_SIZE_CAPTION,
               "background": COLOR_SURFACE_2,
               "border": f"1px solid {COLOR_DIVIDER}",
               "borderRadius": RADIUS_SM, "padding": f"{SPACE_3}"},
    )]


def _build_overview_thumbnail(store: Store, config: dict | None,
                                session_dir: str, trace_mode: str
                                ) -> list:
    """Return the children list for the Overview waveform-thumbnail
    Div. *trace_mode* ∈ {"mean", "sem", "overlay"}.

    Both left (stim ±1 ms) and right (analysis window) panels show the
    LFP mean_trace -- the left panel is just a zoomed view around t=0,
    not the stim copy channel. Y-axis on every panel scales so the
    data spans 80% of the plot height.
    """
    waveforms = []
    if session_dir:
        try:
            # Only the overlay view draws prior files; mean/SEM render the newest
            # recording alone. Fetching the whole session's traces for mean/SEM
            # was ~150 MB / ~4 s of JSON thrown away -- scope the read to match.
            # Labelled by mode so the ledger separates the cheap latest-only read
            # from the heavier overlay history read.
            with _perf.Timer(f"overview:thumb-fetch-db:{trace_mode}"):
                if trace_mode == "overlay":
                    waveforms = store.get_evoked_waveforms_for_session(
                        session_dir, limit_files=_OVERLAY_MAX_FILES)
                else:
                    waveforms = store.get_evoked_waveforms_for_session(
                        session_dir, latest_only=True)
        except Exception as e:
            logger.debug("Could not load waveform thumbnail: %s", e)
            waveforms = []
    if not waveforms:
        # No DB evoked_waveforms for the newest session yet (the daemon is still
        # processing it). We used to fall back to h5py-reading the latest .mat off
        # the SMB share here -- but that read HOLDS THIS (dashboard) process's GIL
        # for seconds, and it fired on EVERY refresh (the newest session is
        # perpetually unprocessed while recording), starving the Overview DB
        # cards behind it. So skip the share read on the refresh path and show a
        # placeholder; the thumbnail fills once the daemon writes the waveforms.
        # (The "Recent recordings" overlay mode still reads the share, but only on
        # explicit user selection and off a background worker.)
        return _thumb_unavailable("Latest recording still processing "
                                  "(no evoked data in the DB yet)")

    fa_cfg = (config or {}).get("feature_analysis", {}) or {}
    evoked_x0 = float(fa_cfg.get("analysis_start_ms", 2.0))
    evoked_x1 = float(fa_cfg.get("analysis_end_ms", 200.0))
    stim_x0, stim_x1 = -1.0, 1.0

    # Latest waveform per channel + the full per-channel history so
    # overlay mode has something to draw under the mean.
    latest_by_ch: dict[int, dict] = {}
    history_by_ch: dict[int, list[dict]] = {}
    latest_datetime = ""
    for wf in waveforms:
        ch = int(wf.get("channel", 0))
        latest_by_ch[ch] = wf
        history_by_ch.setdefault(ch, []).append(wf)
        dt = wf.get("chunk_datetime", "") or ""
        if dt > latest_datetime:
            latest_datetime = dt
    sorted_chs = sorted(latest_by_ch.keys())
    n_ch = len(sorted_chs)
    if n_ch == 0:
        return [html.Div()]

    colors_list = ["#636EFA", "#00CC96", "#FFA15A", "#EF553B", "#AB63FA"]
    chan_label = {
        ch: latest_by_ch[ch].get("channel_name", f"Ch{ch}")
        for ch in sorted_chs
    }
    # Per-cell titles dropped (channel name lives on y-axis, column
    # name lives on the figure-level annotations below). Spacing
    # bumped to 0.08 + non-bottom rows hide their x ticks, so tick
    # labels from row N can no longer collide with row N+1's plot
    # top or with the next row's title region.
    thumb_fig = make_subplots(
        rows=n_ch, cols=2, column_widths=[0.16, 0.84],
        shared_xaxes=False, subplot_titles=None,
        vertical_spacing=0.08, horizontal_spacing=0.035,
    )

    def _hex_to_rgba(hex_str: str, alpha: float) -> str:
        h = hex_str.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    import numpy as np
    for ri, ch in enumerate(sorted_chs, 1):
        wf = latest_by_ch[ch]
        nm = chan_label[ch]
        n_ep = wf.get("n_epochs", 0)
        time_ms = wf["time_axis_ms"]
        mean_tr = wf["mean_trace"]
        sem_tr = wf.get("sem_trace")
        # Hand plotly NUMPY arrays (not Python lists) so Dash figure
        # serialization takes plotly's fast ndarray branch (one C call) instead
        # of the per-element list recursion that pins the GIL on the waitress
        # request thread for seconds. Kept SEPARATE from the list vars above so
        # every `if not x`/`len()` guard below stays correct.
        _t = np.asarray(time_ms, dtype=float)
        _m = np.asarray(mean_tr, dtype=float)
        color = colors_list[(ri - 1) % len(colors_list)]
        band_color = _hex_to_rgba(color, 0.18)
        overlay_color = _hex_to_rgba(color, 0.12)
        # Group every trace for this channel under one legendgroup so
        # clicking the legend entry hides stim+evoked+SEM band+overlay
        # simultaneously. Lets the user isolate one channel cleanly
        # (Plotly: single-click hide, double-click isolate).
        lg = f"ch_{ch}"

        # Overlay mode: draw every file's mean_trace under this
        # channel's latest mean. Cheap stylized backdrop, no legend.
        if trace_mode == "overlay":
            # Cap the overlaid history: was EVERY historical file at full
            # resolution across 2 panels -> ~420k points, a huge figure JSON
            # payload. Draw only the newest _OVERLAY_MAX_FILES, each decimated.
            _hist = sorted(history_by_ch[ch],
                           key=lambda w: (w.get("chunk_datetime") or ""),
                           reverse=True)[:_OVERLAY_MAX_FILES]
            for prev in _hist:
                if prev is wf:
                    continue
                pt = prev.get("time_axis_ms")
                pm = prev.get("mean_trace")
                if not pt or not pm or len(pt) != len(pm):
                    continue
                _step = max(1, len(pt) // 800)   # cap each trace to ~800 pts
                pt, pm = pt[::_step], pm[::_step]
                # Left + right panels both get the overlay so the
                # stim-window context matches the evoked window.
                for col_idx in (1, 2):
                    thumb_fig.add_trace(go.Scatter(
                        x=pt, y=pm, mode="lines",
                        line=dict(color=overlay_color, width=0.7),
                        hoverinfo="skip", showlegend=False,
                        legendgroup=lg,
                    ), row=ri, col=col_idx)

        # SEM band: filled ribbon mean±sem, mean drawn on top.
        if trace_mode == "sem" and sem_tr and len(sem_tr) == len(mean_tr):
            _s = np.asarray(sem_tr, dtype=float)
            upper = _m + _s
            lower = _m - _s
            for col_idx in (1, 2):
                thumb_fig.add_trace(go.Scatter(
                    x=_t, y=upper, mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip", showlegend=False,
                    legendgroup=lg,
                ), row=ri, col=col_idx)
                thumb_fig.add_trace(go.Scatter(
                    x=_t, y=lower, mode="lines",
                    fill="tonexty", fillcolor=band_color,
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip", showlegend=False,
                    legendgroup=lg,
                ), row=ri, col=col_idx)

        # Left panel: LFP mean, zoomed to ±1 ms around stim.
        thumb_fig.add_trace(go.Scatter(
            x=_t, y=_m, mode="lines",
            name=f"{nm} stim",
            line=dict(color=color, width=1.5),
            showlegend=False, legendgroup=lg,
        ), row=ri, col=1)
        thumb_fig.add_vline(
            x=0, line=dict(color="white", width=0.5, dash="dash"),
            row=ri, col=1)
        thumb_fig.update_xaxes(range=[stim_x0, stim_x1], row=ri, col=1)
        thumb_fig.update_yaxes(title_text=nm, title_font_size=10,
                                row=ri, col=1)
        y_left = _yrange_pad(time_ms, mean_tr, stim_x0, stim_x1)
        if y_left is not None:
            thumb_fig.update_yaxes(range=list(y_left), row=ri, col=1)

        # Right panel: LFP mean, evoked window. The visible legend
        # entry lives on this trace (showlegend=True default) and
        # carries the channel group so clicking it hides the whole
        # row across both columns.
        thumb_fig.add_trace(go.Scatter(
            x=_t, y=_m, mode="lines",
            name=f"{nm} (n={n_ep})",
            line=dict(color=color, width=1.5),
            legendgroup=lg,
        ), row=ri, col=2)
        thumb_fig.update_xaxes(range=[evoked_x0, evoked_x1],
                                row=ri, col=2)
        y_right = _yrange_pad(time_ms, mean_tr, evoked_x0, evoked_x1)
        if y_right is not None:
            thumb_fig.update_yaxes(range=list(y_right), row=ri, col=2)

        # Only the bottom row carries x-axis ticks + title. Non-
        # bottom rows hide tick labels so the "-1 -0.5 0 0.5 1"
        # labels of row N can't collide with row N+1's plot top
        # at tight vertical spacing.
        if ri < n_ch:
            thumb_fig.update_xaxes(showticklabels=False, row=ri, col=1)
            thumb_fig.update_xaxes(showticklabels=False, row=ri, col=2)

    thumb_fig.update_xaxes(title_text="Time (ms)", row=n_ch, col=1)
    thumb_fig.update_xaxes(title_text="Time (ms)", row=n_ch, col=2)
    mode_label = {"mean": "mean",
                   "sem": "mean ± SEM",
                   "overlay": "mean + per-file overlay"}.get(
        trace_mode, "mean")
    # Viewport-aware: autosize=True + a CSS calc() height on the
    # Graph container makes Plotly redraw to fill whatever vertical
    # space the viewport allows. minHeight keeps each channel row
    # readable on a 1080p screen; maxHeight prevents the figure from
    # going absurdly tall on a 4K monitor.
    # Legend back on, but as a vertical strip in the right margin --
    # avoids the title collision the top-right horizontal legend
    # caused, and Plotly's built-in legend behavior (click to hide,
    # double-click to isolate) gives the user a way to focus on one
    # channel when three overlapping traces get noisy.
    thumb_fig.update_layout(
        title=dict(
            text=f"Latest Evoked — {latest_datetime[:16]} ({mode_label})",
            font=dict(size=11), x=0.5, xanchor="center",
            y=0.985, yanchor="top"),
        autosize=True,
        margin=dict(l=42, r=130, t=44, b=32),
        showlegend=True,
        legend=dict(
            orientation="v", x=1.005, y=1, xanchor="left",
            yanchor="top", font_size=10, bgcolor="rgba(0,0,0,0)",
            itemclick="toggle",        # single-click = hide/show
            itemdoubleclick="toggleothers",  # double = isolate
        ),
    )
    # Column headers as figure-level annotations, sitting just below
    # the title. x-positions track column_widths=[0.16, 0.84] +
    # horizontal_spacing=0.035; update both if the subplot grid
    # geometry changes.
    thumb_fig.add_annotation(
        xref="paper", yref="paper",
        x=0.04, y=1.01, xanchor="left", yanchor="bottom",
        text=f"Stim window  {stim_x0:g} to {stim_x1:g} ms",
        showarrow=False,
        font=dict(size=10, color="#888"),
    )
    thumb_fig.add_annotation(
        xref="paper", yref="paper",
        x=0.22, y=1.01, xanchor="left", yanchor="bottom",
        text=f"Evoked  {evoked_x0:g}–{evoked_x1:g} ms",
        showarrow=False,
        font=dict(size=10, color="#888"),
    )
    return [dcc.Graph(
        figure=thumb_fig, responsive=True,
        style={
            "marginTop": "4px",
            # 58vh (down from 62) keeps the title strip clearly
            # readable at common viewports without losing channel
            # legibility. min keeps each row readable on a small
            # laptop; max caps absurd 4K heights.
            "height": "58vh",
            "minHeight": f"{max(420, 108 * n_ch)}px",
            "maxHeight": "780px",
        },
    )]


def _parse_end_datetime(s: str | None) -> datetime | None:
    """Parse the sheet's End_DateTime column.

    KMrecorder writes the row when the chunk finishes, so the
    End_DateTime cell is the closest the sheet has to "row submitted
    at." Values look like '6/1/2026 16:06:27' (no leading zeros) but
    we accept the ISO form too in case the column gets normalised.
    """
    if not s:
        return None
    text = str(s).strip()
    if not text:
        return None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace(" ", "T"))
    except ValueError:
        return None


def _km_log_summary(config: dict | None) -> dict:
    """Most-recent KMrecorder data-log entries + freshness in minutes.

    Pulls from the same Google Sheet the data_log_xref tab uses, so
    we share the existing TTL cache (no extra API quota cost). The
    sheet has a Filename column (recording start, derived from the
    .mat filename) and an End_DateTime column (when the chunk
    finished -- KMrecorder submits the row at that point, so we
    use it as "submitted at").
    """
    out = {
        "enabled": False, "rows": [], "latest_iso": None,
        "age_min": None,
        "latest_submitted_iso": None,
        "submitted_age_min": None,
    }
    cfg = (config or {}).get("data_log_xref", {}) or {}
    if not cfg.get("enabled", False):
        return out
    try:
        from src.dashboard.tabs.surgeries import (
            _load_sheet_via_api, _resolve_sa_path, _find_column,
            _normalize_text,
        )
        from src.dashboard.tabs.data_log_xref import _extract_key
    except Exception:
        return out
    sheet_id = cfg.get("sheet_id", "")
    tab_name = cfg.get("tab_name", "")
    sa = _resolve_sa_path(config or {},
                           cfg.get("service_account_file", ""))
    if not (sheet_id and tab_name and sa):
        return out
    ttl_sec = float(cfg.get("refresh_minutes", 15)) * 60.0
    try:
        df = _load_sheet_via_api(sheet_id, tab_name, sa, ttl_sec)
    except Exception as e:
        logger.debug("KM log fetch failed: %s", e)
        return out
    if df is None or df.empty:
        out["enabled"] = True
        return out

    filename_col = _find_column(df, ["Filename", "File", "Path"])
    animal_col = _find_column(df, ["Animal_ID", "Animal"])
    pc_col = _find_column(df, ["PC Name", "PC"])
    end_dt_col = _find_column(df, ["End_DateTime", "EndDateTime",
                                      "End"])
    operator_col = _find_column(df, ["Operator"])
    date_col = _find_column(df, ["Date"])
    rec_end_col = _find_column(df, ["Recording_End", "RecordingEnd"])
    if filename_col is None:
        out["enabled"] = True
        return out

    def _resolve_submitted(row) -> datetime | None:
        """Submission timestamp: prefer End_DateTime column; fall back
        to Date + Recording_End when the recorder doesn't populate
        End_DateTime (recent KMrecorder versions leave it blank)."""
        if end_dt_col:
            dt = _parse_end_datetime(row.get(end_dt_col))
            if dt is not None:
                return dt
        if not (date_col and rec_end_col):
            return None
        date_s = _normalize_text(row.get(date_col))
        end_s = _normalize_text(row.get(rec_end_col))
        if not (date_s and end_s):
            return None
        # Date is "YYYY-MM-DD"; Recording_End is "HH:MM:SS".
        try:
            return datetime.strptime(f"{date_s} {end_s}",
                                       "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    # Build records keyed by recording-start (chunk filename key)
    # so they sort chronologically lexicographically.
    records: dict[str, dict] = {}
    for _idx, row in df.iterrows():
        key = _extract_key(row.get(filename_col))
        if key is None:
            continue
        records[key] = {
            "key": key,
            "animal": (_normalize_text(row.get(animal_col))
                        if animal_col else ""),
            "pc": (_normalize_text(row.get(pc_col))
                    if pc_col else ""),
            "operator": (_normalize_text(row.get(operator_col))
                          if operator_col else ""),
            "submitted_dt": _resolve_submitted(row),
        }
    if not records:
        out["enabled"] = True
        return out

    sorted_keys = sorted(records.keys(), reverse=True)
    rows = [records[k] for k in sorted_keys[:5]]
    latest_key = sorted_keys[0]
    try:
        latest_dt = datetime.strptime(latest_key, _CHUNK_DT_FMT)
        age_min = (datetime.now() - latest_dt).total_seconds() / 60.0
    except ValueError:
        latest_dt, age_min = None, None
    latest_sub_dt = records[latest_key].get("submitted_dt")
    sub_age_min = ((datetime.now()
                     - latest_sub_dt).total_seconds() / 60.0
                    if latest_sub_dt else None)
    out.update({
        "enabled": True, "rows": rows,
        "latest_iso": latest_dt.isoformat() if latest_dt else None,
        "age_min": age_min,
        "latest_submitted_iso": (latest_sub_dt.isoformat()
                                    if latest_sub_dt else None),
        "submitted_age_min": sub_age_min,
    })
    return out


def _fmt_age(minutes: float | None) -> tuple[str, str]:
    """(label, color) for an age in minutes. Green <90 min, amber <24 h, red
    older. On-palette design tokens so the KM-log stamp and the per-card
    freshness footer (_freshness_footer) share one scale. Used for 'recorded',
    'submitted', and card build-age."""
    if minutes is None:
        return "unknown", COLOR_TEXT_TERTIARY
    if minutes < 90:
        return f"{minutes:.0f} min ago", COLOR_SUCCESS
    if minutes < 24 * 60:
        return f"{minutes / 60:.1f} h ago", COLOR_WARNING
    return f"{minutes / (24 * 60):.1f} d ago", COLOR_DANGER


def _build_km_log_section(config: dict | None) -> html.Div:
    """KM Recorder freshness header + last-5-entries strip. Each row
    shows the recording-start timestamp (from the filename) AND the
    sheet's End_DateTime, which is when KMrecorder finished the chunk
    and submitted the row -- the closest proxy this lab has for
    'last contact' since the recorder is what writes to the sheet."""
    summary = _km_log_summary(config)
    if not summary["enabled"]:
        return html.Div()

    rec_text, rec_color = _fmt_age(summary["age_min"])
    sub_text, sub_color = _fmt_age(summary["submitted_age_min"])

    rows = summary["rows"]
    if rows:
        tail = []
        for r in rows:
            ts_str = r.get("key", "")[:16].replace("_", "/")
            sub_dt = r.get("submitted_dt")
            sub_str = (sub_dt.strftime("%m/%d %H:%M")
                        if sub_dt else "?")
            label = (
                f"rec {ts_str}  ->  submitted {sub_str}  ·  "
                f"{r.get('animal') or '?'}  ·  {r.get('pc') or '?'}"
            )
            tail.append(html.Div(label, style={
                "color": "#aaa", "fontSize": "11px",
                "padding": "2px 0",
                "overflow": "hidden", "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
            }))
    else:
        tail = [html.Div("No rows from KMrecorder log yet.",
                          style={"color": "#888", "fontSize": "11px"})]

    return html.Div([
        html.Div([
            html.Span(f"recording {rec_text}",
                       style={"color": rec_color, "fontSize": "11px",
                               "fontWeight": "600",
                               "marginRight": "10px"}),
            html.Span(f"submitted {sub_text}",
                       style={"color": sub_color, "fontSize": "11px",
                               "fontWeight": "600"}),
        ], style={"marginBottom": "4px"}),
        html.Div(tail),
    ])


_CHUNK_DT_FMT = "%Y_%m_%d__%H_%M_%S"


def _parse_chunk_dt(ts: str | None) -> datetime | None:
    """Parse a chunk_datetime string into a datetime.

    chunk_datetime is stored in filename format (YYYY_MM_DD__HH_MM_SS)
    not ISO -- the recorder writes the timestamp into the filename and
    the watcher persists it as-is. Falls back to fromisoformat() so a
    future schema change doesn't silently break this.
    """
    if not ts:
        return None
    try:
        return datetime.strptime(ts, _CHUNK_DT_FMT)
    except ValueError:
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None


def _recording_arrivals(store: Store, hours: int
                          ) -> list[tuple[str, float]]:
    """Back-compat shim returning (chunk_datetime, duration_sec)."""
    return [(ts, dur) for ts, dur, _sd, _sn
            in _recording_arrivals_with_session(store, hours)]


def _recording_arrivals_with_session(
        store: Store, hours: int
        ) -> list[tuple[str, float, str, str]]:
    """(chunk_datetime, duration_sec, session_dir, session_name) for
    every chunk whose coverage might overlap the last *hours* window.

    The cutoff is widened by 2h so a chunk that started just before
    the window and runs INTO it still appears.
    """
    assert hours > 0, "hours must be positive"
    cutoff_dt = datetime.now() - timedelta(hours=hours + 2)
    cutoff_str = cutoff_dt.strftime(_CHUNK_DT_FMT)
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT chunk_datetime, duration_sec, session_dir, "
            "       session_name "
            "FROM processed_files "
            "WHERE chunk_datetime >= ? ORDER BY chunk_datetime",
            (cutoff_str,),
        ).fetchall()
        out: list[tuple[str, float, str, str]] = []
        for r in rows:
            ts = r["chunk_datetime"]
            if not ts:
                continue
            dur = float(r["duration_sec"] or 0.0)
            if dur <= 0:
                dur = 3600.0
            sd = str(r["session_dir"] or "")
            sn = str(r["session_name"] or sd.rsplit("/", 1)[-1] or "?")
            out.append((ts, dur, sd, sn))
        return out


# Per-session color palette for the 24h band. Cycles when there are
# more sessions than colors. Greens/blues/oranges/etc. are deliberate
# -- the "no recording" gray (#2a2a40) stays clearly distinct.
_SESSION_PALETTE = [
    "#00CC96", "#636EFA", "#FFA15A", "#EF553B", "#AB63FA",
    "#19D3F3", "#FF6692", "#FECB52", "#B6E880", "#7be3c0",
]
_NO_REC_COLOR = "#2a2a40"


def _session_color_map(arrivals_with_session: list[tuple]
                        ) -> dict[str, str]:
    """Assign a stable color to each session_dir in arrival order.

    The first session to appear gets the first palette color, etc.
    Sessions beyond the palette wrap around -- the visual gives a hint
    that a new session started, even if two distant ones reuse a color.
    """
    order: list[str] = []
    seen: set[str] = set()
    for _ts, _dur, sd, _sn in arrivals_with_session:
        if sd and sd not in seen:
            seen.add(sd)
            order.append(sd)
    return {sd: _SESSION_PALETTE[i % len(_SESSION_PALETTE)]
            for i, sd in enumerate(order)}


def _build_recording_24h_fig(
        arrivals_with_session: list[tuple[str, float, str, str]],
        session_colors: dict[str, str] | None = None
        ) -> go.Figure:
    """24-hour recording-uptime band. 5-minute bins (288 across the
    day) so short interruptions and quick session swaps are visible.
    Color = the session whose chunk covers that bin; gray = no chunk
    covered the bin. If a bin is covered by multiple sessions (rare
    -- only at hand-off seams), the LATEST-starting chunk wins."""
    now = datetime.now()
    bin_min = 5
    n_bins = 24 * 60 // bin_min   # 288
    bin_starts = [now - timedelta(minutes=(n_bins - i) * bin_min)
                   for i in range(n_bins)]
    bin_ends = [b + timedelta(minutes=bin_min) for b in bin_starts]
    sess_per_bin: list[str | None] = [None] * n_bins
    # Latest chunk start wins on overlap, so iterate in original
    # (ascending-by-start) order and overwrite.
    cover_start: list[datetime | None] = [None] * n_bins
    for ts, dur, sd, _sn in arrivals_with_session:
        dt = _parse_chunk_dt(ts)
        if dt is None or not sd:
            continue
        ch_end = dt + timedelta(seconds=dur)
        if ch_end <= bin_starts[0] or dt >= bin_ends[-1]:
            continue
        for i in range(n_bins):
            if dt >= bin_ends[i]:
                continue
            if ch_end <= bin_starts[i]:
                break
            if (cover_start[i] is None or dt >= cover_start[i]):
                sess_per_bin[i] = sd
                cover_start[i] = dt
    colors = (session_colors
               if session_colors is not None
               else _session_color_map(arrivals_with_session))
    sess_short = {sd: sn for ts, dur, sd, sn in arrivals_with_session
                   if sd}
    x_labels = [b.strftime("%H:%M") for b in bin_starts]
    bar_colors = [colors.get(s, _SESSION_PALETTE[0])
                   if s else _NO_REC_COLOR
                   for s in sess_per_bin]
    hovertext = []
    for b, s in zip(bin_starts, sess_per_bin):
        if s:
            label = sess_short.get(s, "?")[:48]
            hovertext.append(f"{b.strftime('%a %H:%M')}: {label}")
        else:
            hovertext.append(f"{b.strftime('%a %H:%M')}: no recording")
    fig = go.Figure(go.Bar(
        x=x_labels, y=[1] * n_bins,
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertext=hovertext, hoverinfo="text",
    ))
    fig.update_layout(
        height=32, margin=dict(l=8, r=8, t=2, b=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, bargap=0,
        xaxis=dict(showgrid=False,
                    tickfont=dict(size=8, color="#888"),
                    tickmode="array",
                    tickvals=[x_labels[0],
                              x_labels[n_bins // 2],
                              x_labels[-1]],
                    ticktext=["24h ago", "12h ago", "now"]),
        yaxis=dict(showgrid=False, showticklabels=False,
                    zeroline=False, range=[0, 1]),
    )
    return fig


def _build_recording_24h_legend(
        arrivals_with_session: list[tuple[str, float, str, str]],
        session_colors: dict[str, str]) -> html.Div:
    """Horizontal swatch strip naming the colors used in the 24h band.

    Always includes "no recording" + one swatch per session that
    actually appears. Sessions are listed in arrival order; long
    session names are truncated to 28 chars so the strip fits in
    the queue block at narrow viewports.
    """
    seen: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    for _ts, _dur, sd, sn in arrivals_with_session:
        if sd and sd not in seen_keys:
            seen_keys.add(sd)
            seen.append((sd, sn))
    swatches = [
        html.Span([
            html.Span(style={
                "display": "inline-block",
                "width": "10px", "height": "10px",
                "background": _NO_REC_COLOR, "borderRadius": "2px",
                "marginRight": "4px",
                "verticalAlign": "middle",
            }),
            html.Span("no recording",
                      style={"verticalAlign": "middle"}),
        ], style={"marginRight": "12px"}),
    ]
    for sd, sn in seen:
        color = session_colors.get(sd, _SESSION_PALETTE[0])
        label = sn if len(sn) <= 28 else sn[:25] + "..."
        swatches.append(html.Span([
            html.Span(style={
                "display": "inline-block",
                "width": "10px", "height": "10px",
                "background": color, "borderRadius": "2px",
                "marginRight": "4px",
                "verticalAlign": "middle",
            }),
            html.Span(label, style={"verticalAlign": "middle"}),
        ], style={"marginRight": "12px"}))
    return html.Div(swatches, style={
        "fontSize": "10px", "color": "#aaa", "marginTop": "4px",
        "display": "flex", "flexWrap": "wrap",
        "rowGap": "4px",
    })


def _build_recording_7d_fig(
        arrivals: list[tuple[str, float]]) -> go.Figure:
    """7-day recording uptime as a day × hour heatmap. Each cell = one
    hour on one day; value = minutes of chunk coverage within that
    hour (0..60). Color saturates at full-hour coverage so partial-
    hour chunks still show clearly. The current-hour cell on today's
    row gets a yellow outline so the user can spot 'now' instantly."""
    today = date.today()
    now = datetime.now()
    # Coverage in MINUTES per (day, hour) cell.
    z = [[0.0] * 24 for _ in range(7)]
    for ts, dur in arrivals:
        dt = _parse_chunk_dt(ts)
        if dt is None:
            continue
        ch_end = dt + timedelta(seconds=dur)
        # Walk hour-by-hour from chunk start to end.
        cur = dt
        guard = 0
        while cur < ch_end and guard < 200:   # NASA rule 2
            guard += 1
            hour_end = cur.replace(minute=0, second=0,
                                    microsecond=0) + timedelta(hours=1)
            seg_end = min(ch_end, hour_end)
            offset = (today - cur.date()).days
            if 0 <= offset < 7:
                mins = (seg_end - cur).total_seconds() / 60.0
                z[6 - offset][cur.hour] += min(60.0, mins)
                # Clamp at 60 so two-back-to-back chunks in the same
                # hour don't shift the colorscale.
                if z[6 - offset][cur.hour] > 60.0:
                    z[6 - offset][cur.hour] = 60.0
            cur = seg_end
    y_labels = [(today - timedelta(days=d)).strftime("%a %m/%d")
                for d in range(6, -1, -1)]
    x_labels = [f"{h:02d}" for h in range(24)]
    # customdata is a 7×24 grid of uptime percent (0..100) -- z gives
    # minutes (0..60) and we want the hover to surface BOTH so a
    # 45-min hour reads "45 min recorded (75%)".
    custom = [[round(z[r][c] / 60.0 * 100.0, 0) for c in range(24)]
               for r in range(7)]
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels, zmin=0, zmax=60,
        customdata=custom,
        colorscale=[[0.0, "#2a2a40"], [0.001, "#1f5a44"],
                     [0.5, "#00CC96"], [1.0, "#7be3c0"]],
        showscale=False, xgap=1, ygap=1,
        hovertemplate="%{y} %{x}:00 — %{z:.0f} min recorded "
                       "(%{customdata:.0f}%)<extra></extra>",
    ))
    # Current-hour highlight: a thin gold outline around the cell at
    # (today, current_hour). Using numeric indices because shape
    # x/y on a categorical axis maps to the underlying index, where
    # hour 17 -> x=17 and row "Mon 06/01" -> y=6 (last row index).
    days_ago = (today - now.date()).days
    if 0 <= days_ago < 7:
        x_idx = now.hour
        y_idx = 6 - days_ago
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=x_idx - 0.5, x1=x_idx + 0.5,
            y0=y_idx - 0.5, y1=y_idx + 0.5,
            line=dict(color="#FFD700", width=2),
            fillcolor="rgba(255, 215, 0, 0.18)",
            layer="above",
        )
    fig.update_layout(
        height=120, margin=dict(l=52, r=8, t=2, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False,
                    tickfont=dict(size=9, color="#888"),
                    tickmode="array",
                    tickvals=["00", "06", "12", "18", "23"],
                    ticktext=["00", "06", "12", "18", "23"],
                    title=dict(text="hour of day",
                                font=dict(size=9, color="#666"))),
        yaxis=dict(showgrid=False,
                    tickfont=dict(size=9, color="#aaa"),
                    autorange="reversed"),
    )
    return fig


def _overview_today_stats(store: Store, session_dir: str) -> dict:
    """One-shot snapshot for the Overview Today block.

    Today is wall-clock local; pending count is total (not today-only).
    """
    today = date.today()
    start = datetime.combine(today, datetime.min.time()).isoformat()
    end = datetime.combine(today, datetime.max.time()).isoformat()
    last_24h = (datetime.now() - timedelta(hours=24)).isoformat()
    last_1h = (datetime.now() - timedelta(hours=1)).isoformat()
    with store.connection() as conn:
        row = conn.execute(
            """SELECT
                 SUM(CASE WHEN status='done'  THEN 1 ELSE 0 END) AS done,
                 SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors
               FROM processed_files
               WHERE processed_at BETWEEN ? AND ?""",
            (start, end),
        ).fetchone()
        done_today = int((row["done"] if row and row["done"] else 0) or 0)
        errs_today = int((row["errors"] if row and row["errors"] else 0) or 0)

        prow = conn.execute(
            "SELECT COUNT(*) AS n, MIN(chunk_datetime) AS oldest "
            "FROM processed_files WHERE status='pending'"
        ).fetchone()
        pending = int(prow["n"] or 0) if prow else 0
        oldest_pending_iso = (prow["oldest"] if prow else None)

        # 24h hourly throughput.
        hourly_rows = conn.execute(
            """SELECT strftime('%Y-%m-%d %H', processed_at) AS bucket,
                      COUNT(*) AS n
               FROM processed_files
               WHERE status='done' AND processed_at >= ?
               GROUP BY bucket
               ORDER BY bucket""",
            (last_24h,),
        ).fetchall()

        # Active-session "this hour" count.
        sess_hour = 0
        if session_dir:
            srow = conn.execute(
                "SELECT COUNT(*) AS n FROM processed_files "
                "WHERE session_dir = ? AND status='done' "
                "AND processed_at >= ?",
                (session_dir, last_1h),
            ).fetchone()
            sess_hour = int(srow["n"] or 0) if srow else 0

    # Oldest-pending age in minutes (or None when nothing pending).
    oldest_age_min: float | None = None
    if oldest_pending_iso:
        oldest_dt = _parse_chunk_dt(oldest_pending_iso)
        if oldest_dt is not None:
            oldest_age_min = (
                datetime.now() - oldest_dt).total_seconds() / 60.0

    return {
        "done_today": done_today,
        "errors_today": errs_today,
        "pending": pending,
        "oldest_pending_age_min": oldest_age_min,
        "hourly_done_24h": [dict(r) for r in hourly_rows],
        "session_hour_done": sess_hour,
    }


def _build_overview_queue(store: Store):
    """Stacked Today + Active Session block.

    The old "lifetime queue progress" bar always pinned at 100% once
    the backlog finished, so it was useless for spotting current
    problems. This replaces it with today-only counters + a 24h
    hourly throughput sparkline + an active-session progress strip.
    """
    sessions = store.get_sessions()
    active = sessions[0] if sessions else {}
    session_dir = active.get("session_dir", "")
    stats = _overview_today_stats(store, session_dir)

    # ----- Today section ----- #
    pending = stats["pending"]
    oldest_min = stats["oldest_pending_age_min"]
    if oldest_min is None:
        oldest_str = "—"
    elif oldest_min < 60:
        oldest_str = f"{oldest_min:.0f} min"
    else:
        oldest_str = f"{oldest_min / 60:.1f} h"

    today_counters = html.Div([
        html.Span(f"{stats['done_today']} done today",
                  style={"color": "#00CC96", "marginRight": "16px",
                          "fontWeight": "600"}),
        html.Span(f"{stats['errors_today']} errors",
                  style={"color": "#EF553B" if stats["errors_today"]
                          else "#888", "marginRight": "16px"}),
        html.Span(f"{pending} pending",
                  style={"color": "#FFA15A" if pending else "#888",
                          "marginRight": "16px"}),
        html.Span(f"oldest pending: {oldest_str}",
                  style={"color": "#888"}),
    ], style={"fontSize": "13px", "marginTop": "4px"})

    # 24-hour recording-uptime band: chunk_datetime not processed_at;
    # 5-min bins (288 across the day) so short interruptions and
    # session-to-session hand-offs are visible. Each session paints a
    # different color from the palette; gaps where no chunk's coverage
    # interval touches stay gray.
    arrivals_24h_full = _recording_arrivals_with_session(store, hours=24)
    session_colors_24h = _session_color_map(arrivals_24h_full)
    rec_24h = dcc.Graph(
        figure=_build_recording_24h_fig(arrivals_24h_full,
                                          session_colors_24h),
        config={"displayModeBar": False},
        style={"marginTop": "4px"},
    )
    rec_24h_legend = _build_recording_24h_legend(arrivals_24h_full,
                                                   session_colors_24h)
    arrivals_7d = _recording_arrivals(store, hours=24 * 7)
    rec_7d = dcc.Graph(
        figure=_build_recording_7d_fig(arrivals_7d),
        config={"displayModeBar": False},
        style={"marginTop": "4px"},
    )

    today_section = html.Div([
        today_counters,
        html.Div("24h recording",
                  style={"color": "#666", "fontSize": "10px",
                          "marginTop": "6px",
                          "letterSpacing": "0.4px",
                          "textTransform": "uppercase"}),
        rec_24h,
        rec_24h_legend,
    ])

    recording_7d_section = html.Div([
        html.Div("Last 7 days",
                style={"color": "#666", "marginTop": "10px",
                        "marginBottom": "2px", "fontSize": "10px",
                        "letterSpacing": "0.4px",
                        "textTransform": "uppercase"}),
        rec_7d,
        html.Div([
            html.Span([
                html.Span(style={
                    "display": "inline-block",
                    "width": "10px", "height": "10px",
                    "background": _NO_REC_COLOR, "borderRadius": "2px",
                    "marginRight": "4px",
                    "verticalAlign": "middle",
                }),
                html.Span("no recording",
                          style={"verticalAlign": "middle"}),
            ], style={"marginRight": "12px"}),
            html.Span([
                html.Span(style={
                    "display": "inline-block",
                    "width": "10px", "height": "10px",
                    "background":
                        "linear-gradient(90deg, #1f5a44, #7be3c0)",
                    "borderRadius": "2px", "marginRight": "4px",
                    "verticalAlign": "middle",
                }),
                html.Span("recording (darker = partial hour, "
                           "brighter = full hour)",
                          style={"verticalAlign": "middle"}),
            ]),
        ], style={"fontSize": "10px", "color": "#aaa",
                   "marginTop": "4px", "display": "flex",
                   "flexWrap": "wrap", "rowGap": "4px"}),
    ])

    # ----- Active Session section ----- #
    if active:
        num = int(active.get("num_files", 0) or 0)
        done = int(active.get("processed", 0) or 0)
        errs = int(active.get("errors", 0) or 0)
        pct = (100.0 * done / num) if num > 0 else 0.0
        bar = html.Div([
            html.Div(
                f"{pct:.0f}%" if pct > 5 else "",
                style={
                    "width": f"{max(pct, 1):.1f}%",
                    "background":
                        "linear-gradient(90deg, #00CC96 0%, #00AA80 100%)",
                    "height": "22px", "borderRadius": "5px",
                    "transition":
                        "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
                    "display": "flex", "alignItems": "center",
                    "justifyContent": "center",
                    "fontSize": "11px", "fontWeight": "bold",
                    "color": "white",
                    "textShadow": "0 1px 2px rgba(0,0,0,0.5)",
                }),
        ], style={"backgroundColor": "#1a1a2e", "borderRadius": "5px",
                   "overflow": "hidden", "marginTop": "4px",
                   "marginBottom": "6px",
                   "boxShadow": "inset 0 1px 4px rgba(0,0,0,0.4)"})
        active_counters = html.Div([
            html.Span(active.get("session_name", "(none)"),
                      style={"color": "white", "fontWeight": "600",
                              "marginRight": "12px"}),
            html.Span(f"{done} / {num} files",
                      style={"color": "#aaa", "marginRight": "12px"}),
            html.Span(f"{errs} errors",
                      style={"color": "#EF553B" if errs else "#888",
                              "marginRight": "12px"}),
            html.Span(f"+{stats['session_hour_done']} in last hour",
                      style={"color": "#888"}),
        ], style={"fontSize": "12px"})
        active_section = html.Div([
            html.Div("Active session",
                    style={"color": "#666", "marginTop": "10px",
                            "marginBottom": "2px", "fontSize": "10px",
                            "letterSpacing": "0.4px",
                            "textTransform": "uppercase"}),
            bar, active_counters,
        ])
    else:
        active_section = html.Div([
            html.Div("Active session",
                    style={"color": "#666", "marginTop": "10px",
                            "fontSize": "10px",
                            "letterSpacing": "0.4px",
                            "textTransform": "uppercase"}),
            html.Div("No sessions yet",
                     style={"color": "#888", "fontSize": "11px"}),
        ])

    return [today_section, active_section, recording_7d_section]


def _overview_tab(store: Store, config: dict | None = None):
    # Progressive shell: only the CHEAP reads (sessions, alerts, channel map)
    # run here; every heavy card is a placeholder filled by a mount/refresh
    # callback. render_tab used to build ~12 DB-backed cards synchronously here
    # AND refresh_overview_dynamic rebuilt 8 of them concurrently on mount --
    # the self-contending double-build that made stage 1 (a 0.01s query) take
    # 15-33s. Now render_tab returns fast and cards populate in the background.
    sessions = store.get_sessions()
    active_session = sessions[0] if sessions else {}
    session_dir = active_session.get("session_dir", "")

    ch_map = _get_channel_map(store, session_dir) if session_dir else {}
    recent_alerts = store.get_recent_alerts(hours=24)

    # Pills as a full-width horizontal strip at the very top of
    # the Overview tab. flexWrap on so the strip wraps to a second
    # row on narrow viewports rather than overflowing.
    # Heavy cards below are deferred: the shell ships placeholders and the fill
    # callbacks (mount + refresh-trigger) populate them. Keeps render_tab light.
    cards = html.Div(
        _pending_placeholder("status pills"),
        id="overview-cards",
        style={"display": "flex", "flexWrap": "wrap",
                "gap": "6px", "marginBottom": "10px"},
    )

    # Per-animal behavioral seizure analysis status. Sits
    # between the pill strip and the 3-column grid so it's
    # always above the fold. Glance-able: top line answers
    # "are we keeping up?" with the rate delta; the table
    # below breaks it down per animal so the user can spot
    # which animal is the bottleneck.
    bsz_status = html.Div(
        _pending_placeholder("behavioral seizure status"),
        id="overview-bsz-status",
        style={"marginBottom": "12px"},
    )
    impedance_status = html.Div(
        _pending_placeholder("impedance trend"),
        id="overview-impedance",
    )
    zss_status = html.Div(
        _pending_placeholder("stim-step consistency"),
        id="overview-zss",
    )
    fidelity_status = html.Div(
        _pending_placeholder("current fidelity"),
        id="overview-current-fidelity",
    )
    region_status = html.Div(
        _pending_placeholder("region drift"),
        id="overview-region-drift",
    )
    # Stim-artifact overlay: lazily loaded on button click (loading many
    # traces is heavy), in its own section so a refresh tick doesn't wipe it.
    artifact_overlay = _collapsible(
        "Stim-artifact overlay",
        html.Div([
            html.Div("Overlays the recorded stim artifacts (oldest→newest) so "
                     "you can see how the pulse response + ohmic step shift "
                     "alongside the Rₐ trend. Pick any animal + stim "
                     "location(s) and a recording date range, then Load.",
                     style={"color": "#a0a0b0", "fontSize": "11px",
                             "marginBottom": "6px"}),
            _artifact_overlay_controls(store, config),
            html.Button("Load / refresh overlay",
                        id="overview-artifact-btn", n_clicks=0,
                        style={"fontSize": "12px", "cursor": "pointer",
                                "padding": "4px 10px", "borderRadius": "5px",
                                "color": "#f0f0f5", "background": "#3a3a4a",
                                "border": "1px solid #555"}),
            dcc.Loading(html.Div(id="overview-artifact-body"), type="dot"),
        ]),
        open_default=False)
    matlab_failed = _build_matlab_failed_card(store, config)
    gain_blocked = _build_gain_blocked_card(store, config)

    # No SECTION_STYLE on these wrappers -- the _collapsible they're
    # placed inside is the visible card. Nested chrome was making
    # the dashboard read as "card-on-card-on-card."
    km_section = html.Div(
        _pending_placeholder("Kaplan–Meier log summary"),
        id="overview-km-log",
    )
    queue_section = html.Div(
        _pending_placeholder("review queue"),
        id="overview-queue",
    )

    # Evoked thumbnail: radio-selected trace mode +
    # callback-rendered figure. The mode and figure persist across
    # refresh ticks because the radio lives outside the container Div
    # that gets re-children'd by refresh_overview_thumbnail.
    trace_mode_radio = html.Div([
        html.Span("Trace mode: ",
                  style={"color": "#888", "fontSize": "12px",
                          "marginRight": "10px"}),
        dcc.RadioItems(
            id="overview-trace-mode",
            options=[
                {"label": " Mean", "value": "mean"},
                {"label": " Mean ± SEM", "value": "sem"},
                {"label": " Overlay all files", "value": "overlay"},
                {"label": " Recent recordings", "value": "hist24"},
            ],
            value="mean",
            inline=True,
            labelStyle={"color": "#ddd", "fontSize": "12px",
                         "marginRight": "16px"},
            inputStyle={"marginRight": "4px"},
        ),
        # How many recent recordings the overlay spans (only used in the
        # 'Recent recordings' mode). Defaults from config.overview.
        html.Span("  N: ", style={"color": "#888", "fontSize": "12px",
                                    "marginLeft": "8px"}),
        dcc.Input(id="overview-hist24-n", type="number", min=1,
                  max=_HIST24_MAX, step=1, value=_hist24_default_n(config),
                  debounce=True,
                  style={"width": "64px", "backgroundColor": "#262638",
                          "color": "#f0f0f5",
                          "border": "1px solid #444",
                          "borderRadius": "4px", "fontSize": "12px"}),
    ], style={"marginTop": "16px", "marginBottom": "4px"})
    waveform_thumbnail = html.Div([
        trace_mode_radio,
        # Polls only while the Last-24 background read is in flight; the
        # thumbnail callback flips its `disabled` once the cache is ready.
        dcc.Interval(id="overview-hist24-poll", interval=1200, disabled=True),
        html.Div(
            _pending_placeholder("evoked thumbnail"),
            id="overview-thumbnail",
        ),
        # Per-client signature of the last-rendered thumbnail (mode + newest
        # recording) so a refresh tick with nothing new skips the rebuild.
        dcc.Store(id="overview-thumb-sig"),
    ])

    # Active session basics (session name / file counts / errors) now
    # live in the queue block's Active Session section, so just emit
    # an empty placeholder here to keep the channel-map branch below.
    session_info = html.Div()
    if active_session:
        if ch_map:
            ch_rows = []
            for idx in sorted(ch_map.keys()):
                info = ch_map[idx]
                ch_rows.append({
                    "index": idx,
                    "name": info["name"],
                    "role": info["role"],
                })
            channel_table = html.Div([
                dash_table.DataTable(
                    data=ch_rows,
                    columns=[{"name": "Index", "id": "index"},
                             {"name": "Name", "id": "name"},
                             {"name": "Role", "id": "role"}],
                    **DARK_TABLE_STYLE,
                    style_data_conditional=[ZEBRA_STRIPE,
                        {"if": {"filter_query": "{role} = eeg"},
                         "color": ROLE_COLORS["eeg"]},
                        {"if": {"filter_query": "{role} = stim_copy"},
                         "color": ROLE_COLORS["stim_copy"]},
                        {"if": {"filter_query": "{role} = reference"},
                         "color": ROLE_COLORS["reference"]},
                    ],
                    page_size=8,
                ),
            ], style={"maxHeight": "300px", "overflow": "hidden"})
        else:
            channel_table = html.Div()
    else:
        session_info = html.P("No sessions yet", style={"color": "#888"})
        channel_table = html.Div()

    # Recent alerts -- capped + scrollable so it doesn't push the
    # rest of the Overview off the viewport. Title lives in the
    # collapsible summary so we drop the inner header.
    if recent_alerts:
        alerts_section = html.Div([
            dash_table.DataTable(
                data=[{"time": a["sent_at"][:19], "severity": a["severity"],
                       "type": a["alert_type"], "message": a["message"][:120]}
                      for a in recent_alerts[:15]],
                columns=[{"name": c, "id": c} for c in ["time", "severity", "type", "message"]],
                **DARK_TABLE_STYLE,
                style_data_conditional=[ZEBRA_STRIPE,
                    {"if": {"filter_query": "{severity} = critical"},
                     "backgroundColor": "#3d1111", "color": "#ff6b6b"},
                    {"if": {"filter_query": "{severity} = warning"},
                     "backgroundColor": "#3d3011", "color": "#ffd93d"},
                    {"if": {"filter_query": "{severity} = info"},
                     "backgroundColor": "#112233", "color": "#6bb5ff"},
                ],
                page_size=5,
            ),
        ], style={"maxHeight": "220px", "overflow": "hidden"})
    else:
        alerts_section = html.Div(
            "No alerts in the last 24 hours",
            style={"color": "#666", "fontSize": "11px",
                    "padding": "4px 2px"})

    today = date.today()
    # Home grid (surgeries / schedule / maintenance / incidents /
    # data log) sits at the TOP of the Overview now, in a 3-col-when-
    # wide auto-fit grid. Each tile keeps its existing internal
    # header. minmax(380px, 1fr) means the layout shows 1 column under
    # ~760px, 2 under ~1140px, 3 above that.
    # Lab tiles stacked vertically -- they live in the first
    # column of the new 3-col top section (Lab | Pills | Evoked).
    # The container is a flex column so each tile gets the full
    # column width and stacks predictably regardless of how many
    # tiles _build_home_grid_children returns.
    home_grid = html.Div(
        _pending_placeholder("lab home grid"),
        id="overview-home-grid",
        style={
            "display": "flex", "flexDirection": "column",
            "gap": "8px",
        },
    )

    # Recent alerts becomes a single header row with the count so it
    # stays as a visible divider even when collapsed.
    n_alerts = len(recent_alerts) if recent_alerts else 0
    alerts_collapsible = _collapsible(
        f"Recent Alerts",
        alerts_section,
        open_default=bool(recent_alerts),
        badge=str(n_alerts) if n_alerts else "0",
        badge_color=("#EF553B" if any(
            a.get("severity") == "critical" for a in (recent_alerts or []))
            else "#FFA15A" if recent_alerts else "rgba(255,255,255,0.08)"),
    )

    km_wrapped = _collapsible(
        "KM Recorder log", km_section, open_default=True,
    )
    # Latest video snapshot: most recent decodable frame from the
    # newest companion video. Renders as a small thumbnail by
    # default so it doesn't push the column tall; click the image
    # to expand it inline. The src is cache-busted on every refresh-
    # trigger tick so we always see fresh frames without reload.
    snapshot_card = html.Div([
        html.Img(
            id="overview-snapshot-img",
            src="/media/latest-snapshot.jpg",
            title="Click to expand / collapse",
            n_clicks=0,
            style={
                # 320 px max width fits a 2-camera side-by-side
                # collage without crushing it; single-camera
                # sessions still cap at 180 px via maxHeight.
                "display": "block",
                "maxWidth": "320px",
                "maxHeight": "180px",
                "width": "100%",
                "height": "auto",
                "objectFit": "contain",
                "borderRadius": "4px",
                "background": "#0a0a14",
                "cursor": "zoom-in",
            },
        ),
        html.Div(
            id="overview-snapshot-caption",
            style={"color": "#888", "fontSize": "10px",
                    "marginTop": "4px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.4px"},
        ),
    ])
    snapshot_wrapped = _collapsible(
        "Latest snapshot", snapshot_card, open_default=True,
    )
    queue_wrapped = _collapsible(
        "Today · Recording uptime", queue_section, open_default=True,
    )
    # Latest Evoked lives in column 3 of the top section; not full
    # width. User explicitly asked for it to be narrower than the
    # whole page. The 1fr grid cell still leaves it the widest
    # visible element after the two narrow sidebar columns.
    thumb_wrapped = _collapsible(
        "Latest Evoked", waveform_thumbnail,
        open_default=True,
    )
    channel_wrapped = _collapsible(
        "Channel Map", channel_table,
        open_default=False,
    )
    # Status pills are a full-width row at the very top, above
    # everything else. The 3-column grid below holds:
    #   Col 1 (280px): Lab tiles
    #   Col 2 (1fr):   Latest Evoked
    #   Col 3 (1fr):   Today + KM stacked
    # Channel Map drops underneath Evoked in column 2; Recent
    # Alerts drops underneath the snapshot in column 3. Both stay
    # collapsed by default so they just add a thin header strip,
    # but they prevent the ragged uneven dead zone the user saw
    # below the three columns. No separate bottom row anymore --
    # the page now reads as a single 3-column slab.
    evoked_col = html.Div([
        thumb_wrapped, channel_wrapped,
    ], style={
        "display": "flex", "flexDirection": "column", "gap": "8px",
    })
    ops_col = html.Div([
        queue_wrapped, km_wrapped, snapshot_wrapped,
        alerts_collapsible,
    ], style={
        "display": "flex", "flexDirection": "column", "gap": "8px",
    })
    top_section = html.Div([
        home_grid,
        evoked_col,
        ops_col,
    ], style={
        "display": "grid",
        # Sidebar (lab tiles) at 320 px. minmax(0, 1fr) on cols 2
        # and 3 stops nowrap content (KM log filenames, active-
        # session names) from forcing those columns wider.
        "gridTemplateColumns": "320px minmax(0, 1fr) minmax(0, 1fr)",
        "gap": "12px",
        "alignItems": "start",
        "marginBottom": "10px",
    })

    # Umbrella: every stimulation/electrode-derived diagnostic under one
    # collapsed section at the bottom of the page, with a glanceable badge so
    # the user knows whether anything inside needs attention.
    stim_badge, stim_badge_color = _stim_health_badge(store, config)
    stim_health = _collapsible(
        "Electrode & stimulation health",
        html.Div([
            html.Div("Everything derived from the stimulation pulse — "
                     "electrode series resistance (Rₐ), stim consistency "
                     "(Z_ss), current-delivery fidelity, placement-region "
                     "drift, and the raw stim-artifact overlay.",
                     style={"color": "#a0a0b0", "fontSize": "11px",
                             "marginBottom": "8px"}),
            impedance_status,  # per-channel access-resistance (Rₐ) drift trend
            _build_electrode_compare_card(store, config),  # pick & overlay
            zss_status,        # slow-phase steady-state impedance (consistency)
            fidelity_status,   # current-delivery fidelity (two-edge ratio)
            region_status,     # drift by placement region (SLM vs SR)
            artifact_overlay,  # lazy stim-artifact overlay across recordings
        ]),
        open_default=False, badge=stim_badge,
        badge_color=stim_badge_color)

    return html.Div([
        # PRIORITY LOAD ORDER. Each card-fill callback is kicked by one of these
        # staggered one-shot intervals so the page fills top-to-bottom by
        # importance instead of all ~10 callbacks firing at once (which convoyed
        # the 8-thread pool and left the high-value pills waiting). Highest first:
        #   120ms  status pills + review queue   (overview-mount)
        #   450ms  behavioral-seizure status     (overview-mount-bsz)
        #   900ms  Latest-Evoked thumbnail       (overview-mount-thumb)
        #   1400ms lab home grid + KM log        (overview-mount-mid)
        #   2400ms stim/electrode umbrella (bottom, lowest) (overview-mount-stim)
        # All fills ALSO listen to refresh-trigger, so this only orders the
        # INITIAL paint; periodic refresh keeps everything fresh thereafter.
        dcc.Interval(id="overview-mount", interval=120, max_intervals=1),
        dcc.Interval(id="overview-mount-bsz", interval=450, max_intervals=1),
        dcc.Interval(id="overview-mount-thumb", interval=900, max_intervals=1),
        dcc.Interval(id="overview-mount-mid", interval=1400, max_intervals=1),
        dcc.Interval(id="overview-mount-stim", interval=2400, max_intervals=1),
        # SLOW refresh (90s) for the heavy, low-priority cards. The 10s
        # refresh-trigger drives ONLY the cheap pills + queue; the expensive
        # builds (stim/electrode umbrella 12-38s, Google-Sheets home-grid + KM
        # log 6-38s, behavioral-seizure status 17s) were re-firing on every 10s
        # tick and, being slower than the interval, saturated the dashboard's
        # single GIL and starved the pills + left column. Those cards move here
        # so they still auto-refresh, just at a cadence that leaves the GIL free
        # for the high-priority cards. Each still paints immediately via its
        # overview-mount-* one-shot on tab open.
        dcc.Interval(id="overview-slow-refresh", interval=90_000),
        cards,         # pills strip, full width
        matlab_failed,  # ⚠ recordings that failed MATLAB processing (or empty)
        gain_blocked,   # ⚠ impedance blocked on missing File_Records gain (or empty)
        bsz_status,    # per-animal seizure analysis status
        top_section,   # sidebar | (Evoked + Channel Map) | (Today + KM + Snapshot + Alerts)
        stim_health,   # collapsed umbrella of all stim/electrode diagnostics
    ])


def _excluded_animals(config) -> list[str]:
    """Subjects to omit from the behavioral-seizure status card. Defaults
    to excluding 'Randles' (a non-experimental subject); overridable via
    config.overview.exclude_animals."""
    ov = (config or {}).get("overview", {}) or {}
    val = ov.get("exclude_animals")
    return list(val) if isinstance(val, list) else ["Randles"]


def _matlab_fail_rows(failed, cap=10):
    """One line per failed recording: basename + attempt count + latest error."""
    rows = []
    for f in failed[:cap]:
        name = os.path.basename(f.get("file_path") or "") or f"#{f.get('file_id')}"
        err = (f.get("error_message") or "").strip().replace("\n", " ")
        n = f.get("attempts") or 0
        rows.append(html.Div([
            html.Span(name, style={"color": "#f0f0f5", "fontSize": "12px",
                                    "fontWeight": "600"}),
            html.Span(f"  ×{n}" if n else "",
                      style={"color": "#7a7a8a", "fontSize": "11px"}),
            html.Span(f"  {err[:140]}" if err else "",
                      style={"color": "#a0a0b0", "fontSize": "11px"}),
        ], style={"padding": "3px 0",
                  "borderBottom": "1px solid rgba(255,255,255,0.06)"}))
    return rows


def _build_matlab_failed_card(store, config=None):
    """A red banner listing recordings whose MATLAB (Tier-2) step errored and are
    still being AUTO-RETRIED, so a per-file failure isn't invisible (an empty
    Evoked tab otherwise looks identical to a legitimate no-stimuli file). Files
    that exhausted their retries drop into a collapsed 'gave up' list below so
    they stop re-nagging. Empty Div when nothing is failing."""
    cap = int((config or {}).get("matlab_max_attempts", 4))
    try:
        failed = store.matlab_failed_files(hours=168, max_attempts=cap)
        exhausted = store.matlab_exhausted_files(max_attempts=cap)
    except Exception:  # noqa: BLE001
        failed, exhausted = [], []
    if not failed and not exhausted:
        return html.Div()

    children = []
    if failed:
        extra = (f"  (+{len(failed) - 10} more)" if len(failed) > 10 else "")
        header = html.Div([
            html.Span(f"⚠ {len(failed)} recording(s) failed MATLAB processing "
                      f"(last 7 days){extra}",
                      style={"color": "#ff453a", "fontWeight": "700",
                             "fontSize": "13px"}),
            html.Button("↻ Retry all", id="overview-retry-matlab", n_clicks=0,
                        style={"marginLeft": "auto", "padding": "2px 10px",
                               "fontSize": "12px", "cursor": "pointer",
                               "borderRadius": "5px", "color": "#f0f0f5",
                               "background": "#3a3a4a",
                               "border": "1px solid #555"}),
        ], style={"display": "flex", "alignItems": "center",
                  "marginBottom": "6px"})
        children += [
            header,
            html.Div("These produced no evoked data. They are auto-retried "
                     f"(up to {cap} attempts, backing off between tries); fix "
                     "the cause or click Retry all to re-queue now.",
                     style={"color": "#a0a0b0", "fontSize": "11px",
                            "marginBottom": "8px"}),
            html.Div(id="overview-retry-matlab-status",
                     style={"color": "#30d158", "fontSize": "11px"}),
            *_matlab_fail_rows(failed),
        ]
    if exhausted:
        gextra = (f"  (+{len(exhausted) - 10} more)"
                  if len(exhausted) > 10 else "")
        children.append(_collapsible(
            f"⋯ {len(exhausted)} recording(s) gave up after {cap} attempts"
            f"{gextra}",
            html.Div([
                html.Div(f"These exceeded MATLAB processing {cap} times "
                         "(likely too heavy for matlab_timeout_sec). No longer "
                         "auto-retried; raise the timeout / enable the local "
                         "mirror, then Retry all to try again.",
                         style={"color": "#a0a0b0", "fontSize": "11px",
                                "marginBottom": "6px"}),
                *_matlab_fail_rows(exhausted),
            ]),
            open_default=False, badge=str(len(exhausted)),
            badge_color="rgba(120,120,140,0.85)", title_color="#9a9aa8"))
    return html.Div(children, style={
        "marginBottom": "12px", "padding": "10px 14px",
        "background": "rgba(255,69,58,0.08)",
        "border": "1px solid rgba(255,69,58,0.35)",
        "borderRadius": "8px"})


def _build_gain_blocked_card(store, config=None):
    """Amber banner naming animals whose impedance can't be computed because their
    amplifier GAIN is missing from the File_Records sheet -- a silent data-entry
    gap that otherwise just looks like 'no impedance yet' forever (BCH040's 663
    files were invisibly stuck this way). Legacy animals listed in
    ``impedance.ignore_blocked_animals`` are excluded (their gain will never
    resolve). Empty Div when nothing is blocked."""
    ignore = ((config or {}).get("impedance", {}) or {}).get(
        "ignore_blocked_animals", []) or []
    try:
        blocked = store.files_blocked_on_missing_gain(exclude=ignore)
    except Exception:  # noqa: BLE001
        blocked = []
    if not blocked:
        return html.Div()
    total = sum(int(b["files"]) for b in blocked)
    who = ", ".join(f"{b['animal_id']} ({b['files']})" for b in blocked[:8])
    extra = (f"  (+{len(blocked) - 8} more animal(s))" if len(blocked) > 8 else "")
    body = html.Div([
        html.Div(f"Blocked animals: {who}{extra}",
                 style={"color": "#f0f0f5", "fontSize": "12px",
                        "marginBottom": "4px"}),
        html.Div("Add each animal's gain to the File_Records sheet; the daemon "
                 "recomputes access resistance on the next impedance sweep — no "
                 "restart needed.",
                 style={"color": "#a0a0b0", "fontSize": "11px"}),
    ])
    # Collapsed by default -- it's a standing data-entry advisory, not an alert
    # that needs to occupy the fold every load. The ⚠ + amber title + count badge
    # keep it noticeable in the collapsed summary.
    return _collapsible(
        f"⚠ {total} recording(s) have no impedance — amplifier gain missing "
        f"from File_Records",
        body,
        open_default=False,
        badge=str(total),
        badge_color="rgba(255,214,10,0.85)",
        title_color="#ffd60a")


def _build_behavioral_seizure_status_card(store, config=None):
    """Per-animal behavioral seizure analysis status.

    Designed so the user can answer two questions at a glance:
      1. Are we keeping up?  -- rate delta on top.
      2. Which animal is the bottleneck?  -- per-row breakdown.

    Sorted so animals with the steepest backlog growth float
    to the top.
    """
    rows = store.behavioral_seizure_status_per_animal(
        days=7, exclude=_excluded_animals(config))
    if not rows:
        return html.Div(
            "No animals ingested yet.",
            style={"padding": "12px",
                    "color": "#a0a0b0",
                    "fontSize": "12px",
                    "background": COLOR_SURFACE_1,
                    "border": f"1px solid {COLOR_DIVIDER}",
                    "borderRadius": RADIUS_MD})
    # Totals + rate verdict.
    total_created = sum(r["created_window"] for r in rows)
    total_approved = sum(r["approved_window"] for r in rows)
    total_queue = sum(r["n_queue"] for r in rows)
    total_pending = sum(r["n_pending_pi"] for r in rows)
    total_needs = sum(r.get("n_needs_scoring", 0) for r in rows)
    total_threshold = sum(r.get("n_flag_pool", 0) for r in rows)
    rate_delta = total_created - total_approved
    rate_per_day = rate_delta / 7.0
    if rate_per_day > 5:
        verdict_text = (
            f"Backlog growing by +{rate_per_day:.1f} files/day. "
            "Consider hiring more reviewers.")
        verdict_color = "#ff453a"
    elif rate_per_day > 0.5:
        verdict_text = (
            f"Backlog growing slowly (+{rate_per_day:.1f} "
            "files/day). Team marginal.")
        verdict_color = "#ff9f0a"
    elif rate_per_day > -0.5:
        verdict_text = (
            "Team keeping up. Throughput matches arrivals.")
        verdict_color = "#30d158"
    else:
        verdict_text = (
            f"Backlog shrinking ({rate_per_day:.1f} files/day). "
            "Team ahead of schedule.")
        verdict_color = "#30d158"
    # Collapsed = this compressed summary (totals + verdict); expanding
    # reveals the per-animal cards/charts. The +/- marker + hover come from
    # the shared `details > summary` CSS (theme.css).
    summary = html.Summary([
        html.Div([
            html.Span("Behavioral seizure analysis status",
                       style={"color": "#f0f0f5", "fontWeight": "600",
                               "fontSize": "13px"}),
            html.Span("  (last 7 days)",
                       style={"color": "#888", "fontSize": "11px"}),
        ]),
        html.Div([
            html.Span(f"{total_created} created  ·  ",
                       style={"color": "#cfd0d6"}),
            html.Span(f"{total_approved} approved  ·  ",
                       style={"color": "#cfd0d6"}),
            html.Span(f"{total_queue} in queue  ·  ",
                       style={"color": "#cfd0d6"}),
            html.Span(f"{total_pending} pending PI  ·  ",
                       style={"color": "#cfd0d6"}),
            html.Span(f"🚩 {total_threshold} flagged  ·  ",
                       style={"color": "#f0b429" if total_threshold
                              else "#cfd0d6"}),
            html.Span(f"⚠️ {total_needs} needs more onsets",
                       style={"color": "#ff9f0a" if total_needs
                              else "#cfd0d6"}),
        ], style={"fontSize": "12px", "marginTop": "2px"}),
        html.Div(verdict_text,
                  style={"color": verdict_color, "fontWeight": "600",
                          "fontSize": "12px", "marginTop": "6px"}),
    ], style={"cursor": "pointer", "userSelect": "none",
               "padding": "12px 14px", "listStyle": "none"})
    # Body (expanded): Cards <-> Charts toggle + the per-animal detail.
    body = html.Div([
        dcc.RadioItems(
            id="bsz-view-toggle",
            options=[{"label": " Cards", "value": "cards"},
                      {"label": " Charts", "value": "charts"},
                      {"label": " Calendar", "value": "calendar"}],
            value="cards", inline=True,
            style={"margin": "2px 0 4px", "fontSize": "11px"},
            labelStyle={"color": "#cfd0d6", "marginRight": "14px",
                         "cursor": "pointer"},
            inputStyle={"marginRight": "4px"}),
        html.Div(_bsz_cards(rows, store.flag_email_subscribed_animals()),
                  id="overview-bsz-body"),
    ], style={"borderTop": f"1px solid {COLOR_DIVIDER}",
               "padding": "4px 4px 6px"})
    return html.Details([summary, body], open=False,
                         style={"background": COLOR_SURFACE_1,
                                 "border": f"1px solid {COLOR_DIVIDER}",
                                 "borderRadius": RADIUS_MD,
                                 "marginBottom": "6px"})


def _flag_email_checkbox(animal_id: str, subscribed: bool):
    """The per-animal 'email me when a seizure event is flagged' opt-in shown on
    each seizure-status card. Global toggle -> Store.set_flag_email_subscription;
    the background loop (AlertRuleEngine.check_flagged_events) does the sending."""
    return html.Div([
        dcc.Checklist(
            id={"type": "bsz-flag-email", "animal": animal_id},
            options=[{"label": " 📧 email me on a new flagged event",
                       "value": "on"}],
            value=["on"] if subscribed else [],
            style={"fontSize": "10px"},
            labelStyle={"color": "#cfd0d6", "cursor": "pointer",
                         "display": "inline-flex", "alignItems": "center"},
            inputStyle={"marginRight": "5px", "cursor": "pointer"}),
        html.Span(id={"type": "bsz-flag-email-status", "animal": animal_id},
                   style={"fontSize": "10px", "color": "#30d158",
                           "marginLeft": "4px"}),
    ], style={"marginTop": "6px", "paddingTop": "6px",
               "borderTop": f"1px solid {COLOR_DIVIDER}"})


def _bsz_cards(rows, subscribed: set | None = None):
    """Per-animal stat cards (the default 'Cards' view).

    *subscribed* is the set of animals opted in to flagged-event emails (from
    ``Store.flag_email_subscribed_animals``); each card carries a checkbox
    reflecting and toggling that opt-in."""
    subscribed = subscribed or set()
    cells = []
    for r in rows:
        delta = (r["created_window"]
                  - r["approved_window"])
        delta_color = ("#ff453a" if delta > 5
                        else "#ff9f0a" if delta > 0
                        else "#30d158")
        delta_label = (f"+{delta}" if delta > 0
                        else f"{delta}")
        last = r["last_activity_at"][:16] or "—"
        cells.append(html.Div([
            html.Div(r["animal_id"],
                      style={"color": "#f0f0f5",
                              "fontWeight": "600",
                              "fontSize": "13px",
                              "marginBottom": "4px"}),
            html.Div([
                html.Div([
                    html.Span("queue",
                               style={"color": "#888",
                                       "fontSize": "10px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
                    html.Div(f"{r['n_queue']}",
                              style={"color": "#cfd0d6",
                                      "fontSize": "16px",
                                      "fontWeight": "600"}),
                ]),
                html.Div([
                    html.Span("pending PI",
                               style={"color": "#888",
                                       "fontSize": "10px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
                    html.Div(f"{r['n_pending_pi']}",
                              style={"color": "#5e7ce2",
                                      "fontSize": "16px",
                                      "fontWeight": "600"}),
                ]),
                html.Div([
                    html.Span("approved",
                               style={"color": "#888",
                                       "fontSize": "10px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
                    html.Div(f"{r['n_approved']}",
                              style={"color": "#30d158",
                                      "fontSize": "16px",
                                      "fontWeight": "600"}),
                ]),
                html.Div([
                    html.Span("🚩 flagged",
                               style={"color": "#888",
                                       "fontSize": "10px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
                    html.Div(f"{r.get('n_flag_pool', 0)}",
                              style={"color": "#f0b429"
                                     if r.get("n_flag_pool")
                                     else "#cfd0d6",
                                      "fontSize": "16px",
                                      "fontWeight": "600"}),
                ]),
                html.Div([
                    html.Span("⚠️ needs more onsets",
                               style={"color": "#888",
                                       "fontSize": "10px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
                    html.Div(f"{r.get('n_needs_scoring', 0)}",
                              style={"color": "#ff9f0a"
                                     if r.get("n_needs_scoring")
                                     else "#cfd0d6",
                                      "fontSize": "16px",
                                      "fontWeight": "600"}),
                ]),
                html.Div([
                    html.Span("7 d net",
                               style={"color": "#888",
                                       "fontSize": "10px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
                    html.Div(delta_label,
                              style={"color": delta_color,
                                      "fontSize": "16px",
                                      "fontWeight": "600"}),
                ]),
            ], style={"display": "grid",
                       "gridTemplateColumns":
                           "repeat(3, 1fr)",
                       "gap": "8px",
                       "marginBottom": "4px"}),
            html.Div(
                f"created 7 d: {r['created_window']}  ·  "
                f"approved 7 d: {r['approved_window']}  ·  "
                f"last activity: {last}",
                style={"color": "#888",
                        "fontSize": "10px"}),
            _flag_email_checkbox(r["animal_id"],
                                 r["animal_id"] in subscribed),
        ], style={"padding": "10px 12px",
                   "background": COLOR_SURFACE_1,
                   "border": f"1px solid {COLOR_DIVIDER}",
                   "borderRadius": RADIUS_SM,
                   "borderLeft": f"3px solid {delta_color}"}))
    return html.Div(
        cells,
        style={"display": "grid",
                "gridTemplateColumns":
                    "repeat(auto-fit, minmax(300px, 1fr))",
                "gap": "8px",
                "padding": "12px 14px"})


def _bsz_charts(rows):
    """Per-animal donut charts (the 'Charts' view) -- one donut per animal
    showing its review-status composition (queue / pending PI / approved /
    needs scoring), with the total in the centre. Colours match the cards.
    (Threshold detections are a detector metric that overlaps the queue, so
    they are shown on the cards but not in this pipeline-composition pie.)"""
    labels = ["queue", "pending PI", "approved", "needs more onsets"]
    colors = ["#8a8a99", "#5e7ce2", "#30d158", "#ff9f0a"]
    n = len(rows)
    cols = min(n, 5) if n else 1
    n_rows = (n + cols - 1) // cols
    specs = [[{"type": "domain"} for _ in range(cols)]
             for _ in range(n_rows)]
    fig = make_subplots(
        rows=n_rows, cols=cols, specs=specs,
        subplot_titles=[r["animal_id"] for r in rows],
        vertical_spacing=0.14, horizontal_spacing=0.03)
    for i, r in enumerate(rows):
        rr, cc = i // cols + 1, i % cols + 1
        vals = [r["n_queue"], r["n_pending_pi"], r["n_approved"],
                r.get("n_needs_scoring", 0)]
        fig.add_trace(go.Pie(
            labels=labels, values=vals, sort=False, hole=0.58,
            marker=dict(colors=colors,
                         line=dict(color="#13131f", width=1)),
            textinfo="percent", textfont=dict(size=9),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            showlegend=(i == 0)), rr, cc)
        dom = fig.data[-1].domain
        fig.add_annotation(
            x=(dom.x[0] + dom.x[1]) / 2, y=(dom.y[0] + dom.y[1]) / 2,
            text=f"<b>{sum(vals):,}</b>", showarrow=False,
            font=dict(size=13, color="#f0f0f5"))
    for ann in fig.layout.annotations[:n]:   # animal-id subplot titles
        ann.font.size = 12
        ann.font.color = "#f0f0f5"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=130 + 170 * n_rows,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08,
                     xanchor="center", x=0.5, font=dict(size=10),
                     bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#cfd0d6"))
    return dcc.Graph(
        figure=fig, config={"displayModeBar": False},
        style={"padding": "8px 6px"})


def _bsz_calendar(cal: dict):
    """Per-animal seizure calendar (GitHub-contribution style): one grid per
    animal, columns = weeks, rows = weekday, cell colour = number of scored
    seizures (EEG onset + Racine) that day. All animals share ONE week axis so
    the reviewer can compare WHEN across animals at a glance.

    *cal* is ``{animal: {'YYYY-MM-DD': n_seizures}}`` from
    ``Store.seizure_days_per_animal``."""
    if not cal:
        return html.Div(
            "No scored seizures yet — the calendar fills in as EEG onsets get "
            "an onset time + Racine score.",
            style={"padding": "12px", "color": "#a0a0b0", "fontSize": "12px"})
    from datetime import datetime as _dt, timedelta as _td
    animals = sorted(cal.keys())
    all_days = sorted({d for m in cal.values() for d in m})
    d0 = _dt.strptime(all_days[0], "%Y-%m-%d")
    d1 = _dt.strptime(all_days[-1], "%Y-%m-%d")
    start = d0 - _td(days=d0.weekday())          # Monday on/before first day
    n_weeks = ((d1 - start).days // 7) + 1
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    # A month label at the first week-column of each month.
    ticks, labels, seen = [], [], set()
    for w in range(n_weeks):
        wk = start + _td(days=7 * w)
        key = (wk.year, wk.month)
        if key not in seen:
            seen.add(key)
            ticks.append(w)
            labels.append(wk.strftime("%b %y"))
    zmax = max(2, max((c for m in cal.values() for c in m.values()),
                       default=1))
    n = len(animals)
    fig = make_subplots(rows=n, cols=1, subplot_titles=animals,
                         vertical_spacing=(0.11 if n > 1 else 0.0))
    for i, a in enumerate(animals):
        z = [[None] * n_weeks for _ in range(7)]
        cd = [[""] * n_weeks for _ in range(7)]
        for day, cnt in cal[a].items():
            dd = _dt.strptime(day, "%Y-%m-%d")
            w = (dd - start).days // 7
            if 0 <= w < n_weeks:
                z[dd.weekday()][w] = cnt
                cd[dd.weekday()][w] = day
        fig.add_trace(go.Heatmap(
            z=z, x=list(range(n_weeks)), y=weekdays, customdata=cd,
            zmin=1, zmax=zmax, xgap=2, ygap=2,
            colorscale=[[0.0, "#f0b429"], [1.0, "#ff453a"]],   # amber -> red
            showscale=(i == 0),
            colorbar=(dict(title=dict(text="seizures/day", side="right"),
                           thickness=10, len=0.85, tickfont=dict(size=9))
                      if i == 0 else None),
            hovertemplate="%{customdata} · %{z} seizure(s)<extra>"
                          + a + "</extra>"),
            i + 1, 1)
        fig.update_xaxes(tickvals=ticks, ticktext=labels, tickangle=0,
                         tickfont=dict(size=8), showgrid=False,
                         row=i + 1, col=1)
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=8),
                         showgrid=False, row=i + 1, col=1)
    for ann in fig.layout.annotations[:n]:       # left-align animal titles
        ann.font.size = 12
        ann.font.color = "#f0f0f5"
        ann.x = 0
        ann.xanchor = "left"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=70 + 120 * n, margin=dict(l=34, r=10, t=28, b=16),
        font=dict(color="#cfd0d6"))
    return dcc.Graph(figure=fig, config={"displayModeBar": False},
                     style={"padding": "8px 6px"})



# ===================================================================== #
#  Transfer-impedance trend card
# ===================================================================== #

# Max channels charted (keeps the figure a sane height); overflow is
# reported in the header rather than silently dropped.
_IMPEDANCE_MAX_CHANNELS = 30

# Graph config for the access-R figures: keep a hover mode bar (with a
# reset-axes button) + double-click-to-reset so pan/zoom is recoverable.
_IMPEDANCE_GRAPH_CONFIG = {
    "displayModeBar": "hover", "displaylogo": False,
    "doubleClick": "reset", "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d",
                                "autoScale2d", "toggleSpikelines"],
}


def _impedance_cfg(config) -> dict:
    """Drift thresholds + baseline window, shared with the alert rule."""
    rules = ((config or {}).get("alerting", {}) or {}).get("rules", {}) or {}
    return {
        "pct": float(rules.get("impedance_shift_pct", 40)),
        "pct_crit": float(rules.get("impedance_shift_pct_critical", 75)),
        "window": int(rules.get("impedance_baseline_window", 10)),
        "min_history": int(rules.get("impedance_min_history", 4)),
        "sag_pct": float(rules.get("current_sag_pct", -15)),
    }


def _median(vals: list) -> float | None:
    s = sorted(v for v in vals if v is not None)
    if not s:
        return None
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _parse_chunk_dt(s):
    """Parse a recorder chunk_datetime ('YYYY_MM_DD__HH_MM_SS' or ISO) to a
    datetime, else None -- lets the impedance trend use a real time axis."""
    if not s:
        return None
    from datetime import datetime as _dt
    for fmt in ("%Y_%m_%d__%H_%M_%S", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return _dt.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _impedance_span_note(channels: list[dict]):
    """A one-line 'these span X → Y' caption so the measurement timescale is
    obvious at a glance."""
    dts = [_parse_chunk_dt(r["chunk_datetime"])
           for c in channels for r in c["rows"]]
    dts = [d for d in dts if d is not None]
    if not dts:
        return html.Div()
    lo, hi = min(dts), max(dts)
    n_max = max((len(c["rows"]) for c in channels), default=0)
    span_days = (hi - lo).total_seconds() / 86400.0
    span_txt = (f"{span_days:.1f} days" if span_days >= 1
                else f"{(hi - lo).total_seconds() / 3600.0:.1f} h")
    return html.Div(
        f"Timescale: {lo:%Y-%m-%d %H:%M} → {hi:%Y-%m-%d %H:%M}  "
        f"({span_txt}, up to {n_max} recordings/channel)",
        style={"color": "#888", "fontSize": "10px", "padding": "0 2px 6px"})


def _drift_stat(values: list, window: int, min_history: int) -> dict | None:
    """Rolling-median drift of the newest point vs the prior *window*.

    Returns ``{latest, baseline, drift_pct}`` or None when there isn't
    enough history / the baseline is degenerate.
    """
    vals = [v for v in values if v is not None]
    if len(vals) < max(2, min_history):
        return None
    latest = vals[-1]
    baseline = _median(vals[max(0, len(vals) - 1 - window):-1])
    if baseline is None or baseline == 0:
        return None
    return {"latest": latest, "baseline": baseline,
            "drift_pct": (latest - baseline) / baseline * 100.0}


def _build_zss_consistency_card(store, config=None):
    """Slow-phase steady-state impedance (Z_ss) per channel: a stim-consistency
    check across time AND between animals (gain-corrected so it's comparable).
    Cross-animal bars (latest per channel) + per-channel trend, collapsed."""
    title = "Stim consistency — slow-phase steady-state impedance (Z_ss)"
    try:
        series = store.impedance_series_by_channel(
            exclude=_excluded_animals(config))
        active = store.active_impedance_channel_keys()
        latest = store.latest_zss_per_channel()
    except Exception as e:  # noqa: BLE001
        logger.warning("zss card failed: %s", e)
        series, active, latest = {}, set(), {}
    if active:
        series = {k: v for k, v in series.items() if k in active}
        latest = {k: v for k, v in latest.items() if k in active}
    icfg = _impedance_cfg(config)
    channels = []
    for (animal, ch), rows in series.items():
        vals = [r.get("slow_ss_kohm") for r in rows]
        if not any(v is not None for v in vals):
            continue
        channels.append({
            "animal": animal, "channel": ch, "rows": rows,
            "stat": _drift_stat(vals, icfg["window"], icfg["min_history"]),
            "latest": latest.get((animal, ch))})
    if not channels:
        return _collapsible(title, html.Div(
            "No slow-phase steady-state data for the current session yet.",
            style={"color": "#a0a0b0", "fontSize": "12px"}),
            open_default=False, badge="0")
    channels.sort(key=lambda c: (c["latest"] is None, c["latest"] or 0))
    body = html.Div([
        _zss_header(),
        _impedance_span_note(channels),
        _zss_bar_figure(channels),
        _zss_trend_figure(channels, icfg),
        _zss_artifact_overlay(store, channels),
        _zss_example_block(store, channels, icfg),
    ])
    return _collapsible(title, body, open_default=False,
                        badge=f"{len(channels)} channels")


def _zss_header():
    return html.Div([
        html.Span("Animals currently on the rig · Z_ss = |V_ss(slow plateau)| "
                  "÷ commanded I_slow (kΩ, gain-corrected → comparable between "
                  "animals). ", style={"color": "#888", "fontSize": "11px"}),
        html.Span("Bars = latest per channel (dashed = median, so an outlier "
                  "animal pops); trend = over time. At 20 kHz the slow phase "
                  "is still settling, so Z_ss is a plateau estimate — valid "
                  "for drift + like-for-like comparison.",
                  style={"color": "#888", "fontSize": "11px"}),
    ], style={"padding": "2px 2px 8px"})


def _zss_bar_figure(channels: list[dict]):
    """Horizontal bar of the latest Z_ss per channel + a median guide line."""
    labeled = [(f"{c['animal']} {c['channel']}", c["latest"])
               for c in channels if c["latest"] is not None]
    if not labeled:
        return html.Div()
    labeled.sort(key=lambda x: x[1])
    names = [x[0] for x in labeled]
    vals = [x[1] for x in labeled]
    med = _median(vals)
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color="#5e7ce2"),
        hovertemplate="%{y}: %{x:.3f} kΩ<extra></extra>"))
    if med is not None:
        fig.add_vline(x=med, line=dict(color="#ff9f0a", width=1, dash="dot"),
                      annotation_text="median",
                      annotation=dict(font=dict(size=9, color="#ff9f0a")))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=60 + 28 * len(names), margin=dict(l=12, r=16, t=10, b=30),
        showlegend=False, font=dict(color="#cfd0d6"),
        xaxis=dict(title="latest Z_ss (kΩ)", gridcolor="#2a2a3a",
                    automargin=True),
        yaxis=dict(automargin=True, tickfont=dict(size=9)))
    return dcc.Graph(figure=fig, config=_IMPEDANCE_GRAPH_CONFIG,
                     style={"padding": "4px 2px"})


def _zss_trend_figure(channels: list[dict], icfg: dict):
    """Per-channel Z_ss-vs-date small-multiples (reuses the Rₐ trace helper)."""
    n = len(channels)
    if n == 0:
        return html.Div()
    cols = min(n, 3)
    n_rows = (n + cols - 1) // cols
    titles = [f"{c['animal']} {c['channel']}" for c in channels]
    fig = make_subplots(rows=n_rows, cols=cols, subplot_titles=titles,
                        vertical_spacing=0.16, horizontal_spacing=0.06)
    for i, c in enumerate(channels):
        rr, cc = i // cols + 1, i % cols + 1
        dates = [r["chunk_datetime"] for r in c["rows"]]
        dts = [_parse_chunk_dt(d) for d in dates]
        x = dts if all(d is not None for d in dts) else list(
            range(len(c["rows"])))
        _add_access_r_trace(fig, rr, cc, x, dates,
                            [r.get("slow_ss_kohm") for r in c["rows"]],
                            c["stat"], icfg)
    for ann in fig.layout.annotations[:n]:
        ann.font.size = 11
        ann.font.color = "#f0f0f5"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=120 + 165 * n_rows,
        margin=dict(l=52, r=16, t=42, b=34), showlegend=False,
        font=dict(color="#cfd0d6"))
    fig.update_xaxes(showticklabels=True, tickfont=dict(size=8), nticks=4,
                     tickformat="%b %d", tickangle=0, gridcolor="#2a2a3a",
                     automargin=True)
    fig.update_yaxes(title_text="kΩ", title_font=dict(size=9),
                     tickfont=dict(size=8), nticks=5, gridcolor="#2a2a3a",
                     automargin=True)
    return dcc.Graph(figure=fig, config=_IMPEDANCE_GRAPH_CONFIG,
                     style={"padding": "8px 10px"})


def _stim_health_badge(store, config):
    """Glanceable badge for the umbrella section: how many active (currently-
    recording) channels are drifting in Rₐ or sagging in current fidelity, so
    the user knows whether to open the collapsed 'Electrode & stimulation
    health' section. Reuses one series+active query for both checks."""
    try:
        series = store.impedance_series_by_channel(
            exclude=_excluded_animals(config))
        active = store.active_impedance_channel_keys()
    except Exception:  # noqa: BLE001
        return None, None
    if active:
        series = {k: v for k, v in series.items() if k in active}
    icfg = _impedance_cfg(config)
    n_drift = n_sag = 0
    for _key, rows in series.items():
        st = _drift_stat([r.get("access_r_kohm") for r in rows],
                         icfg["window"], icfg["min_history"])
        if st and abs(st["drift_pct"]) >= icfg["pct"]:
            n_drift += 1
        fst = _drift_stat_abs([r.get("current_fidelity_pct") for r in rows],
                              icfg["window"], icfg["min_history"])
        if fst and fst["median"] is not None and fst["median"] < icfg["sag_pct"]:
            n_sag += 1
    if n_drift or n_sag:
        parts = []
        if n_drift:
            parts.append(f"⚠{n_drift} Rₐ drift")
        if n_sag:
            parts.append(f"⚠{n_sag} current sag")
        return " · ".join(parts), "#EF553B"
    return "all ok", "rgba(48,209,88,0.5)"


def _build_current_fidelity_card(store, config=None):
    """Current-delivery fidelity — the two-edge ratio. Each biphasic pulse
    hands us two ohmic edges with different commanded ΔI; each backs out the
    SAME series resistance, so if the current is delivered at setpoint they
    agree. The normalized difference is impedance-invariant. ~0 => honest
    current; NEGATIVE => the reversal (high-ΔI) edge sags first => commanded
    current not fully delivered (compliance) — the earliest warning, before
    the impedance itself drifts. Collapsed, active channels only."""
    title = "Current delivery fidelity (edge ratio)"
    try:
        series = store.impedance_series_by_channel(
            exclude=_excluded_animals(config))
        active = store.active_impedance_channel_keys()
        latest = store.latest_current_fidelity_per_channel()
    except Exception as e:  # noqa: BLE001
        logger.warning("current-fidelity card failed: %s", e)
        series, active, latest = {}, set(), {}
    if active:
        series = {k: v for k, v in series.items() if k in active}
        latest = {k: v for k, v in latest.items() if k in active}
    icfg = _impedance_cfg(config)
    channels = []
    for (animal, ch), rows in series.items():
        vals = [r.get("current_fidelity_pct") for r in rows]
        if not any(v is not None for v in vals):
            continue
        channels.append({
            "animal": animal, "channel": ch, "rows": rows,
            "stat": _drift_stat_abs(vals, icfg["window"], icfg["min_history"]),
            "latest": latest.get((animal, ch))})
    if not channels:
        return _collapsible(title, html.Div(
            "No two-edge current-fidelity data for the current session yet "
            "(needs both ohmic edges — reversal and offset — of the stim "
            "pulse).", style={"color": "#a0a0b0", "fontSize": "12px"}),
            open_default=False, badge="0")
    # A channel is "sagging" when its rolling-median fidelity is below the
    # (negative) sag threshold. Those float to the top and open the card.
    sag = icfg["sag_pct"]
    for c in channels:
        med = c["stat"]["median"] if c["stat"] else None
        c["sagging"] = bool(med is not None and med < sag)
    channels.sort(key=lambda c: (not c["sagging"],
                                 c["latest"] if c["latest"] is not None else 0))
    n_sag = sum(1 for c in channels if c["sagging"])
    body = html.Div([
        _fidelity_header(sag),
        _impedance_span_note(channels),
        _fidelity_example_block(store, channels, sag),
        _fidelity_bar_figure(channels, sag),
        _fidelity_trend_figure(channels, icfg),
    ])
    badge = (f"{len(channels)} channels · ⚠{n_sag} sagging" if n_sag
             else f"{len(channels)} channels · ok")
    badge_color = "#EF553B" if n_sag else "rgba(48,209,88,0.5)"
    return _collapsible(title, body, open_default=bool(n_sag),
                        badge=badge, badge_color=badge_color)


def _fidelity_header(sag_pct: float):
    return html.Div([
        html.Span("Animals currently on the rig · fidelity = 100·(Rₐ_reversal − "
                  "Rₐ_offset) / mean, from the pulse's two ohmic edges. ",
                  style={"color": "#888", "fontSize": "11px"}),
        html.Span("Both edges back out the SAME series R using their exact "
                  "commanded ΔI, so this is impedance-invariant: ~0 = current "
                  "at setpoint; ", style={"color": "#888", "fontSize": "11px"}),
        html.Span(f"negative = reversal (high-ΔI) edge sagging = current not "
                  f"fully delivered (compliance). Alert below {sag_pct:.0f}%.",
                  style={"color": "#ff9f0a", "fontSize": "11px"}),
    ], style={"padding": "2px 2px 8px"})


def _fidelity_bar_figure(channels: list[dict], sag_pct: float):
    """Horizontal bar of the latest fidelity per channel, zero reference +
    a shaded negative 'sagging' band."""
    labeled = [(f"{c['animal']} {c['channel']}", c["latest"])
               for c in channels if c["latest"] is not None]
    if not labeled:
        return html.Div()
    labeled.sort(key=lambda x: x[1])
    names = [x[0] for x in labeled]
    vals = [x[1] for x in labeled]
    colors = ["#ff453a" if v < sag_pct else "#5e7ce2" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=colors),
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
    lo = min(vals + [sag_pct]) - 3
    fig.add_vrect(x0=lo, x1=0, line_width=0, fillcolor="#ff453a",
                  opacity=0.06)
    fig.add_vline(x=0, line=dict(color="#8a8a99", width=1))
    fig.add_vline(x=sag_pct, line=dict(color="#ff9f0a", width=1, dash="dot"),
                  annotation_text="sag limit",
                  annotation=dict(font=dict(size=9, color="#ff9f0a")))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=60 + 28 * len(names), margin=dict(l=12, r=16, t=10, b=30),
        showlegend=False, font=dict(color="#cfd0d6"),
        xaxis=dict(title="latest fidelity (%)", gridcolor="#2a2a3a",
                   zeroline=False, automargin=True),
        yaxis=dict(automargin=True, tickfont=dict(size=9)))
    return dcc.Graph(figure=fig, config=_IMPEDANCE_GRAPH_CONFIG,
                     style={"padding": "4px 2px"})


def _fidelity_trend_figure(channels: list[dict], icfg: dict):
    """Per-channel fidelity-vs-date small-multiples with a zero reference line
    and a shaded negative sagging band."""
    n = len(channels)
    if n == 0:
        return html.Div()
    cols = min(n, 3)
    n_rows = (n + cols - 1) // cols
    titles = [f"{c['animal']} {c['channel']}" for c in channels]
    fig = make_subplots(rows=n_rows, cols=cols, subplot_titles=titles,
                        vertical_spacing=0.16, horizontal_spacing=0.06)
    sag = icfg["sag_pct"]
    for i, c in enumerate(channels):
        rr, cc = i // cols + 1, i % cols + 1
        dates = [r["chunk_datetime"] for r in c["rows"]]
        dts = [_parse_chunk_dt(d) for d in dates]
        x = dts if all(d is not None for d in dts) else list(
            range(len(c["rows"])))
        yvals = [r.get("current_fidelity_pct") for r in c["rows"]]
        fig.add_trace(go.Scatter(
            x=x, y=yvals, mode="lines+markers", showlegend=False,
            line=dict(color="#5e7ce2", width=1.5), marker=dict(size=3),
            customdata=dates,
            hovertemplate="fidelity: %{y:+.1f}%<br>%{customdata}<extra></extra>"),
            rr, cc)
        fig.add_hline(y=0, line=dict(color="#8a8a99", width=1), row=rr, col=cc)
        fig.add_hline(y=sag, line=dict(color="#ff9f0a", width=1, dash="dot"),
                      row=rr, col=cc, opacity=0.7)
        # Highlight the latest point red when it's below the sag limit.
        if yvals and yvals[-1] is not None and yvals[-1] < sag:
            fig.add_trace(go.Scatter(
                x=[x[-1]], y=[yvals[-1]], mode="markers", showlegend=False,
                marker=dict(size=8, color="#ff453a", symbol="circle-open",
                            line=dict(width=2, color="#ff453a")),
                hovertemplate=(f"SAG {yvals[-1]:+.1f}%<extra></extra>")), rr, cc)
    for ann in fig.layout.annotations[:n]:
        ann.font.size = 11
        ann.font.color = "#f0f0f5"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=120 + 165 * n_rows,
        margin=dict(l=52, r=16, t=42, b=34), showlegend=False,
        font=dict(color="#cfd0d6"))
    fig.update_xaxes(showticklabels=True, tickfont=dict(size=8), nticks=4,
                     tickformat="%b %d", tickangle=0, gridcolor="#2a2a3a",
                     automargin=True)
    fig.update_yaxes(title_text="%", title_font=dict(size=9),
                     tickfont=dict(size=8), nticks=5, gridcolor="#2a2a3a",
                     zeroline=False, automargin=True)
    return dcc.Graph(figure=fig, config=_IMPEDANCE_GRAPH_CONFIG,
                     style={"padding": "8px 10px"})


def _drift_stat_abs(values: list, window: int, min_history: int) -> dict | None:
    """Rolling median of the last *window* points (absolute, not %-of-baseline)
    — for fidelity, which is already a signed percentage and centred on 0, so
    a ratio-to-baseline is meaningless. Returns ``{latest, median}`` or None."""
    vals = [v for v in values if v is not None]
    if len(vals) < max(2, min_history):
        return None
    return {"latest": vals[-1],
            "median": _median(vals[max(0, len(vals) - window):])}


def _build_region_drift_card(store, config=None):
    """Drift by placement region (SR vs SLM …). Per stimulated channel, the CV
    of its access-resistance over its dominant-charge history; grouped by the
    stimulated channel's electrode region. Colony-wide (every channel with
    enough history) so each region has statistical weight."""
    title = "Drift by placement region (SLM vs SR)"
    min_pts = 4
    try:
        series = store.impedance_series_by_channel(
            exclude=_excluded_animals(config))
    except Exception as e:  # noqa: BLE001
        logger.warning("region-drift card failed: %s", e)
        series = {}
    regions: dict = {}
    for (animal, ch), rows in series.items():
        vals = [r.get("access_r_kohm") for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) < min_pts:
            continue
        cv = _coef_of_variation(vals)
        if cv is None:
            continue
        region = _electrode_region(ch)
        regions.setdefault(region, []).append({
            "animal": animal, "channel": ch, "cv_pct": cv * 100.0})
    regions.pop("", None)
    if len(regions) < 1 or not any(regions.values()):
        return _collapsible(title, html.Div(
            "Not enough per-channel history yet to compare regions (need ≥4 "
            "recordings per stimulated channel).",
            style={"color": "#a0a0b0", "fontSize": "12px"}),
            open_default=False, badge="0")
    body = html.Div([
        _region_verdict(regions),
        _region_strip_figure(regions),
    ])
    return _collapsible(title, body, open_default=False,
                        badge=f"{len(regions)} regions")


def _electrode_region(channel: str) -> str:
    """The placement region tag embedded in a stimulated channel name
    (…SLM…, …SR…). Falls back to '' when none is recognisable."""
    up = (channel or "").upper()
    for tag in ("SLM", "SR", "SP", "SO", "SLU"):
        if tag in up:
            return tag
    return ""


def _coef_of_variation(vals: list) -> float | None:
    """Std / mean (population). None when the mean is ~0 or <2 points."""
    xs = [float(v) for v in vals if v is not None]
    if len(xs) < 2:
        return None
    mean = sum(xs) / len(xs)
    if mean == 0:
        return None
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return (var ** 0.5) / abs(mean)


def _region_verdict(regions: dict):
    """One-line comparison of the per-region median CV, worst region first."""
    summ = []
    for region, chans in regions.items():
        cvs = [c["cv_pct"] for c in chans]
        summ.append((region, _median(cvs), len(chans)))
    summ.sort(key=lambda t: (t[1] is None, -(t[1] or 0)))
    parts = [f"{r} median CV ≈{m:.0f}% (n={n})"
             for r, m, n in summ if m is not None]
    verdict = " vs ".join(parts)
    if len(summ) >= 2 and summ[0][1] and summ[-1][1]:
        ratio = summ[0][1] / summ[-1][1]
        verdict += (f" → {summ[0][0]} ~{ratio:.1f}× more drift than "
                    f"{summ[-1][0]}")
    return html.Div([
        html.Div(verdict, style={"color": "#f0f0f5", "fontSize": "12px",
                                 "fontWeight": "600", "marginBottom": "2px"}),
        html.Div("Region = the stimulated channel's electrode placement (where "
                 "we stim + record the artifact). Each point is one channel's "
                 "access-R CV over its dominant-charge history.",
                 style={"color": "#888", "fontSize": "11px"}),
    ], style={"padding": "2px 2px 8px"})


def _region_strip_figure(regions: dict):
    """Per-region strip of per-channel CV (a box + jittered points), sorted by
    median so the higher-drift region is obvious."""
    order = sorted(regions.keys(),
                   key=lambda r: -(_median([c["cv_pct"]
                                            for c in regions[r]]) or 0))
    fig = go.Figure()
    palette = {"SLM": "#ff9f0a", "SR": "#5e7ce2"}
    for region in order:
        chans = regions[region]
        cvs = [c["cv_pct"] for c in chans]
        labels = [f"{c['animal']} {c['channel']}" for c in chans]
        color = palette.get(region, "#8a8a99")
        fig.add_trace(go.Box(
            x=[region] * len(cvs), y=cvs, name=region, boxpoints="all",
            jitter=0.5, pointpos=0, marker=dict(color=color, size=6),
            line=dict(color=color), fillcolor="rgba(0,0,0,0)",
            text=labels,
            hovertemplate="%{text}: CV %{y:.0f}%<extra></extra>"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=48, r=16, t=14, b=30), showlegend=False,
        font=dict(color="#cfd0d6"),
        xaxis=dict(title="placement region", gridcolor="#2a2a3a"),
        yaxis=dict(title="access-R CV (%)", gridcolor="#2a2a3a",
                   rangemode="tozero", automargin=True))
    return dcc.Graph(figure=fig, config=_IMPEDANCE_GRAPH_CONFIG,
                     style={"padding": "4px 2px"})


# --------------------------------------------------------------------- #
#  Electrode-history comparison — overlay any of one animal's electrodes on a
#  common relative axis (recording #, aligned at each electrode's own first
#  recording) so histories over DIFFERENT date ranges still compare.
# --------------------------------------------------------------------- #

_ECMP_COLORS = ["#5e7ce2", "#ff9f0a", "#30d158", "#ff453a", "#bf5af2",
                "#64d2ff", "#ffd60a", "#ff6482"]
_ECMP_LABEL = {"color": "#888", "fontSize": "11px", "display": "block",
               "marginBottom": "2px"}

# The over-time metrics this comparison tool can overlay (field on each series
# row, display label, unit, and whether normalizing to the median is
# meaningful). Current fidelity is a signed % already centred near 0, so a
# ratio-to-median is nonsense → not normalizable (always shown absolute).
_ECMP_METRICS = {
    "access_r_kohm":       {"label": "Access resistance Rₐ", "unit": "kΩ",
                            "normalizable": True},
    "slow_ss_kohm":        {"label": "Slow steady-state Z_ss", "unit": "kΩ",
                            "normalizable": True},
    "current_fidelity_pct": {"label": "Current fidelity", "unit": "%",
                            "normalizable": False},
}


def _impedance_animals_and_electrodes(store, config) -> dict:
    """``{animal: [electrode, ...]}`` for every animal/channel with any
    electrode/stim history. Full colony, not just active — the whole point is
    to compare a current electrode against a retired one."""
    try:
        series = store.impedance_series_by_channel(
            exclude=_excluded_animals(config))
    except Exception:  # noqa: BLE001
        return {}
    fields = tuple(_ECMP_METRICS.keys())
    out: dict = {}
    for (animal, ch), rows in series.items():
        if not any(r.get(f) is not None for r in rows for f in fields):
            continue
        out.setdefault(animal, [])
        if ch not in out[animal]:
            out[animal].append(ch)
    for a in out:
        out[a].sort()
    return out


def _build_electrode_compare_card(store, config=None):
    """Interactive: pick an animal, a metric (Rₐ / Z_ss / current fidelity),
    and any of its electrodes, then overlay their histories on a common
    relative axis (recording #, aligned at each electrode's first recording) so
    electrodes recorded over different date ranges still line up for shape
    comparison. Y toggles absolute / normalized (to each electrode's own
    median) for the kΩ metrics."""
    amap = _impedance_animals_and_electrodes(store, config)
    animals = sorted(amap.keys())
    if not animals:
        return _collapsible(
            "Compare electrode histories",
            html.Div("No electrode history yet.",
                     style={"color": "#a0a0b0", "fontSize": "12px"}),
            open_default=False)
    # Prefer an animal currently on the rig as the default.
    try:
        active = {a for a, _c in store.active_impedance_channel_keys()}
    except Exception:  # noqa: BLE001
        active = set()
    default_animal = next((a for a in animals if a in active), animals[0])
    default_elecs = amap.get(default_animal, [])[:4]
    default_metric = "access_r_kohm"
    controls = html.Div([
        html.Div([
            html.Span("Animal", style=_ECMP_LABEL),
            dcc.Dropdown(
                id="overview-ecmp-animal",
                options=[{"label": a, "value": a} for a in animals],
                value=default_animal, clearable=False,
                style=DROPDOWN_STYLE, className="dark-dropdown"),
        ], style={"minWidth": "140px"}),
        html.Div([
            html.Span("Metric", style=_ECMP_LABEL),
            dcc.Dropdown(
                id="overview-ecmp-metric",
                options=[{"label": f"{m['label']} ({m['unit']})", "value": k}
                         for k, m in _ECMP_METRICS.items()],
                value=default_metric, clearable=False,
                style=DROPDOWN_STYLE, className="dark-dropdown"),
        ], style={"minWidth": "180px"}),
        html.Div([
            html.Span("Electrodes", style=_ECMP_LABEL),
            dcc.Dropdown(
                id="overview-ecmp-electrodes",
                options=[{"label": c, "value": c} for c in default_elecs],
                value=default_elecs, multi=True,
                placeholder="Pick electrodes to overlay",
                style=DROPDOWN_STYLE, className="dark-dropdown"),
        ], style={"flex": "1", "minWidth": "220px"}),
        html.Div([
            html.Span("Y-axis", style=_ECMP_LABEL),
            dcc.RadioItems(
                id="overview-ecmp-ymode",
                options=[{"label": " Normalized (% of median)",
                          "value": "norm"},
                         {"label": " Absolute", "value": "abs"}],
                value="norm", inline=True,
                labelStyle={"color": "#ddd", "fontSize": "12px",
                            "marginRight": "12px"},
                inputStyle={"marginRight": "4px"}),
        ], style={"minWidth": "210px"}),
    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end",
              "gap": "12px", "marginBottom": "8px"})
    body = html.Div([
        html.Div("Overlay any of one animal's electrodes for any stim/electrode "
                 "metric on a common relative axis: x = recording # (each "
                 "electrode aligned at its own first recording, so different "
                 "date ranges still compare). Hover shows the real date. "
                 "Normalized = each electrode scaled to its own median (100%) "
                 "so a low- and a high-value electrode compare by drift shape "
                 "(N/A for current fidelity, which is already a signed %).",
                 style={"color": "#a0a0b0", "fontSize": "11px",
                         "marginBottom": "8px"}),
        controls,
        dcc.Loading(html.Div(
            _electrode_compare_figure(store, config, default_animal,
                                      default_elecs, "norm", default_metric),
            id="overview-ecmp-fig"), type="dot"),
    ])
    return _collapsible("Compare electrode histories", body, open_default=False)


def _electrode_compare_figure(store, config, animal, electrodes, ymode,
                              metric="access_r_kohm"):
    """Overlay the selected electrodes' history for ``metric`` on a recording-
    index x-axis. ``ymode`` 'norm' scales each electrode to its own median
    (=100%) when the metric is normalizable; 'abs' plots raw values."""
    electrodes = electrodes or []
    spec = _ECMP_METRICS.get(metric) or _ECMP_METRICS["access_r_kohm"]
    if not animal or not electrodes:
        return html.Div("Select an animal and at least one electrode.",
                        style={"color": "#888", "fontSize": "12px",
                                "padding": "24px 4px"})
    try:
        series = store.impedance_series_by_channel(
            exclude=_excluded_animals(config))
    except Exception:  # noqa: BLE001
        series = {}
    normalized = (ymode != "abs") and spec["normalizable"]
    unit = spec["unit"]
    fig = go.Figure()
    n_plotted = 0
    for i, ch in enumerate(electrodes):
        rows = series.get((animal, ch)) or []
        pts = [(r.get("chunk_datetime"), r.get(metric))
               for r in rows if r.get(metric) is not None]
        if not pts:
            continue
        dates = [p[0] for p in pts]
        yraw = [float(p[1]) for p in pts]
        color = _ECMP_COLORS[i % len(_ECMP_COLORS)]
        med = _median(yraw) or 0.0
        if normalized and med:
            y = [v / med * 100.0 for v in yraw]
            name = f"{ch} · median {med:.3f} {unit} ({len(y)} recs)"
            hover = ("%{y:.0f}% of median<br>%{customdata[0]} · "
                     "%{customdata[1]:.3f} " + unit + "<extra></extra>")
            cdata = [[d, v] for d, v in zip(dates, yraw)]
        else:
            y = yraw
            name = f"{ch} ({len(y)} recs)"
            hover = ("%{y:.3f} " + unit + "<br>%{customdata}<extra></extra>")
            cdata = dates
        fig.add_trace(go.Scatter(
            x=list(range(len(y))), y=y, mode="lines+markers", name=name,
            line=dict(color=color, width=1.6), marker=dict(size=4),
            customdata=cdata, hovertemplate=hover))
        n_plotted += 1
    if n_plotted == 0:
        return html.Div(f"No {spec['label']} data for the selected electrodes.",
                        style={"color": "#888", "fontSize": "12px",
                                "padding": "24px 4px"})
    if normalized:
        fig.add_hline(y=100, line=dict(color="#8a8a99", width=1, dash="dot"),
                      annotation_text="each electrode's median",
                      annotation=dict(font=dict(size=9, color="#8a8a99")))
    elif metric == "current_fidelity_pct":
        # Zero line = current at setpoint; below it the reversal edge is sagging.
        fig.add_hline(y=0, line=dict(color="#8a8a99", width=1))
    ytitle = (f"% of each electrode's median {spec['label']}" if normalized
              else f"{spec['label']} ({unit})")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=360, margin=dict(l=58, r=16, t=34, b=42),
        font=dict(color="#cfd0d6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left",
                    x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="recording # (aligned at each electrode's first)",
                   gridcolor="#2a2a3a", automargin=True),
        yaxis=dict(title=ytitle, gridcolor="#2a2a3a", automargin=True))
    return dcc.Graph(figure=fig, config=_IMPEDANCE_GRAPH_CONFIG,
                     style={"padding": "4px 2px"})


def _build_impedance_trend_card(store, config=None):
    """Per-channel transfer-impedance trend (both stim phases) over time, for
    the animals currently on the rig, wrapped in a collapsed section with a
    glance-able drift badge (Apple-HIG: one container, collapsed by default)."""
    try:
        series = store.impedance_series_by_channel(
            exclude=_excluded_animals(config))
        active = store.active_impedance_channel_keys()
    except Exception as e:  # noqa: BLE001
        logger.warning("impedance card failed: %s", e)
        series, active = {}, set()
    # Scope to the most-recent stim session's channels (currently recording).
    if active:
        series = {k: v for k, v in series.items() if k in active}
    if not series:
        return _collapsible(
            "Electrode series resistance (Rₐ) drift",
            html.Div("No access-resistance data for the current recording "
                     "session yet. Computed from the ohmic step of the stim "
                     "pulse once amplifier gains (File_Records) are available.",
                     style={"color": "#a0a0b0", "fontSize": "12px"}),
            open_default=False, badge="0")
    icfg = _impedance_cfg(config)
    channels = _impedance_channel_rows(series, icfg)
    n_total = len(channels)
    n_drift = sum(1 for c in channels if c["drifting"])
    shown = channels[:_IMPEDANCE_MAX_CHANNELS]
    body = html.Div([
        _impedance_header(n_total, n_drift, len(shown), icfg),
        _impedance_span_note(shown),
        _impedance_example_block(store, channels, icfg),
        _impedance_figure(shown, icfg),
    ])
    badge = (f"{n_total} active · ⚠{n_drift} drifting" if n_drift
             else f"{n_total} active · ok")
    badge_color = "#EF553B" if n_drift else "rgba(48,209,88,0.5)"
    return _collapsible("Electrode series resistance (Rₐ) drift", body,
                        open_default=bool(n_drift), badge=badge,
                        badge_color=badge_color)


def _impedance_channel_rows(series: dict, icfg: dict) -> list[dict]:
    """One entry per (animal, channel): access-R series + drift stat."""
    out = []
    for (animal, ch), rows in series.items():
        stat = _drift_stat([r["access_r_kohm"] for r in rows],
                           icfg["window"], icfg["min_history"])
        drifting = bool(stat and abs(stat["drift_pct"]) >= icfg["pct"])
        out.append({"animal": animal, "channel": ch, "rows": rows,
                    "stat": stat, "drifting": drifting})
    # Drifting channels float to the top, then by animal/channel.
    out.sort(key=lambda c: (not c["drifting"], c["animal"], c["channel"]))
    return out


def _impedance_header(n_total, n_drift, n_shown, icfg):
    """Compact subhead inside the collapsible body: what the numbers mean +
    counts (the collapsible summary already carries the section title)."""
    line = [
        html.Span("Animals currently on the rig · effective series resistance Rₐ "
                  "(kΩ) from the stim voltage-transient step · ",
                  style={"color": "#888", "fontSize": "11px"}),
        html.Span(f"baseline = rolling median of last {icfg['window']} · ",
                  style={"color": "#888", "fontSize": "11px"}),
        html.Span(f"{n_total} channels · ", style={"color": "#cfd0d6",
                                                    "fontSize": "11px"}),
        html.Span(f"⚠ {n_drift} drifting >{icfg['pct']:.0f}%",
                  style={"color": "#ff9f0a" if n_drift else "#30d158",
                          "fontWeight": "600", "fontSize": "11px"}),
    ]
    if n_shown < n_total:
        line.append(html.Span(f" · showing {n_shown} of {n_total}",
                              style={"color": "#888", "fontSize": "11px"}))
    return html.Div(line, style={"padding": "2px 2px 8px"})


def _impedance_figure(channels: list[dict], icfg: dict):
    """Small-multiples: one subplot per channel, pos+neg traces + baseline
    band. Newest point turns red when its phase has drifted past threshold."""
    n = len(channels)
    if n == 0:
        return html.Div()
    cols = min(n, 3)
    n_rows = (n + cols - 1) // cols
    titles = [f"{c['animal']} {c['channel']}" for c in channels]
    fig = make_subplots(rows=n_rows, cols=cols, subplot_titles=titles,
                        vertical_spacing=0.16, horizontal_spacing=0.06)
    for i, c in enumerate(channels):
        rr, cc = i // cols + 1, i % cols + 1
        dates = [r["chunk_datetime"] for r in c["rows"]]
        # Real recording datetimes on x so the timescale is explicit; fall
        # back to an index only if a channel's stamps won't parse.
        dts = [_parse_chunk_dt(d) for d in dates]
        x = dts if all(d is not None for d in dts) else list(
            range(len(c["rows"])))
        _add_access_r_trace(fig, rr, cc, x, dates,
                            [r["access_r_kohm"] for r in c["rows"]],
                            c["stat"], icfg)
    for ann in fig.layout.annotations[:n]:
        ann.font.size = 11
        ann.font.color = "#f0f0f5"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=120 + 165 * n_rows,
        margin=dict(l=52, r=16, t=42, b=34), showlegend=False,
        font=dict(color="#cfd0d6"))
    # Show real date ticks so the measurement timescale is clear; give the
    # y tick labels + 'kΩ' room so they aren't clipped at the card edge.
    fig.update_xaxes(showticklabels=True, tickfont=dict(size=8),
                     nticks=4, tickformat="%b %d", tickangle=0,
                     gridcolor="#2a2a3a", automargin=True)
    fig.update_yaxes(title_text="kΩ", title_font=dict(size=9),
                     tickfont=dict(size=8), nticks=5,
                     gridcolor="#2a2a3a", automargin=True)
    return dcc.Graph(figure=fig, config=_IMPEDANCE_GRAPH_CONFIG,
                     style={"padding": "8px 10px"})


def _add_access_r_trace(fig, rr, cc, x, dates, yvals, stat, icfg):
    """Add a channel's access-R line + rolling-median baseline band + a
    drift-highlighted latest point."""
    fig.add_trace(go.Scatter(
        x=x, y=yvals, mode="lines+markers", name="Rₐ", showlegend=False,
        line=dict(color="#5e7ce2", width=1.5), marker=dict(size=3),
        customdata=dates,
        hovertemplate="Rₐ: %{y:.3f} kΩ<br>%{customdata}<extra></extra>"),
        rr, cc)
    if stat is None:
        return
    band = stat["baseline"] * icfg["pct"] / 100.0
    fig.add_hline(y=stat["baseline"], line=dict(color="#5e7ce2", width=1,
                  dash="dot"), row=rr, col=cc, opacity=0.5)
    fig.add_hrect(y0=stat["baseline"] - band, y1=stat["baseline"] + band,
                  line_width=0, fillcolor="#5e7ce2", opacity=0.08,
                  row=rr, col=cc)
    if abs(stat["drift_pct"]) >= icfg["pct"]:
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[yvals[-1]], mode="markers", showlegend=False,
            marker=dict(size=8, color="#ff453a", symbol="circle-open",
                        line=dict(width=2, color="#ff453a")),
            hovertemplate=(f"DRIFT {stat['drift_pct']:+.0f}%<br>"
                           f"{stat['latest']:.3f} vs {stat['baseline']:.3f} kΩ"
                           "<extra></extra>")), rr, cc)


_ARTIFACT_LABEL = {"color": "#888", "fontSize": "11px", "display": "block",
                   "marginBottom": "2px"}


def _artifact_default_selection(store, config):
    """(animals, default_animal, default_locs) to seed the stim-artifact overlay
    controls -- newest active animal + its channels. Each figure gets its OWN
    date range (seeded per location), so no global range is returned here."""
    amap = _impedance_animals_and_electrodes(store, config)
    animals = sorted(amap.keys())
    if not animals:
        return [], None, []
    try:
        active = {a for a, _c in store.active_impedance_channel_keys()}
    except Exception:  # noqa: BLE001
        active = set()
    default_animal = next((a for a in animals if a in active), animals[0])
    return animals, default_animal, list(amap.get(default_animal, []))


def _artifact_range_for(store, animal, locs):
    """(start_date, end_date) YYYY-MM-DD for *animal*'s *locs*: the last 30 days
    of the union of their dominant-charge history, or (None, None). Used to seed
    each figure's own picker from that location's own time span."""
    los, his = [], []
    for loc in locs or []:
        try:
            b = store.channel_trace_date_bounds(animal, loc)
        except Exception:  # noqa: BLE001
            b = None
        if b:
            los.append(b[0])
            his.append(b[1])
    if not his:
        return (None, None)
    end = min(max(his)[:10].replace("_", "-"), date.today().isoformat())
    lo = min(los)[:10].replace("_", "-")
    try:
        start = max(lo, (date.fromisoformat(end) - timedelta(days=30)).isoformat())
    except ValueError:
        start = lo
    return (start, end)


def _artifact_overlay_controls(store, config):
    """Animal + stim-location(s) picker for the stim-artifact overlay. The date
    range is PER FIGURE (each location's own picker), because different
    locations are usually recorded over different periods. Mirrors the
    electrode-compare card's animal->electrode cascade."""
    animals, animal0, locs0 = _artifact_default_selection(store, config)
    return html.Div([
        html.Div([
            html.Span("Animal", style=_ARTIFACT_LABEL),
            dcc.Dropdown(
                id="overview-artifact-animal",
                options=[{"label": a, "value": a} for a in animals],
                value=animal0, clearable=False,
                style=DROPDOWN_STYLE, className="dark-dropdown"),
        ], style={"minWidth": "140px"}),
        html.Div([
            html.Span("Stim location(s)", style=_ARTIFACT_LABEL),
            dcc.Dropdown(
                id="overview-artifact-locs",
                options=[{"label": c, "value": c} for c in locs0],
                value=locs0, multi=True,
                placeholder="Pick stim channel(s) to overlay",
                style=DROPDOWN_STYLE, className="dark-dropdown"),
        ], style={"flex": "1", "minWidth": "220px"}),
    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end",
              "gap": "12px", "marginBottom": "8px"})


def _single_location_fig(store, animal, loc, start, end, *, max_traces=200):
    """(figure, subtitle) for ONE stim location's artifact overlay over its own
    date range -- traces oldest→newest, zoomed to the transition window. Ranged
    when both dates are set, else the recent-N fallback."""
    from plotly.colors import sample_colorscale
    ranged = bool(start and end)
    if ranged:
        traces = store.channel_traces_in_range(
            animal, loc, start, end, max_traces=max_traces)
        capped = len(traces) >= max_traces
    else:
        traces = store.recent_channel_traces(animal, loc, 24)
        capped = False
    oldest_first = list(reversed(traces))           # oldest → newest
    m = len(oldest_first)
    lo, hi = -0.3, 1.0
    fig = go.Figure()
    for j, tr in enumerate(oldest_first):
        t, y = tr["time_ms"], tr["mean_trace"]
        xs = [t[k] for k in range(min(len(t), len(y))) if lo <= t[k] <= hi]
        ys = [y[k] for k in range(min(len(t), len(y))) if lo <= t[k] <= hi]
        color = sample_colorscale("Turbo", j / max(1, m - 1))[0]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", showlegend=False,
            line=dict(color=color, width=1), opacity=0.65,
            customdata=[tr["chunk_datetime"]] * len(xs),
            hovertemplate="%{y:.3f}<br>%{customdata}<extra></extra>"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=270, margin=dict(l=52, r=16, t=8, b=34),
        showlegend=False, font=dict(color="#cfd0d6"))
    fig.update_xaxes(title_text="ms from stimulus", title_font=dict(size=9),
                     tickfont=dict(size=8), gridcolor="#2a2a3a", automargin=True)
    fig.update_yaxes(title_text="raw", title_font=dict(size=9),
                     tickfont=dict(size=8), gridcolor="#2a2a3a", automargin=True)
    span = f" · {start} → {end}" if ranged else " · recent 24"
    cap = f" · capped at {max_traces} — narrow the range" if capped else ""
    if m == 0:
        sub = ("No recordings in this range." if ranged
               else "No recordings for this channel.")
    else:
        sub = f"{m} traces · oldest (blue) → newest (red){span}{cap}"
    return fig, sub


def _artifact_location_block(store, animal, loc, start=None, end=None):
    """One figure card: header + its OWN date-range picker + the overlay. The
    picker seeds to this location's own history and redraws only this figure."""
    if start is None and end is None:
        start, end = _artifact_range_for(store, animal, [loc])
    fig, sub = _single_location_fig(store, animal, loc, start, end)
    key = {"animal": animal, "loc": loc}
    return html.Div([
        html.Div([
            html.Span(f"{animal} {loc}",
                      style={"color": "#f0f0f5", "fontWeight": "600",
                             "fontSize": "12px"}),
            dcc.DatePickerRange(
                id={"type": "artifact-loc-range", **key},
                start_date=start, end_date=end,
                display_format="YYYY-MM-DD", className="dark-daterange"),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "flexWrap": "wrap", "gap": "8px",
                  "marginBottom": "4px"}),
        html.Div(sub, id={"type": "artifact-loc-sub", **key},
                 style={"color": "#888", "fontSize": "10px",
                        "marginBottom": "2px"}),
        dcc.Graph(figure=fig, id={"type": "artifact-loc-fig", **key},
                  config=_IMPEDANCE_GRAPH_CONFIG),
    ], style={"padding": "10px 12px", "background": COLOR_SURFACE_1,
              "border": f"1px solid {COLOR_DIVIDER}", "borderRadius": RADIUS_SM})


def _artifact_overlay_body(store, animal, locs):
    """The overlay: one independently-dated figure per selected stim location.
    Falls back to the active channels (recent-N) when no animal is chosen."""
    if not animal:
        try:
            keys = sorted(store.active_impedance_channel_keys())
        except Exception:  # noqa: BLE001
            keys = []
        blocks = [_artifact_location_block(store, a, c) for a, c in keys]
    elif not locs:
        return html.Div("Pick at least one stim location.",
                        style={"color": "#888", "fontSize": "11px",
                               "padding": "6px 2px"})
    else:
        blocks = [_artifact_location_block(store, animal, loc) for loc in locs]
    if not blocks:
        return html.Div("Pick an animal and at least one stim location.",
                        style={"color": "#888", "fontSize": "11px",
                               "padding": "6px 2px"})
    return html.Div(blocks, style={
        "display": "grid", "gap": "10px",
        "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))"})


def _impedance_example_block(store, channels: list[dict], icfg: dict):
    """Optional (collapsed) 'how it's computed' worked example for the newest
    active stimulated channel: annotated trace + step-by-step arithmetic."""
    if not channels:
        return html.Div()
    ex = max(channels, key=lambda c: c["rows"][-1]["chunk_datetime"])
    animal, ch = ex["animal"], ex["channel"]
    try:
        row = store.latest_impedance_row(animal, ch)
    except Exception:  # noqa: BLE001
        row = None
    if not row:
        return html.Div()
    wf = None
    try:
        for w in store.get_evoked_waveform_by_file(row["file_id"]):
            if int(w.get("channel") or -1) == int(row["channel"]):
                wf = w
                break
    except Exception:  # noqa: BLE001
        wf = None
    inner = html.Div([
        _impedance_example_steps(row),
        (_impedance_example_figure(wf, row) if wf else html.Div(
            "Waveform unavailable for this recording.",
            style={"color": "#888", "fontSize": "11px"})),
    ])
    when = (row.get("chunk_datetime") or "")[:16]
    return _collapsible(
        f"How this is computed — example: {animal} {ch} ({when})",
        inner, open_default=False)


def _impedance_example_steps(row: dict):
    """The transparent access-resistance arithmetic for the two transitions,
    using the stored per-transition values."""
    def _f(k):
        v = row.get(k)
        return None if v is None else float(v)
    i_pos, i_neg = _f("i_pos_ua"), _f("i_neg_ua")
    rev, off, ra = (_f("access_r_reversal_kohm"), _f("access_r_offset_kohm"),
                    _f("access_r_kohm"))
    # ΔI is the current STEP at each transition (not a phase amplitude):
    #   reversal flips +I_fast → −I_slow, so the step is the SUM;
    #   off goes −I_slow → 0, so the step is I_slow.
    rev_expr = (f"I_fast+I_slow = {i_pos:.1f}+{i_neg:.1f} = {i_pos + i_neg:.1f}"
                if (i_pos and i_neg) else None)
    off_expr = f"I_slow = {i_neg:.1f}" if i_neg else None

    def _line(label, r, di_expr, color):
        if r is None or di_expr is None:
            return html.Div(f"{label}: n/a", style={"color": "#888",
                            "fontFamily": "monospace", "fontSize": "11px"})
        return html.Div([
            html.Span(f"{label}:  ", style={"color": color,
                                            "fontWeight": "600"}),
            html.Span(f"ΔI = {di_expr} µA   ·   "
                      f"Rₐ = |ΔV| ÷ ΔI = {r:.3f} kΩ"),
        ], style={"fontFamily": "monospace", "fontSize": "11px",
                   "color": "#cfd0d6", "marginBottom": "3px"})
    agree = ("linearity check passed" if ra is not None
             else "rejected (disagree / saturation collapse)")
    return html.Div([
        html.Div("Series resistance Rₐ = |ΔV| ÷ ΔI at a current transition — "
                 "the step is dominated by the ohmic (resistive) drop since "
                 "the double-layer voltage can't jump instantly. At 20 kHz "
                 "(50 µs/sample) it also folds in a little capacitive "
                 "charging, so it's an EFFECTIVE series resistance, not a "
                 "pure Rₐ. Measured at two independent transitions (their "
                 "agreement is a linearity check, not proof of the absolute "
                 "value):",
                 style={"color": "#a0a0b0", "fontSize": "11px",
                         "marginBottom": "5px"}),
        _line("Fast→slow reversal", rev, rev_expr, "#5e7ce2"),
        _line("Slow→off", off, off_expr, "#ff9f0a"),
        html.Div(f"→ Rₐ = mean = {ra:.3f} kΩ  ({agree})" if ra is not None
                 else f"→ Rₐ rejected ({agree})",
                 style={"color": "#f0f0f5", "fontWeight": "600",
                         "fontSize": "11px", "marginTop": "3px",
                         "fontFamily": "monospace"}),
        html.Div("Caveats: at 20 kHz the step is one 50-µs sample, so Rₐ "
                 "folds in ≤50 µs of double-layer charging (effective, not "
                 "absolute — good for DRIFT). Rows where the amp saturated on "
                 "the fast spike and collapsed before the reversal (spurious "
                 "R≈0) are rejected; high-charge pulses saturate, so only the "
                 "dominant test-pulse charge is tracked. The dotted grey line "
                 "is the stim command I(t) — the steps should sit on its "
                 "edges.",
                 style={"color": "#777", "fontSize": "10px",
                         "marginTop": "5px"}),
    ], style={"padding": "4px 2px 8px"})


def _impedance_example_figure(wf: dict, row: dict):
    """Annotated recorded trace near t=0 with the current transitions marked
    and the fast→slow OHMIC STEP (ΔV) highlighted — the Rₐ measurement."""
    from src.utils.impedance import find_current_transitions
    t = wf.get("time_axis_ms") or []
    m = wf.get("mean_trace") or []
    sm = wf.get("stim_mean_trace") or []
    if not t or not m:
        return html.Div()
    pw_ms = (float(row.get("pulse_width_us") or 150.0)) / 1000.0
    ratio = float(row.get("neg_ratio") or 3.0)
    lo, hi = -0.3, max(1.0, pw_ms * (1.0 + ratio) + 0.2)
    xs = [t[i] for i in range(min(len(t), len(m))) if lo <= t[i] <= hi]
    ys = [m[i] for i in range(min(len(t), len(m))) if lo <= t[i] <= hi]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                             line=dict(color="#cfd0d6", width=1.5),
                             name="recorded (raw)"))
    # Overlay the stim command (stim-copy), scaled to the trace, so the reader
    # can verify the marked transitions land on the real current edges.
    smw = [sm[i] for i in range(min(len(t), len(sm))) if lo <= t[i] <= hi]
    if smw and ys:
        span = max((abs(v) for v in ys), default=1.0) or 1.0
        cspan = max((abs(v) for v in smw), default=1.0) or 1.0
        scale = 0.5 * span / cspan
        fig.add_trace(go.Scatter(
            x=xs, y=[v * scale for v in smw], mode="lines",
            line=dict(color="#8a8a99", width=1, dash="dot"),
            name="commanded current I(t) — reference only, scaled (NOT Rₐ)",
            hoverinfo="skip"))
    tr = find_current_transitions(sm, t)
    if tr is not None:
        r_rev = row.get("access_r_reversal_kohm")
        r_off = row.get("access_r_offset_kohm")
        _mark_ohmic_step(fig, t, m, tr, r_rev, r_off)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(l=40, r=10, t=16, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                     xanchor="left", x=0, font=dict(size=9),
                     bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#cfd0d6"),
        xaxis=dict(title="ms from stimulus", gridcolor="#2a2a3a"),
        yaxis=dict(title="raw", gridcolor="#2a2a3a"))
    return dcc.Graph(figure=fig, config=_IMPEDANCE_GRAPH_CONFIG,
                     style={"padding": "4px 2px"})


def _mark_ohmic_step(fig, t, m, tr, r_rev=None, r_off=None):
    """Mark the reversal (blue) and offset (orange) ohmic steps: a bold
    segment across each transition (pre→post sample) + a dotted guide line at
    the transition time, labelled with that step's Rₐ so both are obvious."""
    _onset, reversal, offset = tr
    for idx, color, base, rval in (
            (reversal, "#5e7ce2", "reversal", r_rev),
            (offset, "#ff9f0a", "off", r_off)):
        if idx is None or idx < 1 or idx >= len(m):
            continue
        tag = (f"{base} → Rₐ={rval:.3f} kΩ" if rval is not None
               else f"{base} ΔV")
        fig.add_vline(x=t[idx], line=dict(color=color, width=1, dash="dot"),
                      opacity=0.4)
        fig.add_trace(go.Scatter(
            x=[t[idx - 1], t[idx]], y=[m[idx - 1], m[idx]],
            mode="lines+markers", name=tag,
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color),
            hovertemplate=f"{base} ΔV: %{{y:.3f}}<extra></extra>"))


def _zss_example_block(store, channels: list[dict], icfg: dict):
    """Optional (collapsed) 'how Z_ss is measured' worked example for the newest
    active stimulated channel: the plateau arithmetic + an annotated trace with
    the steady-state window shaded."""
    if not channels:
        return html.Div()
    # Pick the channel with the LARGEST latest Z_ss (the clearest, least-
    # saturated plateau) so the worked example is legible even when the current
    # rig protocol saturates some channels.
    ex = max(channels, key=lambda c: (c.get("latest") or 0))
    animal, ch = ex["animal"], ex["channel"]
    try:
        row = store.latest_impedance_row(animal, ch)
    except Exception:  # noqa: BLE001
        row = None
    if not row:
        return html.Div()
    wf = None
    try:
        for w in store.get_evoked_waveform_by_file(row["file_id"]):
            if int(w.get("channel") or -1) == int(row["channel"]):
                wf = w
                break
    except Exception:  # noqa: BLE001
        wf = None
    inner = html.Div([
        _zss_example_steps(row),
        (_zss_example_figure(wf, row) if wf else html.Div(
            "Waveform unavailable for this recording.",
            style={"color": "#888", "fontSize": "11px"})),
    ])
    when = (row.get("chunk_datetime") or "")[:16]
    return _collapsible(
        f"How Z_ss is measured — example: {animal} {ch} ({when})",
        inner, open_default=False)


def _zss_example_steps(row: dict):
    """The transparent slow-phase steady-state arithmetic: settled plateau
    voltage ÷ commanded slow current, gain-corrected."""
    from src.utils.impedance import SLOW_SS_LO, SLOW_SS_HI

    def _f(k):
        v = row.get(k)
        return None if v is None else float(v)
    vraw, gain, i_slow, zss = (_f("slow_ss_raw"), _f("gain"),
                               _f("i_neg_ua"), _f("slow_ss_kohm"))

    def _mono(txt, color="#cfd0d6"):
        return html.Div(txt, style={"fontFamily": "monospace",
                                    "fontSize": "11px", "color": color,
                                    "marginBottom": "3px"})
    steps = [html.Div(
        "Slow-phase steady-state impedance Z_ss = |V_ss| ÷ I_slow. V_ss is the "
        "SETTLED plateau of the long slow (recharge) phase — the mean of the "
        f"recorded trace over {SLOW_SS_LO * 100:.0f}–{SLOW_SS_HI * 100:.0f}% of "
        "the slow phase, i.e. AFTER the reversal transient has died out (shaded "
        "green below) and before the offset transient. Gain-corrected so it is "
        "comparable between animals — a stim-consistency signal complementary "
        "to Rₐ (Rₐ is the instantaneous ohmic STEP; Z_ss is the settled "
        "PLATEAU):",
        style={"color": "#a0a0b0", "fontSize": "11px", "marginBottom": "5px"})]
    if vraw is not None and gain and i_slow:
        vmv = abs(vraw) / gain * 1000.0
        steps += [
            _mono(f"V_ss (plateau mean) = {vraw:.4f} raw ÷ gain {gain:.0f} × "
                  f"1000 = {vmv:.3f} mV", "#2ca089"),
            _mono(f"I_slow (commanded) = {i_slow:.1f} µA", "#ff9f0a"),
            html.Div(f"→ Z_ss = |V_ss| ÷ I_slow = {vmv:.3f} ÷ {i_slow:.1f} = "
                     f"{zss:.3f} kΩ" if zss is not None else "→ Z_ss = n/a",
                     style={"color": "#f0f0f5", "fontWeight": "600",
                            "fontSize": "11px", "marginTop": "3px",
                            "fontFamily": "monospace"})]
    else:
        steps.append(_mono(f"Z_ss = {zss:.3f} kΩ" if zss is not None
                           else "Z_ss = n/a (inputs missing)", "#f0f0f5"))
    steps.append(html.Div(
        "Caveats: at 20 kHz the slow phase is still decaying, so this is a "
        "settled-PLATEAU estimate — window-dependent absolute value, valid for "
        "drift + like-for-like comparison. High-charge pulses saturate the "
        "plateau (Z_ss ≪ Rₐ) and are excluded; only the dominant test-pulse "
        "charge is tracked. The dotted grey line is the stim command I(t).",
        style={"color": "#777", "fontSize": "10px", "marginTop": "5px"}))
    return html.Div(steps, style={"padding": "4px 2px 8px"})


def _zss_example_figure(wf: dict, row: dict):
    """Annotated recorded trace with the slow-phase STEADY-STATE plateau window
    shaded (the settled region after the reversal transient dies out) and V_ss
    (its mean) drawn — the Z_ss measurement."""
    from src.utils.impedance import (find_current_transitions, SLOW_SS_LO,
                                     SLOW_SS_HI)
    t = wf.get("time_axis_ms") or []
    m = wf.get("mean_trace") or []
    sm = wf.get("stim_mean_trace") or []
    if not t or not m:
        return html.Div()
    pw_ms = (float(row.get("pulse_width_us") or 150.0)) / 1000.0
    ratio = float(row.get("neg_ratio") or 3.0)
    lo, hi = -0.3, max(1.0, pw_ms * (1.0 + ratio) + 0.2)
    n = min(len(t), len(m))
    xs = [t[i] for i in range(n) if lo <= t[i] <= hi]
    ys = [m[i] for i in range(n) if lo <= t[i] <= hi]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                             line=dict(color="#cfd0d6", width=1.5),
                             name="recorded (raw)"))
    smw = [sm[i] for i in range(min(len(t), len(sm))) if lo <= t[i] <= hi]
    if smw and ys:
        span = max((abs(v) for v in ys), default=1.0) or 1.0
        cspan = max((abs(v) for v in smw), default=1.0) or 1.0
        scale = 0.5 * span / cspan
        fig.add_trace(go.Scatter(
            x=xs, y=[v * scale for v in smw], mode="lines",
            line=dict(color="#8a8a99", width=1, dash="dot"),
            name="commanded current I(t) — reference, scaled", hoverinfo="skip"))
    tr = find_current_transitions(sm, t)
    if tr is not None:
        _onset, reversal, offset = tr
        span_s = (offset - reversal) if (reversal is not None
                                         and offset is not None) else 0
        if span_s > 1:
            a = reversal + int(span_s * SLOW_SS_LO)
            b = max(a + 1, reversal + int(span_s * SLOW_SS_HI))
            b = min(b, len(t) - 1, len(m) - 1)
            if 0 <= a < b:
                fig.add_vrect(x0=t[a], x1=t[b], line_width=0,
                              fillcolor="rgba(46,160,137,0.18)",
                              annotation_text="steady-state plateau "
                              "(transients died out)",
                              annotation_position="top left",
                              annotation_font_size=9)
                vss = row.get("slow_ss_raw")
                if vss is None:
                    win = [m[i] for i in range(a, b + 1)]
                    vss = sum(win) / len(win) if win else None
                if vss is not None:
                    fig.add_trace(go.Scatter(
                        x=[t[a], t[b]], y=[vss, vss], mode="lines",
                        line=dict(color="#2ca089", width=3),
                        name=f"V_ss = {vss:.4f} raw (plateau mean)"))
            for idx, color, base in ((reversal, "#5e7ce2", "reversal transient"),
                                     (offset, "#ff9f0a", "offset transient")):
                if idx is not None and 0 <= idx < len(t):
                    fig.add_vline(x=t[idx], line=dict(color=color, width=1,
                                  dash="dot"), opacity=0.35)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(l=40, r=10, t=16, b=30), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left",
                    x=0, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#cfd0d6"),
        xaxis=dict(title="ms from stimulus", gridcolor="#2a2a3a"),
        yaxis=dict(title="raw", gridcolor="#2a2a3a"))
    return dcc.Graph(figure=fig, config=_IMPEDANCE_GRAPH_CONFIG,
                     style={"padding": "4px 2px"})


def _zss_artifact_overlay(store, channels: list[dict]):
    """Recent stim-artifact traces for the clearest Z_ss channel, overlaid
    old→new with the Z_ss plateau window shaded -- so a real plateau shift (a
    genuine Z_ss change) vs saturation noise is visible right behind the trend.
    Collapsed; loads a handful of traces."""
    if not channels:
        return html.Div()
    from plotly.colors import sample_colorscale
    from src.utils.impedance import (find_current_transitions, SLOW_SS_LO,
                                     SLOW_SS_HI)
    ex = max(channels, key=lambda c: (c.get("latest") or 0))
    animal, ch = ex["animal"], ex["channel"]
    try:
        traces = store.recent_channel_traces(animal, ch, limit=8)
    except Exception:  # noqa: BLE001
        traces = []
    if len(traces) < 2:
        return html.Div()
    # Plateau window from the newest recording's transitions (one waveform).
    plateau = None
    try:
        row = store.latest_impedance_row(animal, ch)
        wf = None
        if row:
            for w in store.get_evoked_waveform_by_file(row["file_id"]):
                if (w.get("channel_name") or "") == ch:
                    wf = w
                    break
        if wf:
            tm = wf.get("time_axis_ms") or []
            tr = find_current_transitions(wf.get("stim_mean_trace") or [], tm)
            if tr is not None:
                _o, rev, off = tr
                span = (off - rev) if (rev is not None
                                       and off is not None) else 0
                if span > 1:
                    a = rev + int(span * SLOW_SS_LO)
                    b = min(rev + int(span * SLOW_SS_HI), len(tm) - 1)
                    if 0 <= a < b:
                        plateau = (tm[a], tm[b])
    except Exception:  # noqa: BLE001
        plateau = None
    traces = list(reversed(traces))               # old -> new for the colour ramp
    n = len(traces)
    colors = sample_colorscale("Viridis",
                               [i / max(1, n - 1) for i in range(n)])
    lo, hi = -0.3, 0.8
    fig = go.Figure()
    for i, t in enumerate(traces):
        tm = t.get("time_ms") or []
        mt = t.get("mean_trace") or []
        k = min(len(tm), len(mt))
        xs = [tm[j] for j in range(k) if lo <= tm[j] <= hi]
        ys = [mt[j] for j in range(k) if lo <= tm[j] <= hi]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                                 line=dict(color=colors[i], width=1),
                                 opacity=0.85, hoverinfo="skip"))
    if plateau:
        fig.add_vrect(x0=plateau[0], x1=plateau[1], line_width=0,
                      fillcolor="rgba(46,160,137,0.15)",
                      annotation_text="Z_ss plateau", annotation_font_size=9,
                      annotation_position="top left")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(l=40, r=10, t=16, b=30), showlegend=False,
        font=dict(color="#cfd0d6"),
        xaxis=dict(title="ms from stimulus", gridcolor="#2a2a3a"),
        yaxis=dict(title="raw", gridcolor="#2a2a3a"))
    body = html.Div([
        html.Div(f"{animal} {ch} — last {n} recordings (dark = older → bright = "
                 "newer). A real Z_ss change shows as a SHIFTING plateau in the "
                 "green window; saturation shows as a plateau crushed to ~0 "
                 "while the fast/reversal edges clip.",
                 style={"color": "#a0a0b0", "fontSize": "11px",
                        "marginBottom": "4px"}),
        dcc.Graph(figure=fig, config=_IMPEDANCE_GRAPH_CONFIG,
                  style={"padding": "4px 2px"}),
    ])
    return _collapsible("Stim artifacts behind the Z_ss trend", body,
                        open_default=False)


def _fidelity_example_block(store, channels: list[dict], sag_pct: float):
    """Optional (collapsed) 'how it's computed' worked example for the newest
    active channel: the two-edge arithmetic that yields the fidelity %, plus
    the same annotated trace so the reader can see the two ohmic edges the
    ratio is built from."""
    if not channels:
        return html.Div()
    ex = max(channels, key=lambda c: c["rows"][-1]["chunk_datetime"])
    animal, ch = ex["animal"], ex["channel"]
    try:
        row = store.latest_impedance_row(animal, ch)
    except Exception:  # noqa: BLE001
        row = None
    if not row:
        return html.Div()
    wf = None
    try:
        for w in store.get_evoked_waveform_by_file(row["file_id"]):
            if int(w.get("channel") or -1) == int(row["channel"]):
                wf = w
                break
    except Exception:  # noqa: BLE001
        wf = None
    inner = html.Div([
        _fidelity_example_steps(row, sag_pct),
        (_impedance_example_figure(wf, row) if wf else html.Div(
            "Waveform unavailable for this recording.",
            style={"color": "#888", "fontSize": "11px"})),
    ])
    when = (row.get("chunk_datetime") or "")[:16]
    return _collapsible(
        f"How this is computed — example: {animal} {ch} ({when})",
        inner, open_default=False)


def _fidelity_example_steps(row: dict, sag_pct: float):
    """The transparent two-edge fidelity arithmetic using the stored per-edge
    series-resistance values."""
    def _f(k):
        v = row.get(k)
        return None if v is None else float(v)
    rev, off = _f("access_r_reversal_kohm"), _f("access_r_offset_kohm")
    fid = _current_fidelity_pct(rev, off)

    def _line(label, r, color):
        txt = "n/a" if r is None else f"{r:.3f} kΩ"
        return html.Div([
            html.Span(f"{label}:  ", style={"color": color,
                                            "fontWeight": "600"}),
            html.Span(f"Rₐ = |ΔV| ÷ ΔI(commanded) = {txt}"),
        ], style={"fontFamily": "monospace", "fontSize": "11px",
                   "color": "#cfd0d6", "marginBottom": "3px"})

    if fid is None:
        verdict = "one edge missing — fidelity not computed for this recording"
        vcolor = "#888"
    elif fid < sag_pct:
        verdict = (f"{fid:+.1f}% < {sag_pct:.0f}% → reversal edge sagging = "
                   "commanded current NOT fully delivered (compliance)")
        vcolor = "#ff453a"
    else:
        verdict = f"{fid:+.1f}% → edges agree, current at setpoint"
        vcolor = "#30d158"
    ratio_expr = (f"100 · ({rev:.3f} − {off:.3f}) ÷ mean({rev:.3f}, {off:.3f})"
                  f" = {fid:+.1f}%" if fid is not None else "n/a")
    return html.Div([
        html.Div("The biphasic pulse hands us TWO ohmic edges with different "
                 "commanded ΔI (the fast→slow reversal steps by I_fast+I_slow; "
                 "the slow→off step by I_slow). Each edge backs out the SAME "
                 "series resistance — its voltage step ÷ its OWN commanded ΔI — "
                 "so if the current is delivered at setpoint the two agree:",
                 style={"color": "#a0a0b0", "fontSize": "11px",
                         "marginBottom": "5px"}),
        _line("Fast→slow reversal edge (high ΔI)", rev, "#5e7ce2"),
        _line("Slow→off edge (low ΔI)", off, "#ff9f0a"),
        html.Div(f"fidelity = 100·(Rₐ_reversal − Rₐ_offset) ÷ mean = "
                 f"{ratio_expr}",
                 style={"color": "#cfd0d6", "fontFamily": "monospace",
                         "fontSize": "11px", "marginTop": "4px"}),
        html.Div(f"→ {verdict}",
                 style={"color": vcolor, "fontWeight": "600",
                         "fontSize": "11px", "marginTop": "3px",
                         "fontFamily": "monospace"}),
        html.Div("Why it's impedance-invariant: both edges scale with the same "
                 "R_s, so the normalized difference cancels R_s entirely — the "
                 "metric moves only when the current itself is off, decoupled "
                 "from electrode-impedance drift. Negative means the HIGH-ΔI "
                 "(reversal) edge sags first, the earliest sign of compliance "
                 "before Rₐ itself visibly drifts. The two edges are the same "
                 "ones marked (blue / orange) on the trace below.",
                 style={"color": "#777", "fontSize": "10px",
                         "marginTop": "5px"}),
    ], style={"padding": "4px 2px 8px"})


def layout(store: Store, config: dict | None = None):
    """Public tab entrypoint (mirrors the other tab modules)."""
    return _overview_tab(store, config)


# ===================================================================== #
#  Callbacks
# ===================================================================== #

def start_overview_warmer(store: Store, config: dict) -> None:
    """Bind snapshot persistence, restore each expensive Overview card from
    SQLite, then rebuild them all in a daemon thread so the FIRST user after a
    daemon restart never pays the cold build.

    Restore runs SYNCHRONOUSLY (three small SQLite reads + unpickles) so even a
    request that lands before the warm thread finishes still serves the last-good
    card. The warm thread then revalidates every cache (and kicks the recent-
    recordings overlay) in the background. Idempotent enough to call once at boot.
    Guarded end-to-end: a warm failure only means the old cold-start behavior."""
    _bind_snapshot_store(store, config)
    for cache in (_BSZ_CACHE, _HOME_GRID_CACHE, _KM_LOG_CACHE):
        try:
            cache.restore()
        except Exception as e:  # noqa: BLE001
            logger.debug("overview snapshot restore failed: %s", e)
    ov = (config or {}).get("overview", {}) or {}
    if not ov.get("warm_on_boot", True):
        return

    def _warm():
        from datetime import date as _date
        builds = (
            (_HOME_GRID_CACHE,
             lambda: _build_home_grid_children(store, config, _date.today()),
             "warm:home-grid"),
            (_KM_LOG_CACHE, lambda: _build_km_log_section(config),
             "warm:km-log"),
            (_BSZ_CACHE,
             lambda: _build_behavioral_seizure_status_card(store, config),
             "warm:bsz-status"),
        )
        for cache, build, label in builds:
            try:
                cache.get(build, label=label)
            except Exception as e:  # noqa: BLE001
                logger.warning("overview warm failed (%s): %s", label, e)
        # NOTE: intentionally NOT pre-warming the hist24 "recent recordings"
        # overlay here -- it h5py-reads ~24 .mat files off the SMB share, which
        # holds this process's GIL and stalled the Overview cards on boot (and it
        # was returning 0 channels anyway). It builds on demand when the user
        # selects "Recent recordings".
        logger.info("overview caches warmed")

    threading.Thread(target=_warm, daemon=True, name="overview-warmer").start()


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire the Overview tab's six callbacks (fine-grained refresh,
    thumbnail, snapshot size + src, home-grid Open buttons)."""

    # Bind snapshot persistence + warm/restore the expensive caches at boot so
    # the first open after a restart paints the last-good cards instantly.
    start_overview_warmer(store, config)

    # "Retry all" on the failed-MATLAB card: flip matlab_error rows back to
    # pending so the daemon reprocesses them (after the cause is fixed).
    @app.callback(
        Output("overview-retry-matlab-status", "children"),
        Input("overview-retry-matlab", "n_clicks"),
        prevent_initial_call=True,
    )
    def _retry_matlab(n):
        if not n:
            return no_update
        try:
            n_requeued = store.requeue_failed_files()
        except Exception as e:  # noqa: BLE001
            return f"Retry failed: {e}"
        return (f"Re-queued {n_requeued} recording(s) for reprocessing — "
                "they'll clear as the daemon works through them.")

    # Behavioral-seizure status: swap the body between the per-animal
    # cards and the per-animal donut charts on the Cards/Charts toggle.
    @app.callback(
        Output("overview-bsz-body", "children"),
        Input("bsz-view-toggle", "value"),
        prevent_initial_call=True,
    )
    def _bsz_view(view):
        if view == "calendar":
            # Its own data source (per-animal per-day scored seizures), NOT the
            # 7-day status rows -- so the whole seizure history shows.
            return _bsz_calendar(store.seizure_days_per_animal(
                exclude=_excluded_animals(config)))
        rows = store.behavioral_seizure_status_per_animal(
            days=7, exclude=_excluded_animals(config))
        if not rows:
            return no_update
        if view == "charts":
            return _bsz_charts(rows)
        return _bsz_cards(rows, store.flag_email_subscribed_animals())

    # Per-animal flagged-event email opt-in (the checkbox on each seizure-status
    # card). MATCH so each card's checkbox is independent. On enable we seed the
    # dedup baseline with the CURRENT flag pool, so turning it on does not email
    # the whole standing backlog -- only files flagged afterwards.
    @app.callback(
        Output({"type": "bsz-flag-email-status", "animal": MATCH}, "children"),
        Input({"type": "bsz-flag-email", "animal": MATCH}, "value"),
        State({"type": "bsz-flag-email", "animal": MATCH}, "id"),
        prevent_initial_call=True,
    )
    def _toggle_flag_email(value, cid):
        animal = (cid or {}).get("animal")
        if not animal:
            return no_update
        enabled = bool(value) and "on" in value
        store.set_flag_email_subscription(animal, enabled)
        if enabled:
            store.seed_flag_email_baseline(animal)
            return "✓ on — emails the alert list on new flags"
        return ""

    # Fine-grained refresh: replace just the volatile Overview cards +
    # queue children instead of re-rendering the whole tab. Eliminates
    # the white flash that used to happen on every refresh tick. The
    # callback is a no-op when the user's on any other tab (the target
    # divs don't exist in the rendered tree).
    # FAST DB-backed cards only. The Sheets-backed cards (home grid, KM log) are
    # deliberately SPLIT into their own callbacks below: those read Google Sheets
    # which, on a flaky/VPN network, fail-fast at the socket timeout (~10-20s per
    # sheet) -- bundling them here made the SLOW Sheets cards block the FAST DB
    # cards (status pills + review queue showed the placeholder for as long as
    # the unreachable sheets took). Now the DB cards fill in seconds regardless.
    @app.callback(
        Output("overview-cards", "children"),
        Output("overview-queue", "children"),
        Input("refresh-trigger", "data"),
        Input("overview-mount", "n_intervals"),
        State("tabs", "value"),
        prevent_initial_call=True,
    )
    def refresh_overview_dynamic(_n, _mount, current_tab):
        # HIGHEST-PRIORITY cards: status pills + review queue only. The 4 heavy
        # stim cards were split into _fill_stim_cards (own lock) so the pills no
        # longer wait behind the slow impedance build.
        nout = (no_update, no_update)
        if current_tab != "overview":
            return nout
        if not _REFRESH_LOCK.try_begin():
            _REFRESH_LOCK.warn_if_stalled()
            return nout
        try:
            with _perf.Timer("cb:overview-auto-refresh"):
                with _perf.Timer("overview:cards"):
                    # overview-cards is itself the flex/wrap ROW, so return the
                    # pills as a FLAT list straight into it -- an extra html.Div
                    # wrapper would be a single block child and stack the pills
                    # vertically. The freshness stamp sits at the RIGHT END of the
                    # SAME row (marginLeft:auto pushes it there; alignSelf centers
                    # it against the pills). (Flat list, not a nested [list,
                    # footer]: nesting a list in children triggers React #31 in
                    # Dash 4.1.0 -> blank strip.)
                    pills = [*_build_overview_cards(store),
                             html.Div(_freshness_footer(time.time()),
                                      style={"marginLeft": "auto",
                                             "alignSelf": "center"})]
                with _perf.Timer("overview:queue"):
                    queue = _build_overview_queue(store)
                return (pills, queue)
        finally:
            _REFRESH_LOCK.end()

    # Stim/electrode cards (impedance / Zss / current-fidelity / region drift):
    # OWN callback + lock. They live in the collapsed "Electrode & stimulation
    # health" umbrella and are the heaviest Overview builds; bundling them with
    # the pills made the pills wait for them. Now they fill independently and
    # never block the high-priority pills/queue.
    @app.callback(
        Output("overview-impedance", "children"),
        Output("overview-zss", "children"),
        Output("overview-current-fidelity", "children"),
        Output("overview-region-drift", "children"),
        Input("overview-slow-refresh", "n_intervals"),  # 90s, not the 10s trigger
        Input("overview-mount-stim", "n_intervals"),
        State("tabs", "value"),
        prevent_initial_call=True,
    )
    def _fill_stim_cards(_n, _mount, current_tab):
        nout = (no_update,) * 4
        if current_tab != "overview":
            return nout
        if not _STIM_LOCK.try_begin():
            _STIM_LOCK.warn_if_stalled()
            return nout
        try:
            with _perf.Timer("cb:overview-stim-cards"):
                with _perf.Timer("overview:impedance"):
                    imp = _build_impedance_trend_card(store, config)
                with _perf.Timer("overview:zss"):
                    zss = _build_zss_consistency_card(store, config)
                with _perf.Timer("overview:fidelity"):
                    fid = _build_current_fidelity_card(store, config)
                with _perf.Timer("overview:region"):
                    region = _build_region_drift_card(store, config)
                return (imp, zss, fid, region)
        finally:
            _STIM_LOCK.end()

    # Home grid (Google Sheets) -- own callback + 120s single-flight cache so a
    # slow/unreachable Sheets read never blocks the DB cards above.
    @app.callback(
        Output("overview-home-grid", "children"),
        Input("overview-slow-refresh", "n_intervals"),  # 90s, not the 10s trigger
        Input("overview-mount-mid", "n_intervals"),
        State("tabs", "value"),
        prevent_initial_call=True,
    )
    def _fill_home_grid(_slow, _mount, current_tab):
        if current_tab != "overview":
            return no_update
        from datetime import date as _date
        return _guarded_card(
            _HOME_GRID_CACHE,
            lambda: _build_home_grid_children(store, config, _date.today()),
            label="cb:overview-home-grid", title="Lab home grid")

    # KM-log summary (Google Sheets) -- own callback + 120s cache, same reason.
    @app.callback(
        Output("overview-km-log", "children"),
        Input("overview-slow-refresh", "n_intervals"),  # 90s, not the 10s trigger
        Input("overview-mount-mid", "n_intervals"),
        State("tabs", "value"),
        prevent_initial_call=True,
    )
    def _fill_km_log(_slow, _mount, current_tab):
        if current_tab != "overview":
            return no_update
        return _guarded_card(
            _KM_LOG_CACHE,
            lambda: _build_km_log_section(config),
            label="cb:overview-km-log", title="KM-recorder log")

    # Behavioral-seizure status card: its own fill callback so it loads in
    # PARALLEL with the 8-card refresh above (own thread) instead of adding to
    # that callback's sequential build. Mount + refresh-trigger, tab-gated.
    @app.callback(
        Output("overview-bsz-status", "children"),
        Input("overview-slow-refresh", "n_intervals"),  # 90s, not the 10s trigger
        Input("overview-mount-bsz", "n_intervals"),
        State("tabs", "value"),
        prevent_initial_call=True,
    )
    def _fill_bsz_status(_slow, _mount, current_tab):
        if current_tab != "overview":
            return no_update
        # Single-flight + 45s cache (see _BSZ_CACHE): stops the every-10s pile-up
        # that turned a ~2s build into a ~36 min stall. _guarded_card adds a
        # freshness stamp + a "stalled" state if it ever wedges again.
        return _guarded_card(
            _BSZ_CACHE,
            lambda: _build_behavioral_seizure_status_card(store, config),
            label="cb:overview-bsz-status", title="Behavioral-seizure status")

    # Lazy stim-artifact overlay: only loads (heavy trace reads) when the
    # user clicks, and lives outside the refreshed card so it persists.
    @app.callback(
        Output("overview-artifact-body", "children"),
        Input("overview-artifact-btn", "n_clicks"),
        State("overview-artifact-animal", "value"),
        State("overview-artifact-locs", "value"),
        prevent_initial_call=True,
    )
    def load_artifact_overlay(_n, animal, locs):
        try:
            return _artifact_overlay_body(store, animal, locs)
        except Exception as e:  # noqa: BLE001
            logger.warning("artifact overlay failed: %s", e)
            return html.Div("Overlay failed to load.",
                            style={"color": "#888", "fontSize": "11px"})

    # Animal -> stim-location options. Mirrors ecmp_electrode_options; no
    # prevent_initial_call so the list always matches the current animal. Each
    # figure seeds its own date range at render, so nothing date-related here.
    @app.callback(
        Output("overview-artifact-locs", "options"),
        Output("overview-artifact-locs", "value"),
        Input("overview-artifact-animal", "value"),
    )
    def artifact_location_options(animal):
        try:
            locs = _impedance_animals_and_electrodes(
                store, config).get(animal, [])
        except Exception as e:  # noqa: BLE001 -- never blank the control silently
            logger.warning("artifact location options failed: %s", e)
            locs = []
        return [{"label": c, "value": c} for c in locs], locs

    # Per-figure date range: changing one location's picker redraws ONLY that
    # figure (+ its subtitle). animal + loc are carried in the pattern id, so no
    # cross-figure state is needed.
    @app.callback(
        Output({"type": "artifact-loc-fig", "animal": MATCH, "loc": MATCH},
               "figure"),
        Output({"type": "artifact-loc-sub", "animal": MATCH, "loc": MATCH},
               "children"),
        Input({"type": "artifact-loc-range", "animal": MATCH, "loc": MATCH},
              "start_date"),
        Input({"type": "artifact-loc-range", "animal": MATCH, "loc": MATCH},
              "end_date"),
        State({"type": "artifact-loc-range", "animal": MATCH, "loc": MATCH},
              "id"),
        prevent_initial_call=True,
    )
    def redraw_artifact_location(start, end, cid):
        try:
            fig, sub = _single_location_fig(
                store, cid["animal"], cid["loc"], start, end)
            return fig, sub
        except Exception as e:  # noqa: BLE001
            logger.warning("artifact per-figure redraw failed: %s", e)
            return no_update, "Redraw failed."

    # Electrode-history comparison: picking an animal repopulates its electrode
    # list (default = up to 4 of them); the overlay figure redraws on any of
    # (electrodes, y-mode, metric). No prevent_initial_call here (mirrors the
    # working chronic animal->session pattern) so the list is always populated
    # for the current animal even on first interaction.
    @app.callback(
        Output("overview-ecmp-electrodes", "options"),
        Output("overview-ecmp-electrodes", "value"),
        Input("overview-ecmp-animal", "value"),
    )
    def ecmp_electrode_options(animal):
        try:
            elecs = _impedance_animals_and_electrodes(
                store, config).get(animal, [])
        except Exception as e:  # noqa: BLE001 -- never blank the control silently
            logger.warning("ecmp electrode options failed: %s", e)
            elecs = []
        return [{"label": c, "value": c} for c in elecs], elecs[:4]

    @app.callback(
        Output("overview-ecmp-fig", "children"),
        Input("overview-ecmp-electrodes", "value"),
        Input("overview-ecmp-ymode", "value"),
        Input("overview-ecmp-metric", "value"),
        State("overview-ecmp-animal", "value"),
        prevent_initial_call=True,
    )
    def ecmp_figure(electrodes, ymode, metric, animal):
        # animal is State: switching animals repopulates the electrode list
        # (electrode names are animal-specific), which retriggers this via the
        # electrodes Input with the fresh animal already committed -- so no
        # stale-electrode flash.
        return _electrode_compare_figure(store, config, animal, electrodes,
                                         ymode, metric)

    # Thumbnail has its own callback because the radio adds an
    # additional Input. Re-renders on either trigger; the mean/sem/overlay
    # figures are cheap and synchronous. The "Last 24 recordings" mode reads
    # 24 .mat files off-thread (see _hist24_render): a cache miss kicks the
    # worker, shows a 'Computing...' placeholder, and enables the poll Interval
    # so the count advances; once ready, the overlay renders and the poll stops.
    @app.callback(
        Output("overview-thumbnail", "children"),
        Output("overview-hist24-poll", "disabled"),
        Output("overview-thumb-sig", "data"),
        Input("overview-trace-mode", "value"),
        Input("refresh-trigger", "data"),
        Input("overview-hist24-poll", "n_intervals"),
        Input("overview-hist24-n", "value"),
        Input("overview-mount-thumb", "n_intervals"),
        State("overview-thumb-sig", "data"),
        State("tabs", "value"),
    )
    def refresh_overview_thumbnail(trace_mode, _n, _poll, hist_n, _mount,
                                    last_sig, _tab):
        if _tab != "overview":         # skip off-tab refresh fan-out
            return no_update, no_update, no_update
        mode = trace_mode or "mean"
        if mode == "hist24":
            # hist24 has its own cache + poll; let it manage rebuilds.
            children, poll_off = _hist24_render(config, hist_n)
            return children, poll_off, no_update
        # The Latest-Evoked thumbnail only needs to change when a NEW
        # recording arrives -- not on every refresh tick. Short-circuit when
        # the (mode, newest-recording) signature is unchanged.
        try:
            sig = f"{mode}:{store.max_processed_file_id()}"
        except Exception:  # noqa: BLE001
            sig = None
        if sig is not None and sig == last_sig:
            return no_update, True, no_update
        # Single-flight the heavy build (below the cheap sig check) so
        # overlapping ticks don't stack .mat reads + figure builds. The .mat
        # fallback read inside is now deadline-bounded, so this can't wedge; the
        # watchdog just logs loudly if it ever runs long.
        if not _THUMB_LOCK.try_begin():
            _THUMB_LOCK.warn_if_stalled()
            return no_update, no_update, no_update
        try:
            sessions = store.get_sessions()
            session_dir = sessions[0]["session_dir"] if sessions else ""
            return (_build_overview_thumbnail(
                        store, config, session_dir, mode),
                    True, sig)
        finally:
            _THUMB_LOCK.end()

    @app.callback(
        Output("snapshot-expanded", "data"),
        Input("overview-snapshot-img", "n_clicks"),
        State("snapshot-expanded", "data"),
        prevent_initial_call=True,
    )
    def toggle_snapshot_size(n_clicks, current):
        # Each click flips between thumb and expanded states.
        return not bool(current)

    @app.callback(
        Output("overview-snapshot-img", "style"),
        Input("snapshot-expanded", "data"),
    )
    def apply_snapshot_size(expanded):
        if expanded:
            return {
                "display": "block",
                "width": "100%",
                "height": "auto",
                "maxWidth": "100%",
                "maxHeight": "none",
                "objectFit": "contain",
                "borderRadius": "4px",
                "background": "#0a0a14",
                "cursor": "zoom-out",
            }
        # Small thumbnail: maxWidth caps total width and the
        # browser keeps the aspect ratio via objectFit: contain.
        return {
            "display": "block",
            "maxWidth": "240px",
            "maxHeight": "180px",
            "width": "100%",
            "height": "auto",
            "objectFit": "contain",
            "borderRadius": "4px",
            "background": "#0a0a14",
            "cursor": "zoom-in",
        }

    @app.callback(
        Output("overview-snapshot-img", "src"),
        Output("overview-snapshot-caption", "children"),
        Input("refresh-trigger", "data"),
        State("tabs", "value"),
    )
    def refresh_overview_snapshot(_n, _tab):
        if _tab != "overview":         # skip off-tab refresh fan-out
            return no_update, no_update
        # ?t= cache-buster forces the browser to re-fetch each tick;
        # the Flask side caches the JPEG for ~8s so we don't actually
        # pummel the SMB share.
        import time as _t
        src = f"/media/latest-snapshot.jpg?t={int(_t.time())}"
        # Caption: recording window of the underlying chunk + camera
        # count so the operator can tell single- vs multi-camera
        # sessions apart at a glance.
        with store.connection() as conn:
            row = conn.execute(
                "SELECT file_path, chunk_datetime, duration_sec "
                "FROM processed_files "
                "WHERE has_video = 1 "
                "ORDER BY chunk_datetime DESC LIMIT 1"
            ).fetchone()
        if not row or not row["chunk_datetime"]:
            return src, "No videos available"
        # Camera count for the caption. companion_video_paths GLOBS the SMB
        # share -- a GIL-holding read that used to fire every 10s tick here and
        # stall the Overview builds. Memoize by file_path: the newest recording
        # changes ~hourly, so the glob now runs at most once per new file.
        fp = row["file_path"] or ""
        n_cams = _SNAP_CAM_CACHE.get(fp)
        if n_cams is None:
            from src.utils.video import companion_video_paths
            try:
                n_cams = len(companion_video_paths(fp))
            except Exception:  # noqa: BLE001
                n_cams = 0
            _SNAP_CAM_CACHE[fp] = n_cams
            while len(_SNAP_CAM_CACHE) > 64:      # bound the memo
                _SNAP_CAM_CACHE.pop(next(iter(_SNAP_CAM_CACHE)))
        cam_tag = (f" · {n_cams} cameras" if n_cams > 1
                    else "")
        start_dt = _parse_chunk_dt(row["chunk_datetime"])
        if start_dt is None:
            return src, f"Latest recording{cam_tag}"
        dur = float(row["duration_sec"] or 3600.0)
        end_dt = start_dt + timedelta(seconds=dur)
        return src, (f"Chunk {start_dt.strftime('%H:%M')} – "
                      f"{end_dt.strftime('%H:%M')} (last frame)"
                      f"{cam_tag}")

    # ------------------------------------------------------------------ #
    #  Home page "Open >" buttons -> jump to the matching tab
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("group-tabs", "value", allow_duplicate=True),
        Output("tabs", "value", allow_duplicate=True),
        Input("home-open-surgeries", "n_clicks"),
        Input("home-open-schedule", "n_clicks"),
        Input("home-open-maintenance", "n_clicks"),
        Input("home-open-incidents", "n_clicks"),
        Input("home-open-datalog", "n_clicks"),
        prevent_initial_call=True,
    )
    def _home_open(_s, _sch, _m, _inc, _d):
        # Gate on a real button press. The home grid is re-instantiated
        # by refresh_overview_dynamic on every refresh-trigger tick;
        # without this check Dash interprets the fresh n_clicks=0 buttons
        # as a triggering event and would tab-switch the user out of
        # Overview into Lab on every poll. Also covers the initial-page
        # load case (Dash sometimes ignores prevent_initial_call when
        # multiple callbacks share Output via allow_duplicate).
        if not callback_context.triggered:
            return no_update, no_update
        trig_payload = callback_context.triggered[0]
        if not trig_payload.get("value"):  # n_clicks 0 / None
            return no_update, no_update
        trig = callback_context.triggered_id
        mapping = {
            "home-open-surgeries": ("lab", "surgeries"),
            "home-open-schedule": ("lab", "maintenance"),
            "home-open-maintenance": ("lab", "maintenance"),
            "home-open-incidents": ("lab", "maintenance"),
            "home-open-datalog": ("lab", "data_log_xref"),
        }
        if trig in mapping:
            return mapping[trig]
        return no_update, no_update

