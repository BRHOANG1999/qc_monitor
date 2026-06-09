"""Reviewer assignments — animal -> undergrad mapping loader.

Reads a Google Sheet tab (default name "Reviewer Assignments")
containing one row per (user, animal, active?) triple.

The sheet is loaded via the same TTL-cached helper the Surgeries
+ Data Log Diff tabs use, so we don't pay an API call per
dashboard refresh and a stale read after a PI edit is bounded to
``review_queue.assignments.refresh_minutes`` minutes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from src.dashboard.tabs.surgeries import (
    _load_sheet_via_api, _resolve_sa_path,
    _find_column, _normalize_text,
)

logger = logging.getLogger("qc_monitor.utils.assignments")

# Monotonic version counter bumped by the background warmer each
# time it successfully refreshes the assignment cache. The Dash UI
# polls this through a hidden Store so the picker auto-re-renders
# when fresh data lands -- no need for the user to click Refresh.
_cache_version: int = 0
_warmer_started: bool = False
_warmer_lock = threading.Lock()

# Wall-clock timestamp of the FIRST completed warmer iteration
# (regardless of success or failure of the underlying Sheets API
# call). The picker uses this + a hard timeout so it doesn't spin
# forever on 'Loading...' when the Sheet tab is empty, missing,
# or unreachable. 0.0 means "warmer hasn't completed once yet".
_warmer_first_completed_at: float = 0.0
_warmer_started_at: float = 0.0

# How long the picker is willing to wait for the warmer before
# treating the cache as populated-but-empty. 5 s is comfortably
# longer than a healthy Sheets round trip (~0.5-1 s) and short
# enough that a totally-unreachable Sheet doesn't render the tab
# unusable.
WARMER_TIMEOUT_SEC: float = 5.0


@dataclass(frozen=True)
class Assignment:
    user_email: str
    animal_id: str
    assigned_at: date | None
    notes: str

    def matches_user(self, email: str) -> bool:
        if not email or not self.user_email:
            return False
        return self.user_email.lower().strip() == email.lower().strip()


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #

def _cfg(config: dict) -> dict:
    return ((config or {}).get("review_queue", {}) or {}).get(
        "assignments", {}) or {}


def load_assignments(config: dict,
                       ttl_sec: float | None = None,
                       *, cache_only: bool = False,
                       ) -> list[Assignment] | None:
    """Return every ACTIVE assignment row from the configured tab.

    Inactive rows (``Active`` cell != truthy) are dropped from the
    returned list but kept in the sheet for audit. Returns ``[]``
    on any error or when the feature is disabled in config.

    ``cache_only=True`` skips the network round trip and only reads
    the in-process TTL cache. Returns ``None`` if the cache has
    never been populated (warmer hasn't finished its first
    iteration yet); UI callers should render a "Loading..."
    placeholder and let the assignments-version Store push a
    re-render once the warmer publishes fresh data. Returns ``[]``
    if cached but truly empty.
    """
    cfg = _cfg(config)
    if not cfg:
        return []
    sheet_id = cfg.get("sheet_id")
    tab_name = cfg.get("tab_name") or "Reviewer Assignments"
    if not sheet_id:
        return []
    if cache_only:
        df = _peek_sheet_cache(sheet_id, tab_name)
        if df is None:
            return None
    else:
        sa_path_raw = cfg.get("service_account_file", "")
        if not sa_path_raw:
            # Fall back to the lab's shared service-account file
            # used everywhere else.
            sa_path_raw = (config.get("data_log_xref", {}) or {}).get(
                "service_account_file", "")
        sa_path = _resolve_sa_path(config or {}, sa_path_raw)
        if not sa_path:
            return []
        if ttl_sec is None:
            ttl_sec = float(cfg.get("refresh_minutes", 30)) * 60.0
        df = _load_sheet_via_api(sheet_id, tab_name, sa_path, ttl_sec)
    if df is None or df.empty:
        return []

    user_col = _find_column(df, ["User_Email", "User Email", "Email"])
    animal_col = _find_column(df, ["Animal_ID", "Animal", "Animal Id"])
    active_col = _find_column(df, ["Active", "Is Active",
                                     "Enabled"])
    when_col = _find_column(df, ["Assigned_At", "Assigned At",
                                   "Assigned", "Start"])
    notes_col = _find_column(df, ["Notes", "Note", "Comment"])
    if user_col is None or animal_col is None:
        logger.warning("Assignments sheet missing required "
                        "User_Email / Animal_ID columns "
                        "(have: %s)", list(df.columns))
        return []

    out: list[Assignment] = []
    for _idx, row in df.iterrows():
        email = _normalize_text(row.get(user_col)).lower()
        animal = _normalize_text(row.get(animal_col))
        if not email or not animal:
            continue
        if active_col is not None:
            active_raw = _normalize_text(row.get(active_col)).lower()
            if active_raw in ("false", "0", "no", "n", "off",
                                "inactive"):
                continue
        when = None
        if when_col is not None:
            raw = row.get(when_col)
            when = _parse_date(raw)
        notes = (_normalize_text(row.get(notes_col))
                  if notes_col is not None else "")
        out.append(Assignment(
            user_email=email, animal_id=animal,
            assigned_at=when, notes=notes,
        ))
    return out


def assignments_for_user(config: dict, user_email: str,
                           *, cache_only: bool = False,
                           ) -> list[Assignment] | None:
    """Active assignments for one reviewer. Case-insensitive.

    ``cache_only=True`` returns ``None`` if the cache is cold
    (see ``load_assignments``).
    """
    if not user_email:
        return []
    every = load_assignments(config, cache_only=cache_only)
    if every is None:
        return None
    return [a for a in every if a.matches_user(user_email)]


def animals_for_user(config: dict, user_email: str,
                       *, cache_only: bool = False,
                       ) -> list[str] | None:
    """Sorted distinct animal ids assigned to the user.

    Returns ``None`` (cache cold) only when ``cache_only=True``.
    """
    rows = assignments_for_user(config, user_email,
                                  cache_only=cache_only)
    if rows is None:
        return None
    return sorted({a.animal_id for a in rows})


def assigned_animals(config: dict,
                       *, cache_only: bool = False,
                       ) -> set[str] | None:
    """Every animal id with at least one active assignee.

    Returns ``None`` (cache cold) only when ``cache_only=True``.
    """
    every = load_assignments(config, cache_only=cache_only)
    if every is None:
        return None
    return {a.animal_id for a in every}


def assignees_for_animal(config: dict, animal_id: str
                          ) -> list[str]:
    """Email addresses currently assigned to *animal_id*."""
    if not animal_id:
        return []
    every = load_assignments(config)
    if every is None:
        return []
    return sorted({a.user_email for a in every
                    if a.animal_id == animal_id})


def unassigned_animals(config: dict, store,
                         *, cache_only: bool = False,
                         ) -> list[str] | None:
    """Animal ids that exist in session_config but have no active
    assignee. Used for the "Unassigned pool" UX.

    Returns ``None`` (cache cold) only when ``cache_only=True``.
    """
    assigned = assigned_animals(config, cache_only=cache_only)
    if assigned is None:
        return None
    every_animal = set(store.list_all_animals())
    return sorted(every_animal - assigned)


# --------------------------------------------------------------------- #
# Background cache warmer
# --------------------------------------------------------------------- #

def cache_version() -> int:
    """Monotonic version of the assignment cache.

    Bumped each time the background warmer successfully refreshes
    the sheet. The dashboard polls this through a hidden Store so
    picker callbacks can subscribe to "data changed" without paying
    network latency themselves.
    """
    return _cache_version


def _warmer_interval_sec(config: dict) -> float:
    """Sleep interval between warmer iterations. Floored at 60 s
    so a misconfigured refresh_minutes=0 doesn't busy-loop."""
    cfg = _cfg(config)
    minutes = float(cfg.get("refresh_minutes", 30) or 30)
    return max(60.0, minutes * 60.0)


def start_warmer(config: dict) -> None:
    """Spawn the background cache-warmer thread. Idempotent.

    Refreshes the assignment cache on its own clock (independent
    of the dashboard's 10 s refresh tick) and bumps
    ``_cache_version`` so the UI can react to "data ready" without
    blocking on the Sheets API in any render path.
    """
    assert isinstance(config, dict), "config must be a dict"
    cfg = _cfg(config)
    if not cfg or not cfg.get("sheet_id"):
        logger.debug("assignments warmer not started (no sheet "
                      "configured)")
        return
    global _warmer_started, _warmer_started_at
    with _warmer_lock:
        if _warmer_started:
            return
        _warmer_started = True
        _warmer_started_at = time.time()
    t = threading.Thread(
        target=_warmer_loop, args=(config,),
        daemon=True, name="qc-assignments-warmer",
    )
    t.start()
    logger.info("Started assignments cache warmer "
                 "(refresh every %.0f s)",
                 _warmer_interval_sec(config))


def _warmer_loop(config: dict) -> None:
    """Daemon loop. Refreshes assignments + bumps version counter.

    The ``while True`` here is a service loop by design, but per
    the lab's NASA JPL Rule 2 we still cap iterations and assert
    at the top. At a 30-min refresh the cap covers ~60,000 years.
    """
    global _cache_version, _warmer_first_completed_at
    assert isinstance(config, dict), "config must be a dict"
    interval = _warmer_interval_sec(config)
    max_iter = 10 ** 9
    i = 0
    while True:
        assert i < max_iter, "warmer loop runaway"
        i += 1
        try:
            load_assignments(config, ttl_sec=0.0)
            _cache_version += 1
            logger.debug("assignments cache warmed (v=%d)",
                          _cache_version)
        except Exception:
            logger.exception(
                "assignments warmer iteration failed; "
                "will retry in %.0f s", interval)
        # Stamp first-completion regardless of success: a
        # failed Sheets API call (tab missing, network down,
        # auth expired) should still let the picker fall back
        # to "no assignments" instead of spinning forever on
        # the cold-cache loading message.
        if _warmer_first_completed_at == 0.0:
            _warmer_first_completed_at = time.time()
        time.sleep(interval)


def cache_settled() -> bool:
    """True once the warmer has completed at least one full
    iteration (regardless of whether it succeeded). Lets UI
    code distinguish 'haven't tried yet' from 'tried and got
    nothing' so empty Sheets don't spin the picker forever.
    """
    return _warmer_first_completed_at > 0.0


def warmer_should_have_settled() -> bool:
    """True when the picker should give up waiting for the
    warmer and fall back to treating the cache as empty.

    Two conditions OR'd together:
      * warmer has completed at least one iteration
        (``cache_settled``), OR
      * ``WARMER_TIMEOUT_SEC`` has elapsed since the warmer
        was started (defensive: if the daemon thread itself
        failed to spawn or hung before the first iteration,
        the picker still recovers).
    """
    if cache_settled():
        return True
    if _warmer_started_at <= 0.0:
        return False
    return (time.time() - _warmer_started_at
              ) >= WARMER_TIMEOUT_SEC


def _peek_sheet_cache(sheet_id: str,
                       tab_name: str) -> pd.DataFrame | None:
    """Read the shared sheet TTL cache without triggering a fetch.

    Returns the cached DataFrame regardless of age (it's the
    warmer's job to keep it fresh) or ``None`` if the cache has
    never been populated for this (sheet, tab).
    """
    assert sheet_id, "sheet_id required"
    assert tab_name, "tab_name required"
    from src.dashboard.tabs.surgeries import _cache, _cache_lock
    cache_key = f"api:{sheet_id}:{tab_name}:h1"
    with _cache_lock:
        hit = _cache.get(cache_key)
    return hit[1] if hit is not None else None


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def _parse_date(raw) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    text = _normalize_text(raw)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # pandas can usually rescue weird Sheets serials.
    try:
        return pd.to_datetime(text).date()
    except Exception:
        return None
