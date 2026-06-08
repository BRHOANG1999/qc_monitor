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
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from src.dashboard.tabs.surgeries import (
    _load_sheet_via_api, _resolve_sa_path,
    _find_column, _normalize_text,
)

logger = logging.getLogger("qc_monitor.utils.assignments")


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
                       ttl_sec: float | None = None
                       ) -> list[Assignment]:
    """Return every ACTIVE assignment row from the configured tab.

    Inactive rows (``Active`` cell != truthy) are dropped from the
    returned list but kept in the sheet for audit. Returns ``[]``
    on any error or when the feature is disabled in config.
    """
    cfg = _cfg(config)
    if not cfg:
        return []
    sheet_id = cfg.get("sheet_id")
    tab_name = cfg.get("tab_name") or "Reviewer Assignments"
    sa_path_raw = cfg.get("service_account_file", "")
    if not sheet_id:
        return []
    if not sa_path_raw:
        # Fall back to the lab's shared service-account file used
        # everywhere else.
        sa_path_raw = (config.get("data_log_xref", {}) or {}).get(
            "service_account_file", "")
    sa_path = _resolve_sa_path(config or {}, sa_path_raw)
    if not sa_path:
        return []
    if ttl_sec is None:
        ttl_sec = float(cfg.get("refresh_minutes", 5)) * 60.0
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


def assignments_for_user(config: dict, user_email: str
                           ) -> list[Assignment]:
    """Active assignments for one reviewer. Case-insensitive."""
    if not user_email:
        return []
    every = load_assignments(config)
    return [a for a in every if a.matches_user(user_email)]


def animals_for_user(config: dict, user_email: str) -> list[str]:
    """Sorted distinct animal ids assigned to the user."""
    return sorted({a.animal_id
                    for a in assignments_for_user(config, user_email)})


def assigned_animals(config: dict) -> set[str]:
    """Every animal id with at least one active assignee."""
    return {a.animal_id for a in load_assignments(config)}


def assignees_for_animal(config: dict, animal_id: str
                          ) -> list[str]:
    """Email addresses currently assigned to *animal_id*."""
    if not animal_id:
        return []
    return sorted({a.user_email
                    for a in load_assignments(config)
                    if a.animal_id == animal_id})


def unassigned_animals(config: dict, store) -> list[str]:
    """Animal ids that exist in session_config but have no active
    assignee. Used for the "Unassigned pool" UX."""
    every_animal = set(store.list_all_animals())
    return sorted(every_animal - assigned_animals(config))


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
