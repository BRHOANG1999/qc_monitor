"""Cage lookup pulled from the KM Lab Mice ``JAX Cages`` tab.

That tab is laid out non-tabularly: one section per operator (Lauren,
Madison, Brandon, ...), each section starting with a header row and
followed by one row per housing cage. Each cage row lists multiple
animals in a single free-text cell (``BCH63(1L), BCH64(1L1R)`` style).

This module walks the tab top-to-bottom keeping a small state
machine: track ``current_operator`` from the section headers, parse
each data row for animal IDs, and build ``{animal_id_upper: CageRow}``
so the Surgeries tab can look up per-animal cage context without
re-reading the sheet.

There is no per-cage barcode column in this sheet; the JAX codes at
the top (``-00112935`` etc.) cover entire grants, not individual
cages. ``cage_descriptor`` therefore reports a derived identifier
(operator + father strain + open date).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.dashboard.tabs.surgeries import (
    _load_sheet_via_api, _resolve_sa_path, _normalize_text,
)

logger = logging.getLogger("qc_monitor.dashboard.cage_index")

# Animal-ID extractor for the "Animal ID(s)" free-text cell.
# Conservative: 1-5 letters + 1-4 digits. Avoids matching ear-mark
# notes like "1L1R" (pure digits/letters mix without leading alpha
# block) and strain codes like "Bl6-TD" (5+ chars across hyphens).
_ANIMAL_RE = re.compile(r'\b([A-Za-z]{1,5}\d{1,4})\b')

# Position-based access: JAX Cages always uses column indices 0-12.
# Per-cage header row (the one with "Cage open date / Strain / ...")
# has the same layout every section.
_COL_OPEN_DATE = 0
_COL_STRAIN = 3
_COL_FATHER = 4
_COL_MOTHER = 5
_COL_LITTER_DOB = 6
_COL_NOTES = 11
_COL_ANIMALS = 12


@dataclass(frozen=True)
class CageRow:
    operator: str
    strain: str
    father: str
    mother: str
    litter_dob: str
    cage_open_date: str
    notes: str


def _cfg(config: dict) -> dict:
    return ((config or {}).get("surgeries", {}) or {}).get(
        "cage_index", {}) or {}


def cage_descriptor(cage: CageRow | None) -> str:
    """Compact one-line label for the DataTable cell."""
    if cage is None:
        return ""
    bits: list[str] = []
    if cage.operator:
        bits.append(cage.operator)
    if cage.father:
        bits.append(cage.father)
    if cage.cage_open_date:
        bits.append(f"opened {cage.cage_open_date}")
    elif cage.litter_dob:
        bits.append(f"litter {cage.litter_dob}")
    if cage.strain and cage.strain not in (cage.father, cage.mother):
        bits.append(cage.strain)
    return " · ".join(bits) or "(unknown cage)"


# ===================================================================== #
#  Walker
# ===================================================================== #

def _is_operator_header(row: list) -> str | None:
    """Return the operator name if *row* looks like a section header.

    Heuristic: col 0 has a single first-name string AND col 11 reads
    'Strain Notes:' (or close). The sheet's existing headers all share
    that shape (rows 6, 14, 18, 27, 37 in the live tab).
    """
    if not row:
        return None
    col0 = _normalize_text(row[0] if len(row) > 0 else "")
    col11 = _normalize_text(row[11] if len(row) > 11 else "")
    if not col0:
        return None
    # Reject if col0 looks like a date or contains spaces (e.g. "Cage
    # open date") -- a real operator name is one short token.
    if " " in col0 or "/" in col0 or ":" in col0 or len(col0) > 20:
        return None
    if "Strain Notes" not in col11 and "Earpunches" not in col11:
        return None
    return col0


def _is_table_header(row: list) -> bool:
    """Detect the per-section header row 'Cage open date | ... | Animal
    ID(s)'. We use it to confirm we've entered a data region."""
    if not row or len(row) <= _COL_ANIMALS:
        return False
    col0 = _normalize_text(row[0])
    col12 = _normalize_text(row[_COL_ANIMALS])
    return ("Cage open date" in col0
            and "Animal ID" in col12)


def _animals_in_cell(text: str) -> list[str]:
    if not text:
        return []
    return [m.upper() for m in _ANIMAL_RE.findall(text)]


def _build_index(values: list[list]) -> dict[str, CageRow]:
    """Walk the parsed JAX Cages tab and return the animal->cage map."""
    out: dict[str, CageRow] = {}
    current_operator = ""
    in_data = False

    for raw in values:
        # Normalize: ensure row has at least 13 cells so we can index by
        # position safely.
        row = list(raw) + [""] * max(0, _COL_ANIMALS + 1 - len(raw))

        op = _is_operator_header(row)
        if op is not None:
            current_operator = op
            in_data = False
            continue

        if _is_table_header(row):
            in_data = True
            continue

        if not in_data:
            continue

        animal_cell = _normalize_text(row[_COL_ANIMALS])
        if not animal_cell:
            continue

        animals = _animals_in_cell(animal_cell)
        if not animals:
            continue

        cage = CageRow(
            operator=current_operator,
            strain=_normalize_text(row[_COL_STRAIN]),
            father=_normalize_text(row[_COL_FATHER]),
            mother=_normalize_text(row[_COL_MOTHER]),
            litter_dob=_normalize_text(row[_COL_LITTER_DOB]),
            cage_open_date=_normalize_text(row[_COL_OPEN_DATE]),
            notes=_normalize_text(row[_COL_NOTES]),
        )
        for a in animals:
            # Later occurrences overwrite earlier ones (so a transferred
            # animal lands in its *current* cage). Acceptable.
            out[a] = cage
    return out


# ===================================================================== #
#  Public
# ===================================================================== #

def load_cage_index(config: dict,
                    ttl_sec: float | None = None
                    ) -> dict[str, CageRow]:
    """Return {animal_id (uppercase): CageRow} from the configured tab.

    Empty dict when the tab is missing, the SA file isn't configured,
    or the fetch fails. Lookup keys are uppercased so callers should
    do ``index.get(animal.upper())``.
    """
    cfg = _cfg(config)
    if not cfg.get("enabled", False):
        return {}
    sheet_id = cfg.get("sheet_id", "")
    tab_name = cfg.get("tab_name", "JAX Cages")
    sa_file = _resolve_sa_path(
        config or {},
        ((config or {}).get("surgeries", {}) or {})
        .get("service_account_file", ""),
    )
    if not (sheet_id and sa_file and tab_name):
        return {}
    if ttl_sec is None:
        # Inherit cadence from the surgeries tab.
        refresh_min = float(
            ((config or {}).get("surgeries", {}) or {})
            .get("refresh_minutes", 10)
        )
        ttl_sec = refresh_min * 60.0

    df = _load_sheet_via_api(sheet_id, tab_name, sa_file, ttl_sec)
    if df is None or df.empty:
        return {}

    # Strip the synthetic header (the loader sets header=row 0 by
    # default). Re-render to a list-of-lists so the walker can use
    # position-based access -- the sheet has no stable column names
    # at the *tab* level (only per-section).
    values: list[list] = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
    return _build_index(values)
