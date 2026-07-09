"""Quantify how much scored seizure work was silently lost, per (animal,
reviewer), across the whole DB.

WHY THIS EXISTS
---------------
Reviewers scored events (Racine + light + onset landmarks) in the Video
Review tab, but per-event scores lived only in the browser until an explicit
terminal button was clicked -- and the `n` hotkey ("mark no events + next")
force-discarded them. The result: across the live DB only a handful of events
anywhere carry a Racine score, though reviewers believe they scored hundreds.
This module turns that opaque loss into an explicit, per-(animal, reviewer)
count so recovery (Phase 2/3) can be scoped -- which animals need re-scoring,
and which have a manual log to import from.

The fixes in the Video Review tab (save-guard + autosave) stop *new* loss;
this audit measures the *existing* damage. It is strictly READ-ONLY: it opens
the DB with ``mode=ro`` and never triggers Store migrations or job-reaping.

SIGNALS (per animal, per reviewer)
----------------------------------
- ``zero_finish_on_flagged`` -- ``finish`` audit events that recorded
  ``n_markers=0`` on a (file, animal) the detector had flagged
  (``auto_filter_flag``). The detector is high-sensitivity so a flag isn't
  proof of a real seizure, but a zero-marker finish on a flagged file is the
  signature of the `n`-hotkey discard.
- ``claimed_never_saved`` -- (file, animal) the reviewer opened (``claim``)
  with no ``finish``/``quick_flag`` by anyone: opened to score, abandoned.
- ``unscored_events`` / ``unscored_files`` -- events parked in the latest
  ``needs_scoring`` draft with an EEG onset (``EO_sec``) but no ``racine``:
  the onset survived, the score didn't.
- ``scored_events`` -- events that DO carry a Racine (the survivor set), for
  contrast.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field


def _events(markers_json) -> list:
    """Decode a review_state.markers_json into a list of event dicts."""
    if not markers_json:
        return []
    try:
        d = json.loads(markers_json)
    except (json.JSONDecodeError, TypeError):
        return []
    return [e for e in d if isinstance(e, dict)] if isinstance(d, list) else []


_LEGACY = "(legacy/NULL)"


@dataclass
class LossSummary:
    # (animal, reviewer) -> {signal: count}
    cells: dict = field(default_factory=lambda: defaultdict(
        lambda: defaultdict(int)))
    elapsed_sec: float = 0.0

    def bump(self, animal, reviewer, signal, n=1):
        self.cells[(animal or _LEGACY, reviewer or "(unknown)")][signal] += n

    # --- rollups -------------------------------------------------------- #
    def by_animal(self) -> dict:
        out = defaultdict(lambda: defaultdict(int))
        for (animal, _rev), sig in self.cells.items():
            for k, v in sig.items():
                out[animal][k] += v
        return out

    def totals(self) -> dict:
        out = defaultdict(int)
        for sig in self.cells.values():
            for k, v in sig.items():
                out[k] += v
        return out


_SIGNALS = ("scored_events", "unscored_events", "unscored_files",
            "zero_finish_on_flagged", "claimed_never_saved")


def audit(conn: sqlite3.Connection, progress=None) -> LossSummary:
    """Compute the loss summary from a (read-only) sqlite3 connection.

    ``progress(step, total)`` is called between the four passes for live
    feedback on a large DB.
    """
    conn.row_factory = sqlite3.Row
    s = LossSummary()
    t0 = time.time()
    steps, total = 0, 4

    def _tick():
        nonlocal steps
        steps += 1
        if progress:
            progress(steps, total)

    # Pass 1: (animal, file) the detector flagged.
    flagged = set()
    for r in conn.execute(
            "SELECT DISTINCT animal_id, file_id FROM review_event_log "
            "WHERE action='auto_filter_flag' AND animal_id IS NOT NULL"):
        flagged.add((r["animal_id"], r["file_id"]))
    _tick()

    # Pass 2: zero-marker finishes on flagged (animal, file).
    for r in conn.execute(
            "SELECT animal_id, user_email, file_id, payload_json "
            "FROM review_event_log "
            "WHERE action='finish' AND animal_id IS NOT NULL"):
        try:
            nm = json.loads(r["payload_json"] or "{}").get("n_markers", 0)
        except (json.JSONDecodeError, TypeError):
            nm = 0
        if not nm and (r["animal_id"], r["file_id"]) in flagged:
            s.bump(r["animal_id"], r["user_email"], "zero_finish_on_flagged")
    _tick()

    # Pass 3: claimed-but-never-saved (file, animal).
    claims: dict = {}          # (animal, file) -> set(reviewers who claimed)
    saved = set()              # (animal, file) with a finish/quick_flag
    for r in conn.execute(
            "SELECT animal_id, file_id, user_email, action "
            "FROM review_event_log "
            "WHERE action IN ('claim','finish','quick_flag') "
            "AND animal_id IS NOT NULL"):
        k = (r["animal_id"], r["file_id"])
        if r["action"] == "claim":
            claims.setdefault(k, set()).add(r["user_email"])
        else:
            saved.add(k)
    for (animal, fid), users in claims.items():
        if (animal, fid) not in saved:
            for u in users:
                s.bump(animal, u, "claimed_never_saved")
    _tick()

    # Pass 4: latest needs_scoring draft per (file, animal) -> unscored /
    # scored event counts. Take MAX(id) per (file, animal) so historical
    # duplicate rows aren't double-counted (later id overwrites in the dict).
    latest: dict = {}          # (animal, file) -> row with the max id
    for r in conn.execute(
            "SELECT id, file_id, animal_id, user_email, status, markers_json "
            "FROM review_state "
            "WHERE animal_id IS NOT NULL AND markers_json IS NOT NULL "
            "AND markers_json != '[]' ORDER BY id"):
        latest[(r["animal_id"], r["file_id"])] = r     # later id overwrites
    for r in latest.values():
        evs = _events(r["markers_json"])
        scored = sum(1 for e in evs
                     if e.get("racine") not in (None, "", []))
        unscored = sum(1 for e in evs
                       if e.get("EO_sec") not in (None, "")
                       and e.get("racine") in (None, "", []))
        if scored:
            s.bump(r["animal_id"], r["user_email"], "scored_events", scored)
        if r["status"] == "needs_scoring" and unscored:
            s.bump(r["animal_id"], r["user_email"], "unscored_events",
                   unscored)
            s.bump(r["animal_id"], r["user_email"], "unscored_files", 1)
    _tick()

    s.elapsed_sec = time.time() - t0
    return s


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #

def _fmt_row(label, sig) -> str:
    return (f"  {label:34} "
            f"scored={sig.get('scored_events', 0):<6} "
            f"unscored_ev={sig.get('unscored_events', 0):<6} "
            f"unscored_files={sig.get('unscored_files', 0):<6} "
            f"zero_finish={sig.get('zero_finish_on_flagged', 0):<6} "
            f"abandoned={sig.get('claimed_never_saved', 0)}")


def print_report(s: LossSummary) -> None:
    by_animal = s.by_animal()
    # Rank animals by lost work (unscored events + zero-marker finishes).
    order = sorted(by_animal.items(),
                   key=lambda kv: -(kv[1].get("unscored_events", 0)
                                    + kv[1].get("zero_finish_on_flagged", 0)
                                    + kv[1].get("claimed_never_saved", 0)))
    print("\n=== Scored-work loss audit (per animal, most-affected first) ===")
    for animal, sig in order:
        print(_fmt_row(animal, sig))
        # per-reviewer breakdown under each animal
        revs = sorted(
            ((rev, cell) for (a, rev), cell in s.cells.items() if a == animal),
            key=lambda kv: -(kv[1].get("unscored_events", 0)
                             + kv[1].get("zero_finish_on_flagged", 0)))
        for rev, cell in revs:
            print(_fmt_row(f"    - {rev}", cell))
    t = s.totals()
    print("\n" + _fmt_row("TOTAL", t))
    print(f"\n(audit read-only; {s.elapsed_sec:.1f}s)")
    print("Legend: unscored_ev = onset dropped but Racine never saved "
          "(recoverable if a manual log exists); zero_finish = detector-"
          "flagged file saved as 0 events (the n-hotkey discard); "
          "abandoned = opened, never saved.")


def as_dict(s: LossSummary) -> dict:
    return {
        "by_reviewer": {f"{a}|{r}": dict(sig)
                        for (a, r), sig in s.cells.items()},
        "by_animal": {a: dict(sig) for a, sig in s.by_animal().items()},
        "totals": dict(s.totals()),
        "elapsed_sec": s.elapsed_sec,
    }


# --------------------------------------------------------------------------- #
#  CLI  --  python -m src.maintenance.scoring_loss_audit [--db ...] [--json P]
# --------------------------------------------------------------------------- #

def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Read-only audit of silently-lost seizure scores.")
    ap.add_argument("--db", default="data/monitor.db")
    ap.add_argument("--json", default=None,
                    help="also write the full summary as JSON to this path")
    a = ap.parse_args(argv)

    print(f"Opening {a.db} read-only ...", flush=True)
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)

    def prog(step, total):
        print(f"\r[pass {step}/{total}] scanning ...", end="", flush=True)

    try:
        s = audit(conn, progress=prog)
    finally:
        conn.close()
    print_report(s)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(as_dict(s), fh, indent=1)
        print(f"\nfull summary -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
