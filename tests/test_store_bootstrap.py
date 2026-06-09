"""Boot + migration smoke tests for src/db/store.py.

The Store class boots in three modes that must all stay healthy:

* fresh DB -- schema executes cleanly, every table from SCHEMA_SQL
  exists.
* legacy DB whose ``review_state`` predates the PI-verification flow --
  the CHECK constraint is rebuilt and finalised rows are promoted to
  ``pi_approved`` via the bespoke detect-and-rebuild migration in
  ``Store._migrate_review_state_pi_statuses``.
* DB that has a ``mass_analyze_job`` row stuck in 'running'/'pending'
  across a restart -- the unconditional sweep in ``Store._init_db``
  must reap it without erroring on a brand-new DB that has no rows.

These tests pay for themselves the first time someone widens a CHECK
constraint or renames a column the boot path inspects.

Run with: pytest tests/test_store_bootstrap.py -q
"""

from __future__ import annotations

import os
import sqlite3
import sys

import pytest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.db.schema import SCHEMA_SQL  # noqa: E402


# Tables we expect after a fresh Store() bootstrap. Pulled from the
# CREATE TABLE statements in src/db/schema.py; if you add a table, add
# it here so an accidental name change shows up as a test failure.
EXPECTED_TABLES = {
    "settings_versions", "session_config", "processed_files",
    "chunk_qc", "stim_qc", "evoked_features", "evoked_summary",
    "criticality", "seizure_events", "matlab_results", "alerts",
    "evoked_waveforms", "processing_log", "annotations",
    "system_health", "users", "video_qc",
    "review_state", "review_event_log", "envelope_peak_cache",
    "mass_analyze_job", "event_clip_job",
}


def _db_path(tmp_path) -> str:
    return str(tmp_path / "data" / "monitor.db")


# ===================================================================== #
#  Fresh bootstrap
# ===================================================================== #

def test_fresh_bootstrap_creates_every_expected_table(tmp_path):
    Store(_db_path(tmp_path))
    conn = sqlite3.connect(_db_path(tmp_path))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        conn.close()
    actual = {r[0] for r in rows if not r[0].startswith("sqlite_")}
    missing = EXPECTED_TABLES - actual
    assert not missing, f"Missing tables after bootstrap: {missing}"


def test_fresh_bootstrap_is_idempotent(tmp_path):
    """Second Store(path) on the same DB is a no-op."""
    Store(_db_path(tmp_path))
    # The second construction reruns _init_db; if any migration was
    # not idempotent it would either raise or drop+rebuild a populated
    # table.
    Store(_db_path(tmp_path))


def test_fresh_bootstrap_does_not_break_on_empty_mass_analyze_job(tmp_path):
    """The unconditional UPDATE that reaps stuck mass_analyze jobs
    must tolerate a brand-new DB with zero rows."""
    Store(_db_path(tmp_path))  # would raise if the UPDATE failed
    with Store(_db_path(tmp_path)).connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM mass_analyze_job").fetchone()[0]
    assert n == 0


# ===================================================================== #
#  Migration: review_state pre-PI -> post-PI
# ===================================================================== #

# The pre-PI review_state CHECK constraint, captured verbatim from
# the schema as it existed before commit b682621.
_OLD_REVIEW_STATE_SQL = """
CREATE TABLE review_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    user_email TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK(status IN ('claimed', 'no_events',
                          'has_events', 'abandoned')),
    markers_json TEXT,
    note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _seed_legacy_db(path: str) -> None:
    """Create a DB with the pre-PI review_state shape + a couple of
    finalised rows ready to be promoted by the migration.

    Strategy: run the current SCHEMA_SQL (so referenced tables -- in
    particular processed_files with all its current columns + indices
    -- exist as the migration expects), then DROP and re-create
    ``review_state`` with the pre-PI CHECK constraint and reseed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.execute("DROP INDEX IF EXISTS idx_review_state_file")
        conn.execute("DROP INDEX IF EXISTS idx_review_state_user")
        conn.execute("DROP INDEX IF EXISTS idx_review_state_status")
        conn.execute("DROP TABLE review_state")
        conn.execute(_OLD_REVIEW_STATE_SQL)
        # Two finalised rows (should be promoted), one in-flight
        # (should NOT be promoted), and a parent processed_files row
        # for each.
        conn.executescript("""
            INSERT INTO processed_files (id, file_path) VALUES
              (1, '/fake/a.mat'),
              (2, '/fake/b.mat'),
              (3, '/fake/c.mat');
            INSERT INTO review_state
              (file_id, user_email, status, markers_json, note,
               created_at, updated_at)
            VALUES
              (1, 'r@lab', 'no_events',  NULL, NULL,
               '2026-01-01T00:00:00', '2026-01-01T00:00:00'),
              (2, 'r@lab', 'has_events', '[]',  NULL,
               '2026-01-02T00:00:00', '2026-01-02T00:00:00'),
              (3, 'r@lab', 'claimed',    NULL, NULL,
               '2026-01-03T00:00:00', '2026-01-03T00:00:00');
        """)
        conn.commit()
    finally:
        conn.close()


def test_legacy_review_state_is_promoted_to_pi_approved(tmp_path):
    db = _db_path(tmp_path)
    _seed_legacy_db(db)

    # Sanity-check the seed: the old CHECK rejects 'pi_approved'.
    conn = sqlite3.connect(db)
    try:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='review_state'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert "pi_approved" not in sql, "seed should use the pre-PI shape"

    # Boot Store -- this triggers _migrate_review_state_pi_statuses.
    Store(db)

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT file_id, status FROM review_state ORDER BY file_id"
        ).fetchall()
        new_sql = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='review_state'"
        ).fetchone()[0]
        audit_rows = conn.execute(
            "SELECT file_id, action FROM review_event_log "
            "ORDER BY file_id"
        ).fetchall()
    finally:
        conn.close()

    by_file = {fid: st for (fid, st) in rows}
    assert by_file == {1: "pi_approved", 2: "pi_approved", 3: "claimed"}
    assert "pi_approved" in new_sql, \
        "CHECK constraint should now include pi_approved"
    # The migration writes one audit row per promoted file.
    audit_files = sorted(r[0] for r in audit_rows
                          if r[1] == "pi_migration_auto")
    assert audit_files == [1, 2]


def test_migration_is_idempotent_across_boots(tmp_path):
    """Once the rebuild has run, a second boot must be a no-op:
    no duplicate audit rows, no row count changes."""
    db = _db_path(tmp_path)
    _seed_legacy_db(db)

    Store(db)
    conn = sqlite3.connect(db)
    try:
        n_state_first = conn.execute(
            "SELECT COUNT(*) FROM review_state"
        ).fetchone()[0]
        n_audit_first = conn.execute(
            "SELECT COUNT(*) FROM review_event_log "
            "WHERE action='pi_migration_auto'"
        ).fetchone()[0]
    finally:
        conn.close()

    # Second boot must not rerun the rebuild.
    Store(db)
    conn = sqlite3.connect(db)
    try:
        n_state_second = conn.execute(
            "SELECT COUNT(*) FROM review_state"
        ).fetchone()[0]
        n_audit_second = conn.execute(
            "SELECT COUNT(*) FROM review_event_log "
            "WHERE action='pi_migration_auto'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert n_state_first == n_state_second
    assert n_audit_first == n_audit_second


# ===================================================================== #
#  Migration: stuck mass_analyze_job rows
# ===================================================================== #

def test_stuck_mass_analyze_jobs_are_reaped(tmp_path):
    db = _db_path(tmp_path)
    Store(db)  # populate schema
    # Inject pretend stuck rows and a pre-completed one.
    conn = sqlite3.connect(db)
    try:
        conn.executescript("""
            INSERT INTO mass_analyze_job
              (pi_email, animal_id, cutoff, status, created_at)
            VALUES
              ('pi@lab', 'A1', 1.0, 'pending', '2026-01-01T00:00:00'),
              ('pi@lab', 'A2', 1.0, 'running', '2026-01-01T00:00:00'),
              ('pi@lab', 'A3', 1.0, 'done',    '2026-01-01T00:00:00');
        """)
        conn.commit()
    finally:
        conn.close()

    # Reboot: the unconditional sweep should flip pending+running to
    # failed but leave 'done' alone.
    Store(db)
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT animal_id, status, error FROM mass_analyze_job "
            "ORDER BY animal_id"
        ).fetchall()
    finally:
        conn.close()
    by_animal = {a: (s, e) for (a, s, e) in rows}
    assert by_animal["A1"][0] == "failed"
    assert by_animal["A1"][1] == "worker restart"
    assert by_animal["A2"][0] == "failed"
    assert by_animal["A2"][1] == "worker restart"
    assert by_animal["A3"][0] == "done"


# ===================================================================== #
#  Public connection() contextmanager
# ===================================================================== #

def test_connection_yields_a_usable_connection(tmp_path):
    s = Store(_db_path(tmp_path))
    with s.connection() as conn:
        # Row factory should be installed so existing callsites that
        # do row["col"] still work.
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "LIMIT 1"
        ).fetchall()
    assert rows  # non-empty since the schema was just run
    assert rows[0]["name"]  # dict-style access via Row factory


def test_connection_closes_on_exception(tmp_path):
    s = Store(_db_path(tmp_path))
    with pytest.raises(RuntimeError):
        with s.connection() as conn:
            assert conn is not None
            raise RuntimeError("kaboom")
    # If the contextmanager leaked the connection, the next WAL
    # checkpoint might block. A trivial round-trip proves it didn't.
    with s.connection() as conn:
        conn.execute("SELECT 1").fetchone()


# ===================================================================== #
#  Manual entry point
# ===================================================================== #

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
