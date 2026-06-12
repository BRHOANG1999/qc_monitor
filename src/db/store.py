"""SQLite data access layer for the QC monitor — v2 with evoked, criticality, versioning."""

import hashlib
import json
import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from .schema import SCHEMA_SQL


class Store:
    # Soft-claim TTL: a reviewer who opens a file holds it (hidden from
    # other reviewers' queues) for this long. Generous vs. the ~40s
    # videos / few-minute reviews so a heartbeat isn't needed; if a
    # reviewer wanders off the claim simply expires and the file
    # returns to the pool.
    CLAIM_TTL_MINUTES = 20

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def connection(self):
        """Public contextmanager handle for the SQLite connection.

        Used by callers (dashboard tabs, notifications, background
        workers) that need to run ad-hoc reads/writes against tables
        the Store class doesn't yet expose a method for. Callers must
        still ``conn.commit()`` explicitly for writes -- this matches
        the connect-and-commit-by-hand pattern used inside Store and
        keeps the sweep from this commit mechanical. The connection
        is closed on exit regardless of whether the block raised.
        """
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        conn = self._connect()
        conn.executescript(SCHEMA_SQL)
        # Idempotent forward-compat migrations: add columns to old DBs
        # that pre-date the audit-trail work.
        existing_ann_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(annotations)")
        }
        if "user_email" not in existing_ann_cols:
            conn.execute("ALTER TABLE annotations ADD COLUMN user_email TEXT")
        # PI verification rollout migration: widen the review_state
        # status CHECK + promote pre-existing finalised rows to
        # 'pi_approved' (their CSV row was already written under the
        # pre-PI flow). SQLite can't ALTER a CHECK in place, so we
        # detect-and-rebuild only when the old constraint is on disk.
        self._migrate_review_state_pi_statuses(conn)
        # Reap mass_analyze_job rows stuck in 'running' across a
        # restart -- without this they'd never re-progress because
        # the worker that started them no longer exists. Safe to
        # run unconditionally; idempotent on a clean shutdown.
        try:
            conn.execute(
                """UPDATE mass_analyze_job
                   SET status='failed', error='worker restart',
                       finished_at=datetime('now')
                   WHERE status IN ('pending', 'running')"""
            )
        except Exception:
            # Table doesn't exist yet on a brand-new DB; fine.
            pass
        conn.commit()
        conn.close()

    def _migrate_review_state_pi_statuses(self, conn) -> None:
        """Promote pre-existing review_state rows to the PI flow.

        Detects the old CHECK constraint via sqlite_master.sql and
        rebuilds the table only when it's the pre-PI shape. Idempotent
        across boots; once migrated the rebuild is skipped.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='review_state'"
        ).fetchone()
        if not row:
            return
        current_sql = (row["sql"] or "").lower()
        if "pi_approved" in current_sql:
            return  # already migrated
        # Rebuild with the wider CHECK + promote in one go.
        conn.executescript(
            """
            CREATE TABLE review_state_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL
                    REFERENCES processed_files(id),
                user_email TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK(status IN ('claimed', 'no_events',
                                      'has_events', 'abandoned',
                                      'pending_pi_review',
                                      'pi_approved',
                                      'pi_flagged')),
                markers_json TEXT,
                note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            -- Promote: every row whose pre-PI status was a finalised
            -- one is treated as already-PI-approved (the CSV row was
            -- written under the old _save_review path). Claimed +
            -- abandoned stay where they are.
            INSERT INTO review_state_new
              (id, file_id, user_email, status, markers_json, note,
               created_at, updated_at)
            SELECT id, file_id, user_email,
                   CASE WHEN status IN ('no_events', 'has_events')
                        THEN 'pi_approved'
                        ELSE status END,
                   markers_json, note, created_at, updated_at
              FROM review_state;
            DROP TABLE review_state;
            ALTER TABLE review_state_new RENAME TO review_state;
            CREATE INDEX IF NOT EXISTS idx_review_state_file
                ON review_state(file_id);
            CREATE INDEX IF NOT EXISTS idx_review_state_user
                ON review_state(user_email);
            CREATE INDEX IF NOT EXISTS idx_review_state_status
                ON review_state(status);
            INSERT INTO review_event_log
              (file_id, user_email, action, payload_json, at)
            SELECT file_id, user_email, 'pi_migration_auto',
                   '{"reason": "pre-PI promote to pi_approved"}',
                   datetime('now')
              FROM review_state
              WHERE status = 'pi_approved';
            """
        )

    # ------------------------------------------------------------------ #
    #  settings_versions
    # ------------------------------------------------------------------ #

    def create_settings_version(self, config_dict: dict, label: str = "") -> int:
        """Compute SHA-256 of the canonical JSON and insert a new version row.

        If a row with the same hash already exists the existing id is returned
        without inserting a duplicate.
        """
        canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
        version_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO settings_versions
                   (version_hash, settings_json, label, created_at, is_active)
                   VALUES (?, ?, ?, ?, 1)""",
                (version_hash, canonical, label, datetime.now().isoformat()),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT id FROM settings_versions WHERE version_hash = ?",
                    (version_hash,),
                ).fetchone()
                vid = row["id"]
            else:
                vid = cursor.lastrowid
            conn.commit()
            return vid
        finally:
            conn.close()

    def insert_settings_version(self, version_hash: str, settings_json: str,
                                label: str = "", is_active: int = 1) -> int:
        """Insert a pre-computed settings version row."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO settings_versions
                   (version_hash, settings_json, label, created_at, is_active)
                   VALUES (?, ?, ?, ?, ?)""",
                (version_hash, settings_json, label, datetime.now().isoformat(), is_active),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT id FROM settings_versions WHERE version_hash = ?",
                    (version_hash,),
                ).fetchone()
                vid = row["id"]
            else:
                vid = cursor.lastrowid
            conn.commit()
            return vid
        finally:
            conn.close()

    def get_active_settings_version(self) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM settings_versions WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_all_settings_versions(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM settings_versions ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def set_active_settings_version(self, version_id: int):
        """Mark *version_id* as the sole active version."""
        conn = self._connect()
        try:
            conn.execute("UPDATE settings_versions SET is_active = 0")
            conn.execute("UPDATE settings_versions SET is_active = 1 WHERE id = ?", (version_id,))
            conn.commit()
        finally:
            conn.close()

    def get_settings_version_by_hash(self, version_hash: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM settings_versions WHERE version_hash = ?", (version_hash,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  session_config
    # ------------------------------------------------------------------ #

    def upsert_session_config(self, session_dir: str, config: dict) -> int:
        """Insert or update auto-discovered session configuration."""
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id FROM session_config WHERE session_dir = ?", (session_dir,)
            ).fetchone()

            if existing:
                sets, vals = [], []
                for col in (
                    "channel_names", "eeg_channels", "stim_copy_channels",
                    "reference_channels", "stim_frequency_hz", "stim_charge_nC",
                    "stim_pulse_width_us", "stim_neg_ratio", "stim_active_outputs",
                    "sampling_rate", "num_channels", "spike_threshold",
                    "power_threshold", "coastline_threshold",
                ):
                    if col in config:
                        sets.append(f"{col} = ?")
                        val = config[col]
                        # store lists/dicts as JSON text
                        if isinstance(val, (list, dict)):
                            val = json.dumps(val)
                        vals.append(val)
                if sets:
                    vals.append(session_dir)
                    conn.execute(
                        f"UPDATE session_config SET {', '.join(sets)} WHERE session_dir = ?",
                        vals,
                    )
                conn.commit()
                return existing["id"]
            else:
                def _v(key):
                    val = config.get(key)
                    if isinstance(val, (list, dict)):
                        return json.dumps(val)
                    return val

                cursor = conn.execute(
                    """INSERT INTO session_config
                       (session_dir, channel_names, eeg_channels, stim_copy_channels,
                        reference_channels, stim_frequency_hz, stim_charge_nC,
                        stim_pulse_width_us, stim_neg_ratio, stim_active_outputs,
                        sampling_rate, num_channels, spike_threshold,
                        power_threshold, coastline_threshold, discovered_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        session_dir,
                        _v("channel_names"), _v("eeg_channels"),
                        _v("stim_copy_channels"), _v("reference_channels"),
                        _v("stim_frequency_hz"), _v("stim_charge_nC"),
                        _v("stim_pulse_width_us"), _v("stim_neg_ratio"),
                        _v("stim_active_outputs"), _v("sampling_rate"),
                        _v("num_channels"), _v("spike_threshold"),
                        _v("power_threshold"), _v("coastline_threshold"),
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
                return cursor.lastrowid
        finally:
            conn.close()

    def get_session_config(self, session_dir: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM session_config WHERE session_dir = ?", (session_dir,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_all_session_configs(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM session_config ORDER BY discovered_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  processed_files
    # ------------------------------------------------------------------ #

    def is_processed(self, file_path: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT status FROM processed_files WHERE file_path = ?", (file_path,)
            ).fetchone()
            return row is not None and row["status"] == "done"
        finally:
            conn.close()

    def get_file_status(self, file_path: str) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT status FROM processed_files WHERE file_path = ?", (file_path,)
            ).fetchone()
            return row["status"] if row else None
        finally:
            conn.close()

    def register_file(self, file_path: str, file_size: int, file_mtime: float,
                      session_dir: str, session_name: str, chunk_datetime: str,
                      version_id: int | None = None) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO processed_files
                   (file_path, file_size, file_mtime, session_dir, session_name,
                    chunk_datetime, status, version_id)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (file_path, file_size, file_mtime, session_dir, session_name,
                 chunk_datetime, version_id),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT id FROM processed_files WHERE file_path = ?", (file_path,)
                ).fetchone()
                file_id = row["id"]
            else:
                file_id = cursor.lastrowid
            conn.commit()
            return file_id
        finally:
            conn.close()

    def update_file_status(self, file_id: int, status: str, **kwargs):
        conn = self._connect()
        try:
            sets = ["status = ?", "processed_at = ?"]
            vals = [status, datetime.now().isoformat()]
            for k, v in kwargs.items():
                sets.append(f"{k} = ?")
                vals.append(v)
            vals.append(file_id)
            conn.execute(
                f"UPDATE processed_files SET {', '.join(sets)} WHERE id = ?", vals
            )
            conn.commit()
        finally:
            conn.close()

    def reset_stale_processing(self):
        """Reset files stuck in 'processing' (crashed) back to 'pending'."""
        conn = self._connect()
        try:
            conn.execute("UPDATE processed_files SET status = 'pending' WHERE status = 'processing'")
            conn.commit()
        finally:
            conn.close()

    def get_pending_files(self, limit: int = 50) -> list[dict]:
        """Pending files, newest chunk_datetime first.

        LIFO on chunk_datetime so the dispatcher always crunches the
        most recent recordings first -- the Overview "Latest Evoked"
        thumbnail tracks the live experiment instead of waiting for
        a multi-day backfill to drain. Backfill still runs, just
        from the present moving back rather than from the past
        moving forward.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM processed_files WHERE status = 'pending' "
                "ORDER BY chunk_datetime DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  chunk_qc
    # ------------------------------------------------------------------ #

    def insert_chunk_qc(self, file_id: int, channel: int, metrics: dict,
                        channel_name: str | None = None,
                        channel_role: str | None = None,
                        version_id: int | None = None):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO chunk_qc
                   (file_id, channel, channel_name, channel_role,
                    rms_amplitude, line_noise_power, line_noise_ratio,
                    artifact_pct, has_nan, has_inf, has_clipping, flat_line_pct, dc_offset,
                    delta_power, theta_power, alpha_power, beta_power, gamma_power,
                    high_gamma_power, total_power, computed_at, version_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_id, channel, channel_name, channel_role,
                    metrics.get("rms_amplitude"),
                    metrics.get("line_noise_power"),
                    metrics.get("line_noise_ratio"),
                    metrics.get("artifact_pct"),
                    metrics.get("has_nan", 0),
                    metrics.get("has_inf", 0),
                    metrics.get("has_clipping", 0),
                    metrics.get("flat_line_pct"),
                    metrics.get("dc_offset"),
                    metrics.get("delta_power"),
                    metrics.get("theta_power"),
                    metrics.get("alpha_power"),
                    metrics.get("beta_power"),
                    metrics.get("gamma_power"),
                    metrics.get("high_gamma_power"),
                    metrics.get("total_power"),
                    datetime.now().isoformat(),
                    version_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  stim_qc
    # ------------------------------------------------------------------ #

    def insert_stim_qc(self, file_id: int, stim_channel: int, metrics: dict):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO stim_qc
                   (file_id, stim_channel, charge_nC, pulse_width_us, neg_pulse_ratio,
                    frequency_hz, gain, total_pulses, total_charge_nC, duration_sec,
                    expected_pulses, delivery_pct, computed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_id, stim_channel,
                    metrics.get("charge_nC"),
                    metrics.get("pulse_width_us"),
                    metrics.get("neg_pulse_ratio"),
                    metrics.get("frequency_hz"),
                    metrics.get("gain"),
                    metrics.get("total_pulses"),
                    metrics.get("total_charge_nC"),
                    metrics.get("duration_sec"),
                    metrics.get("expected_pulses"),
                    metrics.get("delivery_pct"),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  evoked_features
    # ------------------------------------------------------------------ #

    _EVOKED_FEATURE_COLS = (
        "line_length", "log_auc", "peak_amplitude", "trough_amplitude",
        "peak_to_trough", "rms_amplitude", "peak_latency_ms", "trough_latency_ms",
        "max_slope", "max_slope_time_ms", "early_area", "late_area",
        "early_late_ratio", "recovery_tau", "recovery_slope",
        "template_correlation", "pca_recon_error", "variance",
        "autocorrelation", "ac_width", "exp_fit_a", "sum_power_low",
        "freq_moment_low", "sum_power_high", "freq_moment_high",
        "is_artifact", "is_ictal",
    )

    def bulk_insert_evoked_features(self, file_id: int, epochs: list[dict],
                                    version_id: int | None = None):
        """Insert per-epoch evoked feature rows from MATLAB JSON output.

        *epochs* is a list of dicts, each containing at minimum
        ``epoch_index`` and any subset of the feature columns.
        """
        conn = self._connect()
        try:
            cols = ("file_id", "epoch_index", "epoch_time_sec") + self._EVOKED_FEATURE_COLS + ("version_id",)
            placeholders = ", ".join("?" for _ in cols)
            col_str = ", ".join(cols)
            rows = []
            for ep in epochs:
                row = (
                    file_id,
                    ep["epoch_index"],
                    ep.get("epoch_time_sec"),
                    *[ep.get(c) for c in self._EVOKED_FEATURE_COLS],
                    version_id,
                )
                rows.append(row)
            conn.executemany(
                f"INSERT OR REPLACE INTO evoked_features ({col_str}) VALUES ({placeholders})",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def get_evoked_feature_timeseries(self, feature_name: str, session_dir: str,
                                      version_id: int | None = None,
                                      hours: int | None = None) -> list[dict]:
        """Return epoch-level values of a single feature over time.

        This is the primary query for the Evoked Response dashboard.
        Results are ordered by chunk_datetime then epoch_index so they
        form a continuous time series.
        """
        if feature_name not in self._EVOKED_FEATURE_COLS and feature_name != "epoch_time_sec":
            raise ValueError(f"Unknown evoked feature: {feature_name}")

        conn = self._connect()
        try:
            conditions = ["pf.session_dir = ?"]
            params: list = [session_dir]

            if version_id is not None:
                conditions.append("ef.version_id = ?")
                params.append(version_id)

            if hours:
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                conditions.append("pf.chunk_datetime > ?")
                params.append(cutoff)

            where = " AND ".join(conditions)
            sql = f"""
                SELECT pf.chunk_datetime,
                       ef.epoch_index,
                       ef.epoch_time_sec,
                       ef.{feature_name} AS value,
                       ef.is_artifact,
                       ef.is_ictal
                FROM evoked_features ef
                JOIN processed_files pf ON ef.file_id = pf.id
                WHERE {where}
                ORDER BY pf.chunk_datetime, ef.epoch_index
            """
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def query_evoked_features(self, file_id: int,
                              version_id: int | None = None) -> list[dict]:
        """Return all evoked feature rows for a given file."""
        conn = self._connect()
        try:
            if version_id is not None:
                rows = conn.execute(
                    "SELECT * FROM evoked_features WHERE file_id = ? AND version_id = ? ORDER BY epoch_index",
                    (file_id, version_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM evoked_features WHERE file_id = ? ORDER BY epoch_index",
                    (file_id,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  evoked_summary
    # ------------------------------------------------------------------ #

    def insert_evoked_summary(self, file_id: int, summary: dict,
                              version_id: int | None = None):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO evoked_summary
                   (file_id, evoked_output_path, num_stimuli_detected,
                    num_traces_extracted, num_artifact_epochs, num_ictal_epochs,
                    mean_peak_amplitude, std_peak_amplitude, mean_trough_amplitude,
                    mean_peak_latency_ms, mean_line_length, mean_recovery_tau,
                    mean_template_correlation, pipeline_duration_sec, version_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    file_id,
                    summary.get("evoked_output_path"),
                    summary.get("num_stimuli_detected"),
                    summary.get("num_traces_extracted"),
                    summary.get("num_artifact_epochs"),
                    summary.get("num_ictal_epochs"),
                    summary.get("mean_peak_amplitude"),
                    summary.get("std_peak_amplitude"),
                    summary.get("mean_trough_amplitude"),
                    summary.get("mean_peak_latency_ms"),
                    summary.get("mean_line_length"),
                    summary.get("mean_recovery_tau"),
                    summary.get("mean_template_correlation"),
                    summary.get("pipeline_duration_sec"),
                    version_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def query_evoked_summary(self, session_dir: str | None = None,
                             version_id: int | None = None,
                             hours: int | None = None) -> list[dict]:
        conn = self._connect()
        try:
            conditions: list[str] = []
            params: list = []

            if session_dir is not None:
                conditions.append("pf.session_dir = ?")
                params.append(session_dir)
            if version_id is not None:
                conditions.append("es.version_id = ?")
                params.append(version_id)
            if hours is not None:
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                conditions.append("pf.chunk_datetime > ?")
                params.append(cutoff)

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            sql = f"""
                SELECT pf.chunk_datetime, pf.session_dir, es.*
                FROM evoked_summary es
                JOIN processed_files pf ON es.file_id = pf.id
                {where}
                ORDER BY pf.chunk_datetime
            """
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  criticality
    # ------------------------------------------------------------------ #

    def bulk_insert_criticality(self, file_id: int, windows: list[dict],
                                version_id: int | None = None):
        """Insert per-window criticality measurements.

        *windows* is a list of dicts each containing at minimum
        ``channel`` and ``window_index``.
        """
        conn = self._connect()
        try:
            rows = []
            for w in windows:
                rows.append((
                    file_id,
                    w["channel"],
                    w["window_index"],
                    w.get("time_sec"),
                    w.get("db_value"),
                    w.get("db_std"),
                    w.get("sigma"),
                    w.get("ar_order"),
                    w.get("exit_status"),
                    version_id,
                ))
            conn.executemany(
                """INSERT OR REPLACE INTO criticality
                   (file_id, channel, window_index, time_sec, db_value, db_std,
                    sigma, ar_order, exit_status, version_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def query_criticality_timeseries(self, session_dir: str,
                                     channel: int | None = None,
                                     version_id: int | None = None,
                                     hours: int | None = None) -> list[dict]:
        """Return criticality (dB) values over time for a session."""
        conn = self._connect()
        try:
            conditions = ["pf.session_dir = ?"]
            params: list = [session_dir]

            if channel is not None:
                conditions.append("c.channel = ?")
                params.append(channel)
            if version_id is not None:
                conditions.append("c.version_id = ?")
                params.append(version_id)
            if hours:
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                conditions.append("pf.chunk_datetime > ?")
                params.append(cutoff)

            where = " AND ".join(conditions)
            sql = f"""
                SELECT pf.chunk_datetime, c.*
                FROM criticality c
                JOIN processed_files pf ON c.file_id = pf.id
                WHERE {where}
                ORDER BY pf.chunk_datetime, c.window_index
            """
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  seizure_events
    # ------------------------------------------------------------------ #

    def insert_seizure_event(self, file_id: int, event: dict,
                             version_id: int | None = None) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO seizure_events
                   (file_id, channel, channel_name, onset_sec, offset_sec,
                    duration_sec, spike_count, peak_amplitude, spike_rate_hz,
                    severity, version_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    file_id,
                    event.get("channel"),
                    event.get("channel_name"),
                    event.get("onset_sec"),
                    event.get("offset_sec"),
                    event.get("duration_sec"),
                    event.get("spike_count"),
                    event.get("peak_amplitude"),
                    event.get("spike_rate_hz"),
                    event.get("severity"),
                    version_id,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def query_seizure_events(self, session_dir: str,
                             version_id: int | None = None,
                             hours: int | None = None) -> list[dict]:
        conn = self._connect()
        try:
            conditions = ["pf.session_dir = ?"]
            params: list = [session_dir]

            if version_id is not None:
                conditions.append("se.version_id = ?")
                params.append(version_id)
            if hours is not None:
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                conditions.append("pf.chunk_datetime > ?")
                params.append(cutoff)

            where = " AND ".join(conditions)
            sql = f"""
                SELECT pf.chunk_datetime, pf.session_dir, se.*
                FROM seizure_events se
                JOIN processed_files pf ON se.file_id = pf.id
                WHERE {where}
                ORDER BY pf.chunk_datetime, se.onset_sec
            """
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  matlab_results
    # ------------------------------------------------------------------ #

    def insert_video_qc(self, file_id: int, result: dict,
                         version_id: int | None = None) -> int:
        """Persist a video_qc.VideoQCResult dict. Returns the new row id."""
        import json
        conn = self._connect()
        try:
            cur = conn.execute(
                """INSERT INTO video_qc
                   (file_id, version_id, video_path, analyzed_at, fps,
                    n_frames_total, n_frames_sampled, sample_every_n,
                    duration_sec, duration_analyzed_sec,
                    mean_global, var_global,
                    mean_min, mean_max, var_min, var_max,
                    n_dark, n_bright, n_low_var, n_bad, pct_bad,
                    first_bad_idx, first_bad_reason,
                    status, error, thresholds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_id, version_id,
                    result.get("video_path"),
                    datetime.now().isoformat(),
                    result.get("fps"),
                    result.get("n_frames_total"),
                    result.get("n_frames_sampled"),
                    result.get("sample_every_n"),
                    result.get("duration_sec"),
                    result.get("duration_analyzed_sec"),
                    result.get("mean_global"),
                    result.get("var_global"),
                    result.get("mean_min"),
                    result.get("mean_max"),
                    result.get("var_min"),
                    result.get("var_max"),
                    result.get("n_dark"),
                    result.get("n_bright"),
                    result.get("n_low_var"),
                    result.get("n_bad"),
                    result.get("pct_bad"),
                    result.get("first_bad_idx"),
                    result.get("first_bad_reason"),
                    result.get("status"),
                    result.get("error"),
                    json.dumps(result.get("thresholds") or {}),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def get_recent_video_qc(self, hours: int = 24) -> list[dict]:
        """Return video_qc rows analyzed within the last *hours* hours.

        Used by the Overview compact status card and the weekly video
        QC digest. Newest first.
        """
        assert hours > 0, "hours must be positive"
        conn = self._connect()
        try:
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            rows = conn.execute(
                """SELECT id, file_id, video_path, analyzed_at, status,
                          pct_bad, n_dark, n_bright, n_low_var, n_bad,
                          first_bad_reason
                   FROM video_qc
                   WHERE analyzed_at > ?
                   ORDER BY analyzed_at DESC""",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  matlab_results
    # ------------------------------------------------------------------ #

    def insert_matlab_result(self, file_id: int, result: dict,
                             version_id: int | None = None):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO matlab_results
                   (file_id, evoked_output_path, num_stimuli, num_traces,
                    criticality_mean, criticality_std, num_features,
                    pipeline_duration_sec, exit_status, error_message,
                    computed_at, version_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_id,
                    result.get("evoked_output_path"),
                    result.get("num_stimuli"),
                    result.get("num_traces"),
                    result.get("criticality_mean"),
                    result.get("criticality_std"),
                    result.get("num_features"),
                    result.get("pipeline_duration_sec"),
                    result.get("exit_status"),
                    result.get("error_message"),
                    datetime.now().isoformat(),
                    version_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  alerts
    # ------------------------------------------------------------------ #

    def insert_alert(self, alert_type: str, severity: str, message: str,
                     file_id: int = None, session_dir: str = None):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO alerts (alert_type, severity, message, file_id, session_dir, sent_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (alert_type, severity, message, file_id, session_dir, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_last_alert_time(self, alert_type: str) -> datetime | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT sent_at FROM alerts WHERE alert_type = ? ORDER BY sent_at DESC LIMIT 1",
                (alert_type,),
            ).fetchone()
            if row:
                return datetime.fromisoformat(row["sent_at"])
            return None
        finally:
            conn.close()

    def get_recent_alerts(self, hours: int = 24) -> list[dict]:
        conn = self._connect()
        try:
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            rows = conn.execute(
                "SELECT * FROM alerts WHERE sent_at > ? ORDER BY sent_at DESC", (cutoff,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  system_health
    # ------------------------------------------------------------------ #

    def insert_health(self, metrics: dict):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO system_health
                   (timestamp, cpu_pct, memory_pct, disk_free_gb,
                    network_share_accessible, queue_depth, files_processed_last_hour)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    metrics.get("cpu_pct"),
                    metrics.get("memory_pct"),
                    metrics.get("disk_free_gb"),
                    metrics.get("network_share_accessible"),
                    metrics.get("queue_depth"),
                    metrics.get("files_processed_last_hour"),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  compare_versions
    # ------------------------------------------------------------------ #

    def compare_versions(self, version_id_a: int, version_id_b: int,
                         feature_name: str, session_dir: str) -> list[dict]:
        """Return both versions' feature values aligned by epoch for comparison plots.

        Each result row contains:
          chunk_datetime, epoch_index, epoch_time_sec, value_a, value_b
        Rows are aligned on (file_id, epoch_index) so only epochs present
        in both versions appear.
        """
        if feature_name not in self._EVOKED_FEATURE_COLS and feature_name != "epoch_time_sec":
            raise ValueError(f"Unknown evoked feature: {feature_name}")

        conn = self._connect()
        try:
            sql = f"""
                SELECT pf.chunk_datetime,
                       a.epoch_index,
                       a.epoch_time_sec,
                       a.{feature_name} AS value_a,
                       b.{feature_name} AS value_b
                FROM evoked_features a
                JOIN evoked_features b
                    ON a.file_id = b.file_id AND a.epoch_index = b.epoch_index
                JOIN processed_files pf ON a.file_id = pf.id
                WHERE a.version_id = ?
                  AND b.version_id = ?
                  AND pf.session_dir = ?
                ORDER BY pf.chunk_datetime, a.epoch_index
            """
            rows = conn.execute(sql, (version_id_a, version_id_b, session_dir)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  dashboard query helpers
    # ------------------------------------------------------------------ #

    def get_qc_timeseries(self, session_dir: str = None, hours: int = 24,
                          channel: int | None = None,
                          version_id: int | None = None) -> list[dict]:
        conn = self._connect()
        try:
            conditions: list[str] = []
            params: list = []

            if hours:
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                conditions.append("pf.chunk_datetime > ?")
                params.append(cutoff)

            if session_dir:
                conditions.append("pf.session_dir = ?")
                params.append(session_dir)
            if channel is not None:
                conditions.append("cq.channel = ?")
                params.append(channel)
            if version_id is not None:
                conditions.append("cq.version_id = ?")
                params.append(version_id)

            where = (" AND ".join(conditions)) if conditions else "1=1"
            rows = conn.execute(
                f"""SELECT pf.chunk_datetime, cq.*
                    FROM chunk_qc cq JOIN processed_files pf ON cq.file_id = pf.id
                    WHERE {where}
                    ORDER BY pf.chunk_datetime""",
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_sessions(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT session_dir, session_name,
                          COUNT(*) as num_files,
                          MIN(chunk_datetime) as first_chunk,
                          MAX(chunk_datetime) as last_chunk,
                          SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as processed,
                          SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
                   FROM processed_files
                   GROUP BY session_dir
                   ORDER BY last_chunk DESC"""
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_health_history(self, hours: int = 24) -> list[dict]:
        conn = self._connect()
        try:
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            rows = conn.execute(
                "SELECT * FROM system_health WHERE timestamp > ? ORDER BY timestamp",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_files_processed_last_hour(self) -> int:
        conn = self._connect()
        try:
            cutoff = (datetime.now() - timedelta(hours=1)).isoformat()
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM processed_files WHERE processed_at > ? AND status = 'done'",
                (cutoff,),
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  evoked_waveforms
    # ------------------------------------------------------------------ #

    def insert_evoked_waveform(self, file_id: int, time_axis: list, mean_trace: list,
                               sem_trace: list = None, stim_mean_trace: list = None,
                               n_epochs: int = 0, channel: int = 0,
                               channel_name: str = "",
                               analysis_start_ms: float = 5.0,
                               analysis_end_ms: float = 50.0,
                               version_id: int | None = None):
        """Insert a mean evoked waveform for a file+channel (JSON-serialised arrays)."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO evoked_waveforms
                   (file_id, channel, channel_name, time_axis_ms, mean_trace,
                    sem_trace, stim_mean_trace, n_epochs,
                    analysis_start_ms, analysis_end_ms, version_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_id, channel, channel_name,
                    json.dumps(list(time_axis)),
                    json.dumps(list(mean_trace)),
                    json.dumps(list(sem_trace)) if sem_trace else None,
                    json.dumps(list(stim_mean_trace)) if stim_mean_trace else None,
                    n_epochs,
                    analysis_start_ms,
                    analysis_end_ms,
                    version_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_evoked_waveform(self, file_id: int,
                            version_id: int | None = None) -> dict | None:
        """Return the mean evoked waveform for a file with parsed JSON arrays."""
        conn = self._connect()
        try:
            if version_id is not None:
                row = conn.execute(
                    "SELECT * FROM evoked_waveforms WHERE file_id = ? AND version_id = ?",
                    (file_id, version_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM evoked_waveforms WHERE file_id = ? ORDER BY id DESC LIMIT 1",
                    (file_id,),
                ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["time_axis_ms"] = json.loads(d["time_axis_ms"])
            d["mean_trace"] = json.loads(d["mean_trace"])
            if d.get("sem_trace") is not None:
                d["sem_trace"] = json.loads(d["sem_trace"])
            if d.get("stim_mean_trace") is not None:
                d["stim_mean_trace"] = json.loads(d["stim_mean_trace"])
            return d
        finally:
            conn.close()

    def get_evoked_waveform_by_file(self, file_id: int,
                                      version_id: int | None = None) -> list[dict]:
        """Return ALL channel waveforms for a single file."""
        conn = self._connect()
        try:
            if version_id is not None:
                rows = conn.execute(
                    "SELECT * FROM evoked_waveforms WHERE file_id = ? AND version_id = ? ORDER BY channel",
                    (file_id, version_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM evoked_waveforms WHERE file_id = ? ORDER BY channel",
                    (file_id,),
                ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["time_axis_ms"] = json.loads(d["time_axis_ms"])
                d["mean_trace"] = json.loads(d["mean_trace"])
                if d.get("sem_trace") is not None:
                    d["sem_trace"] = json.loads(d["sem_trace"])
                if d.get("stim_mean_trace") is not None:
                    d["stim_mean_trace"] = json.loads(d["stim_mean_trace"])
                results.append(d)
            return results
        finally:
            conn.close()

    def get_evoked_waveforms_for_session(self, session_dir: str,
                                         version_id: int | None = None) -> list[dict]:
        """Return all evoked waveforms for a session, with parsed JSON arrays."""
        conn = self._connect()
        try:
            conditions = ["pf.session_dir = ?"]
            params: list = [session_dir]

            if version_id is not None:
                conditions.append("ew.version_id = ?")
                params.append(version_id)

            where = " AND ".join(conditions)
            sql = f"""
                SELECT pf.chunk_datetime, pf.session_dir, ew.*
                FROM evoked_waveforms ew
                JOIN processed_files pf ON ew.file_id = pf.id
                WHERE {where}
                ORDER BY pf.chunk_datetime
            """
            rows = conn.execute(sql, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["time_axis_ms"] = json.loads(d["time_axis_ms"])
                d["mean_trace"] = json.loads(d["mean_trace"])
                if d.get("sem_trace") is not None:
                    d["sem_trace"] = json.loads(d["sem_trace"])
                if d.get("stim_mean_trace") is not None:
                    d["stim_mean_trace"] = json.loads(d["stim_mean_trace"])
                results.append(d)
            return results
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  processing_log
    # ------------------------------------------------------------------ #

    def log_activity(self, level: str, action: str, message: str,
                     file_id: int | None = None, file_path: str | None = None,
                     duration_sec: float | None = None):
        """Write an entry to the processing_log table."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO processing_log
                   (timestamp, level, file_id, file_path, action, message, duration_sec)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    level,
                    file_id,
                    file_path,
                    action,
                    message,
                    duration_sec,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_activity_log(self, hours: int = 24, level: str | None = None,
                         limit: int = 500) -> list[dict]:
        """Return recent processing log entries.

        *hours* controls the lookback window.  If *hours* is 0, no time
        filter is applied (returns all entries).
        """
        conn = self._connect()
        try:
            conditions: list[str] = []
            params: list = []

            if hours:
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                conditions.append("timestamp > ?")
                params.append(cutoff)
            if level is not None:
                conditions.append("level = ?")
                params.append(level)

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)
            rows = conn.execute(
                f"SELECT * FROM processing_log {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  annotations
    # ------------------------------------------------------------------ #

    def add_annotation(self, timestamp: str, note: str,
                       category: str = "observation",
                       session_dir: str | None = None,
                       file_id: int | None = None,
                       user_email: str | None = None) -> int:
        """Insert a user annotation / note and return its id."""
        assert isinstance(timestamp, str), "timestamp must be a string"
        assert isinstance(note, str) and note.strip(), "note must be a non-empty string"
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO annotations
                   (timestamp, created_at, session_dir, file_id, note, category, user_email)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    timestamp,
                    datetime.now().isoformat(),
                    session_dir,
                    file_id,
                    note,
                    category,
                    user_email,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_annotations(self, session_dir: str | None = None,
                        hours: int = 0) -> list[dict]:
        """Return annotations, optionally filtered by session and time window.

        *hours* = 0 (the default) means return all annotations with no
        time cutoff.
        """
        conn = self._connect()
        try:
            conditions: list[str] = []
            params: list = []

            if session_dir is not None:
                conditions.append("session_dir = ?")
                params.append(session_dir)
            if hours:
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                conditions.append("timestamp > ?")
                params.append(cutoff)

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            rows = conn.execute(
                f"SELECT * FROM annotations {where} ORDER BY timestamp DESC",
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_annotation(self, annotation_id: int):
        """Delete a single annotation by its id."""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  users (identity supplied by Cloudflare Access)
    # ------------------------------------------------------------------ #

    def upsert_user(self, email: str, display_name: str | None = None) -> int:
        """Record/refresh a user; called from the auth hook on every request.

        Returns the user's row id. Updates last_seen_at on every call so the
        Users table doubles as an activity ledger.
        """
        assert isinstance(email, str) and "@" in email, "valid email required"
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO users (email, display_name, role, first_seen_at, last_seen_at)
                   VALUES (?, ?, 'reviewer', ?, ?)
                   ON CONFLICT(email) DO UPDATE SET
                       last_seen_at = excluded.last_seen_at,
                       display_name = COALESCE(users.display_name, excluded.display_name)""",
                (email, display_name, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()
            return row["id"] if row else cursor.lastrowid
        finally:
            conn.close()

    def get_user_role(self, email: str) -> str:
        """Return the user's role ('mentor' or 'reviewer'); 'reviewer' if unknown."""
        if not email:
            return "reviewer"
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT role FROM users WHERE email = ?", (email,)
            ).fetchone()
            return (row["role"] if row else "reviewer") or "reviewer"
        finally:
            conn.close()

    def set_user_role(self, email: str, role: str):
        """Promote/demote a user. Only used by an out-of-band admin action."""
        assert role in ("mentor", "reviewer"), "role must be 'mentor' or 'reviewer'"
        conn = self._connect()
        try:
            conn.execute("UPDATE users SET role = ? WHERE email = ?", (role, email))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  Review queue / state (Video Review tab)
    # ------------------------------------------------------------------ #

    # "Reviewed" means at least one final-state row exists for this
    # file. Claimed-only rows still count the file as unreviewed.
    _REVIEW_FINAL_STATUSES = ("no_events", "has_events")

    def get_review_state(self, file_id: int) -> dict | None:
        """Latest review_state row for *file_id*, any user."""
        assert isinstance(file_id, int), "file_id must be int"
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT * FROM review_state
                   WHERE file_id = ?
                   ORDER BY updated_at DESC LIMIT 1""",
                (file_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_review_state_by_user(self, file_id: int,
                                   user_email: str) -> dict | None:
        """Latest row for this (file, user) pair. Used by the UI to
        show whether the current viewer already reviewed this file."""
        assert isinstance(file_id, int), "file_id must be int"
        if not user_email:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT * FROM review_state
                   WHERE file_id = ? AND user_email = ?
                   ORDER BY updated_at DESC LIMIT 1""",
                (file_id, user_email.lower()),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def insert_review_event(self, file_id: int, user_email: str,
                              action: str,
                              payload: dict | None = None) -> int:
        """Write to the append-only review_event_log; returns row id.

        action ∈ {'claim', 'finish', 'abandon', 'reopen',
                   'backlog_zero'}; payload is JSON-serialised.
        """
        assert isinstance(file_id, int), "file_id must be int"
        assert user_email, "user_email required"
        assert action, "action required"
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            cur = conn.execute(
                """INSERT INTO review_event_log
                   (file_id, user_email, action, payload_json, at)
                   VALUES (?, ?, ?, ?, ?)""",
                (file_id, user_email.lower(), action,
                 json.dumps(payload or {}), now),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def mark_review(self, file_id: int, user_email: str,
                     status: str,
                     markers: list[dict] | None = None,
                     note: str | None = None) -> int:
        """Atomic: write review_state row + matching event-log entry.

        status ∈ {'claimed', 'no_events', 'has_events',
                   'abandoned', 'pending_pi_review',
                   'pi_approved', 'pi_flagged'}.

        Returns the review_state row id. Finalising statuses
        (no_events / has_events / pending_pi_review /
        pi_approved) emit a 'finish' log event; flagged emits
        'pi_flag'; abandoned emits 'abandon'; claiming emits
        'claim'.
        """
        assert isinstance(file_id, int), "file_id must be int"
        assert user_email, "user_email required"
        assert status in ("claimed", "no_events", "has_events",
                           "abandoned", "pending_pi_review",
                           "pi_approved", "pi_flagged"), \
            f"bad status {status!r}"
        now = datetime.now().isoformat()
        markers_json = json.dumps(markers or [])
        log_action = ("claim" if status == "claimed"
                       else "abandon" if status == "abandoned"
                       else "pi_flag" if status == "pi_flagged"
                       else "finish")
        conn = self._connect()
        try:
            cur = conn.execute(
                """INSERT INTO review_state
                   (file_id, user_email, status, markers_json,
                    note, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (file_id, user_email.lower(), status, markers_json,
                 note, now, now),
            )
            conn.execute(
                """INSERT INTO review_event_log
                   (file_id, user_email, action, payload_json, at)
                   VALUES (?, ?, ?, ?, ?)""",
                (file_id, user_email.lower(), log_action,
                 json.dumps({"status": status,
                              "n_markers": len(markers or [])}),
                 now),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Soft-claim: file_claim table (see schema.py). Keeps two reviewers
    # off the same recording without a hard lock.
    # ------------------------------------------------------------------ #

    def claim_cutoff_iso(self) -> str:
        """ISO timestamp marking the oldest still-valid claim. Claims
        with ``claimed_at`` >= this are 'fresh'; older ones have
        expired. Shared by the queue and the Mass Analyze pre-screen so
        both agree on what 'actively being reviewed' means."""
        cutoff = datetime.now() - timedelta(
            minutes=self.CLAIM_TTL_MINUTES)
        return cutoff.isoformat()

    def claim_file(self, file_id: int, user_email: str) -> None:
        """Mark *file_id* as being reviewed by *user_email* (UPSERT --
        one claim per file). Logs a 'claim' audit event only when this
        is a genuinely new claim (no prior row, or a different/expired
        prior claimer) so re-opening the same file doesn't spam
        review_event_log."""
        assert isinstance(file_id, int), "file_id must be int"
        assert user_email, "user_email required"
        email = user_email.lower()
        now = datetime.now().isoformat()
        cutoff = self.claim_cutoff_iso()
        conn = self._connect()
        try:
            prior = conn.execute(
                "SELECT user_email, claimed_at FROM file_claim "
                "WHERE file_id = ?",
                (file_id,),
            ).fetchone()
            is_new_claim = (
                prior is None
                or prior["user_email"] != email
                or (prior["claimed_at"] or "") < cutoff
            )
            conn.execute(
                """INSERT INTO file_claim
                   (file_id, user_email, claimed_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(file_id) DO UPDATE SET
                       user_email = excluded.user_email,
                       claimed_at = excluded.claimed_at""",
                (file_id, email, now),
            )
            if is_new_claim:
                conn.execute(
                    """INSERT INTO review_event_log
                       (file_id, user_email, action, payload_json, at)
                       VALUES (?, ?, 'claim', ?, ?)""",
                    (file_id, email, json.dumps({}), now),
                )
            conn.commit()
        finally:
            conn.close()

    def release_claim(self, file_id: int) -> None:
        """Drop the claim on *file_id* (called after a successful
        submit so the row doesn't linger). Best-effort; a missing row
        is fine."""
        assert isinstance(file_id, int), "file_id must be int"
        conn = self._connect()
        try:
            conn.execute(
                "DELETE FROM file_claim WHERE file_id = ?", (file_id,))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    #  Rejected BHZ candidate peaks (the pink triangles)
    # ------------------------------------------------------------------ #

    def reject_peak(self, file_id: int, channel: int,
                     peak_time_sec: float, user_email: str | None = None
                     ) -> None:
        """Mark a candidate peak at *peak_time_sec* as a false positive
        for (file, channel). Idempotent (UNIQUE)."""
        assert isinstance(file_id, int), "file_id must be int"
        assert isinstance(channel, int), "channel must be int"
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO rejected_peak
                   (file_id, channel, peak_time_sec, rejected_by,
                    rejected_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (file_id, channel, round(float(peak_time_sec), 3),
                 (user_email or "").lower() or None, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_rejected_peaks(self, file_id: int, channel: int
                            ) -> list[float]:
        """Rejected candidate times (seconds) for (file, channel)."""
        assert isinstance(file_id, int), "file_id must be int"
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT peak_time_sec FROM rejected_peak
                   WHERE file_id = ? AND channel = ?
                   ORDER BY peak_time_sec""",
                (file_id, int(channel)),
            ).fetchall()
        finally:
            conn.close()
        return [float(r["peak_time_sec"]) for r in rows]

    def unreject_last_peak(self, file_id: int, channel: int) -> bool:
        """Undo the most-recent rejection for (file, channel). Returns
        True if a row was removed."""
        assert isinstance(file_id, int), "file_id must be int"
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT id FROM rejected_peak
                   WHERE file_id = ? AND channel = ?
                   ORDER BY rejected_at DESC, id DESC LIMIT 1""",
                (file_id, int(channel)),
            ).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM rejected_peak WHERE id = ?",
                         (int(row["id"]),))
            conn.commit()
            return True
        finally:
            conn.close()

    @staticmethod
    def _eeg_channel_names_for_config(channel_names_json: str | None,
                                          eeg_channels_json: str | None
                                          ) -> list[tuple[int, str]]:
        """Return ``(channel_index, channel_name)`` pairs for every
        EEG channel in this session_config row.

        ``eeg_channels`` is a JSON list of channel indices into
        ``channel_names`` (the non-stim_copy electrodes). stim_copy
        + reference channels are intentionally NOT in eeg_channels
        and therefore never appear here, so the reviewer can't pick
        them as an LFP target.
        """
        if not channel_names_json or not eeg_channels_json:
            return []
        try:
            names = json.loads(channel_names_json)
            idxs = json.loads(eeg_channels_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(names, list) or not isinstance(idxs, list):
            return []
        out: list[tuple[int, str]] = []
        for i in idxs:
            try:
                ii = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= ii < len(names) and names[ii]:
                out.append((ii, str(names[ii])))
        return out

    @staticmethod
    def _animal_ids_for_config(channel_names_json: str | None,
                                 eeg_channels_json: str | None,
                                 pattern: str | None = None,
                                 ) -> list[str]:
        """Distinct animal IDs (without electrode suffix) for the
        EEG channels in this row. Skips non-animal labels like
        ``saline`` / ``test`` so the queue and backlog summaries
        only ever surface real animals.

        E.g. given channel_names=``["stimCopy", "BCH062SR",
        "stimCopy", "BCH062SLM"]`` and eeg_channels=``[1, 3]``,
        returns ``["BCH062"]`` (one entry — the two electrode
        locations collapse to a single animal).
        """
        # Local import: keeps schema-only consumers from pulling
        # the parser module.
        from src.utils.animal import (
            split_animal_electrode, is_animal_channel,
        )
        pairs = Store._eeg_channel_names_for_config(
            channel_names_json, eeg_channels_json,
        )
        seen: list[str] = []
        seen_set: set[str] = set()
        for _idx, name in pairs:
            if not is_animal_channel(name):
                continue
            animal, _ = split_animal_electrode(name, pattern)
            if animal and animal not in seen_set:
                seen.append(animal)
                seen_set.add(animal)
        return seen

    def electrodes_for_animal_in_session(
            self, session_dir: str, animal_id: str,
            pattern: str | None = None,
            ) -> list[dict]:
        """Return ``[{channel_index, channel_name, location}, ...]``
        for every EEG channel in *session_dir* whose parsed animal
        prefix matches *animal_id*. Used by the Video Review
        channel picker to default to the assigned animal's first
        electrode and let the user switch to another.
        """
        from src.utils.animal import (
            split_animal_electrode, is_animal_channel,
        )
        if not session_dir or not animal_id:
            return []
        cfg = self.get_session_config(session_dir) or {}
        pairs = self._eeg_channel_names_for_config(
            cfg.get("channel_names"), cfg.get("eeg_channels"),
        )
        out: list[dict] = []
        wanted = animal_id.strip()
        for idx, name in pairs:
            if not is_animal_channel(name):
                continue
            animal, location = split_animal_electrode(name, pattern)
            if animal == wanted:
                out.append({
                    "channel_index": idx,
                    "channel_name": name,
                    "location": location or "",
                })
        return out

    def get_review_queue(self, animal_ids: list[str],
                          user_email: str,
                          limit: int = 100,
                          since_iso: str | None = None
                          ) -> list[dict]:
        """Unreviewed files belonging to *any* of *animal_ids*, FIFO
        (oldest chunk first).

        The behavioral-review queue is intentionally FIFO so a
        reviewer always clears the backlog from the front — the
        oldest unfinished recording surfaces at the top no matter
        how many newer chunks have landed since. Pairs with the
        warn_age / crit_age badges in the UI: the operator sees the
        loudest-aged item every time.

        Rules:
        * File's session_config animal list (channel_names indexed
          by eeg_channels) must intersect *animal_ids*.
        * File must have a companion video (has_video = 1).
        * No row in review_state with status IN ('no_events',
          'has_events') for this file (regardless of user).
        * The current *user_email* hasn't already finalised it.
        * Optional *since_iso* filters chunk_datetime >= that ISO
          string (used to honour a backlog-zero floor).
        """
        if not animal_ids:
            return []
        assert isinstance(limit, int) and limit > 0, "limit must be positive int"
        # animal_ids are PREFIXES (no electrode suffix) — match any
        # quoted channel name that starts with the prefix. The trailing
        # wildcard catches BCH062SR + BCH062SLM under "BCH062" while
        # the leading quote keeps "BCH062" from also matching "BCH0620".
        # A precise Python check below the SQL trims accidental matches.
        like_clauses = " OR ".join(
            ["sc.channel_names LIKE ?"] * len(animal_ids)
        )
        like_args = [f'%"{a}%' for a in animal_ids]
        email_l = (user_email or "").lower()
        claim_cutoff = self.claim_cutoff_iso()
        extra_where = ""
        if since_iso:
            extra_where = " AND pf.chunk_datetime >= ?"
        # Params are appended in the EXACT order their ? placeholders
        # appear in the SQL below. (A prior version appended since_iso
        # before user_email, which transposed the two whenever a
        # since_iso floor was supplied.)
        params: list = list(like_args)        # {like_clauses}
        params.append(email_l)                # rs2.user_email = ?
        params.append(email_l)                # fc.user_email != ?
        params.append(claim_cutoff)           # fc.claimed_at >= ?
        if since_iso:
            params.append(since_iso)          # {extra_where}
        # Over-fetch a little so the post-filter (rejecting rows
        # whose decoded animal list doesn't actually intersect)
        # still has enough hits.
        params.append(limit * 3)              # LIMIT
        sql = f"""
            SELECT pf.id, pf.file_path, pf.session_dir,
                    pf.session_name, pf.chunk_datetime,
                    pf.duration_sec, pf.has_video,
                    sc.eeg_channels, sc.channel_names
            FROM processed_files pf
            JOIN session_config sc
              ON sc.session_dir = pf.session_dir
            WHERE pf.has_video = 1
              AND ({like_clauses})
              -- A file is OUT of the queue once it's been
              -- finalised by anyone (legacy no_events /
              -- has_events) OR submitted to the PI
              -- (pending_pi_review, pi_approved). pi_flagged
              -- INTENTIONALLY does NOT appear here: the PI's
              -- "send back for re-review" puts the file back
              -- in the undergrad's queue. The pi_flagged row's
              -- note is surfaced by the queue-card renderer.
              AND NOT EXISTS (
                SELECT 1 FROM review_state rs
                WHERE rs.file_id = pf.id
                  AND rs.status IN ('no_events', 'has_events',
                                     'pending_pi_review',
                                     'pi_approved')
              )
              AND NOT EXISTS (
                SELECT 1 FROM review_state rs2
                WHERE rs2.file_id = pf.id
                  AND rs2.user_email = ?
                  AND rs2.status IN ('no_events', 'has_events',
                                      'pending_pi_review',
                                      'pi_approved')
              )
              -- Soft-claim: hide a recording another reviewer is
              -- actively reviewing (claimed within the TTL). The
              -- reviewer's OWN claim stays visible so they can resume;
              -- expired claims (claimed_at < cutoff) don't hide it.
              AND NOT EXISTS (
                SELECT 1 FROM file_claim fc
                WHERE fc.file_id = pf.id
                  AND fc.user_email != ?
                  AND fc.claimed_at >= ?
              )
              {extra_where}
            ORDER BY pf.chunk_datetime ASC
            LIMIT ?
        """
        wanted = {a for a in animal_ids}
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        out: list[dict] = []
        for r in rows:
            r_animals = self._animal_ids_for_config(
                r["channel_names"], r["eeg_channels"])
            if wanted & set(r_animals):
                d = dict(r)
                d["animals"] = r_animals
                out.append(d)
            if len(out) >= limit:
                break
        return out

    def user_streak_sessions(self, user_email: str) -> int:
        """Consecutive review *sessions* with >=1 finish event.

        A 'session' is a calendar day; we count consecutive
        distinct review days walking backward from today,
        allowing up to a 2-day gap between days (the lab's
        cadence is bi-daily, so an off-day should not break
        the streak). Returns 0 if the most-recent finish is
        more than 2 days old.
        """
        assert user_email, "user_email required"
        from datetime import date
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT DISTINCT date(at) AS d
                   FROM review_event_log
                   WHERE user_email = ? AND action = 'finish'
                   ORDER BY d DESC LIMIT 60""",
                (user_email.lower(),),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return 0
        dates: list[date] = []
        max_iter = 60
        for i, r in enumerate(rows):
            assert i < max_iter, "row scan runaway"
            try:
                dates.append(date.fromisoformat(r["d"]))
            except (ValueError, TypeError):
                continue
        if not dates:
            return 0
        today = date.today()
        if (today - dates[0]).days > 2:
            return 0
        streak = 1
        for i in range(1, len(dates)):
            if (dates[i - 1] - dates[i]).days <= 2:
                streak += 1
            else:
                break
        return streak

    def user_finishes_today(self, user_email: str) -> int:
        """Count of ``finish`` events emitted today (lab local).

        Used by the queue progress strip ("Today: 12 reviewed").
        """
        assert user_email, "user_email required"
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT COUNT(*) AS n FROM review_event_log
                   WHERE user_email = ? AND action = 'finish'
                     AND date(at) = date('now', 'localtime')""",
                (user_email.lower(),),
            ).fetchone()
        finally:
            conn.close()
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------ #
    #  PI verification layer (Track A2 of the PI verification plan)
    # ------------------------------------------------------------------ #

    def pi_pending_files(self, *,
                           animal_id: str | None = None,
                           limit: int = 200) -> list[dict]:
        """Return files in ``pending_pi_review``, oldest first.

        Each row includes the deserialised events list so the PI
        tab can render without a second query. ``animal_id``
        narrows to a single animal prefix (matched against the
        session_config channel naming convention).
        """
        assert isinstance(limit, int) and limit > 0, "limit > 0"
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT rs.id AS state_id, rs.file_id,
                          rs.user_email, rs.status,
                          rs.markers_json, rs.note,
                          rs.created_at AS submitted_at,
                          pf.file_path, pf.session_dir,
                          pf.chunk_datetime, pf.duration_sec,
                          pf.sampling_rate
                   FROM review_state rs
                   JOIN processed_files pf
                     ON pf.id = rs.file_id
                   WHERE rs.status = 'pending_pi_review'
                   ORDER BY rs.created_at ASC, rs.id ASC
                   LIMIT ?""",
                (limit * 4,),  # over-fetch; animal filter prunes
            ).fetchall()
        finally:
            conn.close()
        out: list[dict] = []
        for r in rows:
            try:
                events = json.loads(r["markers_json"] or "[]")
            except json.JSONDecodeError:
                events = []
            if animal_id:
                names = self._channel_names_for_session(
                    r["session_dir"])
                if not self._session_has_animal(names, animal_id):
                    continue
            d = dict(r)
            d["events"] = events
            out.append(d)
            if len(out) >= limit:
                break
        return out

    def _channel_names_for_session(self, session_dir: str
                                      ) -> list[str]:
        """Helper for animal-prefix filtering."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT channel_names FROM session_config "
                "WHERE session_dir = ?", (session_dir,),
            ).fetchone()
        finally:
            conn.close()
        if not row or not row["channel_names"]:
            return []
        try:
            return list(json.loads(row["channel_names"]))
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _session_has_animal(channel_names: list[str],
                              animal_id: str) -> bool:
        from src.utils.animal import split_animal_electrode
        max_iter = 64  # NASA Rule 2
        for i, name in enumerate(channel_names):
            assert i < max_iter, "channel scan runaway"
            if not isinstance(name, str):
                continue
            a, _ = split_animal_electrode(name)
            if a == animal_id:
                return True
        return False

    def pi_approve(self, file_id: int, pi_email: str,
                     *,
                     events_override: list[dict] | None = None,
                     ) -> int:
        """Move a file from ``pending_pi_review`` to ``pi_approved``.

        When the PI nudges or removes events inline, pass the new
        list as ``events_override``; the original undergrad
        markers are captured in the audit row's payload_json so
        the pre-PI version stays queryable.
        """
        assert isinstance(file_id, int), "file_id must be int"
        assert pi_email, "pi_email required"
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT id, markers_json FROM review_state
                   WHERE file_id = ?
                     AND status = 'pending_pi_review'
                   ORDER BY updated_at DESC LIMIT 1""",
                (file_id,),
            ).fetchone()
            if not row:
                return 0
            pre_edit = row["markers_json"]
            if events_override is not None:
                new_payload = json.dumps(events_override)
                conn.execute(
                    """UPDATE review_state
                       SET status='pi_approved',
                           markers_json=?, updated_at=?
                       WHERE id=?""",
                    (new_payload, now, row["id"]),
                )
            else:
                conn.execute(
                    """UPDATE review_state
                       SET status='pi_approved', updated_at=?
                       WHERE id=?""",
                    (now, row["id"]),
                )
            payload = {
                "pi_email": pi_email.lower(),
                "edited": events_override is not None,
            }
            if events_override is not None:
                payload["pre_edit_markers_json"] = pre_edit
            conn.execute(
                """INSERT INTO review_event_log
                   (file_id, user_email, action, payload_json, at)
                   VALUES (?, ?, 'pi_approve', ?, ?)""",
                (file_id, pi_email.lower(),
                 json.dumps(payload), now),
            )
            conn.commit()
            return int(row["id"])
        finally:
            conn.close()

    def pi_flag(self, file_id: int, pi_email: str,
                  *, note: str,
                  event_indices: list[int] | None = None) -> int:
        """Send a file back to the undergrad's queue with a note.

        ``note`` shows up on the queue card. ``event_indices``
        marks WHICH events the PI wants re-checked; an empty list
        means the whole file needs a second look.
        """
        assert isinstance(file_id, int), "file_id must be int"
        assert pi_email, "pi_email required"
        assert isinstance(note, str), "note must be str"
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT id FROM review_state
                   WHERE file_id = ?
                     AND status IN ('pending_pi_review',
                                     'pi_approved')
                   ORDER BY updated_at DESC LIMIT 1""",
                (file_id,),
            ).fetchone()
            if not row:
                return 0
            conn.execute(
                """UPDATE review_state
                   SET status='pi_flagged', note=?, updated_at=?
                   WHERE id=?""",
                (note, now, row["id"]),
            )
            conn.execute(
                """INSERT INTO review_event_log
                   (file_id, user_email, action, payload_json, at)
                   VALUES (?, ?, 'pi_flag', ?, ?)""",
                (file_id, pi_email.lower(),
                 json.dumps({"note": note,
                              "event_indices":
                                  list(event_indices or [])}),
                 now),
            )
            conn.commit()
            return int(row["id"])
        finally:
            conn.close()

    def pi_bulk_approve(self, file_ids: list[int],
                          pi_email: str) -> int:
        """Approve many files; emits one ``pi_bulk_approve`` row
        in addition to a per-file ``pi_approve`` entry each.
        """
        assert isinstance(file_ids, list), "file_ids must be list"
        assert pi_email, "pi_email required"
        max_iter = 1024
        n_done = 0
        for i, fid in enumerate(file_ids):
            assert i < max_iter, "bulk loop runaway"
            if self.pi_approve(int(fid), pi_email):
                n_done += 1
        if n_done:
            self.insert_review_event(
                int(file_ids[0]), pi_email,
                "pi_bulk_approve",
                {"file_ids": [int(f) for f in file_ids],
                 "n_approved": n_done},
            )
        return n_done

    def pi_bulk_flag(self, file_ids: list[int], pi_email: str,
                       *, note: str) -> int:
        """Flag many files with a shared note."""
        assert isinstance(file_ids, list), "file_ids must be list"
        assert pi_email, "pi_email required"
        max_iter = 1024
        n_done = 0
        for i, fid in enumerate(file_ids):
            assert i < max_iter, "bulk loop runaway"
            if self.pi_flag(int(fid), pi_email, note=note):
                n_done += 1
        if n_done:
            self.insert_review_event(
                int(file_ids[0]), pi_email,
                "pi_bulk_flag",
                {"file_ids": [int(f) for f in file_ids],
                 "n_flagged": n_done, "note": note},
            )
        return n_done

    def get_adjacent_files(self, file_id: int,
                              direction: str) -> dict | None:
        """Return the previous or next ``processed_files`` row in
        the same ``session_dir`` by ``chunk_datetime``.

        Powers the event-clip stitcher when the EO-5min /
        BB+5min window crosses a recording boundary.
        """
        assert isinstance(file_id, int), "file_id must be int"
        assert direction in ("prev", "next"), \
            "direction in {'prev', 'next'}"
        conn = self._connect()
        try:
            base = conn.execute(
                """SELECT session_dir, chunk_datetime
                   FROM processed_files WHERE id = ?""",
                (file_id,),
            ).fetchone()
            if not base:
                return None
            cmp = "<" if direction == "prev" else ">"
            order = "DESC" if direction == "prev" else "ASC"
            row = conn.execute(
                f"""SELECT id, file_path, session_dir,
                           chunk_datetime, duration_sec,
                           sampling_rate
                    FROM processed_files
                    WHERE session_dir = ?
                      AND chunk_datetime {cmp} ?
                    ORDER BY chunk_datetime {order}
                    LIMIT 1""",
                (base["session_dir"], base["chunk_datetime"]),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def file_row(self, file_id: int) -> dict | None:
        """Single ``processed_files`` row lookup. Used by the
        event-clip resolver."""
        assert isinstance(file_id, int), "file_id must be int"
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT id, file_path, session_dir,
                          chunk_datetime, duration_sec,
                          sampling_rate
                   FROM processed_files WHERE id = ?""",
                (file_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def reopen_review(self, file_id: int, user_email: str) -> bool:
        """Flip the most recent ``no_events`` / ``has_events`` row
        for this (file, user) pair to ``abandoned`` and append a
        ``reopen`` audit entry.

        Powers the Undo toast in Video Review. The queue filter
        treats ``abandoned`` rows as un-finalised, so the file
        re-appears in the user's queue right after this call.

        Returns ``True`` if a row was flipped, ``False`` if there
        was nothing to reopen (idempotent UI noise).
        """
        assert isinstance(file_id, int), "file_id must be int"
        assert user_email, "user_email required"
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT id, status FROM review_state
                   WHERE file_id = ? AND user_email = ?
                     AND status IN ('no_events', 'has_events')
                   ORDER BY updated_at DESC LIMIT 1""",
                (file_id, user_email.lower()),
            ).fetchone()
            if not row:
                return False
            conn.execute(
                "UPDATE review_state SET status='abandoned', "
                "updated_at=? WHERE id=?",
                (now, row["id"]),
            )
            conn.execute(
                """INSERT INTO review_event_log
                   (file_id, user_email, action, payload_json, at)
                   VALUES (?, ?, 'reopen', ?, ?)""",
                (file_id, user_email.lower(),
                 json.dumps({"from_status": row["status"]}), now),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def neighbor_queue_file(self, *, current_file_id: int | None,
                              animal_ids: list[str],
                              user_email: str,
                              direction: int,
                              since_iso: str | None = None,
                              limit: int = 100,
                              ) -> int | None:
        """File id +/- one slot from ``current_file_id`` in the
        FIFO queue. Returns ``None`` when there's nowhere to go.

        Powers ``J`` / ``K`` queue cycling and the auto-advance
        after ``N``-mark-done. If ``current_file_id`` isn't in
        the queue (e.g. it was just finalised), we return the
        queue head for direction=+1 / the queue tail for -1, so
        the reviewer never gets stuck.
        """
        assert direction in (-1, 1), "direction must be -1 or +1"
        assert isinstance(animal_ids, list), "animal_ids list"
        rows = self.get_review_queue(
            animal_ids, user_email,
            limit=limit, since_iso=since_iso,
        )
        if not rows:
            return None
        ids = [int(r["id"]) for r in rows]
        if current_file_id is None or int(current_file_id) not in ids:
            return ids[0] if direction == 1 else ids[-1]
        idx = ids.index(int(current_file_id))
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(ids):
            return None
        return ids[new_idx]

    def review_backlog_summary(self, since_iso: str | None = None
                                ) -> list[dict]:
        """Per-animal backlog snapshot for the PI tab + weekly digest.

        Returns rows of {animal_id, n_unreviewed,
        oldest_chunk_datetime}. animal_id is derived from
        session_config.eeg_channels split on comma — one row per
        distinct token.
        """
        # Pull every (file, channel_names, eeg_channels) tuple not
        # finalised. The animal-id resolution happens in Python:
        # eeg_channels indexes into channel_names to get the actual
        # electrode/animal labels.
        conn = self._connect()
        try:
            extra_where = ""
            params: list = []
            if since_iso:
                extra_where = " AND pf.chunk_datetime >= ?"
                params.append(since_iso)
            rows = conn.execute(
                f"""SELECT pf.id, pf.chunk_datetime,
                            sc.channel_names, sc.eeg_channels
                    FROM processed_files pf
                    JOIN session_config sc
                      ON sc.session_dir = pf.session_dir
                    WHERE pf.has_video = 1
                      AND sc.channel_names IS NOT NULL
                      AND sc.eeg_channels IS NOT NULL
                      AND NOT EXISTS (
                        SELECT 1 FROM review_state rs
                        WHERE rs.file_id = pf.id
                          AND rs.status IN ('no_events',
                                             'has_events')
                      ){extra_where}""",
                params,
            ).fetchall()
        finally:
            conn.close()
        by_animal: dict[str, dict] = {}
        for r in rows:
            animals = self._animal_ids_for_config(
                r["channel_names"], r["eeg_channels"])
            for a in animals:
                slot = by_animal.setdefault(a, {
                    "animal_id": a,
                    "n_unreviewed": 0,
                    "oldest_chunk_datetime": r["chunk_datetime"],
                })
                slot["n_unreviewed"] += 1
                if (r["chunk_datetime"] or "") < (
                        slot["oldest_chunk_datetime"] or ""):
                    slot["oldest_chunk_datetime"] = r["chunk_datetime"]
        return sorted(by_animal.values(),
                       key=lambda d: -d["n_unreviewed"])

    def behavioral_seizure_status_per_animal(self,
                                                  days: int = 7
                                                  ) -> list[dict]:
        """Per-animal pipeline state for the Overview status card.

        For every animal that has at least one ingested file,
        compute:
          * queue depth (files with no review_state or
            'pi_flagged');
          * pending_pi_review count;
          * pi_approved count (already in CSV);
          * pi_flagged count;
          * files CREATED in the last *days* (rate-of-arrival);
          * files APPROVED in the last *days* (rate-of-
            throughput);
          * last_activity_at (max updated_at across all the
            animal's review_state rows).

        The rate delta = created_window - approved_window lets
        the Overview card tell the user 'team keeping up' vs
        'backlog growing' at a glance.

        One round-trip; SQL does the join + aggregation.
        """
        assert isinstance(days, int) and days > 0, "days > 0"
        cutoff_iso = (datetime.now()
                       - timedelta(days=days)).isoformat()
        conn = self._connect()
        try:
            # Pull every (file, session, status) tuple we'll
            # need. Animal resolution happens in Python via
            # _animal_ids_for_config so the LIKE-fuzzy match
            # of get_review_queue isn't repeated here.
            rows = conn.execute(
                """SELECT pf.id AS file_id,
                          pf.session_dir, pf.chunk_datetime,
                          sc.channel_names, sc.eeg_channels,
                          (
                            SELECT status FROM review_state rs
                            WHERE rs.file_id = pf.id
                            ORDER BY rs.updated_at DESC LIMIT 1
                          ) AS latest_status,
                          (
                            SELECT updated_at FROM review_state rs
                            WHERE rs.file_id = pf.id
                            ORDER BY rs.updated_at DESC LIMIT 1
                          ) AS latest_updated_at
                   FROM processed_files pf
                   JOIN session_config sc
                     ON sc.session_dir = pf.session_dir
                   WHERE pf.has_video = 1
                     AND sc.channel_names IS NOT NULL
                     AND sc.eeg_channels IS NOT NULL"""
            ).fetchall()
        finally:
            conn.close()
        # Aggregate per animal.
        per_animal: dict[str, dict] = {}
        max_iter = len(rows) + 1
        for i, r in enumerate(rows):
            assert i < max_iter, "row scan runaway"
            animals = self._animal_ids_for_config(
                r["channel_names"], r["eeg_channels"])
            status = r["latest_status"]
            chunk_dt = r["chunk_datetime"] or ""
            updated_at = r["latest_updated_at"] or ""
            for a in animals:
                slot = per_animal.setdefault(a, {
                    "animal_id": a,
                    "n_queue": 0,
                    "n_pending_pi": 0,
                    "n_approved": 0,
                    "n_flagged": 0,
                    "created_window": 0,
                    "approved_window": 0,
                    "last_activity_at": "",
                })
                # Status buckets.
                if status is None or status == "pi_flagged":
                    slot["n_queue"] += 1
                if status == "pending_pi_review":
                    slot["n_pending_pi"] += 1
                if status == "pi_approved":
                    slot["n_approved"] += 1
                if status == "pi_flagged":
                    slot["n_flagged"] += 1
                # Rate windows.
                if (chunk_dt and chunk_dt >= cutoff_iso):
                    slot["created_window"] += 1
                if (status == "pi_approved"
                        and updated_at
                        and updated_at >= cutoff_iso):
                    slot["approved_window"] += 1
                # Last activity.
                if (updated_at
                        and updated_at > slot["last_activity_at"]):
                    slot["last_activity_at"] = updated_at
        out = list(per_animal.values())
        # Sort: animals with growing backlog first (positive
        # net delta), then by queue size descending.
        out.sort(key=lambda s: (
            -(s["created_window"] - s["approved_window"]),
            -s["n_queue"],
        ))
        return out

    def list_all_animals(self) -> list[str]:
        """Distinct animal ids across every session_config row.

        Used to compute the unassigned pool (animals - assigned).
        Names are returned in a stable sorted order.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT channel_names, eeg_channels
                   FROM session_config
                   WHERE channel_names IS NOT NULL
                     AND eeg_channels IS NOT NULL"""
            ).fetchall()
        finally:
            conn.close()
        seen: set[str] = set()
        for r in rows:
            for a in self._animal_ids_for_config(
                    r["channel_names"], r["eeg_channels"]):
                seen.add(a)
        return sorted(seen)

    def review_user_throughput(self, days: int = 7
                                 ) -> list[dict]:
        """Per-user finished-count over the last *days*."""
        assert isinstance(days, int) and days > 0
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT user_email,
                          SUM(CASE WHEN status='no_events'
                                   THEN 1 ELSE 0 END) AS n_no_events,
                          SUM(CASE WHEN status='has_events'
                                   THEN 1 ELSE 0 END) AS n_has_events,
                          COUNT(*) AS n_total,
                          MAX(updated_at) AS last_finalised_at
                   FROM review_state
                   WHERE status IN ('no_events', 'has_events')
                     AND updated_at >= ?
                   GROUP BY user_email
                   ORDER BY n_total DESC""",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def review_backlog_floor(self) -> str | None:
        """Return the most recent 'backlog_zero' sentinel timestamp,
        or None if the PI hasn't set a floor yet. The queue + backlog
        summary use this to ignore prehistoric files."""
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT MAX(at) AS at FROM review_event_log
                   WHERE action = 'backlog_zero'"""
            ).fetchone()
            return (row["at"] if row and row["at"] else None)
        finally:
            conn.close()

    def set_video_quality_threshold(self, *,
                                       threshold: float,
                                       sampled_variance: float,
                                       set_by_email: str,
                                       file_id: int | None = None,
                                       note: str = "") -> int:
        """Persist a new pixel-variance quality threshold.

        Uses ``review_event_log`` as a single-source audit
        trail; ``get_video_quality_threshold`` reads back the
        latest payload. ``file_id`` is the recording the sample
        was taken from when available; without it we attach the
        sentinel row to the most recent processed_files id so
        the existing FK constraint stays honored.
        """
        assert isinstance(threshold, (int, float)), "threshold num"
        assert isinstance(sampled_variance, (int, float)), \
            "sampled_variance num"
        assert set_by_email, "set_by_email required"
        payload = {
            "threshold": float(threshold),
            "sampled_variance": float(sampled_variance),
            "note": str(note or ""),
        }
        # Resolve a valid file_id for the FK. Prefer the caller's
        # explicit id; fall back to "any existing file" so the
        # threshold can be set before/without a specific
        # recording in view.
        resolved_id = file_id
        if not resolved_id:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id FROM processed_files "
                    "ORDER BY id LIMIT 1").fetchone()
                resolved_id = int(row["id"]) if row else None
            finally:
                conn.close()
        if not resolved_id:
            # Empty DB -- no audit row possible. Caller can retry
            # after first ingest.
            raise RuntimeError(
                "No processed_files row exists to anchor the "
                "quality-threshold audit row to.")
        return self.insert_review_event(
            int(resolved_id), set_by_email,
            "video_quality_threshold", payload,
        )

    def get_video_quality_threshold(self) -> dict | None:
        """Return the latest video quality threshold record
        (``{threshold, sampled_variance, note, at, user_email}``)
        or ``None`` if a PI has never set one."""
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT user_email, payload_json, at
                   FROM review_event_log
                   WHERE action = 'video_quality_threshold'
                   ORDER BY at DESC LIMIT 1"""
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            return None
        payload["at"] = row["at"]
        payload["user_email"] = row["user_email"]
        return payload
