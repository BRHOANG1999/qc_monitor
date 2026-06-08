"""SQLite data access layer for the QC monitor — v2 with evoked, criticality, versioning."""

import hashlib
import json
import sqlite3
import os
from datetime import datetime, timedelta
from .schema import SCHEMA_SQL


class Store:
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
        conn.commit()
        conn.close()

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
                   'abandoned'}. Returns the review_state row id.
        Finalising (no_events / has_events / abandoned) emits a
        'finish' (or 'abandon') log event; claiming emits 'claim'.
        """
        assert isinstance(file_id, int), "file_id must be int"
        assert user_email, "user_email required"
        assert status in ("claimed", "no_events", "has_events",
                           "abandoned"), f"bad status {status!r}"
        now = datetime.now().isoformat()
        markers_json = json.dumps(markers or [])
        log_action = ("claim" if status == "claimed"
                       else "abandon" if status == "abandoned"
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
        params: list = list(like_args)
        extra_where = ""
        if since_iso:
            extra_where += " AND pf.chunk_datetime >= ?"
            params.append(since_iso)
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
              AND NOT EXISTS (
                SELECT 1 FROM review_state rs
                WHERE rs.file_id = pf.id
                  AND rs.status IN ('no_events', 'has_events')
              )
              AND NOT EXISTS (
                SELECT 1 FROM review_state rs2
                WHERE rs2.file_id = pf.id
                  AND rs2.user_email = ?
              )
              {extra_where}
            ORDER BY pf.chunk_datetime ASC
            LIMIT ?
        """
        params.append((user_email or "").lower())
        # Over-fetch a little so the post-filter (rejecting rows
        # whose decoded animal list doesn't actually intersect)
        # still has enough hits.
        params.append(limit * 3)
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
