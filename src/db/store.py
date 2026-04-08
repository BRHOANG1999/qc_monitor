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
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM processed_files WHERE status = 'pending' ORDER BY chunk_datetime ASC LIMIT ?",
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

            if hours is not None:
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
            if hours is not None:
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
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            conditions = ["pf.chunk_datetime > ?"]
            params: list = [cutoff]

            if session_dir:
                conditions.append("pf.session_dir = ?")
                params.append(session_dir)
            if channel is not None:
                conditions.append("cq.channel = ?")
                params.append(channel)
            if version_id is not None:
                conditions.append("cq.version_id = ?")
                params.append(version_id)

            where = " AND ".join(conditions)
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
                   ORDER BY first_chunk DESC"""
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
