"""SQLite data access layer for the QC monitor."""

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

    # --- processed_files ---

    def is_processed(self, file_path: str) -> bool:
        conn = self._connect()
        row = conn.execute(
            "SELECT status FROM processed_files WHERE file_path = ?", (file_path,)
        ).fetchone()
        conn.close()
        return row is not None and row["status"] == "done"

    def get_file_status(self, file_path: str) -> str | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT status FROM processed_files WHERE file_path = ?", (file_path,)
        ).fetchone()
        conn.close()
        return row["status"] if row else None

    def register_file(self, file_path: str, file_size: int, file_mtime: float,
                      session_dir: str, session_name: str, chunk_datetime: str) -> int:
        conn = self._connect()
        cursor = conn.execute(
            """INSERT OR IGNORE INTO processed_files
               (file_path, file_size, file_mtime, session_dir, session_name,
                chunk_datetime, status)
               VALUES (?, ?, ?, ?, ?, ?, 'pending')""",
            (file_path, file_size, file_mtime, session_dir, session_name, chunk_datetime)
        )
        if cursor.rowcount == 0:
            row = conn.execute(
                "SELECT id FROM processed_files WHERE file_path = ?", (file_path,)
            ).fetchone()
            file_id = row["id"]
        else:
            file_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return file_id

    def update_file_status(self, file_id: int, status: str, **kwargs):
        conn = self._connect()
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
        conn.close()

    def reset_stale_processing(self):
        """Reset files stuck in 'processing' (crashed) back to 'pending'."""
        conn = self._connect()
        conn.execute("UPDATE processed_files SET status = 'pending' WHERE status = 'processing'")
        conn.commit()
        conn.close()

    def get_pending_files(self, limit: int = 50) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM processed_files WHERE status = 'pending' ORDER BY chunk_datetime ASC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # --- chunk_qc ---

    def insert_chunk_qc(self, file_id: int, channel: int, metrics: dict):
        conn = self._connect()
        conn.execute(
            """INSERT OR REPLACE INTO chunk_qc
               (file_id, channel, rms_amplitude, line_noise_power, line_noise_ratio,
                artifact_pct, has_nan, has_inf, has_clipping, flat_line_pct, dc_offset,
                delta_power, theta_power, alpha_power, beta_power, gamma_power,
                high_gamma_power, total_power, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_id, channel,
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
             datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    # --- stim_qc ---

    def insert_stim_qc(self, file_id: int, stim_channel: int, metrics: dict):
        conn = self._connect()
        conn.execute(
            """INSERT OR REPLACE INTO stim_qc
               (file_id, stim_channel, charge_nC, pulse_width_us, neg_pulse_ratio,
                frequency_hz, gain, total_pulses, total_charge_nC, duration_sec,
                expected_pulses, delivery_pct, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_id, stim_channel,
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
             datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    # --- matlab_results ---

    def insert_matlab_result(self, file_id: int, result: dict):
        conn = self._connect()
        conn.execute(
            """INSERT INTO matlab_results
               (file_id, evoked_output_path, num_stimuli, num_traces,
                criticality_mean, criticality_std, num_features,
                pipeline_duration_sec, exit_status, error_message, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_id,
             result.get("evoked_output_path"),
             result.get("num_stimuli"),
             result.get("num_traces"),
             result.get("criticality_mean"),
             result.get("criticality_std"),
             result.get("num_features"),
             result.get("pipeline_duration_sec"),
             result.get("exit_status"),
             result.get("error_message"),
             datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    # --- alerts ---

    def insert_alert(self, alert_type: str, severity: str, message: str,
                     file_id: int = None, session_dir: str = None):
        conn = self._connect()
        conn.execute(
            """INSERT INTO alerts (alert_type, severity, message, file_id, session_dir, sent_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (alert_type, severity, message, file_id, session_dir, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    def get_last_alert_time(self, alert_type: str) -> datetime | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT sent_at FROM alerts WHERE alert_type = ? ORDER BY sent_at DESC LIMIT 1",
            (alert_type,)
        ).fetchone()
        conn.close()
        if row:
            return datetime.fromisoformat(row["sent_at"])
        return None

    def get_recent_alerts(self, hours: int = 24) -> list[dict]:
        conn = self._connect()
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        rows = conn.execute(
            "SELECT * FROM alerts WHERE sent_at > ? ORDER BY sent_at DESC", (cutoff,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # --- system_health ---

    def insert_health(self, metrics: dict):
        conn = self._connect()
        conn.execute(
            """INSERT INTO system_health
               (timestamp, cpu_pct, memory_pct, disk_free_gb,
                network_share_accessible, queue_depth, files_processed_last_hour)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(),
             metrics.get("cpu_pct"),
             metrics.get("memory_pct"),
             metrics.get("disk_free_gb"),
             metrics.get("network_share_accessible"),
             metrics.get("queue_depth"),
             metrics.get("files_processed_last_hour"))
        )
        conn.commit()
        conn.close()

    # --- query helpers for dashboard ---

    def get_qc_timeseries(self, session_dir: str = None, hours: int = 24) -> list[dict]:
        conn = self._connect()
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        if session_dir:
            rows = conn.execute(
                """SELECT pf.chunk_datetime, cq.*
                   FROM chunk_qc cq JOIN processed_files pf ON cq.file_id = pf.id
                   WHERE pf.session_dir = ? AND pf.chunk_datetime > ?
                   ORDER BY pf.chunk_datetime""",
                (session_dir, cutoff)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT pf.chunk_datetime, cq.*
                   FROM chunk_qc cq JOIN processed_files pf ON cq.file_id = pf.id
                   WHERE pf.chunk_datetime > ?
                   ORDER BY pf.chunk_datetime""",
                (cutoff,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_sessions(self) -> list[dict]:
        conn = self._connect()
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
        conn.close()
        return [dict(r) for r in rows]

    def get_health_history(self, hours: int = 24) -> list[dict]:
        conn = self._connect()
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        rows = conn.execute(
            "SELECT * FROM system_health WHERE timestamp > ? ORDER BY timestamp",
            (cutoff,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_files_processed_last_hour(self) -> int:
        conn = self._connect()
        cutoff = (datetime.now() - timedelta(hours=1)).isoformat()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM processed_files WHERE processed_at > ? AND status = 'done'",
            (cutoff,)
        ).fetchone()
        conn.close()
        return row["cnt"] if row else 0
