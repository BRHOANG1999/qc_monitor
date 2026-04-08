"""SQLite schema for the QC monitoring database."""

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS processed_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_size INTEGER,
    file_mtime REAL,
    session_dir TEXT,
    session_name TEXT,
    chunk_datetime TEXT,
    num_channels INTEGER,
    num_samples INTEGER,
    sampling_rate REAL,
    duration_sec REAL,
    has_stim_report INTEGER DEFAULT 0,
    has_video INTEGER DEFAULT 0,
    processed_at TEXT,
    status TEXT DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_pf_status ON processed_files(status);
CREATE INDEX IF NOT EXISTS idx_pf_session ON processed_files(session_dir);
CREATE INDEX IF NOT EXISTS idx_pf_datetime ON processed_files(chunk_datetime);

CREATE TABLE IF NOT EXISTS chunk_qc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    channel INTEGER NOT NULL,
    rms_amplitude REAL,
    line_noise_power REAL,
    line_noise_ratio REAL,
    artifact_pct REAL,
    has_nan INTEGER,
    has_inf INTEGER,
    has_clipping INTEGER,
    flat_line_pct REAL,
    dc_offset REAL,
    delta_power REAL,
    theta_power REAL,
    alpha_power REAL,
    beta_power REAL,
    gamma_power REAL,
    high_gamma_power REAL,
    total_power REAL,
    computed_at TEXT NOT NULL,
    UNIQUE(file_id, channel)
);

CREATE TABLE IF NOT EXISTS stim_qc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    stim_channel INTEGER,
    charge_nC REAL,
    pulse_width_us REAL,
    neg_pulse_ratio REAL,
    frequency_hz REAL,
    gain REAL,
    total_pulses INTEGER,
    total_charge_nC REAL,
    duration_sec REAL,
    expected_pulses INTEGER,
    delivery_pct REAL,
    computed_at TEXT NOT NULL,
    UNIQUE(file_id, stim_channel)
);

CREATE TABLE IF NOT EXISTS matlab_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    evoked_output_path TEXT,
    num_stimuli INTEGER,
    num_traces INTEGER,
    criticality_mean REAL,
    criticality_std REAL,
    num_features INTEGER,
    pipeline_duration_sec REAL,
    exit_status TEXT,
    error_message TEXT,
    computed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    file_id INTEGER REFERENCES processed_files(id),
    session_dir TEXT,
    sent_at TEXT NOT NULL,
    acknowledged INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);

CREATE TABLE IF NOT EXISTS system_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cpu_pct REAL,
    memory_pct REAL,
    disk_free_gb REAL,
    network_share_accessible INTEGER,
    queue_depth INTEGER,
    files_processed_last_hour INTEGER
);
"""
