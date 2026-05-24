"""SQLite schema for the QC monitoring database — v2 with evoked features, criticality, versioning."""

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Settings versions for analysis provenance tracking
CREATE TABLE IF NOT EXISTS settings_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_hash TEXT UNIQUE NOT NULL,
    settings_json TEXT NOT NULL,
    label TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
);

-- Session-level auto-discovered config
CREATE TABLE IF NOT EXISTS session_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_dir TEXT UNIQUE NOT NULL,
    channel_names TEXT,
    eeg_channels TEXT,
    stim_copy_channels TEXT,
    reference_channels TEXT,
    stim_frequency_hz REAL,
    stim_charge_nC REAL,
    stim_pulse_width_us REAL,
    stim_neg_ratio REAL,
    stim_active_outputs TEXT,
    sampling_rate INTEGER,
    num_channels INTEGER,
    spike_threshold REAL,
    power_threshold REAL,
    coastline_threshold REAL,
    discovered_at TEXT NOT NULL
);

-- Processed files
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
    status TEXT DEFAULT 'pending',
    version_id INTEGER REFERENCES settings_versions(id)
);

CREATE INDEX IF NOT EXISTS idx_pf_status ON processed_files(status);
CREATE INDEX IF NOT EXISTS idx_pf_session ON processed_files(session_dir);
CREATE INDEX IF NOT EXISTS idx_pf_datetime ON processed_files(chunk_datetime);

-- Per-channel QC metrics (Tier 1 Python)
CREATE TABLE IF NOT EXISTS chunk_qc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    channel INTEGER NOT NULL,
    channel_name TEXT,
    channel_role TEXT,
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
    version_id INTEGER REFERENCES settings_versions(id),
    UNIQUE(file_id, channel, version_id)
);

-- Stimulation QC from STIM_REPORT.txt
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

-- Per-epoch evoked features (30 features x N epochs per file)
CREATE TABLE IF NOT EXISTS evoked_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    epoch_index INTEGER NOT NULL,
    epoch_time_sec REAL,
    line_length REAL,
    log_auc REAL,
    peak_amplitude REAL,
    trough_amplitude REAL,
    peak_to_trough REAL,
    rms_amplitude REAL,
    peak_latency_ms REAL,
    trough_latency_ms REAL,
    max_slope REAL,
    max_slope_time_ms REAL,
    early_area REAL,
    late_area REAL,
    early_late_ratio REAL,
    recovery_tau REAL,
    recovery_slope REAL,
    template_correlation REAL,
    pca_recon_error REAL,
    variance REAL,
    autocorrelation REAL,
    ac_width REAL,
    exp_fit_a REAL,
    sum_power_low REAL,
    freq_moment_low REAL,
    sum_power_high REAL,
    freq_moment_high REAL,
    is_artifact INTEGER DEFAULT 0,
    is_ictal INTEGER DEFAULT 0,
    version_id INTEGER REFERENCES settings_versions(id),
    UNIQUE(file_id, epoch_index, version_id)
);

CREATE INDEX IF NOT EXISTS idx_ef_file ON evoked_features(file_id);

-- Evoked summary per file (aggregated)
CREATE TABLE IF NOT EXISTS evoked_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    evoked_output_path TEXT,
    num_stimuli_detected INTEGER,
    num_traces_extracted INTEGER,
    num_artifact_epochs INTEGER,
    num_ictal_epochs INTEGER,
    mean_peak_amplitude REAL,
    std_peak_amplitude REAL,
    mean_trough_amplitude REAL,
    mean_peak_latency_ms REAL,
    mean_line_length REAL,
    mean_recovery_tau REAL,
    mean_template_correlation REAL,
    pipeline_duration_sec REAL,
    version_id INTEGER REFERENCES settings_versions(id),
    UNIQUE(file_id, version_id)
);

-- Per-window criticality measurements
CREATE TABLE IF NOT EXISTS criticality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    channel INTEGER NOT NULL,
    window_index INTEGER NOT NULL,
    time_sec REAL,
    db_value REAL,
    db_std REAL,
    sigma REAL,
    ar_order INTEGER,
    exit_status INTEGER,
    version_id INTEGER REFERENCES settings_versions(id),
    UNIQUE(file_id, channel, window_index, version_id)
);

-- Seizure events detected offline
CREATE TABLE IF NOT EXISTS seizure_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    channel INTEGER,
    channel_name TEXT,
    onset_sec REAL,
    offset_sec REAL,
    duration_sec REAL,
    spike_count INTEGER,
    peak_amplitude REAL,
    spike_rate_hz REAL,
    severity TEXT,
    version_id INTEGER REFERENCES settings_versions(id)
);

-- MATLAB pipeline results (status tracking)
CREATE TABLE IF NOT EXISTS matlab_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    evoked_output_path TEXT,
    num_stimuli INTEGER,
    num_traces INTEGER,
    num_features INTEGER,
    criticality_mean REAL,
    criticality_std REAL,
    pipeline_duration_sec REAL,
    exit_status TEXT,
    error_message TEXT,
    computed_at TEXT NOT NULL,
    version_id INTEGER REFERENCES settings_versions(id)
);

-- Alert log
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

-- Mean evoked waveforms per file per channel (for plotting)
CREATE TABLE IF NOT EXISTS evoked_waveforms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    channel INTEGER NOT NULL,
    channel_name TEXT,
    time_axis_ms TEXT NOT NULL,
    mean_trace TEXT NOT NULL,
    sem_trace TEXT,
    stim_mean_trace TEXT,
    n_epochs INTEGER,
    analysis_start_ms REAL,
    analysis_end_ms REAL,
    version_id INTEGER REFERENCES settings_versions(id),
    UNIQUE(file_id, channel, version_id)
);

-- Processing activity log
CREATE TABLE IF NOT EXISTS processing_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    file_id INTEGER,
    file_path TEXT,
    action TEXT NOT NULL,
    message TEXT,
    duration_sec REAL
);

CREATE INDEX IF NOT EXISTS idx_proclog_ts ON processing_log(timestamp);

-- User annotations / notes
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL,
    session_dir TEXT,
    file_id INTEGER,
    note TEXT NOT NULL,
    category TEXT DEFAULT 'observation'
);

CREATE INDEX IF NOT EXISTS idx_annotations_ts ON annotations(timestamp);

-- System health snapshots
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

-- Authenticated users (identity comes from Cloudflare Access)
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    display_name TEXT,
    role TEXT DEFAULT 'reviewer',  -- 'mentor' | 'reviewer'
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Video QC: one row per processed video. Captures frame-level
-- brightness/variance stats so the dispatcher can decide whether to
-- alert (too dark, too bright, or uniform footage).
CREATE TABLE IF NOT EXISTS video_qc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    version_id INTEGER,
    video_path TEXT NOT NULL,
    analyzed_at TEXT NOT NULL,
    fps REAL,
    n_frames_total INTEGER,
    n_frames_sampled INTEGER,
    sample_every_n INTEGER,
    duration_sec REAL,
    duration_analyzed_sec REAL,
    mean_global REAL,
    var_global REAL,
    mean_min REAL,
    mean_max REAL,
    var_min REAL,
    var_max REAL,
    n_dark INTEGER,
    n_bright INTEGER,
    n_low_var INTEGER,
    n_bad INTEGER,
    pct_bad REAL,
    first_bad_idx INTEGER,
    first_bad_reason TEXT,
    status TEXT,
    error TEXT,
    thresholds TEXT,
    FOREIGN KEY (file_id) REFERENCES processed_files(id),
    FOREIGN KEY (version_id) REFERENCES settings_versions(id)
);
CREATE INDEX IF NOT EXISTS idx_video_qc_file ON video_qc(file_id);
CREATE INDEX IF NOT EXISTS idx_video_qc_status ON video_qc(status);
"""
