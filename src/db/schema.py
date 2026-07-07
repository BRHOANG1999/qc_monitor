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
    slow_gamma_wavelet_power REAL,
    gamma_wavelet_power REAL,
    high_gamma_wavelet_power REAL,
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
    wavelet_power_slow_gamma REAL,
    wavelet_power_gamma REAL,
    wavelet_power_high_gamma REAL,
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

-- Per-(file, channel) transfer impedance, both stimulus phases.
-- Z_kOhm = (v_raw/gain*1000) / I_uA, where I_uA = charge_nC/pulse_width_us
-- *1000. Positive (fast) + negative (slow, 1:ratio recharge) phases have
-- different currents, so both impedances are stored. Gain comes from the
-- File_Records sheet (recording amplifier gain), NOT from any .mat file.
CREATE TABLE IF NOT EXISTS channel_impedance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    channel INTEGER NOT NULL,
    channel_name TEXT,
    animal_id TEXT,
    electrode TEXT,
    gain REAL,
    charge_nc REAL,
    pulse_width_us REAL,
    neg_ratio REAL,
    v_pos_raw REAL,
    v_neg_raw REAL,
    i_pos_ua REAL,
    i_neg_ua REAL,
    impedance_pos_kohm REAL,
    impedance_neg_kohm REAL,
    -- Electrode ACCESS RESISTANCE (kΩ) from the ohmic voltage step at a
    -- current transition (the metric; the peak/mean impedance_* above is
    -- legacy and no longer written). r_access = mean of the fast→slow
    -- reversal step and the slow→off step.
    access_r_kohm REAL,
    access_r_reversal_kohm REAL,
    access_r_offset_kohm REAL,
    -- Slow-phase STEADY-STATE impedance (kΩ): settled plateau voltage of the
    -- long slow phase ÷ commanded slow current. Gain-corrected -> comparable
    -- between animals; a stim-consistency signal complementary to Rₐ.
    slow_ss_raw REAL,
    slow_ss_kohm REAL,
    computed_at TEXT NOT NULL,
    UNIQUE(file_id, channel)
);
CREATE INDEX IF NOT EXISTS idx_chimp_animal_chan
    ON channel_impedance(animal_id, channel_name);

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

-- ----------------------------------------------------------------------
-- user_activity: per-user "who did what when" across the dashboard. Filled
-- ONLY for actions not already audited elsewhere (analysis tabs, file
-- browser, CSV finalize, auto-filter, training PI actions, tab navigation);
-- the User Activity tab UNIONs this with review_event_log + training_attempt
-- so the feed is complete without double-logging already-audited verbs.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email TEXT NOT NULL,
    at         TEXT NOT NULL,
    area       TEXT NOT NULL,
    action     TEXT NOT NULL,
    target     TEXT,
    detail_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_user_activity_user
    ON user_activity(user_email, at);
CREATE INDEX IF NOT EXISTS idx_user_activity_at
    ON user_activity(at);

-- User annotations / notes
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL,
    session_dir TEXT,
    file_id INTEGER,
    note TEXT NOT NULL,
    category TEXT DEFAULT 'observation',
    -- JSON blob of the review context active when the note was saved
    -- (animal, file, channel, feature, window, video position) so a
    -- reviewer can click a note to jump back to that exact state.
    context TEXT,
    -- Stable de-dup key for externally-synced rows (e.g. the KMrecorder
    -- operator log). NULL for hand-written notes. SQLite treats NULLs as
    -- distinct, so its UNIQUE index only constrains synced rows. NOTE: the
    -- index is created in Store._init_db AFTER the column-add migration --
    -- it must NOT live here, because on an existing DB this CREATE TABLE
    -- is a no-op and an index here would reference a not-yet-added column.
    source_key TEXT
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

-- ----------------------------------------------------------------------
-- Reviewer queue: per-(file, user) review state + append-only audit log.
-- The dashboard's Video Review tab pulls a queue of unreviewed files for
-- the user's assigned animals; finishing a file inserts one row here and
-- one row in review_event_log so the PI can audit who did what when.
-- ----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS review_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    user_email TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK(status IN ('claimed', 'no_events',
                          'has_events', 'abandoned',
                          'pending_pi_review',
                          'pi_approved', 'pi_flagged',
                          'needs_scoring')),
    markers_json TEXT,
    note TEXT,
    -- The animal this review row is for. NULL = legacy/whole-file row
    -- (written before review went per-animal); treated as a wildcard that
    -- covers every animal on the file. New rows always carry a concrete id.
    animal_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_state_file
    ON review_state(file_id);
CREATE INDEX IF NOT EXISTS idx_review_state_user
    ON review_state(user_email);
CREATE INDEX IF NOT EXISTS idx_review_state_status
    ON review_state(status);
-- NOTE: idx_review_state_animal is created by the _migrate_review_add_animal_id
-- migration (NOT here) so executescript(SCHEMA_SQL) on a pre-migration DB
-- doesn't reference the not-yet-added animal_id column.

CREATE TABLE IF NOT EXISTS review_event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    user_email TEXT NOT NULL,
    action TEXT NOT NULL,
    payload_json TEXT,
    animal_id TEXT,
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_event_log_at
    ON review_event_log(at);
CREATE INDEX IF NOT EXISTS idx_review_event_log_user
    ON review_event_log(user_email);
-- Covers the "Flag" queue's EXISTS(auto_filter_flag for this animal on this
-- file) and auto_filter_flagged_file_ids' (action, animal_id -> file_id) scan,
-- which otherwise full-scan review_event_log per candidate file.
CREATE INDEX IF NOT EXISTS idx_review_event_log_action_animal_file
    ON review_event_log(action, animal_id, file_id);

-- ----------------------------------------------------------------------
-- file_claim: soft-lock so two reviewers don't double-score the same
-- recording. When a reviewer opens a file the queue claims it; the
-- queue hides files claimed by OTHER reviewers within a TTL window
-- (see Store.CLAIM_TTL_MINUTES). Kept separate from review_state on
-- purpose -- review_state is append-only and Mass Analyze excludes on
-- 'claimed' there, so writing claims into it would permanently hide
-- files. Claims are per-(file, animal) so two reviewers can hold different
-- animals of the same multi-animal recording at once; claims UPSERT and
-- expire by timestamp, so no row bloat and no manual cleanup. animal_id ''
-- (empty string, not NULL — composite PKs can't span NULLs in SQLite) is the
-- legacy whole-file claim.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS file_claim (
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    animal_id TEXT NOT NULL DEFAULT '',
    user_email TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    PRIMARY KEY (file_id, animal_id)
);
CREATE INDEX IF NOT EXISTS idx_file_claim_claimed_at
    ON file_claim(claimed_at);

-- ----------------------------------------------------------------------
-- rejected_peak: BHZ candidate peaks (the pink triangles) a reviewer
-- has hand-rejected as false positives. The detector is tuned for high
-- sensitivity, so per file the reviewer prunes the candidates not tied
-- to a real seizure. Persisted so the curated set survives reload.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rejected_peak (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    channel INTEGER NOT NULL,
    peak_time_sec REAL NOT NULL,
    rejected_by TEXT,
    rejected_at TEXT NOT NULL,
    UNIQUE(file_id, channel, peak_time_sec)
);
CREATE INDEX IF NOT EXISTS idx_rejected_peak_file
    ON rejected_peak(file_id, channel);

-- ----------------------------------------------------------------------
-- envelope_peak_cache: per-(file, channel, cutoff) peakseek result so
-- the PI's Mass Analyze panel can re-sweep thresholds without paying
-- the FFT cost twice. UNIQUE constraint makes the cache-aside lookup a
-- single indexed read.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS envelope_peak_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    channel INTEGER NOT NULL,
    cutoff REAL NOT NULL,
    min_peak_dist_sec REAL NOT NULL DEFAULT 150.0,
    n_peaks INTEGER NOT NULL,
    peak_times_json TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    UNIQUE(file_id, channel, cutoff, min_peak_dist_sec)
);
CREATE INDEX IF NOT EXISTS idx_envelope_peak_cache_lookup
    ON envelope_peak_cache(file_id, channel, cutoff);

-- ----------------------------------------------------------------------
-- envelope_auc_cache: sibling of envelope_peak_cache for the AUC screen.
-- Per-(file, channel, auc_threshold, window_sec) sliding-window-AUC
-- peakseek count, so re-sweeping the AUC threshold / window is a single
-- indexed read instead of re-paying the FFT.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS envelope_auc_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES processed_files(id),
    channel INTEGER NOT NULL,
    auc_threshold REAL NOT NULL,
    window_sec REAL NOT NULL,
    min_peak_dist_sec REAL NOT NULL DEFAULT 150.0,
    n_events INTEGER NOT NULL,
    computed_at TEXT NOT NULL,
    UNIQUE(file_id, channel, auc_threshold, window_sec, min_peak_dist_sec)
);
CREATE INDEX IF NOT EXISTS idx_envelope_auc_cache_lookup
    ON envelope_auc_cache(file_id, channel, auc_threshold, window_sec);

-- ----------------------------------------------------------------------
-- animal_screen_config: the PI's per-animal auto-filter thresholds. A
-- background sweep screens each enabled animal's pending files; a file
-- crossing NONE of that animal's active thresholds is auto-marked
-- pending_pi_review (the PI still rubber-stamps it). screen_mode picks
-- which screens are active: 'both' = OR(envelope, auc). peak_cutoff is the
-- raw-envelope (Hilbert) threshold; auc_threshold/auc_window_sec the
-- sliding-window-AUC screen. NULL threshold = that screen unset.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS animal_screen_config (
    animal_id      TEXT PRIMARY KEY,
    peak_cutoff    REAL,
    auc_threshold  REAL,
    auc_window_sec REAL,
    electrode      INTEGER NOT NULL DEFAULT 0,
    screen_mode    TEXT NOT NULL DEFAULT 'both'
        CHECK(screen_mode IN ('both', 'envelope', 'auc')),
    enabled        INTEGER NOT NULL DEFAULT 1,
    updated_by     TEXT,
    updated_at     TEXT
);

-- ----------------------------------------------------------------------
-- mass_analyze_job: background scan queue + progress for the PI's
-- Mass Analyze panel. One row per (PI, animal, cutoff) run; the panel
-- polls scanned_files / n_zero_peaks / n_with_peaks for live progress.
-- auc_threshold / auc_window_sec drive the second (AUC) screen so the
-- scan can populate both pools in one pass.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mass_analyze_job (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pi_email TEXT NOT NULL,
    animal_id TEXT NOT NULL,
    cutoff REAL NOT NULL,
    auc_threshold REAL,
    auc_window_sec REAL,
    electrode INTEGER DEFAULT 0,
    status TEXT NOT NULL
        CHECK(status IN ('pending', 'running',
                          'done', 'failed', 'cancelled')),
    total_files INTEGER,
    scanned_files INTEGER DEFAULT 0,
    n_zero_peaks INTEGER DEFAULT 0,
    n_with_peaks INTEGER DEFAULT 0,
    n_auc_pos INTEGER DEFAULT 0,
    n_disagree INTEGER DEFAULT 0,
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mass_analyze_job_status
    ON mass_analyze_job(status);

-- ----------------------------------------------------------------------
-- screen_eval_job: background benchmark of the two file-screens against
-- human labels. One row per (PI, animal, peak_cutoff, auc_threshold,
-- window) run. The PI panel polls the per-screen confusion counters
-- (p1_* = envelope screen, p2_* = AUC screen) + the disagreement
-- breakdown (*_only_* = file flagged by exactly one screen).
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS screen_eval_job (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pi_email TEXT NOT NULL,
    animal_id TEXT NOT NULL,
    peak_cutoff REAL NOT NULL,
    auc_threshold REAL NOT NULL,
    auc_window_sec REAL NOT NULL,
    electrode INTEGER DEFAULT 0,
    status TEXT NOT NULL
        CHECK(status IN ('pending', 'running',
                          'done', 'failed', 'cancelled')),
    total_files INTEGER,
    scanned_files INTEGER DEFAULT 0,
    p1_tp INTEGER DEFAULT 0,
    p1_fp INTEGER DEFAULT 0,
    p1_tn INTEGER DEFAULT 0,
    p1_fn INTEGER DEFAULT 0,
    p2_tp INTEGER DEFAULT 0,
    p2_fp INTEGER DEFAULT 0,
    p2_tn INTEGER DEFAULT 0,
    p2_fn INTEGER DEFAULT 0,
    env_only_true INTEGER DEFAULT 0,
    env_only_false INTEGER DEFAULT 0,
    auc_only_true INTEGER DEFAULT 0,
    auc_only_false INTEGER DEFAULT 0,
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_screen_eval_job_status
    ON screen_eval_job(status);

-- ----------------------------------------------------------------------
-- event_clip_job: ffmpeg job queue + cache index. The PI tab kicks off
-- a video-clip extraction (5 min before EO to 5 min after BB) and
-- polls this table for completion. spec_hash is the SHA-256 of the
-- normalised ClipSpec + segment list so cache hits across PIs are
-- instant.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS event_clip_job (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_hash TEXT UNIQUE NOT NULL,
    file_id INTEGER REFERENCES processed_files(id),
    eo_sec REAL,
    bb_sec REAL,
    status TEXT NOT NULL
        CHECK(status IN ('pending', 'running',
                          'done', 'failed')),
    cache_path TEXT,
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_clip_job_status
    ON event_clip_job(status);
CREATE INDEX IF NOT EXISTS idx_event_clip_job_hash
    ON event_clip_job(spec_hash);

-- ----------------------------------------------------------------------
-- Training: a student blind-re-scores PI-validated recordings and gets an
-- agreement score. Each graded submission is one training_attempt row;
-- training_progress tracks which stage (1-3) each student has unlocked
-- (advancement is PI-confirmed) and when they were certified.
-- ----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS training_attempt (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_email TEXT NOT NULL,
    stage INTEGER NOT NULL,
    file_id INTEGER REFERENCES processed_files(id),
    submitted_json TEXT,
    agreement REAL,
    breakdown_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_attempt_student
    ON training_attempt(student_email, stage);
CREATE INDEX IF NOT EXISTS idx_training_attempt_file
    ON training_attempt(student_email, file_id);

CREATE TABLE IF NOT EXISTS training_progress (
    student_email TEXT PRIMARY KEY,
    unlocked_stage INTEGER NOT NULL DEFAULT 1,
    certified_at TEXT,
    updated_at TEXT NOT NULL
);

-- A "round" = a balanced batch of examples a student works through before
-- the confidence prompt. started_after_id is the RESET boundary: the
-- student's current grade counts only attempts with id > started_after_id,
-- so "review more" restarts the grade while training_attempt keeps full
-- history. confidence (1-5) + outcome captured at the prompt; notified_at
-- dedups the PI "finished a stage" email.
CREATE TABLE IF NOT EXISTS training_round (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_email TEXT NOT NULL,
    stage INTEGER NOT NULL,
    started_after_id INTEGER NOT NULL,
    files_json TEXT NOT NULL,
    examples INTEGER NOT NULL,
    confidence INTEGER,
    outcome TEXT,
    notified_at TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_round_student
    ON training_round(student_email, stage, id);

-- ----------------------------------------------------------------------
-- training_external_example: historical scored seizures imported from the
-- lab's BHZ CSV exports (src/utils/past_events.py). Each row collapses one
-- recording's scored events into the Training marker schema so the file
-- becomes an extra practice candidate WITHOUT polluting the live
-- review_state pipeline. Keyed 1:1 to the processed_files row created for
-- the resolved .mat; training_candidate_pool UNIONs these in and
-- validated_events_for_file falls back to their markers.
CREATE TABLE IF NOT EXISTS training_external_example (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL UNIQUE REFERENCES processed_files(id),
    folder TEXT,
    filename TEXT,
    animal TEXT,
    fs REAL,
    has_seizure INTEGER NOT NULL DEFAULT 1,
    rep_type TEXT,
    rep_racine INTEGER,
    markers_json TEXT NOT NULL,
    source_csv TEXT,
    peak_stamp REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_external_animal
    ON training_external_example(animal);

-- eeg_file_location: persistent cache of where a recording filename was
-- found on disk (or NULL = not found, with checked_at so the store can
-- expire and re-search offline files). Lets the recursive multi-HDD search
-- in past_events.EEGLocator walk each drive once, not once per event.
CREATE TABLE IF NOT EXISTS eeg_file_location (
    filename TEXT PRIMARY KEY,
    path TEXT,
    checked_at TEXT NOT NULL
);

-- Restart-proof digest dedup: one row per (digest, calendar day) the moment a
-- recurring email is claimed. The UNIQUE constraint + INSERT OR IGNORE makes
-- "send at most once per day" atomic across daemon restarts, backup reverts,
-- and multiple daemon instances -- unlike the JSON state file, which could
-- revert and re-fire the same digest on every restart.
CREATE TABLE IF NOT EXISTS notification_log (
    digest TEXT NOT NULL,
    sent_date TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    UNIQUE(digest, sent_date)
);
"""
