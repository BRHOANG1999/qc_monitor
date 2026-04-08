function run_pipeline(input_file, output_json, config_json)
% RUN_PIPELINE  Full auto-discovering analysis pipeline for QC monitor.
%
%   run_pipeline(INPUT_FILE, OUTPUT_JSON)
%   run_pipeline(INPUT_FILE, OUTPUT_JSON, CONFIG_JSON)
%
%   1. Loads KMrecorder .mat via load_chunk_data
%   2. Auto-discovers EEG/stim channels from fnstr
%   3. Runs batchExtractEvokedResponses with proper channel config
%   4. Extracts ALL 30 features per epoch
%   5. Runs criticality analysis on EEG channel
%   6. Writes comprehensive results to JSON
%
%   CONFIG_JSON (optional): path to JSON with overrides for evoked/criticality params

    t_start = tic;
    result = struct();
    result.input_file = input_file;
    result.exit_status = 'success';
    result.error_message = '';

    % --- Load config overrides if provided ---
    cfg = struct();
    if nargin >= 3 && ~isempty(config_json) && exist(config_json, 'file')
        try
            cfg = jsondecode(fileread(config_json));
        catch
            fprintf('[PIPELINE] Warning: could not parse config JSON\n');
        end
    end

    % Default analysis params (overridable via config)
    pre_ms = get_cfg(cfg, 'pre_stimulus_ms', -100);
    post_ms = get_cfg(cfg, 'post_stimulus_ms', 500);
    analysis_start_ms = get_cfg(cfg, 'analysis_start_ms', 5);
    analysis_end_ms = get_cfg(cfg, 'analysis_end_ms', 50);
    baseline_on = get_cfg(cfg, 'baseline_correction', true);
    baseline_win = get_cfg(cfg, 'baseline_window_ms', [-60, -10]);
    notch60 = get_cfg(cfg, 'notch_60hz', true);
    notch50 = get_cfg(cfg, 'notch_50hz', false);
    hp_on = get_cfg(cfg, 'highpass_enabled', false);
    hp_hz = get_cfg(cfg, 'highpass_cutoff_hz', 1.0);
    lp_on = get_cfg(cfg, 'lowpass_enabled', false);
    lp_hz = get_cfg(cfg, 'lowpass_cutoff_hz', 1000.0);
    stim_thresh_std = get_cfg(cfg, 'stimulus_threshold_std', 3.0);
    min_stim_dist = get_cfg(cfg, 'min_stimulus_distance_sec', 0.1);
    crit_ar = get_cfg(cfg, 'ar_order', 5);
    crit_win = get_cfg(cfg, 'window_sec', 2.0);
    crit_overlap = get_cfg(cfg, 'overlap_pct', 50);

    try
        % --- Add STiM-NET paths ---
        stimnet_root = get_cfg(cfg, 'stimnet_root', ...
            'D:/code/Stimulation-Telemetry-Modulation-NeuroEngineering-Toolkit/daqSignalGenerator');
        addpath(genpath(fullfile(stimnet_root, 'src')));

        % --- Step 1: Load data ---
        fprintf('[PIPELINE] Loading: %s\n', input_file);
        [sbuf, fs, trdata] = load_chunk_data(input_file);
        [n_samples, n_channels] = size(sbuf);
        fprintf('[PIPELINE] Loaded: %d samples x %d ch @ %d Hz (%.1f sec)\n', ...
            n_samples, n_channels, fs, n_samples/fs);

        result.sampling_rate = fs;
        result.num_samples = n_samples;
        result.num_channels = n_channels;
        result.duration_sec = n_samples / fs;

        % --- Step 2: Auto-discover channel roles from fnstr ---
        raw = load(input_file);
        eeg_ch = 1;  % 1-indexed, default
        stim_ch = 2; % 1-indexed, default

        if isfield(raw, 'fnstr')
            fnstr = raw.fnstr;
            if iscell(fnstr)
                result.channel_names = fnstr(1:min(n_channels, length(fnstr)));
                for i = 1:min(n_channels, length(fnstr))
                    name = fnstr{i};
                    if startsWith(name, 'BCH') || startsWith(name, 'EEG') || startsWith(name, 'LFP')
                        eeg_ch = i;
                    end
                    if contains(lower(name), 'stim') && stim_ch == 2
                        stim_ch = i;
                    end
                end
            end
        end

        result.eeg_channel = eeg_ch;
        result.stim_channel = stim_ch;
        fprintf('[PIPELINE] Auto-discovered: EEG=Ch%d, Stim=Ch%d\n', eeg_ch, stim_ch);

        % --- Step 3: Save temp file for batchExtractEvokedResponses ---
        [~, fname, ~] = fileparts(input_file);
        temp_mat = fullfile(tempdir, [fname '_qc_temp.mat']);
        EEG = sbuf;
        save(temp_mat, 'EEG', 'fs', '-v7');

        evoked_output_dir = fullfile(stimnet_root, 'evokedOutput');
        if ~exist(evoked_output_dir, 'dir'), mkdir(evoked_output_dir); end

        % --- Step 4: Evoked extraction ---
        result.evoked_success = false;
        result.num_stimuli = 0;
        result.num_traces = 0;
        result.evoked_output_path = '';

        try
            [er, outfiles] = batchExtractEvokedResponses({temp_mat}, ...
                'EEGChannel', eeg_ch, ...
                'StimulusChannel', stim_ch, ...
                'PreStimulusMS', pre_ms, ...
                'PostStimulusMS', post_ms, ...
                'StimulusThreshold', stim_thresh_std, ...
                'MinStimulusDistance', min_stim_dist, ...
                'BaselineCorrection', baseline_on, ...
                'BaselineWindow', baseline_win, ...
                'NotchFilter60Hz', notch60, ...
                'NotchFilter50Hz', notch50, ...
                'HighPassEnabled', hp_on, ...
                'HighPassCutoff', hp_hz, ...
                'LowPassEnabled', lp_on, ...
                'LowPassCutoff', lp_hz, ...
                'OutputFolder', evoked_output_dir, ...
                'OutputSuffix', '_evoked');

            if ~isempty(er) && er(1).success
                result.evoked_success = true;
                result.num_stimuli = er(1).numStimuli;
                result.num_traces = er(1).numTraces;
                if ~isempty(outfiles)
                    result.evoked_output_path = outfiles{1};
                end
                fprintf('[PIPELINE] Evoked: %d stimuli, %d traces\n', er(1).numStimuli, er(1).numTraces);
            end
        catch e
            result.evoked_message = e.message;
            fprintf('[PIPELINE] Evoked failed: %s\n', e.message);
        end

        % --- Step 5: Extract ALL 30 features per epoch ---
        result.features = struct();
        result.num_features = 0;
        result.epoch_times = [];

        if result.evoked_success && ~isempty(result.evoked_output_path) && exist(result.evoked_output_path, 'file')
            try
                ev = load(result.evoked_output_path);
                traces = ev.evokedData.traces;   % [samples x epochs]
                timeAxis = ev.evokedData.timeAxis;
                stimTimes = ev.evokedData.stimulusTimes;
                n_epochs = size(traces, 2);

                result.epoch_times = stimTimes(:)';

                % --- Sub-slice to analysis window for feature computation ---
                % Extraction window is the full epoch (-100 to +500ms)
                % Analysis window is the evoked response portion (e.g., 5 to 50ms)
                aw_mask = timeAxis >= analysis_start_ms & timeAxis <= analysis_end_ms;
                if sum(aw_mask) > 10
                    analysis_traces = traces(aw_mask, :);
                    analysis_time = timeAxis(aw_mask);
                else
                    % Fallback to full window if analysis window too small
                    analysis_traces = traces;
                    analysis_time = timeAxis;
                end
                result.analysis_window_ms = [analysis_start_ms, analysis_end_ms];
                result.analysis_n_samples = size(analysis_traces, 1);
                fprintf('[PIPELINE] Analysis window: [%d, %d] ms (%d samples)\n', ...
                    analysis_start_ms, analysis_end_ms, size(analysis_traces, 1));

                fe = FeatureExtractor();
                ALL_FEATURES = {'Line Length','Log(AUC)','Peak Amplitude','Trough Amplitude', ...
                    'Peak-to-Trough','RMS Amplitude','Peak Latency','Trough Latency', ...
                    'Max Slope','Max Slope Time','Early Area','Late Area','Early/Late Ratio', ...
                    'Recovery Tau','Recovery Slope','Template Correlation','PCA Recon Error', ...
                    'Variance','Autocorrelation','AC Width', ...
                    'Exp Fit A','Sum Power Low','Freq Moment Low','Sum Power High','Freq Moment High'};

                % Features computed on analysis window (not full extraction window)
                for fi = 1:length(ALL_FEATURES)
                    fname_clean = regexprep(ALL_FEATURES{fi}, '[^a-zA-Z0-9]', '_');
                    try
                        vals = fe.calculateFeatureMetric(ALL_FEATURES{fi}, analysis_traces, analysis_time, fs);
                        result.features.(fname_clean) = vals(:)';
                    catch
                        result.features.(fname_clean) = NaN(1, n_epochs);
                    end
                end
                result.num_features = length(ALL_FEATURES);
                fprintf('[PIPELINE] Features: %d metrics x %d epochs\n', length(ALL_FEATURES), n_epochs);

                % --- Mean evoked waveform (for dashboard plotting) ---
                result.mean_trace = mean(traces, 2, 'omitnan')';  % [1 x samples]
                result.sem_trace = (std(traces, 0, 2, 'omitnan') / sqrt(n_epochs))';
                result.time_axis_ms = timeAxis(:)';  % ms relative to stimulus
                result.n_trace_samples = length(timeAxis);

            catch e
                fprintf('[PIPELINE] Feature extraction failed: %s\n', e.message);
            end
        end

        % --- Step 6: Criticality analysis on EEG channel ---
        result.criticality = struct();
        result.criticality.time_points = [];
        result.criticality.db_values = [];
        result.criticality.db_stds = [];
        result.criticality.sigmas = [];

        try
            eeg_signal = sbuf(:, eeg_ch);
            analyzer = CriticalityAnalyzer([]);
            crit = analyzer.analyzeCriticalityFromEEG(eeg_signal, fs, crit_win, crit_overlap);

            if isfield(crit, 'criticalityDB')
                result.criticality.time_points = crit.timePoints(:)';
                result.criticality.db_values = crit.criticalityDB(:)';
                if isfield(crit, 'criticalitySD')
                    result.criticality.db_stds = crit.criticalitySD(:)';
                end
                if isfield(crit, 'sigma')
                    result.criticality.sigmas = crit.sigma(:)';
                end
                valid = crit.criticalityDB(~isnan(crit.criticalityDB));
                result.criticality_mean = mean(valid);
                result.criticality_std = std(valid);
                result.criticality_n_valid = length(valid);
                result.criticality_n_total = length(crit.criticalityDB);
                fprintf('[PIPELINE] Criticality: %d/%d valid windows, mean=%.3f\n', ...
                    length(valid), length(crit.criticalityDB), mean(valid));
            end
        catch e
            fprintf('[PIPELINE] Criticality failed: %s\n', e.message);
            result.criticality_mean = NaN;
            result.criticality_std = NaN;
        end

        % --- Step 7: LFP summary (PSD, stim artifact, per-channel stats) ---
        try
            eeg_signal = sbuf(:, eeg_ch);

            % PSD via Welch (subsample to ~500 points for JSON)
            nfft = min(2*fs, length(eeg_signal));
            [pxx, f] = pwelch(eeg_signal, hamming(nfft), [], [], fs);
            step = max(1, floor(length(f) / 500));
            result.lfp_psd_freqs = f(1:step:end)';
            result.lfp_psd_power = pxx(1:step:end)';

            % Stim artifact amplitude on stim copy channel
            if stim_ch <= n_channels
                stim_sig = sbuf(:, stim_ch);
                result.stim_artifact_rms = rms(stim_sig);
                result.stim_artifact_peak = max(abs(stim_sig));
            end

            % Per-channel LFP stats
            result.lfp_stats = struct();
            for ci = 1:n_channels
                ch_sig = sbuf(:, ci);
                ch_name = sprintf('ch%d', ci);
                result.lfp_stats.(ch_name).rms = rms(ch_sig);
                result.lfp_stats.(ch_name).mean = mean(ch_sig);
                result.lfp_stats.(ch_name).std_val = std(ch_sig);
                result.lfp_stats.(ch_name).peak = max(ch_sig);
                result.lfp_stats.(ch_name).trough = min(ch_sig);
                result.lfp_stats.(ch_name).p2p = max(ch_sig) - min(ch_sig);
            end
            fprintf('[PIPELINE] LFP stats: %d channels\n', n_channels);
        catch e
            fprintf('[PIPELINE] LFP summary failed: %s\n', e.message);
        end

        % --- Cleanup ---
        if exist(temp_mat, 'file'), delete(temp_mat); end

    catch main_err
        result.exit_status = 'error';
        result.error_message = main_err.message;
        fprintf('[PIPELINE] FATAL: %s\n', main_err.message);
    end

    result.pipeline_duration_sec = toc(t_start);
    fprintf('[PIPELINE] Done in %.1f sec\n', result.pipeline_duration_sec);

    % --- Write JSON ---
    try
        json_str = jsonencode(result);
        fid = fopen(output_json, 'w');
        fprintf(fid, '%s', json_str);
        fclose(fid);
    catch e
        fprintf('[PIPELINE] JSON write failed: %s\n', e.message);
    end
end


function val = get_cfg(cfg, field, default)
    if isstruct(cfg) && isfield(cfg, field)
        val = cfg.(field);
    else
        val = default;
    end
end
