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

    % --- Step 1 params: Epoch Extraction ---
    pre_ms = get_cfg(cfg, 'pre_stimulus_ms', -100);   % negative = before stimulus
    post_ms = get_cfg(cfg, 'post_stimulus_ms', 500);
    stim_thresh_std = get_cfg(cfg, 'stimulus_threshold_std', 3.0);
    min_stim_dist = get_cfg(cfg, 'min_stimulus_distance_sec', 0.1);
    baseline_on = get_cfg(cfg, 'baseline_correction', true);
    baseline_start = get_cfg(cfg, 'baseline_start_ms', -60);
    baseline_end = get_cfg(cfg, 'baseline_end_ms', -10);
    baseline_win = [baseline_start, baseline_end];
    notch60 = get_cfg(cfg, 'notch_60hz', true);
    notch50 = get_cfg(cfg, 'notch_50hz', false);
    hp_on = get_cfg(cfg, 'highpass_enabled', false);
    hp_hz = get_cfg(cfg, 'highpass_cutoff_hz', 1.0);
    lp_on = get_cfg(cfg, 'lowpass_enabled', false);
    lp_hz = get_cfg(cfg, 'lowpass_cutoff_hz', 1000.0);

    % --- Step 2 params: Feature Analysis (sub-window within epoch) ---
    analysis_start_ms = get_cfg(cfg, 'analysis_start_ms', 5);
    analysis_end_ms = get_cfg(cfg, 'analysis_end_ms', 50);
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
        % Rule: stimCopy channels = stimulus artifact. ALL others = LFP.
        raw = load(input_file);
        stim_channels = [];   % 1-indexed
        lfp_channels = [];    % 1-indexed
        channel_names = {};

        if isfield(raw, 'fnstr') && iscell(raw.fnstr)
            channel_names = raw.fnstr(1:min(n_channels, length(raw.fnstr)));
            for i = 1:length(channel_names)
                if contains(lower(channel_names{i}), 'stim')
                    stim_channels(end+1) = i;
                else
                    lfp_channels(end+1) = i;
                end
            end
        else
            % Fallback: ch1=stim, ch2+=LFP
            stim_channels = 1;
            lfp_channels = 2:n_channels;
            for i = 1:n_channels
                channel_names{i} = sprintf('Ch%d', i);
            end
        end

        stim_ch = stim_channels(1);  % use first stimCopy for detection
        result.channel_names = channel_names;
        result.stim_channels = stim_channels;
        result.lfp_channels = lfp_channels;
        fprintf('[PIPELINE] StimCopy: [%s], LFP: [%s]\n', ...
            num2str(stim_channels), num2str(lfp_channels));

        % --- Step 3: Save temp file for batchExtractEvokedResponses ---
        [~, fname, ~] = fileparts(input_file);
        temp_mat = fullfile(tempdir, [fname '_qc_temp.mat']);
        EEG = sbuf;
        save(temp_mat, 'EEG', 'fs', '-v7');

        evoked_output_dir = fullfile(stimnet_root, 'evokedOutput');
        if ~exist(evoked_output_dir, 'dir'), mkdir(evoked_output_dir); end

        % --- Step 4: Evoked extraction for EACH LFP channel ---
        result.per_channel = struct();
        result.num_stimuli = 0;
        result.num_traces = 0;
        result.evoked_success = false;

        for li = 1:length(lfp_channels)
            lfp_ch = lfp_channels(li);
            ch_name = channel_names{lfp_ch};
            ch_key = sprintf('ch%d', lfp_ch);

            ch_result = struct();
            ch_result.channel = lfp_ch;
            ch_result.channel_name = ch_name;
            ch_result.evoked_success = false;

            try
                [er, outfiles] = batchExtractEvokedResponses({temp_mat}, ...
                    'EEGChannel', lfp_ch, ...
                    'StimulusChannel', stim_ch, ...
                    'PreStimulusMS', pre_ms, ...  % pass as-is (negative = before stimulus)
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
                    'OutputSuffix', sprintf('_ch%d_evoked', lfp_ch));

                if ~isempty(er) && er(1).success
                    ch_result.evoked_success = true;
                    ch_result.num_stimuli = er(1).numStimuli;
                    ch_result.num_traces = er(1).numTraces;
                    ch_result.evoked_output_path = '';
                    if ~isempty(outfiles)
                        ch_result.evoked_output_path = outfiles{1};
                    end
                    result.num_stimuli = er(1).numStimuli;
                    result.num_traces = er(1).numTraces;
                    result.evoked_success = true;
                    fprintf('[PIPELINE] Ch%d (%s): %d stimuli, %d traces\n', ...
                        lfp_ch, ch_name, er(1).numStimuli, er(1).numTraces);

                    % Load evoked data for waveform + features
                    ev = load(ch_result.evoked_output_path);
                    traces = ev.evokedData.traces;
                    timeAxis = ev.evokedData.timeAxis;
                    stimTimes = ev.evokedData.stimulusTimes;
                    n_epochs = size(traces, 2);

                    % Mean LFP evoked waveform
                    ch_result.mean_trace = mean(traces, 2, 'omitnan')';
                    ch_result.sem_trace = (std(traces, 0, 2, 'omitnan') / sqrt(n_epochs))';
                    ch_result.time_axis_ms = timeAxis(:)';
                    ch_result.epoch_times = stimTimes(:)';

                    % Mean stim copy waveform (same epochs from stim channel)
                    stim_signal = sbuf(:, stim_ch);
                    stim_indices = round(stimTimes * fs) + 1;
                    pre_samp = round(abs(pre_ms) * fs / 1000);
                    post_samp = round(post_ms * fs / 1000);
                    n_stim_samp = pre_samp + post_samp;
                    stim_traces = zeros(n_stim_samp, length(stim_indices));
                    valid = 0;
                    for si = 1:length(stim_indices)
                        s = stim_indices(si) - pre_samp;
                        e = stim_indices(si) + post_samp - 1;
                        if s >= 1 && e <= length(stim_signal)
                            valid = valid + 1;
                            stim_traces(:, valid) = stim_signal(s:e);
                        end
                    end
                    if valid > 0
                        stim_traces = stim_traces(:, 1:valid);
                        ch_result.stim_mean_trace = mean(stim_traces, 2, 'omitnan')';
                    end
                end
            catch e
                ch_result.evoked_message = e.message;
                fprintf('[PIPELINE] Ch%d evoked failed: %s\n', lfp_ch, e.message);
            end

            result.per_channel.(ch_key) = ch_result;
        end

        % --- Step 5: Extract ALL 30 features per epoch (from first LFP channel) ---
        result.features = struct();
        result.num_features = 0;
        result.epoch_times = [];

        % Find first successful LFP channel for feature extraction
        primary_evoked_path = '';
        if result.evoked_success
            fnames = fieldnames(result.per_channel);
            for fi_ch = 1:length(fnames)
                ch_r = result.per_channel.(fnames{fi_ch});
                if isfield(ch_r, 'evoked_success') && ch_r.evoked_success && ...
                   isfield(ch_r, 'evoked_output_path') && exist(ch_r.evoked_output_path, 'file')
                    primary_evoked_path = ch_r.evoked_output_path;
                    result.epoch_times = ch_r.epoch_times;
                    break;
                end
            end
        end

        if ~isempty(primary_evoked_path)
            try
                ev = load(primary_evoked_path);
                traces = ev.evokedData.traces;
                timeAxis = ev.evokedData.timeAxis;
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

                % (Mean waveforms are stored per-channel in result.per_channel)

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
            primary_lfp = lfp_channels(1);  % first LFP channel
            eeg_signal = sbuf(:, primary_lfp);
            result.criticality_channel = primary_lfp;
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
