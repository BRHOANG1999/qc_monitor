function run_pipeline(input_file, output_json, config_json)
% RUN_PIPELINE  Full analysis pipeline for QC monitor.
%
%   Uses batchEvokedWorkerFcn (identical to STiM-NET Chronic tab batch extraction)
%   to produce allAnimalResults format output in evokedOutput/.
%
%   1. Auto-discovers channels from fnstr
%   2. Calls batchEvokedWorkerFcn → saves allAnimalResults + metadata to evokedOutput/
%   3. Extracts features per epoch from the evoked data
%   4. Runs criticality analysis
%   5. Writes results JSON for Python dashboard

    t_start = tic;
    result = struct();
    result.input_file = input_file;
    result.exit_status = 'success';
    result.error_message = '';

    % --- Load config overrides ---
    cfg = struct();
    if nargin >= 3 && ~isempty(config_json) && exist(config_json, 'file')
        try
            cfg = jsondecode(fileread(config_json));
        catch
        end
    end

    % --- Extraction params ---
    pre_ms = get_cfg(cfg, 'pre_stimulus_ms', -100);
    post_ms = get_cfg(cfg, 'post_stimulus_ms', 500);
    stim_thresh = get_cfg(cfg, 'stimulus_threshold_std', 3.0);
    min_stim_dist = get_cfg(cfg, 'min_stimulus_distance_sec', 0.1);
    baseline_on = get_cfg(cfg, 'baseline_correction', true);
    bl_start = get_cfg(cfg, 'baseline_start_ms', -60);
    bl_end = get_cfg(cfg, 'baseline_end_ms', -10);
    notch60 = get_cfg(cfg, 'notch_60hz', true);
    notch50 = get_cfg(cfg, 'notch_50hz', false);
    hp_on = get_cfg(cfg, 'highpass_enabled', false);
    hp_hz = get_cfg(cfg, 'highpass_cutoff_hz', 1.0);
    lp_on = get_cfg(cfg, 'lowpass_enabled', false);
    lp_hz = get_cfg(cfg, 'lowpass_cutoff_hz', 1000.0);

    % --- Feature analysis params ---
    analysis_start_ms = get_cfg(cfg, 'analysis_start_ms', 5);
    analysis_end_ms = get_cfg(cfg, 'analysis_end_ms', 50);
    crit_ar = get_cfg(cfg, 'ar_order', 5);
    crit_win = get_cfg(cfg, 'window_sec', 2.0);
    crit_overlap = get_cfg(cfg, 'overlap_pct', 50);

    try
        % --- Add STiM-NET paths ---
        stimnet_root = get_cfg(cfg, 'stimnet_root', ...
            'D:/code/Stimulation-Telemetry-Modulation-NeuroEngineering-Toolkit/daqSignalGenerator');
        addpath(genpath(fullfile(stimnet_root, 'src')));

        % --- Step 1: Load & auto-discover channels ---
        fprintf('[PIPELINE] Loading: %s\n', input_file);
        [sbuf, fs, ~] = load_chunk_data(input_file);
        [n_samples, n_channels] = size(sbuf);
        fprintf('[PIPELINE] %d samples x %d ch @ %d Hz\n', n_samples, n_channels, fs);

        result.sampling_rate = fs;
        result.num_samples = n_samples;
        result.num_channels = n_channels;
        result.duration_sec = n_samples / fs;

        raw = load(input_file);
        stim_channels = [];
        lfp_channels = [];
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
            stim_channels = 1;
            lfp_channels = 2:n_channels;
            for i = 1:n_channels
                channel_names{i} = sprintf('Ch%d', i);
            end
        end

        result.channel_names = channel_names;
        result.stim_channels = stim_channels;
        result.lfp_channels = lfp_channels;
        fprintf('[PIPELINE] StimCopy: [%s], LFP: [%s]\n', ...
            num2str(stim_channels), num2str(lfp_channels));

        % --- Step 2: Build animalConfigs for batchEvokedWorkerFcn ---
        % Each LFP channel is an "animal" with paired stim channel
        animalConfigs = [];
        for i = 1:length(lfp_channels)
            ac = struct();
            ac.eegChannel = lfp_channels(i);
            ac.stimChannel = stim_channels(1);
            ac.animalID = channel_names{lfp_channels(i)};
            if isempty(animalConfigs)
                animalConfigs = ac;
            else
                animalConfigs(end+1) = ac;
            end
        end

        % Build params struct matching batchEvokedWorkerFcn format
        evParams = struct();
        evParams.StartMS = pre_ms;
        evParams.EndMS = post_ms;
        evParams.StimulusThreshold = stim_thresh;
        evParams.MinStimulusDistance = min_stim_dist;
        evParams.BaselineCorrection = baseline_on;
        evParams.BaselineWindow = [bl_start, bl_end];
        evParams.HighPassEnabled = hp_on;
        evParams.HighPassCutoff = hp_hz;
        evParams.LowPassEnabled = lp_on;
        evParams.LowPassCutoff = lp_hz;
        evParams.NotchFilter50Hz = notch50;
        evParams.NotchFilter60Hz = notch60;
        evParams.OutputSuffix = '_evoked';

        evoked_output_dir = fullfile(stimnet_root, 'evokedOutput');
        if ~exist(evoked_output_dir, 'dir'), mkdir(evoked_output_dir); end
        evParams.OutputFolder = evoked_output_dir;

        % --- Step 3: Call batchEvokedWorkerFcn (identical to Chronic tab) ---
        fprintf('[PIPELINE] Running batchEvokedWorkerFcn...\n');
        worker_result = batchEvokedWorkerFcn(input_file, evParams, animalConfigs, length(animalConfigs));

        result.evoked_success = worker_result.success;
        result.num_stimuli = 0;
        result.num_traces = worker_result.totalEpochs;
        result.evoked_output_path = worker_result.outputPath;
        fprintf('[PIPELINE] Worker: %s, %d epochs, output: %s\n', ...
            worker_result.statusText, worker_result.totalEpochs, worker_result.outputPath);

        % --- Step 4: Extract waveforms + features from the output ---
        result.per_channel = struct();
        result.features = struct();
        result.num_features = 0;
        result.epoch_times = [];

        if worker_result.success && ~isempty(worker_result.outputPath) && exist(worker_result.outputPath, 'file')
            evoked_data = load(worker_result.outputPath);

            if isfield(evoked_data, 'allAnimalResults')
                animals = fieldnames(evoked_data.allAnimalResults);

                for ai = 1:length(animals)
                    animal = evoked_data.allAnimalResults.(animals{ai});
                    ch_key = matlab.lang.makeValidName(animal.animalID);

                    ch_result = struct();
                    ch_result.channel = animal.eegChannel;
                    ch_result.channel_name = animal.animalID;
                    ch_result.evoked_success = true;
                    ch_result.num_stimuli = animal.numEpochs;
                    ch_result.num_traces = animal.numEpochs;

                    % Mean waveforms
                    traces = animal.evokedData;
                    n_epochs = size(traces, 2);
                    ch_result.mean_trace = mean(traces, 2, 'omitnan')';
                    ch_result.sem_trace = (std(traces, 0, 2, 'omitnan') / sqrt(n_epochs))';
                    ch_result.time_axis_ms = animal.timeAxis(:)';
                    ch_result.epoch_times = animal.stimulusTimes(:)';

                    % Stim copy mean trace
                    if isfield(animal, 'stimulusTraces') && ~isempty(animal.stimulusTraces)
                        ch_result.stim_mean_trace = mean(animal.stimulusTraces, 2, 'omitnan')';
                    end

                    result.per_channel.(ch_key) = ch_result;
                    result.num_stimuli = max(result.num_stimuli, animal.numEpochs);

                    % Feature extraction on FIRST animal (primary LFP)
                    if ai == 1
                        result.epoch_times = animal.stimulusTimes(:)';
                        try
                            timeAxis = animal.timeAxis;
                            aw_mask = timeAxis >= analysis_start_ms & timeAxis <= analysis_end_ms;
                            if sum(aw_mask) > 10
                                a_traces = traces(aw_mask, :);
                                a_time = timeAxis(aw_mask);
                            else
                                a_traces = traces;
                                a_time = timeAxis;
                            end

                            fe = FeatureExtractor();
                            ALL_FEATURES = {'Line Length','Log(AUC)','Peak Amplitude','Trough Amplitude', ...
                                'Peak-to-Trough','RMS Amplitude','Peak Latency','Trough Latency', ...
                                'Max Slope','Max Slope Time','Early Area','Late Area','Early/Late Ratio', ...
                                'Recovery Tau','Recovery Slope','Template Correlation','PCA Recon Error', ...
                                'Variance','Autocorrelation','AC Width', ...
                                'Exp Fit A','Sum Power Low','Freq Moment Low','Sum Power High','Freq Moment High'};

                            for fi = 1:length(ALL_FEATURES)
                                fn = regexprep(ALL_FEATURES{fi}, '[^a-zA-Z0-9]', '_');
                                try
                                    vals = fe.calculateFeatureMetric(ALL_FEATURES{fi}, a_traces, a_time, fs);
                                    result.features.(fn) = vals(:)';
                                catch
                                    result.features.(fn) = NaN(1, n_epochs);
                                end
                            end
                            result.num_features = length(ALL_FEATURES);
                            fprintf('[PIPELINE] Features: %d x %d epochs\n', length(ALL_FEATURES), n_epochs);
                        catch e
                            fprintf('[PIPELINE] Features failed: %s\n', e.message);
                        end
                    end
                end
            end
        end

        % --- Step 5: Criticality ---
        result.criticality = struct();
        result.criticality.time_points = [];
        result.criticality.db_values = [];
        result.criticality.db_stds = [];
        result.criticality.sigmas = [];

        try
            primary_lfp = lfp_channels(1);
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
                valid_db = crit.criticalityDB(~isnan(crit.criticalityDB));
                result.criticality_mean = mean(valid_db);
                result.criticality_std = std(valid_db);
                fprintf('[PIPELINE] Criticality: %d/%d valid\n', length(valid_db), length(crit.criticalityDB));
            end
        catch e
            fprintf('[PIPELINE] Criticality failed: %s\n', e.message);
            result.criticality_mean = NaN;
            result.criticality_std = NaN;
        end

        % --- Step 6: LFP summary ---
        try
            primary_lfp = lfp_channels(1);
            eeg_signal = sbuf(:, primary_lfp);
            nfft = min(2*fs, length(eeg_signal));
            [pxx, f] = pwelch(eeg_signal, hamming(nfft), [], [], fs);
            step = max(1, floor(length(f) / 500));
            result.lfp_psd_freqs = f(1:step:end)';
            result.lfp_psd_power = pxx(1:step:end)';

            stim_ch = stim_channels(1);
            stim_sig = sbuf(:, stim_ch);
            result.stim_artifact_rms = rms(stim_sig);
            result.stim_artifact_peak = max(abs(stim_sig));
        catch e
            fprintf('[PIPELINE] LFP summary failed: %s\n', e.message);
        end

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
