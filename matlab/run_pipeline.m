function run_pipeline(input_file, output_json)
% RUN_PIPELINE  Headless analysis pipeline for QC monitor.
%   run_pipeline(INPUT_FILE, OUTPUT_JSON)
%
%   Loads a KMrecorder .mat file, runs evoked extraction + feature extraction
%   + criticality analysis, and writes results to a JSON file for Python.
%
%   INPUT_FILE  - path to KMrecorder .mat (sbuf format) or STiM-NET chunk
%   OUTPUT_JSON - path where results JSON will be written
%
%   Called by Python via: matlab -batch "run_pipeline('file.mat','result.json')"

    t_start = tic;
    result = struct();
    result.input_file = input_file;
    result.exit_status = 'success';
    result.error_message = '';

    try
        % Add STiM-NET paths
        stimnet_root = 'D:/code/Stimulation-Telemetry-Modulation-NeuroEngineering-Toolkit/daqSignalGenerator';
        addpath(genpath(fullfile(stimnet_root, 'src')));

        % --- Step 1: Load data using load_chunk_data (handles both formats) ---
        fprintf('[PIPELINE] Loading: %s\n', input_file);
        [sbuf, fs, trdata] = load_chunk_data(input_file);
        fprintf('[PIPELINE] Loaded: %d samples x %d channels @ %d Hz\n', ...
            size(sbuf, 1), size(sbuf, 2), fs);

        result.sampling_rate = fs;
        result.num_samples = size(sbuf, 1);
        result.num_channels = size(sbuf, 2);
        result.duration_sec = size(sbuf, 1) / fs;

        % --- Step 2: Prepare temp file for batchExtractEvokedResponses ---
        % batchExtractEvokedResponses expects 'EEG' field + 'fs' field
        [~, fname, ~] = fileparts(input_file);
        temp_dir = tempdir;
        temp_mat = fullfile(temp_dir, [fname '_pipeline_temp.mat']);

        EEG = sbuf;  % [samples x channels] - use column 1 as EEG, column 2 as stim
        save(temp_mat, 'EEG', 'fs', '-v7');
        fprintf('[PIPELINE] Temp file: %s\n', temp_mat);

        % --- Step 3: Run evoked extraction ---
        evoked_output_dir = fullfile(stimnet_root, 'evokedOutput');
        if ~exist(evoked_output_dir, 'dir')
            mkdir(evoked_output_dir);
        end

        eeg_channel = 1;
        stim_channel = min(2, size(sbuf, 2));  % channel 2 if available

        try
            [extract_results, output_files] = batchExtractEvokedResponses({temp_mat}, ...
                'PreStimulusMS', -100, ...
                'PostStimulusMS', 500, ...
                'EEGChannel', eeg_channel, ...
                'StimulusChannel', stim_channel, ...
                'BaselineCorrection', true, ...
                'BaselineWindow', [-60, -10], ...
                'NotchFilter60Hz', true, ...
                'OutputFolder', evoked_output_dir, ...
                'OutputSuffix', '_evoked');

            if ~isempty(extract_results)
                result.evoked_success = extract_results(1).success;
                result.num_stimuli = extract_results(1).numStimuli;
                result.num_traces = extract_results(1).numTraces;
                result.evoked_message = extract_results(1).message;
                if ~isempty(output_files)
                    result.evoked_output_path = output_files{1};
                end
            end
        catch evoked_err
            result.evoked_success = false;
            result.evoked_message = evoked_err.message;
            result.num_stimuli = 0;
            result.num_traces = 0;
            fprintf('[PIPELINE] Evoked extraction failed: %s\n', evoked_err.message);
        end

        % --- Step 4: Feature extraction (if evoked data exists) ---
        result.num_features = 0;
        result.feature_names = {};
        result.feature_means = [];

        if isfield(result, 'evoked_output_path') && exist(result.evoked_output_path, 'file')
            try
                evoked = load(result.evoked_output_path);
                if isfield(evoked, 'evokedData') && isfield(evoked.evokedData, 'traces')
                    traces = evoked.evokedData.traces;
                    timeAxis = evoked.evokedData.timeAxis;

                    fe = FeatureExtractor();
                    feature_names = {'Line Length', 'Peak Amplitude', 'Trough Amplitude', ...
                                     'RMS Amplitude', 'Peak Latency', 'Trough Latency'};
                    feature_means = zeros(1, length(feature_names));

                    for fi = 1:length(feature_names)
                        try
                            vals = fe.calculateFeatureMetric(feature_names{fi}, traces, timeAxis, fs);
                            feature_means(fi) = nanmean(vals);
                        catch
                            feature_means(fi) = NaN;
                        end
                    end

                    result.num_features = length(feature_names);
                    result.feature_names = feature_names;
                    result.feature_means = feature_means;
                end
            catch feat_err
                fprintf('[PIPELINE] Feature extraction failed: %s\n', feat_err.message);
            end
        end

        % --- Step 5: Criticality analysis ---
        result.criticality_mean = NaN;
        result.criticality_std = NaN;

        try
            eeg_signal = sbuf(:, eeg_channel);
            analyzer = CriticalityAnalyzer([]);
            crit_results = analyzer.analyzeCriticalityFromEEG(eeg_signal, fs, 2.0, 50);

            if isfield(crit_results, 'criticalityDB')
                valid_db = crit_results.criticalityDB(~isnan(crit_results.criticalityDB));
                if ~isempty(valid_db)
                    result.criticality_mean = mean(valid_db);
                    result.criticality_std = std(valid_db);
                end
            end
        catch crit_err
            fprintf('[PIPELINE] Criticality failed: %s\n', crit_err.message);
        end

        % Clean up temp file
        if exist(temp_mat, 'file')
            delete(temp_mat);
        end

    catch main_err
        result.exit_status = 'error';
        result.error_message = main_err.message;
        fprintf('[PIPELINE] FATAL: %s\n', main_err.message);
    end

    result.pipeline_duration_sec = toc(t_start);
    fprintf('[PIPELINE] Done in %.1f sec\n', result.pipeline_duration_sec);

    % --- Write results as JSON ---
    try
        json_str = jsonencode(result);
        fid = fopen(output_json, 'w');
        fprintf(fid, '%s', json_str);
        fclose(fid);
        fprintf('[PIPELINE] Results written to: %s\n', output_json);
    catch json_err
        fprintf('[PIPELINE] JSON write failed: %s\n', json_err.message);
    end
end
