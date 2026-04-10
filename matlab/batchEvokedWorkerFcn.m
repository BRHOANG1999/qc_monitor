function result = batchEvokedWorkerFcn(filePath, params, animalConfigs, numAnimals)
    % batchEvokedWorkerFcn  Process one file for batch evoked extraction (parfeval-compatible)
    %
    %   result = batchEvokedWorkerFcn(filePath, params, animalConfigs, numAnimals)
    %
    %   Standalone worker function that processes a single file for all animals.
    %   Designed to run on a parallel worker without any UI dependencies.
    %
    %   Returns a lightweight struct with:
    %       success         - logical
    %       totalEpochs     - total epochs extracted across all animals
    %       animalSuccessCount - number of animals successfully processed
    %       outputPath      - path to saved output file (empty if failed)
    %       errorMessage    - error description (empty if success)
    %       statusText      - human-readable status string for table display

    result = struct('success', false, 'totalEpochs', 0, ...
                    'animalSuccessCount', 0, 'outputPath', '', ...
                    'errorMessage', '', 'statusText', 'Failed');

    try
        [~, fileName] = fileparts(filePath);
        allAnimalResults = struct();
        totalEpochs = 0;
        animalSuccessCount = 0;

        for a = 1:numAnimals
            animalConfig = animalConfigs(a);
            animalParams = params;
            animalParams.EEGChannel = animalConfig.eegChannel;
            animalParams.StimulusChannel = animalConfig.stimChannel;
            animalParams.AnimalID = animalConfig.animalID;

            fileResult = processSingleFileCore(filePath, animalParams);

            if fileResult.success
                fieldName = matlab.lang.makeValidName(animalConfig.animalID);
                animalStruct = struct(...
                    'evokedData', fileResult.evokedData, ...
                    'timeAxis', fileResult.timeAxis, ...
                    'stimulusIndices', fileResult.stimulusIndices, ...
                    'stimulusTimes', fileResult.stimulusTimes, ...
                    'numEpochs', fileResult.numEpochs, ...
                    'eegChannel', animalConfig.eegChannel, ...
                    'stimChannel', animalConfig.stimChannel, ...
                    'animalID', animalConfig.animalID);

                if isfield(fileResult, 'stimulusPeakAmplitudes')
                    animalStruct.stimulusTraces = fileResult.stimulusTraces;
                    animalStruct.stimulusPeakAmplitudes = fileResult.stimulusPeakAmplitudes;
                    animalStruct.stimulusTroughAmplitudes = fileResult.stimulusTroughAmplitudes;
                end

                allAnimalResults.(fieldName) = animalStruct;
                totalEpochs = totalEpochs + fileResult.numEpochs;
                animalSuccessCount = animalSuccessCount + 1;
            end
        end

        if animalSuccessCount > 0
            combinedResult = struct();
            combinedResult.allAnimalResults = allAnimalResults;
            combinedResult.metadata = struct(...
                'sourceFile', filePath, ...
                'numAnimals', numAnimals, ...
                'animalConfigs', animalConfigs, ...
                'processingParams', params, ...
                'extractionDatetime', datetime('now'));

            outputFileName = [fileName, params.OutputSuffix, '.mat'];
            outputPath = fullfile(params.OutputFolder, outputFileName);
            save(outputPath, '-struct', 'combinedResult', '-v7.3');

            if animalSuccessCount < numAnimals
                statusText = sprintf('Done (%d/%d animals, %d epochs)', animalSuccessCount, numAnimals, totalEpochs);
            else
                statusText = sprintf('Done (%d animals, %d epochs)', animalSuccessCount, totalEpochs);
            end

            result.success = true;
            result.totalEpochs = totalEpochs;
            result.animalSuccessCount = animalSuccessCount;
            result.outputPath = outputPath;
            result.statusText = statusText;
        else
            result.statusText = 'Failed';
            result.errorMessage = 'No animals processed successfully';
        end

    catch ex
        result.errorMessage = ex.message;
        result.statusText = 'Error';
    end
end

%% ========================================================================
%  Local helper functions (extracted from EvokedResponseWindow methods)
%  ========================================================================

function fileResult = processSingleFileCore(filePath, params)
    % Process a single file for one animal's channel configuration
    fileResult = struct();
    fileResult.success = false;
    fileResult.numEpochs = 0;
    fileResult.sourceFile = filePath;

    try
        data = load(filePath);

        recordingStartDatetime = extractRecordingDatetimeCore(filePath, data);

        % Find EEG data
        eegData = [];
        samplingRate = [];

        possibleEEGFields = {'sbuf', 'EEG', 'EEGData', 'eegData', 'eeg', 'data', 'signal', 'signals', 'raw'};
        for f = 1:length(possibleEEGFields)
            if isfield(data, possibleEEGFields{f})
                candidate = data.(possibleEEGFields{f});
                if isnumeric(candidate) && ~isempty(candidate)
                    eegData = candidate;
                    break;
                end
            end
        end

        if isempty(eegData) && isfield(data, 'chunk') && isfield(data.chunk, 'sbuf')
            eegData = data.chunk.sbuf;
        end

        if isempty(eegData)
            return;
        end

        % Find sampling rate
        possibleSRFields = {'SamplingRate', 'samplingRate', 'Fs', 'fs', 'sampleRate', 'sr'};
        for f = 1:length(possibleSRFields)
            if isfield(data, possibleSRFields{f})
                candidate = data.(possibleSRFields{f});
                if isnumeric(candidate) && candidate > 0
                    samplingRate = candidate;
                    break;
                end
            end
        end

        if isempty(samplingRate)
            samplingRate = 10000;
        end

        % Ensure [samples x channels]
        if size(eegData, 1) < size(eegData, 2)
            eegData = eegData';
        end

        numChannels = size(eegData, 2);

        if params.EEGChannel > numChannels
            return;
        end

        eegChannel = eegData(:, params.EEGChannel);

        % Apply filters
        if params.HighPassEnabled || params.LowPassEnabled || params.NotchFilter50Hz || params.NotchFilter60Hz
            eegChannel = applyFiltersCore(eegChannel, samplingRate, params);
        end

        % Get stimulus indices
        stimulusIndices = [];

        % NOTE (QC Monitor fix): KMrecorder trdata contains only recording
        % start/end timestamps (2 per channel), NOT stimulus event times.
        % Skip trdata entirely and fall through to stimulus channel detection.
        % Original code checked trdata first, which produced 2 garbage indices
        % instead of ~1800 real stimulus detections per hour at 0.5 Hz.
        %
        % if isfield(data, 'trdata') ...  % SKIPPED for KMrecorder data
        % end

        if isempty(stimulusIndices) && isfield(data, 'PulseEvents') && ~isempty(data.PulseEvents)
            if isnumeric(data.PulseEvents)
                stimulusIndices = round(data.PulseEvents * samplingRate);
            end
        end

        if isempty(stimulusIndices) && isfield(data, 'pulseEvents') && ~isempty(data.pulseEvents)
            if isnumeric(data.pulseEvents)
                stimulusIndices = round(data.pulseEvents * samplingRate);
            end
        end

        if isempty(stimulusIndices) && isfield(data, 'stimulusIndices') && ~isempty(data.stimulusIndices)
            stimulusIndices = data.stimulusIndices;
        end
        if isempty(stimulusIndices) && isfield(data, 'stimulusTimes') && ~isempty(data.stimulusTimes)
            stimulusIndices = round(data.stimulusTimes * samplingRate);
        end

        if isempty(stimulusIndices) && params.StimulusChannel <= numChannels
            stimChannel = eegData(:, params.StimulusChannel);
            stimulusIndices = detectStimuliCore(stimChannel, samplingRate, params);
        end

        if ~isempty(stimulusIndices)
            stimulusIndices = stimulusIndices(stimulusIndices > 0 & stimulusIndices <= size(eegChannel, 1));
        end

        if isempty(stimulusIndices)
            return;
        end

        % Extract evoked windows
        startSamples = round(params.StartMS * samplingRate / 1000);
        endSamples = round(params.EndMS * samplingRate / 1000);
        windowLength = endSamples - startSamples + 1;

        validEpochs = [];
        evokedData = [];

        for i = 1:length(stimulusIndices)
            stimIdx = stimulusIndices(i);
            startIdx = stimIdx + startSamples;
            endIdx = stimIdx + endSamples;

            if startIdx >= 1 && endIdx <= length(eegChannel)
                epoch = eegChannel(startIdx:endIdx);

                if params.BaselineCorrection
                    baselineStartSample = round((params.BaselineWindow(1) - params.StartMS) * samplingRate / 1000) + 1;
                    baselineEndSample = round((params.BaselineWindow(2) - params.StartMS) * samplingRate / 1000) + 1;
                    baselineStartSample = max(1, min(baselineStartSample, windowLength));
                    baselineEndSample = max(1, min(baselineEndSample, windowLength));

                    if baselineStartSample < baselineEndSample
                        baselineMean = mean(epoch(baselineStartSample:baselineEndSample));
                        epoch = epoch - baselineMean;
                    end
                end

                evokedData = [evokedData, epoch(:)]; %#ok<AGROW>
                validEpochs = [validEpochs, i]; %#ok<AGROW>
            end
        end

        if isempty(evokedData)
            return;
        end

        % Extract stimulus channel traces
        stimulusTraces = [];
        if params.StimulusChannel >= 1 && params.StimulusChannel <= numChannels
            stimChannelData = eegData(:, params.StimulusChannel);
            for i = 1:length(validEpochs)
                epochIdx = validEpochs(i);
                stimIdx = stimulusIndices(epochIdx);
                startIdx = stimIdx + startSamples;
                endIdx = stimIdx + endSamples;

                if startIdx >= 1 && endIdx <= length(stimChannelData)
                    stimEpoch = stimChannelData(startIdx:endIdx);
                    stimulusTraces = [stimulusTraces, stimEpoch(:)]; %#ok<AGROW>
                end
            end
        end

        % Build time axis
        timeAxis = linspace(params.StartMS, params.EndMS, windowLength);

        % Build result
        fileResult.evokedData = evokedData;
        fileResult.timeAxis = timeAxis;
        fileResult.stimulusIndices = stimulusIndices(validEpochs);
        fileResult.stimulusTimes = stimulusIndices(validEpochs) / samplingRate;
        fileResult.numEpochs = size(evokedData, 2);
        fileResult.success = true;

        if ~isempty(stimulusTraces)
            fileResult.stimulusTraces = stimulusTraces;
            fileResult.stimulusPeakAmplitudes = max(stimulusTraces, [], 1);
            fileResult.stimulusTroughAmplitudes = min(stimulusTraces, [], 1);
        end

        % Metadata
        fileResult.metadata = struct();
        fileResult.metadata.sourceFile = filePath;
        fileResult.metadata.samplingRate = samplingRate;
        fileResult.metadata.eegChannel = params.EEGChannel;
        fileResult.metadata.stimulusChannel = params.StimulusChannel;
        fileResult.metadata.windowStartMS = params.StartMS;
        fileResult.metadata.windowEndMS = params.EndMS;
        fileResult.metadata.baselineCorrection = params.BaselineCorrection;
        fileResult.metadata.baselineWindow = params.BaselineWindow;
        fileResult.metadata.extractionTimestamp = datetime('now');
        fileResult.metadata.recordingStartDatetime = recordingStartDatetime;

        % Filter settings
        fileResult.filterSettings = struct();
        fileResult.filterSettings.highPassEnabled = params.HighPassEnabled;
        fileResult.filterSettings.highPassCutoff = params.HighPassCutoff;
        fileResult.filterSettings.lowPassEnabled = params.LowPassEnabled;
        fileResult.filterSettings.lowPassCutoff = params.LowPassCutoff;
        fileResult.filterSettings.notch50Hz = params.NotchFilter50Hz;
        fileResult.filterSettings.notch60Hz = params.NotchFilter60Hz;

    catch ex
        fileResult.success = false;
        fileResult.errorMessage = ex.message;
    end
end

function filteredData = applyFiltersCore(data, samplingRate, params)
    % Apply filters to data for batch evoked extraction
    filteredData = data;
    nyquist = samplingRate / 2;

    try
        if params.HighPassEnabled && params.HighPassCutoff < nyquist
            [b, a] = butter(4, params.HighPassCutoff / nyquist, 'high');
            filteredData = filtfilt(b, a, filteredData);
        end

        if params.LowPassEnabled && params.LowPassCutoff < nyquist
            [b, a] = butter(4, params.LowPassCutoff / nyquist, 'low');
            filteredData = filtfilt(b, a, filteredData);
        end

        if params.NotchFilter50Hz && 50 < nyquist
            wo = 50 / nyquist;
            bw = wo / 35;
            [b, a] = iirnotch(wo, bw);
            filteredData = filtfilt(b, a, filteredData);
        end

        if params.NotchFilter60Hz && 60 < nyquist
            wo = 60 / nyquist;
            bw = wo / 35;
            [b, a] = iirnotch(wo, bw);
            filteredData = filtfilt(b, a, filteredData);
        end
    catch
        % If filtering fails, return original data
    end
end

function stimulusIndices = detectStimuliCore(stimChannel, samplingRate, params)
    % Detect stimulus events using first sample of peak plateau
    stimulusIndices = [];

    try
        stimMean = mean(stimChannel);
        stimStd = std(stimChannel);
        threshold = stimMean + params.StimulusThreshold * stimStd;

        minDistanceSamples = round(params.MinStimulusDistance * samplingRate);

        aboveThreshold = stimChannel > threshold;
        edges = diff([0; aboveThreshold(:)]);
        candidates = find(edges == 1);

        if isempty(candidates)
            return;
        end

        alignedIndices = zeros(length(candidates), 1);
        peakAmplitudes = zeros(length(candidates), 1);

        for i = 1:length(candidates)
            startIdx = candidates(i);

            endIdx = find(~aboveThreshold(startIdx:end), 1, 'first');
            if isempty(endIdx)
                endIdx = length(stimChannel);
            else
                endIdx = startIdx + endIdx - 2;
            end

            [peakVal, ~] = max(stimChannel(startIdx:endIdx));
            peakAmplitudes(i) = peakVal;

            peakTolerance = 0.01 * peakVal;
            peakRegion = stimChannel(startIdx:endIdx);
            firstPeakSample = find(peakRegion >= (peakVal - peakTolerance), 1, 'first');

            if ~isempty(firstPeakSample)
                alignedIndices(i) = startIdx + firstPeakSample - 1;
            else
                [~, peakRelIdx] = max(peakRegion);
                alignedIndices(i) = startIdx + peakRelIdx - 1;
            end
        end

        [~, sortOrder] = sort(peakAmplitudes, 'descend');
        alignedIndices = alignedIndices(sortOrder);

        selectedIndices = [];
        for i = 1:length(alignedIndices)
            idx = alignedIndices(i);
            if isempty(selectedIndices) || all(abs(selectedIndices - idx) >= minDistanceSamples)
                selectedIndices(end+1) = idx; %#ok<AGROW>
            end
        end

        stimulusIndices = sort(selectedIndices(:));

        % --- Validate: reject if detections look like noise, not real stimuli ---
        if length(stimulusIndices) < 3
            % Too few detections to be periodic stimulation
            stimulusIndices = [];
            return;
        end

        % Check ISI consistency: real stimulation has regular intervals
        isis = diff(stimulusIndices) / samplingRate;  % in seconds
        isi_cv = std(isis) / mean(isis);  % coefficient of variation

        % Also check: peak amplitudes should be well above noise
        % If median peak is < 5x the std of the stim channel, it's noise
        medianPeak = median(peakAmplitudes(peakAmplitudes > 0));
        snr = (medianPeak - stimMean) / stimStd;

        if isi_cv > 0.5 && snr < 5
            % Irregular timing AND low SNR = noise, not stimulation
            fprintf('[STIM DETECT] Rejected: CV=%.2f, SNR=%.1f — likely no stimulation\n', isi_cv, snr);
            stimulusIndices = [];
            return;
        end
    catch
        % Return empty if detection fails
    end
end

function stimulusIndices = extractStimulusIndicesFromTrdataCore(trdata, samplingRate)
    % Extract stimulus sample indices from trdata structure
    stimulusIndices = [];

    try
        if ~isstruct(trdata) || isempty(trdata)
            return;
        end

        numEvents = length(trdata);
        stimulusIndices = zeros(1, numEvents);

        for i = 1:numEvents
            event = trdata(i);

            if isfield(event, 'sampleIndex') && ~isempty(event.sampleIndex)
                stimulusIndices(i) = event.sampleIndex;
            elseif isfield(event, 'timestamp')
                ts = event.timestamp;
                if isdatetime(ts)
                    if i == 1
                        startTime = ts;
                        stimulusIndices(i) = 1;
                    else
                        stimulusIndices(i) = round(seconds(ts - startTime) * samplingRate) + 1;
                    end
                elseif isnumeric(ts)
                    stimulusIndices(i) = round(ts * samplingRate);
                end
            elseif isfield(event, 'time') && isnumeric(event.time)
                stimulusIndices(i) = round(event.time * samplingRate);
            elseif isfield(event, 'onset') && isnumeric(event.onset)
                stimulusIndices(i) = round(event.onset * samplingRate);
            end
        end

        stimulusIndices = stimulusIndices(stimulusIndices > 0);
    catch
        stimulusIndices = [];
    end
end

function recordingDT = extractRecordingDatetimeCore(filePath, data)
    % Extract recording start datetime from file metadata or filename
    recordingDT = NaT;

    % Check metadata struct
    if isfield(data, 'metadata')
        md = data.metadata;
        recordingFields = {'recordingStartDatetime', 'recordingTimestamp', ...
                           'startTime', 'sessionStartTime', 'recordingDateTime', ...
                           'dateTime', 'acquisitionTime', 'startDatetime'};

        for i = 1:length(recordingFields)
            if isfield(md, recordingFields{i})
                val = md.(recordingFields{i});
                if isdatetime(val) && ~isnat(val)
                    recordingDT = val;
                    return;
                elseif ischar(val) || isstring(val)
                    try
                        recordingDT = datetime(val);
                        if ~isnat(recordingDT)
                            return;
                        end
                    catch
                    end
                elseif isnumeric(val) && val > 7e5 && val < 8e5
                    try
                        recordingDT = datetime(val, 'ConvertFrom', 'datenum');
                        return;
                    catch
                    end
                end
            end
        end
    end

    % Check chunk struct
    if isfield(data, 'chunk')
        chunkFields = {'dateTime', 'recordingDateTime', 'startTime', 'timestamp'};
        for i = 1:length(chunkFields)
            if isfield(data.chunk, chunkFields{i})
                val = data.chunk.(chunkFields{i});
                if isdatetime(val) && ~isnat(val)
                    recordingDT = val;
                    return;
                end
            end
        end
    end

    % Check top-level fields
    topLevelFields = {'recordingDateTime', 'recordingTimestamp', 'sessionDateTime', 'startTime'};
    for i = 1:length(topLevelFields)
        if isfield(data, topLevelFields{i})
            val = data.(topLevelFields{i});
            if isdatetime(val) && ~isnat(val)
                recordingDT = val;
                return;
            end
        end
    end

    % Try parsing from filename pattern: *_YYYY_MM_DD__HH_MM_SS*
    [~, fileName] = fileparts(filePath);
    pattern = '(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})';
    tokens = regexp(fileName, pattern, 'tokens');

    if ~isempty(tokens)
        t = tokens{1};
        try
            recordingDT = datetime(str2double(t{1}), str2double(t{2}), str2double(t{3}), ...
                                   str2double(t{4}), str2double(t{5}), str2double(t{6}));
            return;
        catch
        end
    end

    % Fallback: use file modification time
    try
        fileInfo = dir(filePath);
        if ~isempty(fileInfo)
            recordingDT = datetime(fileInfo.datenum, 'ConvertFrom', 'datenum');
        end
    catch
    end
end
