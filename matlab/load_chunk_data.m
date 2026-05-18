function [sbuf, fs, trdata] = load_chunk_data(filepath)
% LOAD_CHUNK_DATA  Local shadow of the STiM-NET helper.
%
% Identical behavior except the final "channels(sbuf) == length(trdata)"
% guard is replaced with a non-fatal reconciliation step. We have seen
% files where the recorder wrote sbuf with N channels but the saved
% trdata struct still carried M >= N entries from a prior rig config
% (e.g. a 4-channel rig reduced to 2). The original helper hard-errors
% on that mismatch and aborts the entire pipeline, even though the
% downstream detector (batchEvokedWorkerFcn) does not consume trdata
% for KMrecorder data.
%
% Local copy lives in the qc_monitor matlab/ directory which run_pipeline
% prepends to the MATLAB path so this version takes precedence.

    try
        fprintf('Loading chunk file: %s\n', filepath);
        chunkData = load(filepath);

        if isfield(chunkData, 'data') && isfield(chunkData, 'chunkInfo')
            fprintf('Detected chunked data format\n');
            sbuf = double(chunkData.data);
            [numSamples, numChannels] = size(sbuf);
            fprintf('Data dimensions: %d samples x %d channels\n', numSamples, numChannels);

            if isfield(chunkData.chunkInfo, 'samplingRate')
                fs = chunkData.chunkInfo.samplingRate;
                fprintf('Sampling rate: %d Hz\n', fs);
            else
                fs = 20000;
                fprintf('Warning: No sampling rate found, using default %d Hz\n', fs);
            end

            if isfield(chunkData, 'timestamps') && ~isempty(chunkData.timestamps)
                timestamps = chunkData.timestamps;
                fprintf('Using existing timestamps\n');
            elseif isfield(chunkData, 'daqTimestamps') && ~isempty(chunkData.daqTimestamps)
                timestamps = chunkData.daqTimestamps;
                fprintf('Using DAQ timestamps\n');
            else
                if isfield(chunkData.chunkInfo, 'startIndex')
                    startTime = chunkData.chunkInfo.startIndex / fs;
                    timestamps = startTime + (0:numSamples-1)' / fs;
                else
                    timestamps = (0:numSamples-1)' / fs;
                end
            end

            if length(timestamps) == numSamples
                if timestamps(1) > 1e6
                    baseDate = datenum(1970, 1, 1);
                    timestamps = baseDate + timestamps / (24 * 3600);
                else
                    baseDate = now;
                    timestamps = baseDate + timestamps / (24 * 3600);
                end
            end

            trdata = struct('timestamp', []);
            for iChannel = 1:numChannels
                trdata(iChannel).timestamp = timestamps;
            end

        elseif isfield(chunkData, 'sbuf')
            fprintf('Detected original GUIDE format\n');
            sbuf = double(chunkData.sbuf);

            if isfield(chunkData, 'fs')
                fs = chunkData.fs;
            else
                fs = 20000;
                fprintf('Warning: No fs found, using default %d Hz\n', fs);
            end

            if isfield(chunkData, 'trdata')
                trdata = chunkData.trdata;
            else
                [numSamples, ~] = size(sbuf);
                timestamps = now + (0:numSamples-1)' / (fs * 24 * 3600);
                trdata = struct('timestamp', []);
                for iChannel = 1:size(sbuf, 2)
                    trdata(iChannel).timestamp = timestamps;
                end
                fprintf('Created minimal trdata structure\n');
            end

        else
            error('Unrecognized data format: file must contain either ''data'' or ''sbuf'' field');
        end

        if isempty(sbuf)
            error('No signal data found in file');
        end

        if fs <= 0
            error('Invalid sampling rate: %f', fs);
        end

        % --- Reconcile trdata length with sbuf channel count ---------------
        nChan = size(sbuf, 2);
        nTr = length(trdata);
        if nTr ~= nChan
            fprintf(['Warning: trdata length (%d) != sbuf channels (%d). ', ...
                     'Reconciling (downstream does not consume trdata for stim detect).\n'], ...
                     nTr, nChan);
            if nTr > nChan
                trdata = trdata(1:nChan);
            else
                [numSamples, ~] = size(sbuf);
                fillTimestamps = now + (0:numSamples-1)' / (fs * 24 * 3600);
                for iChannel = (nTr + 1):nChan
                    trdata(iChannel).timestamp = fillTimestamps;
                end
            end
        end

        fprintf('Data loading successful: %d samples, %d channels, %d Hz\n', ...
            size(sbuf, 1), size(sbuf, 2), fs);

    catch ME
        error('Failed to load chunk data from %s: %s', filepath, ME.message);
    end
end
