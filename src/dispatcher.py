"""Dispatch queue: manages processing of new files through analysis tiers."""

import gc
import json
import logging
import os
import time

from src.db.store import Store
from src.utils.mat_loader import load_mat
from src.utils.session_config import SessionConfig, discover_from_session_dir
from src.analyzers.qc_basic import analyze_basic_qc
from src.analyzers.spectral import analyze_spectral
from src.analyzers.artifact import analyze_artifact
from src.analyzers.stim_qc import analyze_stim_report
from src.utils.matlab_bridge import run_pipeline as matlab_run_pipeline
from src.watcher import NewFile

logger = logging.getLogger("qc_monitor.dispatcher")


class Dispatcher:
    def __init__(self, store: Store, config: dict):
        self.store = store
        self.config = config
        self._session_configs: dict[str, SessionConfig] = {}

    def get_session_config(self, session_dir: str) -> SessionConfig:
        """Get or auto-discover session config for a session directory."""
        if session_dir not in self._session_configs:
            cfg = discover_from_session_dir(session_dir)
            self._session_configs[session_dir] = cfg
            # Persist to DB
            from dataclasses import asdict
            self.store.upsert_session_config(session_dir, asdict(cfg))
        return self._session_configs[session_dir]

    def process_file(self, new_file: NewFile, version_id: int = None) -> bool:
        """Process a single file through Tier 1 (Python QC) + Tier 2 (MATLAB pipeline)."""
        file_id = self.store.register_file(
            file_path=new_file.path,
            file_size=new_file.size,
            file_mtime=new_file.mtime,
            session_dir=new_file.session_dir,
            session_name=new_file.session_name,
            chunk_datetime=new_file.chunk_datetime,
        )

        self.store.update_file_status(file_id, "processing")

        # Auto-discover session config
        sess_cfg = self.get_session_config(new_file.session_dir)

        try:
            self.store.log_activity("INFO", "PROCESSING_START",
                                    f"Starting: {os.path.basename(new_file.path)}",
                                    file_id=file_id, file_path=new_file.path)

            # --- Tier 1: Python QC (channel-aware) with per-step timing ---
            t1_start = time.time()
            timings = {}

            t = time.time()
            chunk = load_mat(new_file.path)
            timings["load"] = time.time() - t

            self.store.update_file_status(file_id, "processing",
                                          num_channels=chunk.num_channels,
                                          num_samples=chunk.num_samples,
                                          sampling_rate=chunk.fs,
                                          duration_sec=chunk.duration_sec)

            qc_cfg = self.config.get("qc_thresholds", {})
            art_cfg = self.config.get("artifact", {})

            t = time.time()
            basic_results = analyze_basic_qc(
                chunk,
                clipping_voltage=qc_cfg.get("clipping_voltage", 10.0),
                flatline_std_threshold=qc_cfg.get("flatline_std", 1e-6),
            )
            timings["basic_qc"] = time.time() - t

            t = time.time()
            spectral_results = analyze_spectral(chunk, line_noise_freq=60.0)
            timings["spectral"] = time.time() - t

            t = time.time()
            # Only run artifact detection on LFP channels (skip stimCopy — saves ~30s)
            from src.utils.mat_loader import ChunkData
            import numpy as np
            lfp_only_signal = chunk.signal[:, sess_cfg.eeg_channels] if sess_cfg.eeg_channels else chunk.signal
            lfp_chunk = ChunkData(
                signal=lfp_only_signal, fs=chunk.fs,
                num_channels=lfp_only_signal.shape[1],
                num_samples=chunk.num_samples,
                duration_sec=chunk.duration_sec,
                source_path=chunk.source_path,
            )
            lfp_artifact = analyze_artifact(
                lfp_chunk,
                method=art_cfg.get("method", "fixed"),
                threshold=art_cfg.get("fixed_threshold", 500.0),
                mad_k=art_cfg.get("mad_k", 4.0),
                merge_gap_sec=art_cfg.get("merge_gap_sec", 2.0),
            )
            # Map back: stimCopy channels get 0% artifact, LFP channels get real values
            artifact_results = [{"artifact_pct": 0.0}] * chunk.num_channels
            for li, ch_idx in enumerate(sess_cfg.eeg_channels):
                if li < len(lfp_artifact) and ch_idx < chunk.num_channels:
                    artifact_results[ch_idx] = lfp_artifact[li]
            timings["artifact"] = time.time() - t

            t = time.time()
            for ch in range(chunk.num_channels):
                metrics = {}
                if ch < len(basic_results):
                    metrics.update(basic_results[ch])
                if ch < len(spectral_results):
                    metrics.update(spectral_results[ch])
                if ch < len(artifact_results):
                    metrics.update(artifact_results[ch])

                ch_name = sess_cfg.channel_names[ch] if ch < len(sess_cfg.channel_names) else f"Ch{ch}"
                ch_role = "unknown"
                if ch in sess_cfg.eeg_channels:
                    ch_role = "eeg"
                elif ch in sess_cfg.stim_copy_channels:
                    ch_role = "stim_copy"

                self.store.insert_chunk_qc(file_id, ch, metrics,
                                           channel_name=ch_name,
                                           channel_role=ch_role,
                                           version_id=version_id)
            timings["db_store"] = time.time() - t

            t1_elapsed = time.time() - t1_start
            timing_str = " | ".join(f"{k}={v:.1f}s" for k, v in timings.items())
            logger.info("Tier 1: %s (%.1fs: %s)",
                        os.path.basename(new_file.path), t1_elapsed, timing_str)

            # Log detailed timings to activity log
            self.store.log_activity("INFO", "TIER1_TIMING",
                                    f"{os.path.basename(new_file.path)} — {timing_str}",
                                    file_id=file_id, file_path=new_file.path,
                                    duration_sec=t1_elapsed)

            # Free the large signal array before MATLAB runs
            del chunk
            gc.collect()

            # Stim QC
            stim_results = analyze_stim_report(new_file.path)
            if stim_results:
                self.store.update_file_status(file_id, "processing", has_stim_report=1)
                for sr in stim_results:
                    self.store.insert_stim_qc(file_id, sr["stim_channel"], sr)

            # Video check
            video_path = new_file.path.rsplit(".", 1)[0] + "_v1.mp4"
            if os.path.isfile(video_path):
                self.store.update_file_status(file_id, "processing", has_video=1)

            # --- Tier 2: MATLAB Pipeline (channel-aware, configurable) ---
            matlab_enabled = self.config.get("matlab_exe", None) is not None
            if matlab_enabled:
                t2_start = time.time()

                # Build flat config for MATLAB from both extraction + analysis sections
                matlab_config = {
                    "stimnet_root": self.config.get("stimnet_root", ""),
                    **self.config.get("epoch_extraction", {}),
                    **self.config.get("feature_analysis", {}),
                    **self.config.get("criticality", {}),
                }

                matlab_exe = self.config.get("matlab_exe", "matlab")
                result = matlab_run_pipeline(
                    new_file.path, matlab_exe=matlab_exe,
                    timeout=300, config=matlab_config
                )
                t2_elapsed = time.time() - t2_start

                exit_status = result.get("exit_status", "unknown")
                matlab_dur = result.get("pipeline_duration_sec", t2_elapsed)
                n_stim = result.get("num_stimuli", 0)
                n_traces = result.get("num_traces", 0)
                n_feat = result.get("num_features", 0)
                logger.info("Tier 2: %s (%.1fs, MATLAB=%.1fs): %s, %d stimuli, %d traces, %d features",
                            os.path.basename(new_file.path), t2_elapsed, matlab_dur,
                            exit_status, n_stim, n_traces, n_feat)

                self.store.log_activity("INFO", "TIER2_TIMING",
                                        f"{os.path.basename(new_file.path)} — "
                                        f"total={t2_elapsed:.1f}s, matlab={matlab_dur:.1f}s, "
                                        f"stim={n_stim}, traces={n_traces}, feat={n_feat}, "
                                        f"status={exit_status}",
                                        file_id=file_id, file_path=new_file.path,
                                        duration_sec=t2_elapsed)

                # Store MATLAB results
                self.store.insert_matlab_result(file_id, result, version_id=version_id)

                # Store per-epoch evoked features
                features = result.get("features", {})
                epoch_times = result.get("epoch_times", [])
                if features and epoch_times:
                    n_epochs = len(epoch_times)
                    epochs = []
                    for ei in range(n_epochs):
                        epoch = {
                            "epoch_index": ei,
                            "epoch_time_sec": epoch_times[ei] if ei < len(epoch_times) else 0,
                        }
                        for fname, vals in features.items():
                            # Normalize feature name to DB column
                            col = _feature_to_column(fname)
                            if col and ei < len(vals):
                                epoch[col] = vals[ei]
                        epochs.append(epoch)
                    self.store.bulk_insert_evoked_features(file_id, epochs, version_id)

                    # Evoked summary
                    summary = {
                        "evoked_output_path": result.get("evoked_output_path", ""),
                        "num_stimuli_detected": result.get("num_stimuli", 0),
                        "num_traces_extracted": result.get("num_traces", 0),
                    }
                    pa = features.get("Peak_Amplitude", [])
                    if pa:
                        import numpy as np
                        arr = np.array([x for x in pa if x is not None and x == x])
                        if len(arr) > 0:
                            summary["mean_peak_amplitude"] = float(np.mean(arr))
                            summary["std_peak_amplitude"] = float(np.std(arr))
                    self.store.insert_evoked_summary(file_id, summary, version_id)

                # Store per-channel evoked waveforms (LFP + stim trace)
                per_channel = result.get("per_channel", {})
                evoked_cfg = self.config.get("feature_analysis", {})
                for ch_key, ch_data in per_channel.items():
                    if not isinstance(ch_data, dict):
                        continue
                    mean_trace = ch_data.get("mean_trace", [])
                    time_axis = ch_data.get("time_axis_ms", [])
                    if mean_trace and time_axis:
                        self.store.insert_evoked_waveform(
                            file_id,
                            channel=ch_data.get("channel", 0) - 1,  # 0-indexed
                            channel_name=ch_data.get("channel_name", ch_key),
                            time_axis=time_axis,
                            mean_trace=mean_trace,
                            sem_trace=ch_data.get("sem_trace", []),
                            stim_mean_trace=ch_data.get("stim_mean_trace", []),
                            n_epochs=ch_data.get("num_traces", 0),
                            analysis_start_ms=evoked_cfg.get("analysis_start_ms", 5.0),
                            analysis_end_ms=evoked_cfg.get("analysis_end_ms", 50.0),
                            version_id=version_id,
                        )

                # Store criticality windows
                crit = result.get("criticality", {})
                tp = crit.get("time_points", [])
                db = crit.get("db_values", [])
                if tp and db:
                    windows = []
                    stds = crit.get("db_stds", [])
                    sigs = crit.get("sigmas", [])
                    for wi in range(len(tp)):
                        windows.append({
                            "channel": result.get("criticality_channel", result.get("lfp_channels", [1])[0]) - 1,
                            "window_index": wi,
                            "time_sec": tp[wi],
                            "db_value": db[wi] if db[wi] == db[wi] else None,  # NaN check
                            "db_std": stds[wi] if wi < len(stds) and stds[wi] == stds[wi] else None,
                            "sigma": sigs[wi] if wi < len(sigs) else None,
                            "ar_order": self.config.get("criticality", {}).get("ar_order", 5),
                        })
                    self.store.bulk_insert_criticality(file_id, windows, version_id)

            total_elapsed = time.time() - t1_start
            self.store.update_file_status(file_id, "done")
            self.store.log_activity("INFO", "PROCESSING_DONE",
                                    f"Done: {os.path.basename(new_file.path)}",
                                    file_id=file_id, file_path=new_file.path,
                                    duration_sec=total_elapsed)
            return True

        except Exception as e:
            logger.error("Processing failed for %s: %s", new_file.path, e, exc_info=True)
            self.store.update_file_status(file_id, "error")
            self.store.log_activity("ERROR", "PROCESSING_FAILED",
                                    f"Failed: {os.path.basename(new_file.path)}: {e}",
                                    file_id=file_id, file_path=new_file.path)
            return False


# Map MATLAB feature struct field names to DB column names
_FEATURE_MAP = {
    "Line_Length": "line_length",
    "Log_AUC_": "log_auc",
    "Peak_Amplitude": "peak_amplitude",
    "Trough_Amplitude": "trough_amplitude",
    "Peak_to_Trough": "peak_to_trough",
    "RMS_Amplitude": "rms_amplitude",
    "Peak_Latency": "peak_latency_ms",
    "Trough_Latency": "trough_latency_ms",
    "Max_Slope": "max_slope",
    "Max_Slope_Time": "max_slope_time_ms",
    "Early_Area": "early_area",
    "Late_Area": "late_area",
    "Early_Late_Ratio": "early_late_ratio",
    "Recovery_Tau": "recovery_tau",
    "Recovery_Slope": "recovery_slope",
    "Template_Correlation": "template_correlation",
    "PCA_Recon_Error": "pca_recon_error",
    "Variance": "variance",
    "Autocorrelation": "autocorrelation",
    "AC_Width": "ac_width",
    "Exp_Fit_A": "exp_fit_a",
    "Sum_Power_Low": "sum_power_low",
    "Freq_Moment_Low": "freq_moment_low",
    "Sum_Power_High": "sum_power_high",
    "Freq_Moment_High": "freq_moment_high",
}


def _feature_to_column(matlab_field: str) -> str | None:
    return _FEATURE_MAP.get(matlab_field)
