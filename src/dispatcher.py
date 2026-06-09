"""Dispatch queue: manages processing of new files through analysis tiers.

A single ``process_file`` call walks one .mat through three tiers in
order, persisting at each stage:

* **Tier 1 (Python QC)** -- load the chunk, run the four analyzers
  (basic, spectral, artifact, stim_qc), insert one ``chunk_qc`` row
  per channel. ``_run_tier1_python``.
* **Tier 2 (MATLAB pipeline)** -- shell out to ``run_pipeline.m``,
  parse the result JSON, and fan it out into ``matlab_results``,
  ``evoked_features``, ``evoked_summary``, ``evoked_waveforms``, and
  ``criticality``. ``_run_tier2_matlab``.
* **Tier 3 (Video frame QC)** -- sample frames of the companion mp4,
  classify each as dark/bright/uniform, insert a ``video_qc`` row,
  and optionally email an alert when bad-frame % is critical.
  ``_run_tier3_video``.

The tiers used to be one ~280-line method. Splitting them out is
purely a readability move -- behaviour, ordering, and error
boundaries are unchanged. Each tier still aborts the whole file on
exception (caught at the bottom of ``process_file``); Tier 3 also
swallows its own analyzer / email failures internally so a broken
codec can't kill a file that already has Tier 1 + 2 results.
"""

import gc
import logging
import os
import time

import numpy as np

from src.db.store import Store
from src.utils.mat_loader import ChunkData, load_mat
from src.utils.session_config import SessionConfig, discover_from_session_dir
from src.analyzers.qc_basic import analyze_basic_qc
from src.analyzers.spectral import analyze_spectral
from src.analyzers.artifact import analyze_artifact
from src.analyzers.stim_qc import analyze_stim_report
from src.analyzers.video_qc import analyze_video
from src.utils.feature_map import feature_to_column
from src.utils.matlab_bridge import run_pipeline as matlab_run_pipeline
from src.utils.video import video_path_for_mat
from src.watcher import NewFile

logger = logging.getLogger("qc_monitor.dispatcher")


class Dispatcher:
    def __init__(self, store: Store, config: dict, emailer=None):
        """*emailer* is optional so unit tests can drive the dispatcher
        without an SMTP stack. When None, Tier 3 video alerts log-and-
        drop instead of falling back to ad-hoc EmailAlerter construction
        (the previous behaviour built a second EmailAlerter inline,
        which was wasteful and untestable).
        """
        self.store = store
        self.config = config
        self.emailer = emailer
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

    # ------------------------------------------------------------------ #
    #  Orchestrator
    # ------------------------------------------------------------------ #

    def process_file(self, new_file: NewFile, version_id: int = None) -> bool:
        """Walk one file through Tier 1 -> 2 -> 3 in order.

        Returns True on a clean pass, False if any tier raised. Tier 3
        failures don't propagate (see ``_run_tier3_video`` docstring).
        """
        file_id = self.store.register_file(
            file_path=new_file.path,
            file_size=new_file.size,
            file_mtime=new_file.mtime,
            session_dir=new_file.session_dir,
            session_name=new_file.session_name,
            chunk_datetime=new_file.chunk_datetime,
        )
        self.store.update_file_status(file_id, "processing")
        sess_cfg = self.get_session_config(new_file.session_dir)

        try:
            self.store.log_activity(
                "INFO", "PROCESSING_START",
                f"Starting: {os.path.basename(new_file.path)}",
                file_id=file_id, file_path=new_file.path,
            )
            t_total_start = time.time()

            # Tier 1: load + analyze + persist per-channel metrics.
            chunk = self._load_chunk(file_id, new_file)
            self._run_tier1_python(file_id, new_file, chunk,
                                    sess_cfg, version_id)
            # Free the (often >500 MB) signal array before the heavy
            # MATLAB subprocess runs so the parent process doesn't
            # peak twice.
            del chunk
            gc.collect()

            # Cheap metadata sweeps that share Tier-1's "is this file
            # interesting?" question but don't need the loaded signal.
            self._record_stim_and_video_metadata(file_id, new_file)

            # Tier 2 is gated by config -- a deployment without MATLAB
            # still benefits from Tier 1 + Tier 3.
            if self.config.get("matlab_exe") is not None:
                self._run_tier2_matlab(file_id, new_file, version_id)

            # Tier 3: companion video QC + alert.
            video_cfg = self.config.get("video_qc", {}) or {}
            if video_cfg.get("enabled", True):
                video_path = video_path_for_mat(new_file.path)
                if video_path is not None:
                    self._run_tier3_video(file_id, new_file.path,
                                            video_path, video_cfg,
                                            version_id)

            total_elapsed = time.time() - t_total_start
            self.store.update_file_status(file_id, "done")
            self.store.log_activity(
                "INFO", "PROCESSING_DONE",
                f"Done: {os.path.basename(new_file.path)}",
                file_id=file_id, file_path=new_file.path,
                duration_sec=total_elapsed,
            )
            return True

        except Exception as e:
            logger.error("Processing failed for %s: %s",
                          new_file.path, e, exc_info=True)
            self.store.update_file_status(file_id, "error")
            self.store.log_activity(
                "ERROR", "PROCESSING_FAILED",
                f"Failed: {os.path.basename(new_file.path)}: {e}",
                file_id=file_id, file_path=new_file.path,
            )
            return False

    # ------------------------------------------------------------------ #
    #  Tier 1: Python QC
    # ------------------------------------------------------------------ #

    def _load_chunk(self, file_id: int, new_file: NewFile) -> ChunkData:
        """Load the .mat and stamp the file row with chunk metadata."""
        chunk = load_mat(new_file.path)
        self.store.update_file_status(
            file_id, "processing",
            num_channels=chunk.num_channels,
            num_samples=chunk.num_samples,
            sampling_rate=chunk.fs,
            duration_sec=chunk.duration_sec,
        )
        return chunk

    def _run_tier1_python(self, file_id: int, new_file: NewFile,
                          chunk: ChunkData, sess_cfg: SessionConfig,
                          version_id: int | None) -> None:
        """Run the four Tier-1 analyzers and write one chunk_qc row
        per channel.

        Artifact detection only runs on EEG channels -- doing it on
        stim_copy traces would either fire on every pulse (useless)
        or take ~30 s longer per file. The stim_copy channels get a
        zero artifact_pct so the per-channel join in chunk_qc still
        has a row for every channel.
        """
        t1_start = time.time()
        timings: dict[str, float] = {}

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
        artifact_results = self._compute_artifact_results(
            chunk, sess_cfg, art_cfg,
        )
        timings["artifact"] = time.time() - t

        t = time.time()
        self._write_chunk_qc_rows(
            file_id, chunk, sess_cfg, version_id,
            basic_results, spectral_results, artifact_results,
        )
        timings["db_store"] = time.time() - t

        t1_elapsed = time.time() - t1_start
        timing_str = " | ".join(f"{k}={v:.1f}s" for k, v in timings.items())
        logger.info("Tier 1: %s (%.1fs: %s)",
                     os.path.basename(new_file.path), t1_elapsed, timing_str)
        self.store.log_activity(
            "INFO", "TIER1_TIMING",
            f"{os.path.basename(new_file.path)} — {timing_str}",
            file_id=file_id, file_path=new_file.path,
            duration_sec=t1_elapsed,
        )

    def _compute_artifact_results(self, chunk: ChunkData,
                                   sess_cfg: SessionConfig,
                                   art_cfg: dict) -> list[dict]:
        """Run analyze_artifact on EEG channels only and remap the
        results back to the chunk's full channel layout (so the order
        matches chunk.signal exactly). Stim-copy channels get a zero
        artifact_pct row.
        """
        lfp_only_signal = (
            chunk.signal[:, sess_cfg.eeg_channels]
            if sess_cfg.eeg_channels else chunk.signal
        )
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
        artifact_results: list[dict] = [
            {"artifact_pct": 0.0}] * chunk.num_channels
        for li, ch_idx in enumerate(sess_cfg.eeg_channels):
            if li < len(lfp_artifact) and ch_idx < chunk.num_channels:
                artifact_results[ch_idx] = lfp_artifact[li]
        return artifact_results

    def _write_chunk_qc_rows(self, file_id: int, chunk: ChunkData,
                              sess_cfg: SessionConfig,
                              version_id: int | None,
                              basic_results: list[dict],
                              spectral_results: list[dict],
                              artifact_results: list[dict]) -> None:
        """Merge the three analyzer outputs per channel and write a
        single chunk_qc row each."""
        for ch in range(chunk.num_channels):
            metrics: dict = {}
            if ch < len(basic_results):
                metrics.update(basic_results[ch])
            if ch < len(spectral_results):
                metrics.update(spectral_results[ch])
            if ch < len(artifact_results):
                metrics.update(artifact_results[ch])

            ch_name = (sess_cfg.channel_names[ch]
                        if ch < len(sess_cfg.channel_names) else f"Ch{ch}")
            ch_role = "unknown"
            if ch in sess_cfg.eeg_channels:
                ch_role = "eeg"
            elif ch in sess_cfg.stim_copy_channels:
                ch_role = "stim_copy"

            self.store.insert_chunk_qc(
                file_id, ch, metrics,
                channel_name=ch_name, channel_role=ch_role,
                version_id=version_id,
            )

    def _record_stim_and_video_metadata(self, file_id: int,
                                          new_file: NewFile) -> None:
        """Two cheap path-only sweeps that update has_stim_report /
        has_video on the file row without touching the loaded chunk."""
        stim_results = analyze_stim_report(new_file.path)
        if stim_results:
            self.store.update_file_status(file_id, "processing",
                                          has_stim_report=1)
            for sr in stim_results:
                self.store.insert_stim_qc(file_id, sr["stim_channel"], sr)

        if video_path_for_mat(new_file.path) is not None:
            self.store.update_file_status(file_id, "processing", has_video=1)

    # ------------------------------------------------------------------ #
    #  Tier 2: MATLAB pipeline
    # ------------------------------------------------------------------ #

    def _run_tier2_matlab(self, file_id: int, new_file: NewFile,
                           version_id: int | None) -> None:
        """Shell out to MATLAB, parse the result, fan it out across the
        five evoked / criticality tables."""
        t2_start = time.time()

        # Flatten the three config sections MATLAB needs into one dict
        # so run_pipeline.m doesn't have to walk nested groups.
        matlab_config = {
            "stimnet_root": self.config.get("stimnet_root", ""),
            **self.config.get("epoch_extraction", {}),
            **self.config.get("feature_analysis", {}),
            **self.config.get("criticality", {}),
        }
        matlab_exe = self.config.get("matlab_exe", "matlab")
        result = matlab_run_pipeline(
            new_file.path, matlab_exe=matlab_exe,
            timeout=300, config=matlab_config,
        )
        t2_elapsed = time.time() - t2_start

        self._log_tier2(new_file, result, t2_elapsed, file_id)
        self.store.insert_matlab_result(file_id, result, version_id=version_id)
        self._persist_evoked_features(file_id, result, version_id)
        self._persist_evoked_waveforms(file_id, result, version_id)
        self._persist_criticality(file_id, result, version_id)

    def _log_tier2(self, new_file: NewFile, result: dict,
                    t2_elapsed: float, file_id: int) -> None:
        exit_status = result.get("exit_status", "unknown")
        matlab_dur = result.get("pipeline_duration_sec", t2_elapsed)
        n_stim = result.get("num_stimuli", 0)
        n_traces = result.get("num_traces", 0)
        n_feat = result.get("num_features", 0)
        logger.info(
            "Tier 2: %s (%.1fs, MATLAB=%.1fs): %s, "
            "%d stimuli, %d traces, %d features",
            os.path.basename(new_file.path), t2_elapsed, matlab_dur,
            exit_status, n_stim, n_traces, n_feat,
        )
        self.store.log_activity(
            "INFO", "TIER2_TIMING",
            f"{os.path.basename(new_file.path)} — "
            f"total={t2_elapsed:.1f}s, matlab={matlab_dur:.1f}s, "
            f"stim={n_stim}, traces={n_traces}, feat={n_feat}, "
            f"status={exit_status}",
            file_id=file_id, file_path=new_file.path,
            duration_sec=t2_elapsed,
        )

    def _persist_evoked_features(self, file_id: int, result: dict,
                                   version_id: int | None) -> None:
        """Write one evoked_features row per epoch + one evoked_summary
        row aggregating peak amplitude. No-op when MATLAB returned no
        epochs (the typical no-stimuli outcome)."""
        features = result.get("features", {})
        epoch_times = result.get("epoch_times", [])
        if not (features and epoch_times):
            return
        n_epochs = len(epoch_times)
        epochs = []
        for ei in range(n_epochs):
            epoch = {
                "epoch_index": ei,
                "epoch_time_sec": (
                    epoch_times[ei] if ei < len(epoch_times) else 0),
            }
            for fname, vals in features.items():
                col = feature_to_column(fname)
                if col and ei < len(vals):
                    epoch[col] = vals[ei]
            epochs.append(epoch)
        self.store.bulk_insert_evoked_features(file_id, epochs, version_id)

        summary = {
            "evoked_output_path": result.get("evoked_output_path", ""),
            "num_stimuli_detected": result.get("num_stimuli", 0),
            "num_traces_extracted": result.get("num_traces", 0),
        }
        pa = features.get("Peak_Amplitude", [])
        if pa:
            arr = np.array([x for x in pa if x is not None and x == x])
            if len(arr) > 0:
                summary["mean_peak_amplitude"] = float(np.mean(arr))
                summary["std_peak_amplitude"] = float(np.std(arr))
        self.store.insert_evoked_summary(file_id, summary, version_id)

    def _persist_evoked_waveforms(self, file_id: int, result: dict,
                                    version_id: int | None) -> None:
        """Per-channel mean evoked waveform + the stim trace, one row
        per channel that MATLAB reported on."""
        per_channel = result.get("per_channel", {})
        evoked_cfg = self.config.get("feature_analysis", {})
        for ch_key, ch_data in per_channel.items():
            if not isinstance(ch_data, dict):
                continue
            mean_trace = ch_data.get("mean_trace", [])
            time_axis = ch_data.get("time_axis_ms", [])
            if not (mean_trace and time_axis):
                continue
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

    def _persist_criticality(self, file_id: int, result: dict,
                               version_id: int | None) -> None:
        """Walk the criticality time series and write one row per
        window. MATLAB's jsonencode collapses a 1-element vector to a
        scalar so lfp_channels can be either an int or a list -- the
        cast on the first line below is the only place that quirk
        leaks into Python.
        """
        crit = result.get("criticality", {})
        tp = crit.get("time_points", [])
        db = crit.get("db_values", [])
        if not (tp and db):
            return
        lfp_ch_raw = result.get("lfp_channels", [1])
        lfp_ch_first = (lfp_ch_raw[0]
                        if isinstance(lfp_ch_raw, (list, tuple))
                        else lfp_ch_raw)
        crit_channel_1based = result.get("criticality_channel", lfp_ch_first)
        ar_order = self.config.get("criticality", {}).get("ar_order", 5)
        stds = crit.get("db_stds", [])
        sigs = crit.get("sigmas", [])
        windows = []
        for wi in range(len(tp)):
            db_val = db[wi]
            db_std = stds[wi] if wi < len(stds) else None
            windows.append({
                "channel": crit_channel_1based - 1,
                "window_index": wi,
                "time_sec": tp[wi],
                # NaN check via self-equality.
                "db_value": db_val if db_val == db_val else None,
                "db_std": (db_std if db_std is not None
                            and db_std == db_std else None),
                "sigma": sigs[wi] if wi < len(sigs) else None,
                "ar_order": ar_order,
            })
        self.store.bulk_insert_criticality(file_id, windows, version_id)

    # ------------------------------------------------------------------ #
    #  Tier 3: Video frame QC + email alert
    # ------------------------------------------------------------------ #

    def _run_tier3_video(self, file_id: int, mat_path: str,
                          video_path: str, cfg: dict,
                          version_id: int | None) -> None:
        """Sample frames, persist results, optionally email an alert.

        Failures here never kill the parent processing run -- a broken
        codec or a vanished video file is a Tier-3 issue, not a reason
        to error out the whole .mat that already produced Tier 1 + 2
        results. They're logged and dropped.
        """
        t = time.time()
        try:
            result = analyze_video(
                video_path,
                sample_every_n=cfg.get("sample_every_n"),
                downsample_to=tuple(cfg.get("downsample_to", (160, 120))),
                min_mean=float(cfg.get("min_mean", 25.0)),
                max_mean=float(cfg.get("max_mean", 230.0)),
                min_variance=float(cfg.get("min_variance", 50.0)),
                alert_pct_bad=float(cfg.get("alert_pct_bad", 25.0)),
            )
        except Exception as e:
            logger.error("Video QC failed for %s: %s", video_path, e,
                          exc_info=True)
            return

        elapsed = time.time() - t
        logger.info(
            "Tier 3: %s (%.1fs): %s, sampled %d/%d frames, "
            "%.1f%% bad (%s)",
            os.path.basename(video_path), elapsed, result.status,
            result.n_frames_sampled, result.n_frames_total,
            result.pct_bad, result.first_bad_reason or "—",
        )
        try:
            self.store.insert_video_qc(file_id, result.to_dict(),
                                        version_id=version_id)
        except Exception as e:
            logger.warning("Could not persist video_qc row: %s", e)

        # Alert on critical only. Rate limiting is handled inside
        # _send_video_alert via the alerts table.
        if (result.status != "critical"
                or not cfg.get("alert_on_critical", True)):
            return
        self._send_video_alert(mat_path, video_path, result)

    def _send_video_alert(self, mat_path: str, video_path: str,
                           result) -> None:
        """Render and dispatch the bad-video email via the shared
        EmailAlerter passed at __init__ time. Skips quietly when the
        dispatcher was constructed without an emailer (typical for
        unit tests) or when alerting is disabled."""
        emailer = self.emailer
        if emailer is None:
            logger.debug("Video alert suppressed (no emailer wired)")
            return
        if not emailer.enabled or not emailer.password:
            logger.debug("Video alert suppressed (alerting disabled "
                          "or password missing)")
            return

        # Rate-limit through the alerts table: if we sent an alert for
        # this video within the last hour, skip.
        recent = (self.store.get_recent_alerts(hours=1)
                  if hasattr(self.store, "get_recent_alerts") else [])
        if any(a.get("alert_type") == "video_qc"
               and os.path.basename(video_path) in (a.get("message") or "")
               for a in (recent or [])):
            logger.debug("Video alert rate-limited for %s", video_path)
            return

        subject = (f"Video QC alert -- {os.path.basename(video_path)} "
                   f"({result.pct_bad:.0f}% bad frames)")
        text_lines = [
            f"Video: {video_path}",
            f"Companion .mat: {mat_path}",
            "",
            f"Status: {result.status.upper()}",
            f"Sampled {result.n_frames_sampled} of {result.n_frames_total} "
            f"frames (1 per {result.sample_every_n}).",
            f"Bad frames: {result.n_bad} ({result.pct_bad:.1f}%)",
            f"  dark     : {result.n_dark}",
            f"  bright   : {result.n_bright}",
            f"  low_var  : {result.n_low_var}",
            f"First bad frame: idx {result.first_bad_idx} "
            f"({result.first_bad_reason})"
            if result.first_bad_idx is not None
            else "First bad frame: —",
            "",
            "Per-frame thresholds:",
            f"  min_mean = {result.thresholds.get('min_mean')}",
            f"  max_mean = {result.thresholds.get('max_mean')}",
            f"  min_variance = {result.thresholds.get('min_variance')}",
        ]
        text_body = "\n".join(text_lines)
        ok = emailer.send(
            subject=subject, body=text_body, severity="warning",
        )
        try:
            if hasattr(self.store, "insert_alert"):
                self.store.insert_alert(
                    alert_type="video_qc",
                    severity="warning",
                    message=f"{os.path.basename(video_path)}: "
                            f"{result.pct_bad:.0f}% bad frames "
                            f"({result.first_bad_reason})",
                    sent_at_iso=None, sent=int(bool(ok)),
                )
        except Exception as e:
            logger.debug("insert_alert skipped: %s", e)
