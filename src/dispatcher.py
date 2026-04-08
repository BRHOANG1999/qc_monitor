"""Dispatch queue: manages processing of new files through analysis tiers."""

import logging
import time
from datetime import datetime

from src.db.store import Store
from src.utils.mat_loader import load_mat
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
        self.tier1_cfg = config.get("analysis", {}).get("tier1", {})
        self.tier2_cfg = config.get("analysis", {}).get("tier2", {})

    def process_file(self, new_file: NewFile) -> bool:
        """Process a single file through Tier 1 (Python QC) + Tier 2 (MATLAB pipeline).

        Returns True if processing succeeded.
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

        try:
            # --- Tier 1: Python QC ---
            t1_start = time.time()
            chunk = load_mat(new_file.path)
            self.store.update_file_status(file_id, "processing",
                                          num_channels=chunk.num_channels,
                                          num_samples=chunk.num_samples,
                                          sampling_rate=chunk.fs,
                                          duration_sec=chunk.duration_sec)

            # Basic QC
            basic_results = analyze_basic_qc(
                chunk,
                clipping_voltage=self.tier1_cfg.get("clipping_voltage", 10.0),
                flatline_std_threshold=self.tier1_cfg.get("flatline_std_threshold", 1e-6),
            )

            # Spectral
            bands = self.tier1_cfg.get("band_definitions", None)
            spectral_results = analyze_spectral(
                chunk, bands=bands,
                line_noise_freq=self.tier1_cfg.get("line_noise_freq", 60.0),
            )

            # Artifact
            artifact_cfg = self.tier1_cfg.get("artifact", {})
            artifact_results = analyze_artifact(
                chunk,
                method=artifact_cfg.get("method", "fixed"),
                threshold=artifact_cfg.get("threshold", 500.0),
                mad_k=artifact_cfg.get("mad_k", 4.0),
                merge_gap_sec=artifact_cfg.get("merge_gap_sec", 2.0),
            )

            # Merge and store per-channel
            for ch in range(chunk.num_channels):
                metrics = {}
                if ch < len(basic_results):
                    metrics.update(basic_results[ch])
                if ch < len(spectral_results):
                    metrics.update(spectral_results[ch])
                if ch < len(artifact_results):
                    metrics.update(artifact_results[ch])
                self.store.insert_chunk_qc(file_id, ch, metrics)

            t1_elapsed = time.time() - t1_start
            logger.info("Tier 1 done for %s (%.1fs, %d ch)", new_file.path, t1_elapsed, chunk.num_channels)

            # Stim QC
            stim_results = analyze_stim_report(new_file.path)
            if stim_results:
                self.store.update_file_status(file_id, "processing", has_stim_report=1)
                for sr in stim_results:
                    self.store.insert_stim_qc(file_id, sr["stim_channel"], sr)

            # Check for video
            video_path = new_file.path.rsplit(".", 1)[0] + "_v1.mp4"
            if _file_exists(video_path):
                self.store.update_file_status(file_id, "processing", has_video=1)

            # --- Tier 2: MATLAB Pipeline ---
            if self.tier2_cfg.get("enabled", True):
                t2_start = time.time()
                matlab_exe = self.tier2_cfg.get("criticality", {}).get(
                    "matlab_exe", "matlab"
                )
                result = matlab_run_pipeline(new_file.path, matlab_exe=matlab_exe, timeout=600)
                t2_elapsed = time.time() - t2_start
                logger.info("Tier 2 done for %s (%.1fs): %s",
                            new_file.path, t2_elapsed, result.get("exit_status", "unknown"))
                self.store.insert_matlab_result(file_id, result)

            self.store.update_file_status(file_id, "done")
            return True

        except Exception as e:
            logger.error("Processing failed for %s: %s", new_file.path, e, exc_info=True)
            self.store.update_file_status(file_id, "error")
            return False


def _file_exists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except OSError:
        return False


import os
