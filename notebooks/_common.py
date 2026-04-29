"""Shared loaders for ML/AI PoC notebooks.

Delegates to production code for DB access and .mat loading so notebooks
stay focused on modeling, not data plumbing.
"""

import json
import os
import sys
import sqlite3

import cv2
import numpy as np
import pandas as pd

# Make project root importable so we can reuse production modules
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.mat_loader import load_mat, ChunkData  # noqa: E402
from src.utils.video import video_path_for_mat as _video_path_for_mat  # noqa: E402
from src.db.schema import SCHEMA_SQL  # noqa: E402

# Default paths — override in notebooks if needed
DEFAULT_DB = os.path.join(_PROJECT_ROOT, "data", "monitor.db")


# ------------------------------------------------------------------ #
#  Database helpers (read-only)
# ------------------------------------------------------------------ #

def load_db(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    """Open a read-only SQLite connection to the monitor database.

    Returns a connection with row_factory=sqlite3.Row for dict-like access.
    """
    assert os.path.isfile(db_path), f"Database not found: {db_path}"
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def load_evoked_feature_matrix(
    db_path: str = DEFAULT_DB,
    version_id: int | None = None,
) -> pd.DataFrame:
    """Load all evoked features joined with file metadata.

    Returns a DataFrame with columns:
        file_id, session_dir, session_name, chunk_datetime, epoch_index,
        epoch_time_sec, + 26 feature columns, is_artifact, is_ictal
    """
    assert os.path.isfile(db_path), f"Database not found: {db_path}"

    conn = load_db(db_path)
    try:
        query = """
            SELECT
                ef.file_id,
                pf.session_dir,
                pf.session_name,
                pf.chunk_datetime,
                ef.epoch_index,
                ef.epoch_time_sec,
                ef.line_length,
                ef.log_auc,
                ef.peak_amplitude,
                ef.trough_amplitude,
                ef.peak_to_trough,
                ef.rms_amplitude,
                ef.peak_latency_ms,
                ef.trough_latency_ms,
                ef.max_slope,
                ef.max_slope_time_ms,
                ef.early_area,
                ef.late_area,
                ef.early_late_ratio,
                ef.recovery_tau,
                ef.recovery_slope,
                ef.template_correlation,
                ef.pca_recon_error,
                ef.variance,
                ef.autocorrelation,
                ef.ac_width,
                ef.exp_fit_a,
                ef.sum_power_low,
                ef.freq_moment_low,
                ef.sum_power_high,
                ef.freq_moment_high,
                ef.is_artifact,
                ef.is_ictal
            FROM evoked_features ef
            JOIN processed_files pf ON pf.id = ef.file_id
        """
        params = []
        if version_id is not None:
            query += " WHERE ef.version_id = ?"
            params.append(version_id)
        query += " ORDER BY pf.chunk_datetime, ef.epoch_index"

        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    assert len(df) > 0, "No evoked features found in database"
    return df


# ------------------------------------------------------------------ #
#  LFP / .mat helpers
# ------------------------------------------------------------------ #

def load_lfp_window(
    file_path: str,
    start_sec: float,
    end_sec: float,
    channels: list[int] | None = None,
) -> tuple[np.ndarray, float]:
    """Load a time window from a .mat file.

    Returns (signal_window, fs) where signal_window is [samples x channels].
    """
    assert os.path.isfile(file_path), f"MAT file not found: {file_path}"
    assert end_sec > start_sec, "end_sec must be > start_sec"

    chunk = load_mat(file_path)
    fs = chunk.fs
    start_idx = max(0, int(start_sec * fs))
    end_idx = min(chunk.num_samples, int(end_sec * fs))

    assert end_idx > start_idx, "Window is empty after clipping to file bounds"

    signal = chunk.signal[start_idx:end_idx, :]
    if channels is not None:
        signal = signal[:, channels]

    return signal, fs


# ------------------------------------------------------------------ #
#  Video helpers
# ------------------------------------------------------------------ #

# Re-export the canonical helper from src.utils.video so notebook callers
# keep working without importing the production module directly.
video_path_for_mat = _video_path_for_mat


def load_video_frames(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    fps_out: float | None = None,
    max_frames: int = 50000,
) -> tuple[list[np.ndarray], float, int]:
    """Load frames from an MP4 file within a time window.

    Args:
        video_path: Path to the .mp4 file.
        start_sec: Start time in seconds.
        end_sec: End time (None = read to end).
        fps_out: Target output FPS. If lower than native, frames are
                 skipped to approximate the target rate.
        max_frames: Hard cap on number of frames returned.

    Returns:
        (frames, native_fps, total_native_frames)
        frames: list of BGR numpy arrays
    """
    assert os.path.isfile(video_path), f"Video not found: {video_path}"

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video: {video_path}"

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert native_fps > 0, f"Invalid FPS: {native_fps}"

    # Compute frame indices to read
    start_frame = int(start_sec * native_fps)
    if end_sec is not None:
        end_frame = min(int(end_sec * native_fps), total_frames)
    else:
        end_frame = total_frames

    # Frame skip for target FPS
    skip = 1
    if fps_out is not None and fps_out < native_fps:
        skip = max(1, int(round(native_fps / fps_out)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_idx = start_frame
    for _ in range(min(end_frame - start_frame, max_frames * skip)):
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % skip == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return frames, native_fps, total_frames


# ------------------------------------------------------------------ #
#  Schema helper (for LLM prompt context)
# ------------------------------------------------------------------ #

def db_schema_text() -> str:
    """Return the full SQL schema as a string for use in LLM prompts."""
    return SCHEMA_SQL
