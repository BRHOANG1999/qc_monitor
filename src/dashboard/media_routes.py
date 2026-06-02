"""Flask routes that stream local media (videos) into the dashboard.

Registered on the Dash server's underlying Flask instance. Auth runs
ahead of these via the ``before_request`` hook in ``auth.py``, so by
the time the handler fires the user is already verified.
"""

import io
import logging
import os
import sqlite3
import threading
import time

from flask import abort, g, send_file

from src.utils.video import video_path_for_mat

logger = logging.getLogger("qc_monitor.dashboard.media")

# Last-frame snapshot cache: (jpeg_bytes, captured_at_ts). The route
# returns the cached bytes if they're younger than _SNAPSHOT_TTL_SEC
# so the dashboard's 10-s refresh-trigger doesn't hammer the SMB
# share + OpenCV pipeline on every tick.
_SNAPSHOT_TTL_SEC = 8.0
_snapshot_cache: dict[str, tuple[bytes, float]] = {}
_snapshot_lock = threading.Lock()


def _lookup_mat_path(db_path: str, file_id: int) -> str | None:
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT file_path, has_video FROM processed_files WHERE id = ?",
            (file_id,),
        ).fetchone()
        if row is None:
            return None
        return row["file_path"]
    finally:
        conn.close()


def _latest_mat_with_video(db_path: str) -> str | None:
    """Return file_path of the most-recently-recorded chunk that has
    a companion video, or None when nothing qualifies."""
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT file_path FROM processed_files "
            "WHERE has_video = 1 "
            "ORDER BY chunk_datetime DESC LIMIT 1"
        ).fetchone()
        return row["file_path"] if row else None
    finally:
        conn.close()


def _extract_last_frame_jpeg(video_path: str,
                               max_width: int = 640) -> bytes | None:
    """Open *video_path* with OpenCV, seek as close to the end as the
    container allows, grab a frame, and return a JPEG byte buffer.
    Returns None on any failure (missing codec, read error, etc.) --
    the caller decides what to surface to the browser."""
    try:
        import cv2  # noqa: WPS433
    except Exception as e:
        logger.warning("OpenCV not available: %s", e)
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            logger.debug("VideoCapture refused %s", video_path)
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 1:
            # Step back one so we land ON a decodable frame instead
            # of past the end.
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 2))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            new_size = (max_width, int(round(h * scale)))
            frame = cv2.resize(frame, new_size,
                                interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return bytes(buf.tobytes())
    finally:
        cap.release()


def register_media_routes(server, store, config: dict) -> None:
    """Register ``GET /media/video/<file_id>`` on the Dash Flask server.

    Streams the companion ``_v1.mp4`` for the given processed_files.id
    with HTTP Range support so the browser can seek without downloading
    the whole file. Returns 404 if the file_id is unknown or the video
    doesn't exist on disk.
    """
    db_path = (config or {}).get("database", {}).get("path") or store.db_path

    @server.route("/media/video/<int:file_id>")
    def serve_video(file_id: int):  # pragma: no cover — exercised by browser
        if not getattr(g, "user", None):
            abort(403)
        mat_path = _lookup_mat_path(db_path, file_id)
        if mat_path is None:
            abort(404, description=f"Unknown file_id {file_id}")
        video_path = video_path_for_mat(mat_path)
        if video_path is None:
            abort(404, description="No companion video for this file")
        # send_file with conditional=True honors Range/If-Modified-Since
        # so the browser's <video> element can seek normally.
        return send_file(
            video_path,
            mimetype="video/mp4",
            conditional=True,
            as_attachment=False,
        )

    @server.route("/media/latest-snapshot.jpg")
    def serve_latest_snapshot():  # pragma: no cover -- exercised by browser
        """Return a JPEG of the most-recently-recorded video's last
        frame. Cached for _SNAPSHOT_TTL_SEC so the dashboard's
        per-tick refresh doesn't pummel the SMB share."""
        if not getattr(g, "user", None):
            abort(403)
        with _snapshot_lock:
            cached = _snapshot_cache.get("latest")
            now = time.time()
            if cached and (now - cached[1]) < _SNAPSHOT_TTL_SEC:
                jpeg, captured_at = cached
            else:
                mat_path = _latest_mat_with_video(db_path)
                if mat_path is None:
                    abort(404,
                            description="No recorded videos available")
                video_path = video_path_for_mat(mat_path)
                if video_path is None or not os.path.exists(video_path):
                    abort(404,
                            description="Companion video not on disk")
                jpeg = _extract_last_frame_jpeg(video_path)
                if jpeg is None:
                    abort(500,
                            description="Could not decode video frame")
                captured_at = now
                _snapshot_cache["latest"] = (jpeg, captured_at)
        resp = send_file(io.BytesIO(jpeg), mimetype="image/jpeg",
                         conditional=False)
        # Tell the browser this is cheap to re-fetch -- cache buster
        # is the ?t=… query string the dashboard appends each tick.
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp
