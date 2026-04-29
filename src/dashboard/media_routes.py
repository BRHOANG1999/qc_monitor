"""Flask routes that stream local media (videos) into the dashboard.

Registered on the Dash server's underlying Flask instance. Auth runs
ahead of these via the ``before_request`` hook in ``auth.py``, so by
the time the handler fires the user is already verified.
"""

import logging
import os
import sqlite3

from flask import abort, g, send_file

from src.utils.video import video_path_for_mat

logger = logging.getLogger("qc_monitor.dashboard.media")


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
