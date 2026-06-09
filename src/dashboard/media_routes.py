"""Flask routes that stream local media (videos) into the dashboard.

Registered on the Dash server's underlying Flask instance. Auth runs
ahead of these via the ``before_request`` hook in ``auth.py``, so by
the time the handler fires the user is already verified.
"""

import io
import logging
import os
import threading
import time

from flask import abort, g, jsonify, send_file

from src.utils import event_clip as _event_clip
from src.utils.video import video_path_for_mat, companion_video_paths

logger = logging.getLogger("qc_monitor.dashboard.media")

# Last-frame snapshot cache: (jpeg_bytes, captured_at_ts). The route
# returns the cached bytes if they're younger than _SNAPSHOT_TTL_SEC
# so the dashboard's 10-s refresh-trigger doesn't hammer the SMB
# share + OpenCV pipeline on every tick.
_SNAPSHOT_TTL_SEC = 8.0
_snapshot_cache: dict[str, tuple[bytes, float]] = {}
_snapshot_lock = threading.Lock()


def _lookup_mat_path(store, file_id: int) -> str | None:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT file_path, has_video FROM processed_files WHERE id = ?",
            (file_id,),
        ).fetchone()
        if row is None:
            return None
        return row["file_path"]


def _latest_mat_with_video(store) -> str | None:
    """Return file_path of the most-recently-recorded chunk that has
    a companion video, or None when nothing qualifies."""
    with store.connection() as conn:
        row = conn.execute(
            "SELECT file_path FROM processed_files "
            "WHERE has_video = 1 "
            "ORDER BY chunk_datetime DESC LIMIT 1"
        ).fetchone()
        return row["file_path"] if row else None


def _read_last_frame(video_path: str):
    """Open *video_path* with OpenCV, seek as close to the end as the
    container allows, return the decoded frame (BGR ndarray) or None.
    """
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 2))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame
    finally:
        cap.release()


def _frames_to_collage_jpeg(video_paths: list[str],
                              max_width: int = 720) -> bytes | None:
    """Stitch each video's last frame side-by-side (cv2.hconcat) and
    encode the collage as a single JPEG. Frames are first resized to
    the shortest input height to avoid upscaling. Skips videos that
    fail to decode; returns None if every video fails.

    Multi-camera sessions (cage A + cage B etc.) drop multiple
    ``_vN.mp4`` next to one .mat; this lets the dashboard show all
    of them in a single thumbnail without juggling separate <img>
    elements + Flask round-trips.
    """
    try:
        import cv2  # noqa: WPS433
    except Exception as e:
        logger.warning("OpenCV not available: %s", e)
        return None
    frames = []
    for vp in video_paths:
        f = _read_last_frame(vp)
        if f is not None:
            frames.append(f)
    if not frames:
        return None
    # Normalise heights so hconcat doesn't fail. Use the shortest
    # input as the target so we never up-scale (preserves detail).
    target_h = min(f.shape[0] for f in frames)
    norm = []
    for f in frames:
        h, w = f.shape[:2]
        if h == target_h:
            norm.append(f)
            continue
        scale = target_h / float(h)
        new_w = max(1, int(round(w * scale)))
        norm.append(cv2.resize(f, (new_w, target_h),
                                interpolation=cv2.INTER_AREA))
    collage = cv2.hconcat(norm)
    # Downscale collage if it exceeds max_width so we don't ship a
    # 3 K pixel-wide JPEG to the dashboard.
    h, w = collage.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        new_size = (max_width, max(1, int(round(h * scale))))
        collage = cv2.resize(collage, new_size,
                              interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", collage,
                             [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return bytes(buf.tobytes())


def register_media_routes(server, store, config: dict) -> None:
    """Register ``GET /media/video/<file_id>`` on the Dash Flask server.

    Streams the companion ``_v1.mp4`` for the given processed_files.id
    with HTTP Range support so the browser can seek without downloading
    the whole file. Returns 404 if the file_id is unknown or the video
    doesn't exist on disk.
    """
    @server.route("/media/video/<int:file_id>")
    def serve_video(file_id: int):  # pragma: no cover — exercised by browser
        if not getattr(g, "user", None):
            abort(403)
        mat_path = _lookup_mat_path(store, file_id)
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

    @server.route("/media/video/<int:file_id>/<int:cam>")
    def serve_video_cam(file_id: int, cam: int):  # pragma: no cover
        """Serve the Nth companion video (_v1, _v2, ...) for a given
        chunk. cam is 1-indexed because that's how the filenames go.
        Falls through to 404 when the camera index is out of range."""
        if not getattr(g, "user", None):
            abort(403)
        if cam < 1:
            abort(400, description="cam must be >= 1")
        mat_path = _lookup_mat_path(store, file_id)
        if mat_path is None:
            abort(404, description=f"Unknown file_id {file_id}")
        paths = companion_video_paths(mat_path)
        if not paths or cam > len(paths):
            abort(404,
                    description=f"Camera v{cam} not available "
                                 f"({len(paths)} companion(s) on disk)")
        return send_file(
            paths[cam - 1],
            mimetype="video/mp4",
            conditional=True,
            as_attachment=False,
        )

    @server.route("/media/clip/<spec_hash>")
    def serve_event_clip(spec_hash: str):  # pragma: no cover
        """Serve a PI event-verification clip by its spec_hash.

        Three response shapes:
        * 200 + video/mp4 -- the cached clip is ready.
        * 202 + JSON status -- still extracting or queued; the
          PI tab polls until status flips to ``done``.
        * 404 -- the spec_hash was never queued.
        """
        if not getattr(g, "user", None):
            abort(403)
        if not spec_hash or len(spec_hash) > 64:
            abort(400, description="bad spec_hash")
        info = _event_clip.poll_video_clip_status(spec_hash, store)
        status = info.get("status")
        if status == "missing":
            abort(404,
                    description=f"No clip job for {spec_hash}")
        if status == "done" and info.get("cache_path"):
            cache_path = info["cache_path"]
            if not os.path.exists(cache_path):
                # Cache file was evicted; respond 410 so the
                # PI tab can re-submit the job.
                abort(410,
                        description="Cached clip evicted; "
                                     "resubmit the job")
            return send_file(
                cache_path,
                mimetype="video/mp4",
                conditional=True,
                as_attachment=False,
            )
        if status in ("pending", "running"):
            return jsonify({"status": status,
                             "spec_hash": spec_hash,
                             "error": info.get("error")}), 202
        if status == "failed":
            return jsonify({"status": "failed",
                             "spec_hash": spec_hash,
                             "error": info.get("error")}), 500
        abort(500,
                description=f"unexpected clip status {status}")

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
                mat_path = _latest_mat_with_video(store)
                if mat_path is None:
                    abort(404,
                            description="No recorded videos available")
                # Glob every companion _vN.mp4 so multi-camera
                # sessions render side-by-side instead of dropping
                # all but the first camera.
                video_paths = companion_video_paths(mat_path)
                if not video_paths:
                    # Fallback to legacy single-video path so old
                    # sessions still resolve cleanly.
                    legacy = video_path_for_mat(mat_path)
                    if legacy is None or not os.path.exists(legacy):
                        abort(404,
                                description="Companion video not on disk")
                    video_paths = [legacy]
                jpeg = _frames_to_collage_jpeg(video_paths)
                if jpeg is None:
                    abort(500,
                            description="Could not decode any video frame")
                captured_at = now
                _snapshot_cache["latest"] = (jpeg, captured_at)
        resp = send_file(io.BytesIO(jpeg), mimetype="image/jpeg",
                         conditional=False)
        # Tell the browser this is cheap to re-fetch -- cache buster
        # is the ?t=… query string the dashboard appends each tick.
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp
