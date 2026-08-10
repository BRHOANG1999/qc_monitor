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
from src.utils import avi_transcode as _avi
from src.utils.snapshot_reader import decode_snapshot_offproc
from src.utils.video import (video_path_for_mat, companion_video_paths,
                              companion_videos)

logger = logging.getLogger("qc_monitor.dashboard.media")

# Last-frame snapshot cache: (jpeg_bytes, captured_at_ts). A BACKGROUND thread
# (started in register_media_routes) rebuilds this every _SNAPSHOT_TTL_SEC; the
# /media/latest-snapshot.jpg route only ever serves the cached bytes and NEVER
# decodes on the request thread. This matters because the route is on the
# Overview 10-s auto-refresh path -- decoding a video off the SMB share (cv2,
# GIL-holding C) on a request thread, under _snapshot_lock, used to stall every
# dashboard thread each time the cache went cold.
_SNAPSHOT_TTL_SEC = 30.0              # background rebuild cadence (was 8s: the cv2
                                     # decode holds the GIL ~2-3s each pass, so a
                                     # tight loop occupied ~30% of the dashboard's
                                     # GIL and starved the Overview cards. A live
                                     # cage snapshot every 30s is plenty.)
_SNAPSHOT_READ_TIMEOUT = 20.0         # bound one decode so a hung share can't wedge the refresher
_snapshot_cache: dict[str, tuple[bytes, float]] = {}
_snapshot_lock = threading.Lock()
_snapshot_refresher_started = False
_snapshot_start_lock = threading.Lock()


def _build_snapshot_jpeg(store):
    """Resolve the newest companion video (DB, fast) then decode its last-frame
    collage in a CHILD process (snapshot_reader.decode_snapshot_offproc), so the
    slow SMB glob + cv2 decode never hold THIS process's GIL -- that decode was
    the last in-dashboard share reader starving the Overview card builds (found
    via py-spy). Returns JPEG bytes or None."""
    mat_path = _latest_mat_with_video(store)
    if mat_path is None:
        return None
    return decode_snapshot_offproc(mat_path)


def _refresh_snapshot_loop(store):
    """Daemon loop: rebuild the snapshot cache every _SNAPSHOT_TTL_SEC. The heavy
    glob + cv2 decode run OFF-PROCESS (decode_snapshot_offproc), which waits with
    the GIL released and is itself deadline-bounded, so a slow/hung share can
    neither wedge the refresher nor freeze the dashboard."""
    it = 0
    while it < 10_000_000:                 # NASA rule 2: explicit loop bound
        it += 1
        try:
            jpeg = _build_snapshot_jpeg(store)
            if jpeg is not None:
                with _snapshot_lock:
                    _snapshot_cache["latest"] = (jpeg, time.time())
        except Exception:                  # noqa: BLE001 -- never kill the daemon
            logger.debug("snapshot refresh failed", exc_info=True)
        time.sleep(_SNAPSHOT_TTL_SEC)


def _start_snapshot_refresher(store):
    """Start the background snapshot refresher once per process (idempotent)."""
    global _snapshot_refresher_started
    with _snapshot_start_lock:
        if _snapshot_refresher_started:
            return
        _snapshot_refresher_started = True
    threading.Thread(target=_refresh_snapshot_loop, args=(store,),
                     daemon=True, name="snapshot-refresher").start()


def _lookup_mat_path(store, file_id: int) -> str | None:
    # live_file_path falls back to the resolved EEG location when the DB
    # row's path is stale (historical files whose drive was remapped and
    # whose relocation was blocked by UNIQUE(file_path)); otherwise the
    # companion video next to the dead path is never found.
    return store.live_file_path(file_id)


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


# NOTE: the cv2 last-frame decode (_read_last_frame / _frames_to_collage_jpeg)
# moved to src/utils/snapshot_reader.py and runs in an OFF-PROCESS worker -- the
# invariant is that the dashboard package never decodes video on a thread it
# owns (see offproc_guard + tests/test_dashboard_offproc_invariant.py). The
# snapshot route here serves only cached bytes produced by that worker.


def register_media_routes(server, store, config: dict) -> None:
    """Register ``GET /media/video/<file_id>`` on the Dash Flask server.

    Streams the companion ``_v1.mp4`` for the given processed_files.id
    with HTTP Range support so the browser can seek without downloading
    the whole file. Returns 404 if the file_id is unknown or the video
    doesn't exist on disk.
    """
    # Rebuild the Overview snapshot JPEG off the request threads (see
    # _refresh_snapshot_loop) so the 10-s auto-refresh route serves cached bytes
    # only. Idempotent per process.
    _start_snapshot_refresher(store)

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

    @server.route("/media/video/<int:file_id>/cam/<int:cam>")
    def serve_video_cam_n(file_id: int, cam: int):  # pragma: no cover
        """Serve camera number *cam* for a chunk, transcoding .avi -> mp4 on
        demand (cached). mp4 companions stream directly; an .avi camera
        returns 202 (JSON 'transcoding') until its cached mp4 is ready, so
        the Training tab can poll and load it when done."""
        if not getattr(g, "user", None):
            abort(403)
        mat_path = _lookup_mat_path(store, file_id)
        if mat_path is None:
            abort(404, description=f"Unknown file_id {file_id}")
        entry = next((e for e in companion_videos(mat_path)
                      if e["cam"] == cam), None)
        if entry is None:
            abort(404, description=f"Camera v{cam} not available")
        if entry["kind"] == "mp4":
            return send_file(entry["path"], mimetype="video/mp4",
                             conditional=True, as_attachment=False)
        # .avi -> transcoded mp4 (background, cached).
        ready = _avi.ensure_async(entry["path"], config)
        if ready:
            return send_file(ready, mimetype="video/mp4",
                             conditional=True, as_attachment=False)
        st = _avi.status(entry["path"], config)
        return jsonify({"status": "error" if st == "error" else
                        "transcoding", "cam": cam}), 202

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
        """Serve the most recent cached last-frame JPEG. The decode happens in
        the background refresher (_refresh_snapshot_loop); this handler NEVER
        touches the SMB share or OpenCV, so a slow decode can't stall the
        request pool even though this route is on the 10-s auto-refresh path."""
        if not getattr(g, "user", None):
            abort(403)
        with _snapshot_lock:
            cached = _snapshot_cache.get("latest")
        if not cached:
            # Cold: the refresher hasn't produced a frame yet (fresh boot) or
            # there's no companion video. Do NOT decode here -- just say so; the
            # <img> shows its broken/placeholder state and recovers on the next
            # tick once the background thread lands a frame.
            abort(404, description="Snapshot not ready")
        jpeg, _captured_at = cached
        resp = send_file(io.BytesIO(jpeg), mimetype="image/jpeg",
                         conditional=False)
        # Cheap to re-fetch -- the cache buster is the ?t=… the dashboard
        # appends each tick.
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp
