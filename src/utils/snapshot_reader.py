"""Decode the latest cage-video last-frame collage in a CHILD PROCESS.

The Overview "Latest snapshot" refresher globs the SMB share for companion
``_vN.mp4`` and cv2-decodes their last frames every cycle. Over the (2-pipeline-
saturated) Tailscale share those ops are slow C calls that HOLD the dashboard's
GIL -- the last remaining in-dashboard share reader, and (via py-spy) the thing
starving the Overview card builds. Here the glob + decode run in a warm child
process; only the tiny JPEG (or None) crosses the pipe, so the dashboard's GIL is
free during the slow share I/O.

Deliberately light (cv2 + numpy + src.utils.video only, no flask) so the spawned
child imports fast.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger("qc_monitor.utils.snapshot_reader")

_DECODE_TIMEOUT_SEC = 25.0
_pool = None
_pool_lock = threading.Lock()


def _read_last_frame(video_path: str):
    """Open *video_path* with OpenCV, seek near the end, return the decoded
    frame (BGR ndarray) or None. Runs in the CHILD process."""
    from src.utils.offproc_guard import require_offproc
    require_offproc("cv2 video decode")   # invariant: never on a dashboard thread
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        logger.warning("OpenCV not available: %s", e)
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
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


def _frames_to_collage_jpeg(video_paths, max_width: int = 720):
    """Stitch each video's last frame side-by-side and JPEG-encode. Skips videos
    that fail; None if all fail. Runs in the CHILD process."""
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        logger.warning("OpenCV not available: %s", e)
        return None
    frames = [f for f in (_read_last_frame(vp) for vp in video_paths) if f is not None]
    if not frames:
        return None
    target_h = min(f.shape[0] for f in frames)
    norm = []
    for f in frames:
        h, w = f.shape[:2]
        if h == target_h:
            norm.append(f)
        else:
            scale = target_h / float(h)
            norm.append(cv2.resize(f, (max(1, int(round(w * scale))), target_h),
                                   interpolation=cv2.INTER_AREA))
    collage = cv2.hconcat(norm)
    h, w = collage.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        collage = cv2.resize(collage, (max_width, max(1, int(round(h * scale)))),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", collage, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return bytes(buf.tobytes()) if ok else None


def _worker(mat_path: str):
    """CHILD process: resolve the companion videos for *mat_path* (share glob) and
    decode their last-frame collage to JPEG bytes. Returns bytes or None."""
    sys.path.insert(0, _REPO_ROOT)
    from src.utils.offproc_guard import mark_worker
    mark_worker()                         # allowed off-proc reader
    import os as _os
    from src.utils.video import companion_video_paths, video_path_for_mat
    video_paths = companion_video_paths(mat_path)
    if not video_paths:
        legacy = video_path_for_mat(mat_path)
        if legacy is None or not _os.path.exists(legacy):
            return None
        video_paths = [legacy]
    return _frames_to_collage_jpeg(video_paths)


def _get_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            _pool = ProcessPoolExecutor(max_workers=1,
                                        mp_context=mp.get_context("spawn"))
        return _pool


def _reset_pool():
    global _pool
    with _pool_lock:
        p, _pool = _pool, None
    if p is not None:
        try:
            p.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass


def decode_snapshot_offproc(mat_path: str):
    """Return the latest-snapshot JPEG bytes for *mat_path*, decoded in a child
    process (off the caller's GIL). None on any failure/timeout. Never raises."""
    from concurrent.futures import TimeoutError as _FTimeout
    if not mat_path:
        return None
    try:
        fut = _get_pool().submit(_worker, mat_path)
    except Exception:  # noqa: BLE001
        return None
    try:
        return fut.result(timeout=_DECODE_TIMEOUT_SEC)   # GIL released while waiting
    except _FTimeout:
        _reset_pool()
        logger.warning("snapshot decode timed out (%.0fs)", _DECODE_TIMEOUT_SEC)
        return None
    except Exception as e:  # noqa: BLE001
        _reset_pool()
        logger.debug("snapshot decode failed: %s", e)
        return None
