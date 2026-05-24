"""Video frame QC -- catch dark / bright / uniform footage.

Reviews every Nth frame of an mp4/avi (default: ~1 frame per second
of recording). Per sampled frame, computes the grayscale mean and
variance on a downsampled thumbnail. Flags a frame as "bad" when:

* mean below ``min_mean`` (camera covered, lights off, IR filter
  failure)
* mean above ``max_mean`` (over-exposed, light fell into the cage)
* variance below ``min_variance`` (uniform field, lens fogged or
  pointed at a wall)

Summary stats roll up to a status -- ``ok`` / ``warning`` /
``critical`` -- so the dispatcher can decide whether to fire an alert.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

logger = logging.getLogger("qc_monitor.analyzers.video_qc")

_THUMB_DEFAULT = (160, 120)
_MIN_MEAN_DEFAULT = 25.0
_MAX_MEAN_DEFAULT = 230.0
_MIN_VAR_DEFAULT = 50.0
_ALERT_PCT_DEFAULT = 25.0   # bad-frame % at/above which we'd email


@dataclass
class VideoQCResult:
    video_path: str
    ok: bool = True
    error: str = ""
    fps: float = 0.0
    n_frames_total: int = 0
    n_frames_sampled: int = 0
    duration_sec: float = 0.0
    duration_analyzed_sec: float = 0.0
    sample_every_n: int = 1
    mean_global: float = 0.0
    var_global: float = 0.0
    mean_min: float = 0.0
    mean_max: float = 0.0
    var_min: float = 0.0
    var_max: float = 0.0
    n_dark: int = 0
    n_bright: int = 0
    n_low_var: int = 0
    n_bad: int = 0
    pct_bad: float = 0.0
    first_bad_idx: int | None = None
    first_bad_reason: str = ""
    status: str = "ok"            # 'ok' | 'warning' | 'critical'
    thresholds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_sample_step(fps: float, sample_every_n: int | None) -> int:
    """1 fps default cadence -- enough to catch a camera failure that
    persists for more than a second. Caller can force a specific step."""
    if sample_every_n is not None and sample_every_n > 0:
        return int(sample_every_n)
    if fps and fps > 1.0:
        return max(1, int(round(fps)))
    return 1


def _classify_frame(mean: float, var: float, min_mean: float,
                     max_mean: float, min_var: float) -> str:
    if mean < min_mean:
        return "dark"
    if mean > max_mean:
        return "bright"
    if var < min_var:
        return "low_var"
    return ""


def analyze_video(path: str, *,
                   sample_every_n: int | None = None,
                   downsample_to: tuple[int, int] = _THUMB_DEFAULT,
                   min_mean: float = _MIN_MEAN_DEFAULT,
                   max_mean: float = _MAX_MEAN_DEFAULT,
                   min_variance: float = _MIN_VAR_DEFAULT,
                   alert_pct_bad: float = _ALERT_PCT_DEFAULT,
                   ) -> VideoQCResult:
    """Walk *path* and compute per-frame brightness + variance stats.

    Returns a :class:`VideoQCResult`. When the file can't be opened or
    decoded, ``ok=False`` and ``error`` describes why. The dispatcher
    decides whether ``status`` warrants an email alert.
    """
    assert isinstance(path, str) and path, "path required"
    result = VideoQCResult(
        video_path=path,
        thresholds={
            "min_mean": min_mean,
            "max_mean": max_mean,
            "min_variance": min_variance,
            "alert_pct_bad": alert_pct_bad,
            "downsample_to": list(downsample_to),
        },
    )

    if not os.path.isfile(path):
        result.ok = False
        result.error = "file missing"
        result.status = "critical"
        return result

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        result.ok = False
        result.error = "cv2.VideoCapture failed to open"
        result.status = "critical"
        return result

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        result.fps = fps
        result.n_frames_total = n_total
        result.duration_sec = (n_total / fps) if fps > 0 else 0.0
        step = _resolve_sample_step(fps, sample_every_n)
        result.sample_every_n = step

        means: list[float] = []
        variances: list[float] = []
        first_bad_idx: int | None = None
        first_bad_reason = ""
        frame_idx = -1
        sampled = 0
        # Walk every frame so the index is canonical. Skip the heavy
        # work (downsample + stats) on frames we don't sample. The
        # decode cost dominates; the modulus check is free.
        # Loop bound: stop after n_total + a small safety margin to
        # avoid an infinite loop on corrupt files where read() returns
        # True forever.
        max_iter = n_total + 10 if n_total > 0 else 200_000
        iter_guard = 0
        while iter_guard < max_iter:
            iter_guard += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if frame_idx % step != 0:
                continue
            sampled += 1
            # Resize then grayscale -- both are O(pixels) but on the
            # downsampled image this is cheap. cv2 uses BGR by default.
            small = cv2.resize(frame, downsample_to,
                                interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mean = float(np.mean(gray))
            var = float(np.var(gray))
            means.append(mean)
            variances.append(var)
            reason = _classify_frame(mean, var, min_mean, max_mean,
                                      min_variance)
            if reason:
                if reason == "dark":
                    result.n_dark += 1
                elif reason == "bright":
                    result.n_bright += 1
                elif reason == "low_var":
                    result.n_low_var += 1
                if first_bad_idx is None:
                    first_bad_idx = frame_idx
                    first_bad_reason = reason
    finally:
        cap.release()

    result.n_frames_sampled = sampled
    result.duration_analyzed_sec = (
        (sampled * result.sample_every_n) / result.fps
        if result.fps > 0 else 0.0
    )

    if not means:
        result.ok = False
        result.error = "no frames decoded"
        result.status = "critical"
        return result

    result.mean_global = float(np.mean(means))
    result.var_global = float(np.mean(variances))
    result.mean_min = float(np.min(means))
    result.mean_max = float(np.max(means))
    result.var_min = float(np.min(variances))
    result.var_max = float(np.max(variances))

    # Distinct "bad" frames -- a single frame can fail multiple
    # criteria (dark + low_var). We don't dedupe here because the
    # individual buckets are still useful for the email body; the
    # union is computed below for pct_bad.
    bad_union = 0
    for m, v in zip(means, variances):
        if _classify_frame(m, v, min_mean, max_mean, min_variance):
            bad_union += 1
    result.n_bad = bad_union
    result.pct_bad = (100.0 * bad_union / sampled) if sampled else 0.0
    result.first_bad_idx = first_bad_idx
    result.first_bad_reason = first_bad_reason

    # Severity buckets. Critical when more than the alert threshold
    # of sampled frames are bad; warning at half that.
    if result.pct_bad >= alert_pct_bad:
        result.status = "critical"
    elif result.pct_bad >= alert_pct_bad / 2:
        result.status = "warning"
    else:
        result.status = "ok"

    return result
