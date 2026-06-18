"""Video helpers shared between the dispatcher, dashboard, and notebooks."""

import glob
import os
import re


def video_path_for_mat(mat_path: str) -> str | None:
    """Derive the companion (browser-playable) video path from a .mat file
    path.

    Convention: ``<basename>_v1.mp4`` co-located with the .mat; older / lab
    recordings sometimes drop a plain ``<basename>.mp4`` instead. Tries both
    and returns the first that exists on disk, ``None`` otherwise. (Only
    ``.mp4`` is returned -- ``.avi`` isn't playable in an HTML5 <video>.)
    """
    assert isinstance(mat_path, str) and mat_path, "mat_path must be a non-empty string"
    if "." not in os.path.basename(mat_path):
        return None
    base = mat_path.rsplit(".", 1)[0]
    for candidate in (base + "_v1.mp4", base + ".mp4"):
        if os.path.isfile(candidate):
            return candidate
    return None


def avi_companion_exists(mat_path: str) -> bool:
    """True when a co-located ``.avi`` companion exists (``<base>_vN.avi`` or
    ``<base>.avi``). Browsers can't play .avi in <video>, so callers use this
    to explain *why* there's no playback rather than show a broken player."""
    if not isinstance(mat_path, str) or not mat_path:
        return False
    if "." not in os.path.basename(mat_path):
        return False
    base = mat_path.rsplit(".", 1)[0]
    return bool(glob.glob(base + "_v*.avi") or glob.glob(base + ".avi"))


_V_SUFFIX_RE = re.compile(r"_v(\d+)\.mp4$", re.IGNORECASE)


def companion_videos(mat_path: str) -> list[dict]:
    """Every companion video for *mat_path*, one entry per camera, as
    ``[{"cam": N, "path": str, "kind": "mp4"|"avi"}]`` sorted by camera N.

    Handles ``<base>_vN.mp4`` / ``<base>_vN.avi`` (multi-camera) and a bare
    ``<base>.mp4`` / ``<base>.avi`` (single camera = cam 1). When a camera has
    both kinds, mp4 wins (no transcode needed). The lab's behavioral
    recordings are typically ``_v1.avi`` + ``_v2.avi``.
    """
    if not isinstance(mat_path, str) or not mat_path:
        return []
    if "." not in os.path.basename(mat_path):
        return []
    base = mat_path.rsplit(".", 1)[0]
    by_cam: dict[int, dict] = {}
    for kind in ("mp4", "avi"):
        for m in glob.glob(base + "_v*." + kind):
            mm = re.search(r"_v(\d+)\." + kind + r"$", m, re.IGNORECASE)
            if not mm:
                continue
            n = int(mm.group(1))
            cur = by_cam.get(n)
            if cur is None or (kind == "mp4" and cur["kind"] == "avi"):
                by_cam[n] = {"cam": n, "path": m, "kind": kind}
    if not by_cam:
        for kind in ("mp4", "avi"):
            cand = base + "." + kind
            if os.path.isfile(cand):
                by_cam[1] = {"cam": 1, "path": cand, "kind": kind}
                break
    return [by_cam[n] for n in sorted(by_cam)]


def companion_video_paths(mat_path: str) -> list[str]:
    """Every ``<basename>_vN.mp4`` companion of *mat_path*, sorted by N.

    Some sessions write more than one camera angle per chunk (cage A,
    cage B, etc.) -- callers that previously hard-coded ``_v1`` miss
    the rest. Glob-based to avoid hammering ``os.path.isfile`` on a
    range of speculative numbers.
    """
    assert isinstance(mat_path, str) and mat_path, "mat_path must be non-empty"
    if "." not in os.path.basename(mat_path):
        return []
    base = mat_path.rsplit(".", 1)[0]
    matches = glob.glob(base + "_v*.mp4")
    keyed: list[tuple[int, str]] = []
    for m in matches:
        if not os.path.isfile(m):
            continue
        match = _V_SUFFIX_RE.search(m)
        keyed.append((int(match.group(1)) if match else 0, m))
    keyed.sort(key=lambda t: t[0])
    return [p for _n, p in keyed]
