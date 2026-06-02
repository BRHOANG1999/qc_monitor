"""Video helpers shared between the dispatcher, dashboard, and notebooks."""

import glob
import os
import re


def video_path_for_mat(mat_path: str) -> str | None:
    """Derive the companion video path from a .mat file path.

    Convention: ``<basename>_v1.mp4`` co-located with the .mat. Returns the
    path if the file exists on disk, ``None`` otherwise.
    """
    assert isinstance(mat_path, str) and mat_path, "mat_path must be a non-empty string"
    if "." not in os.path.basename(mat_path):
        return None
    candidate = mat_path.rsplit(".", 1)[0] + "_v1.mp4"
    if os.path.isfile(candidate):
        return candidate
    return None


_V_SUFFIX_RE = re.compile(r"_v(\d+)\.mp4$", re.IGNORECASE)


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
