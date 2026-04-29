"""Video helpers shared between the dispatcher, dashboard, and notebooks."""

import os


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
