"""Single source of truth for the app version string.

Derived from git -- ``<short-sha>[-dirty]  <YYYY-MM-DD>`` -- because
there is no setup.py / pyproject.toml here. Returns ``unknown`` when
git isn't reachable (deployed without .git, or git not on PATH).

Computed once and cached so the subprocess cost is paid a single time
per process. The cached value is therefore frozen at first call, which
is exactly what provenance wants: every CSV row written during one run
of the app carries the same version stamp.
"""

from __future__ import annotations

import os
import subprocess

_CACHED: str | None = None


def qc_monitor_version() -> str:
    """Return the cached version string (computes on first call)."""
    global _CACHED
    if _CACHED is None:
        _CACHED = _compute_version()
    return _CACHED


def _compute_version() -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "-C", repo_root, "status", "--porcelain"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        dirty = False
    try:
        ts = subprocess.check_output(
            ["git", "-C", repo_root, "log", "-1", "--format=%cs"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        ts = ""
    label = sha + ("-dirty" if dirty else "")
    return f"{label}  {ts}".strip() if ts else label
