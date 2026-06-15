"""Single source of truth for the app version string.

Derived from git -- the ``<short-sha>`` of HEAD -- because there is no
setup.py / pyproject.toml here. Returns ``unknown`` when git isn't
reachable (deployed without .git, or git not on PATH).

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
    return sha or "unknown"
