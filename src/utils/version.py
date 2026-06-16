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


def _repo_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))


def _compute_version() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", _repo_root(), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"
    return sha or "unknown"


def newer_version_available(page_version: str | None) -> bool:
    """True iff the code on disk (current HEAD) is strictly AHEAD of
    *page_version* -- i.e. a newer version has been deployed since the
    page was served, and the user should reload.

    Uses ``git merge-base --is-ancestor``: returns True only when
    *page_version* is an ancestor of HEAD (HEAD is a descendant /
    newer). Diverged branches, an identical SHA, or a missing/unknown
    version all return False, so this never nags spuriously. Safe when
    git is unavailable (returns False).
    """
    if not page_version or page_version == "unknown":
        return False
    head = _compute_version()  # live, uncached -- reflects a fresh pull
    if head == "unknown" or head == page_version:
        return False
    try:
        # Exit 0 -> page_version IS an ancestor of HEAD -> HEAD newer.
        # Exit 1 -> not an ancestor (diverged / older). Either way the
        # boolean return below reflects it.
        result = subprocess.run(
            ["git", "-C", _repo_root(), "merge-base", "--is-ancestor",
             page_version, "HEAD"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False
    return result.returncode == 0
