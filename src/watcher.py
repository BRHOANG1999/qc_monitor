"""File watcher: polls network share for new KMrecorder .mat files."""

import os
import re
import time
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger("qc_monitor.watcher")

# KMrecorder file pattern: <sessionname>__YYYY_MM_DD__HH_MM_SS.mat
# The separator before the timestamp is 2 OR 3 underscores: older baseline
# recordings used ``___`` but newer ones (e.g. stimStability) use ``__``, so
# accept both -- otherwise the newer files are silently skipped and never
# processed into evokedOutput.
MAT_PATTERN = re.compile(r"_{2,3}(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})\.mat$")

# A recorder timestamp ANYWHERE in the name -- used only to spot ".mat that
# looks like a recording but didn't match MAT_PATTERN" (a format-drift canary
# so a future rename can't silently skip files the way the ___/__ change did).
RECORDING_TS_RX = re.compile(r"\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}")


@dataclass
class NewFile:
    path: str
    size: int
    mtime: float
    session_dir: str
    session_name: str
    chunk_datetime: str


class FileWatcher:
    def __init__(self, watch_paths: list[str], min_file_age_sec: int = 60):
        self.watch_paths = watch_paths
        self.min_file_age_sec = min_file_age_sec
        # Summary of the most recent scan, read by the health loop to detect a
        # stall (nothing processed while newer files sit on the share) and to
        # alarm on recording-like .mat that the pattern skipped (format drift).
        self.last_scan_stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict:
        # newest_chunk_dt = the newest recording on the share by its EMBEDDED
        # timestamp (from the filename), NOT file mtime -- the daemon watches a
        # daily backup that re-copies old files with fresh mtimes, so mtime
        # would look "new" every backup. The embedded datetime never changes.
        return {"newest_chunk_dt": "", "n_scanned": 0,
                "n_skipped_recordinglike": 0, "skipped_samples": []}

    def scan(self, known_paths: set[str]) -> list[NewFile]:
        """Scan watch paths for new .mat files not in known_paths.

        Only returns files whose mtime is older than min_file_age_sec
        (to avoid reading partial writes). Also refreshes ``last_scan_stats``
        (newest .mat mtime + count of recording-like files the pattern
        skipped) for the health-loop stall/format-drift alerts.
        """
        new_files = []
        now = time.time()
        stats = self._empty_stats()

        for root_path in self.watch_paths:
            if not os.path.isdir(root_path):
                logger.warning("Watch path not accessible: %s", root_path)
                continue

            try:
                for entry in os.scandir(root_path):
                    if not entry.is_dir():
                        continue
                    # Monthly folders (e.g., MARCH_2026) or session dirs
                    self._scan_directory(entry.path, known_paths, new_files,
                                         now, stats)
            except OSError as e:
                logger.error("Error scanning %s: %s", root_path, e)

        self.last_scan_stats = stats
        if stats["n_skipped_recordinglike"]:
            logger.warning(
                "Scan skipped %d .mat that look like recordings but don't "
                "match the filename pattern (possible recorder format change): "
                "%s", stats["n_skipped_recordinglike"],
                ", ".join(stats["skipped_samples"]))
        return new_files

    def _scan_directory(self, dir_path: str, known_paths: set[str],
                        new_files: list[NewFile], now: float, stats: dict):
        """Recursively scan a directory for session folders containing .mat files."""
        try:
            entries = list(os.scandir(dir_path))
        except OSError:
            return

        has_mat = False
        for entry in entries:
            if not (entry.is_file() and entry.name.endswith(".mat")):
                continue
            has_mat = True    # a session folder with .mat -> don't recurse
            try:
                stat = entry.stat()
            except OSError:
                continue
            stats["n_scanned"] += 1

            m = MAT_PATTERN.search(entry.name)
            if not m:
                # Recording-like name (has a timestamp) that the pattern
                # rejected -> the format-drift canary.
                if RECORDING_TS_RX.search(entry.name):
                    stats["n_skipped_recordinglike"] += 1
                    if len(stats["skipped_samples"]) < 5:
                        stats["skipped_samples"].append(entry.name)
                continue

            # Track the newest recording on the share by its embedded
            # datetime (immune to a backup re-copy bumping mtimes).
            if m.group(1) > stats["newest_chunk_dt"]:
                stats["newest_chunk_dt"] = m.group(1)

            if entry.path in known_paths:
                continue
            # Age gate: skip files still being written
            if (now - stat.st_mtime) < self.min_file_age_sec:
                continue

            new_files.append(NewFile(
                path=entry.path,
                size=stat.st_size,
                mtime=stat.st_mtime,
                session_dir=dir_path,
                session_name=os.path.basename(dir_path),
                chunk_datetime=m.group(1),
            ))

        # Recurse into subdirectories (for monthly folders containing session dirs)
        if not has_mat:
            for entry in entries:
                if entry.is_dir() and not entry.name.startswith("."):
                    self._scan_directory(entry.path, known_paths, new_files,
                                         now, stats)

    def check_network_accessible(self) -> bool:
        """Check if at least one watch path is accessible."""
        for path in self.watch_paths:
            if os.path.isdir(path):
                return True
        return False
