"""File watcher: polls network share for new KMrecorder .mat files."""

import os
import re
import time
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger("qc_monitor.watcher")

# KMrecorder file pattern: <sessionname>___YYYY_MM_DD__HH_MM_SS.mat
MAT_PATTERN = re.compile(r"___(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})\.mat$")


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

    def scan(self, known_paths: set[str]) -> list[NewFile]:
        """Scan watch paths for new .mat files not in known_paths.

        Only returns files whose mtime is older than min_file_age_sec
        (to avoid reading partial writes).
        """
        new_files = []
        now = time.time()

        for root_path in self.watch_paths:
            if not os.path.isdir(root_path):
                logger.warning("Watch path not accessible: %s", root_path)
                continue

            try:
                for entry in os.scandir(root_path):
                    if not entry.is_dir():
                        continue
                    # Monthly folders (e.g., MARCH_2026) or session dirs
                    self._scan_directory(entry.path, known_paths, new_files, now)
            except OSError as e:
                logger.error("Error scanning %s: %s", root_path, e)

        return new_files

    def _scan_directory(self, dir_path: str, known_paths: set[str],
                        new_files: list[NewFile], now: float):
        """Recursively scan a directory for session folders containing .mat files."""
        try:
            entries = list(os.scandir(dir_path))
        except OSError:
            return

        has_mat = False
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".mat") and MAT_PATTERN.search(entry.name):
                has_mat = True
                if entry.path in known_paths:
                    continue

                try:
                    stat = entry.stat()
                except OSError:
                    continue

                # Age gate: skip files still being written
                if (now - stat.st_mtime) < self.min_file_age_sec:
                    continue

                # Extract datetime from filename
                m = MAT_PATTERN.search(entry.name)
                chunk_datetime = m.group(1) if m else ""

                session_dir = dir_path
                session_name = os.path.basename(dir_path)

                new_files.append(NewFile(
                    path=entry.path,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    session_dir=session_dir,
                    session_name=session_name,
                    chunk_datetime=chunk_datetime,
                ))

        # Recurse into subdirectories (for monthly folders containing session dirs)
        if not has_mat:
            for entry in entries:
                if entry.is_dir() and not entry.name.startswith("."):
                    self._scan_directory(entry.path, known_paths, new_files, now)

    def check_network_accessible(self) -> bool:
        """Check if at least one watch path is accessible."""
        for path in self.watch_paths:
            if os.path.isdir(path):
                return True
        return False
