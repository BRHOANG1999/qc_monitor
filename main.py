"""QC Monitor — 24/7 offline analysis daemon for KMrecorder data.

Watches \\\\100.106.104.22\\bhz\\database\\ for new .mat files, runs Tier 1 Python QC
and Tier 2 MATLAB analysis pipeline, stores results in SQLite, and (optionally)
serves a Dash dashboard + sends email alerts.

Usage:
    python main.py                  # Run with default config
    python main.py --config path    # Run with custom config
    python main.py --no-dashboard   # Run without web dashboard
    python main.py --no-matlab      # Skip MATLAB Tier 2 pipeline
"""

import argparse
import os
import sys
import time
import logging
import threading
import psutil
import shutil

# Ensure project root is on sys.path for thread imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

from src.db.store import Store
from src.watcher import FileWatcher
from src.dispatcher import Dispatcher
from src.alerting.email_alert import EmailAlerter
from src.alerting.rules import AlertRuleEngine
from src.notifications.scheduler import DigestScheduler
from src.utils.logging_config import setup_logging


def load_config(path: str) -> dict:
    # Force UTF-8 -- config contains emoji in maintenance tab names and
    # Windows' default cp1252 codec chokes on them.
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def health_snapshot(watcher: FileWatcher, store: Store, queue_depth: int,
                    db_path: str) -> dict:
    # Report free space on the disk the DB actually lives on so the
    # snapshot survives a move off D:\ (e.g., a non-Windows host running
    # tests, or a future deployment on a different drive letter).
    disk = shutil.disk_usage(os.path.dirname(os.path.abspath(db_path)) or ".")
    return {
        "cpu_pct": psutil.cpu_percent(interval=0.5),
        "memory_pct": psutil.virtual_memory().percent,
        "disk_free_gb": disk.free / (1024 ** 3),
        "network_share_accessible": int(watcher.check_network_accessible()),
        "queue_depth": queue_depth,
        "files_processed_last_hour": store.get_files_processed_last_hour(),
    }


def run_dashboard(config: dict, store: Store):
    """Launch Dash dashboard in a background thread."""
    try:
        from src.dashboard.app import create_app
        app = create_app(config, store)
        host = config.get("dashboard", {}).get("host", "0.0.0.0")
        port = config.get("dashboard", {}).get("port", 8050)
        logging.getLogger("qc_monitor").info("Dashboard serving at http://%s:%s", host, port)
        # Suppress Flask/Werkzeug request logging (the POST spam)
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except ImportError as e:
        logging.getLogger("qc_monitor").warning(
            "Dash not installed (%s). Run: pip install dash plotly", e)
    except Exception as e:
        logging.getLogger("qc_monitor").error("Dashboard failed: %s", e, exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="QC Monitor for KMrecorder data")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip dashboard")
    parser.add_argument("--no-matlab", action="store_true", help="Skip MATLAB pipeline")
    args = parser.parse_args()

    # Resolve config path relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config) if not os.path.isabs(args.config) else args.config
    config = load_config(config_path)

    if args.no_matlab:
        config.pop("matlab_exe", None)

    # Setup logging
    log_cfg = config.get("logging", {})
    logger = setup_logging(
        log_dir=log_cfg.get("dir", os.path.join(script_dir, "logs")),
        level=log_cfg.get("level", "INFO"),
        max_bytes=log_cfg.get("max_file_size_mb", 10) * 1024 * 1024,
        backup_count=log_cfg.get("max_files", 5),
    )
    logger.info("=" * 60)
    logger.info("QC Monitor starting")
    logger.info("Config: %s", config_path)

    # Init components
    db_path = config.get("database", {}).get("path",
              os.path.join(script_dir, "data", "monitor.db"))
    store = Store(db_path)
    store.reset_stale_processing()

    watch_cfg = config.get("watch", {})
    watcher = FileWatcher(
        watch_paths=watch_cfg.get("paths", []),
        min_file_age_sec=watch_cfg.get("min_file_age_sec", 60),
    )

    dispatcher = Dispatcher(store, config)

    # Create initial settings version
    version_id = store.create_settings_version(config, label="initial")
    logger.info("Settings version: %d", version_id)

    emailer = EmailAlerter(config)
    alert_engine = AlertRuleEngine(store, emailer, config)
    digest_scheduler = DigestScheduler(config, emailer, store=store)

    poll_interval = watch_cfg.get("poll_interval_sec", 30)
    health_interval = 60  # seconds

    # Launch dashboard in background thread
    if not args.no_dashboard:
        dash_thread = threading.Thread(target=run_dashboard, args=(config, store), daemon=True)
        dash_thread.start()
        logger.info("Dashboard thread started on port %s",
                     config.get("dashboard", {}).get("port", 8050))

    # Build set of already-known file paths + content fingerprints.
    # The same physical .mat is reachable from two SMB shares (//.../bhz/
    # and //.../database/), so dedup by both path AND (basename, size, mtime)
    # to avoid processing the same recording twice through different paths.
    logger.info("Building known file index...")
    known_paths = set()
    known_fingerprints = set()

    with store.connection() as conn:
        for row in conn.execute("SELECT file_path, file_size, file_mtime FROM processed_files"):
            fp, fsize, fmt = row[0], row[1], row[2]
            known_paths.add(fp)
            known_fingerprints.add((os.path.basename(fp), fsize, fmt))
    logger.info("Known files: %d (unique fingerprints: %d)",
                len(known_paths), len(known_fingerprints))

    # Main loop
    last_health_time = 0
    consecutive_network_failures = 0
    files_since_scan = 10  # force initial scan

    logger.info("Entering main watch loop (poll every %ds)...", poll_interval)

    while True:
        try:
            loop_start = time.time()

            # Daily surgery digest -- fires once per day at/after the
            # configured hour. Cheap no-op when disabled or already sent.
            try:
                digest_scheduler.tick()
            except Exception as e:
                logger.error("Digest scheduler tick failed: %s", e)

            # Health check
            if (loop_start - last_health_time) > health_interval:
                accessible = watcher.check_network_accessible()
                alert_engine.check_network(accessible)

                if accessible:
                    consecutive_network_failures = 0
                else:
                    consecutive_network_failures += 1

                health = health_snapshot(watcher, store, 0, db_path)
                store.insert_health(health)
                last_health_time = loop_start

                if not accessible:
                    logger.warning("Network share not accessible (failure #%d)",
                                   consecutive_network_failures)
                    # Exponential backoff
                    backoff = min(300, poll_interval * (2 ** min(consecutive_network_failures, 4)))
                    time.sleep(backoff)
                    continue

            # Scan for new files every 10 processed files or when queue is empty
            if files_since_scan >= 10 or not store.get_pending_files(limit=1):
                new_files = watcher.scan(known_paths)
                if new_files:
                    new_files.sort(key=lambda f: f.chunk_datetime)
                    registered = 0
                    deduped = 0
                    for nf in new_files:
                        fp = (os.path.basename(nf.path), nf.size, nf.mtime)
                        if fp in known_fingerprints:
                            # Same physical file already known under a different
                            # share path. Remember the duplicate path so we don't
                            # rediscover it every scan, but don't register/process.
                            known_paths.add(nf.path)
                            deduped += 1
                            continue
                        known_paths.add(nf.path)
                        known_fingerprints.add(fp)
                        store.register_file(
                            file_path=nf.path, file_size=nf.size, file_mtime=nf.mtime,
                            session_dir=nf.session_dir, session_name=nf.session_name,
                            chunk_datetime=nf.chunk_datetime,
                        )
                        registered += 1
                    logger.info("Registered %d new files (%d deduped by fingerprint)",
                                registered, deduped)
                files_since_scan = 0

            # Process pending files in batch (up to 10), then rescan
            pending = store.get_pending_files(limit=1)
            if pending:
                pf = pending[0]
                total_pending = len(store.get_pending_files(limit=99999))
                logger.info("[%d queued] Processing: %s",
                            total_pending, os.path.basename(pf["file_path"]))
                from src.watcher import NewFile
                nf = NewFile(
                    path=pf["file_path"], size=pf.get("file_size", 0),
                    mtime=pf.get("file_mtime", 0),
                    session_dir=pf.get("session_dir", ""),
                    session_name=pf.get("session_name", ""),
                    chunk_datetime=pf.get("chunk_datetime", ""),
                )
                dispatcher.process_file(nf, version_id=version_id)
                files_since_scan += 1
            else:
                # Nothing to process — sleep until next poll
                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Shutting down (Ctrl+C)")
            break
        except Exception as e:
            logger.error("Main loop error: %s", e, exc_info=True)
            time.sleep(poll_interval)

    # Cleanup: kill any orphaned MATLAB subprocesses (Windows only --
    # the daemon's production host. On other platforms the dispatcher
    # is exercised under tests with a stubbed matlab_bridge, so there
    # is nothing to reap.)
    if sys.platform == "win32":
        logger.info("Cleaning up MATLAB subprocesses...")
        try:
            import subprocess
            subprocess.run(["taskkill", "/F", "/IM", "MATLAB.exe"],
                           capture_output=True, timeout=10)
        except Exception:
            pass
    logger.info("QC Monitor stopped")


if __name__ == "__main__":
    main()
