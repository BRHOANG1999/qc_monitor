"""QC Monitor — 24/7 offline analysis daemon for KMrecorder data.

Watches \\100.106.104.22\bhz\database\ for new .mat files, runs Tier 1 Python QC
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

import yaml

from src.db.store import Store
from src.watcher import FileWatcher
from src.dispatcher import Dispatcher
from src.alerting.email_alert import EmailAlerter
from src.alerting.rules import AlertRuleEngine
from src.utils.logging_config import setup_logging


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def health_snapshot(watcher: FileWatcher, store: Store, queue_depth: int) -> dict:
    disk = shutil.disk_usage("D:\\")
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
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except ImportError:
        logging.getLogger("qc_monitor").warning(
            "Dash not installed. Run: pip install dash plotly. Dashboard disabled.")
    except Exception as e:
        logging.getLogger("qc_monitor").error("Dashboard failed: %s", e)


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
        config.setdefault("analysis", {}).setdefault("tier2", {})["enabled"] = False

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

    emailer = EmailAlerter(config)
    alert_engine = AlertRuleEngine(store, emailer, config)

    poll_interval = watch_cfg.get("poll_interval_sec", 30)
    health_interval = 60  # seconds

    # Launch dashboard in background thread
    if not args.no_dashboard:
        dash_thread = threading.Thread(target=run_dashboard, args=(config, store), daemon=True)
        dash_thread.start()
        logger.info("Dashboard thread started on port %s",
                     config.get("dashboard", {}).get("port", 8050))

    # Build set of already-known file paths
    logger.info("Building known file index...")
    known_paths = set()
    for row in store.get_pending_files(limit=999999):
        known_paths.add(row["file_path"])
    # Also include done/error files
    import sqlite3
    conn = sqlite3.connect(db_path)
    for row in conn.execute("SELECT file_path FROM processed_files"):
        known_paths.add(row[0])
    conn.close()
    logger.info("Known files: %d", len(known_paths))

    # Main loop
    last_health_time = 0
    consecutive_network_failures = 0

    logger.info("Entering main watch loop (poll every %ds)...", poll_interval)

    while True:
        try:
            loop_start = time.time()

            # Health check
            if (loop_start - last_health_time) > health_interval:
                accessible = watcher.check_network_accessible()
                alert_engine.check_network(accessible)

                if accessible:
                    consecutive_network_failures = 0
                else:
                    consecutive_network_failures += 1

                health = health_snapshot(watcher, store, 0)
                store.insert_health(health)
                last_health_time = loop_start

                if not accessible:
                    logger.warning("Network share not accessible (failure #%d)",
                                   consecutive_network_failures)
                    # Exponential backoff
                    backoff = min(300, poll_interval * (2 ** min(consecutive_network_failures, 4)))
                    time.sleep(backoff)
                    continue

            # Scan for new files
            new_files = watcher.scan(known_paths)

            if new_files:
                logger.info("Found %d new files", len(new_files))
                # Sort by datetime (oldest first)
                new_files.sort(key=lambda f: f.chunk_datetime)

            for nf in new_files:
                known_paths.add(nf.path)
                logger.info("Processing: %s", os.path.basename(nf.path))

                success = dispatcher.process_file(nf)

                if success:
                    # Check alerts on the QC results
                    # (alert checking happens inside dispatcher for per-chunk metrics)
                    pass

            # Sleep until next poll
            elapsed = time.time() - loop_start
            sleep_time = max(1, poll_interval - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Shutting down (Ctrl+C)")
            break
        except Exception as e:
            logger.error("Main loop error: %s", e, exc_info=True)
            time.sleep(poll_interval)

    logger.info("QC Monitor stopped")


if __name__ == "__main__":
    main()
