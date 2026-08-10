"""Standalone dashboard process entry point.

Runs ONLY the Dash dashboard (via ``create_app``) in its own OS process,
separate from the daemon watch loop / dispatcher / MATLAB ingest. This is the
fix for the dashboard freezing whenever the daemon's in-process ``load_mat``
holds the GIL: with the two tiers in separate interpreters, the daemon's
minutes-long ``.mat`` reads off the SMB share can no longer starve any dashboard
thread. The two processes share only the SQLite WAL file (``Store`` opens a
fresh per-call connection), so no IPC is needed.

``QC_DASHBOARD_ROLE`` is set BEFORE ``create_app`` runs so this process does NOT
start the sustained-writer workers (``mass_analyze``, ``event_clip``) -- the
daemon (``main.py``) owns those (starting them here too would double-drain the
job queues).

Deploy: run the daemon as service ``QCMonitorDaemon`` = ``main.py
--no-dashboard`` and this as ``QCMonitorDashboard`` = ``dashboard_main.py`` (see
install_service.bat). No ``multiprocessing``/spawn is involved -- NSSM launches
two independent ``python.exe`` processes.
"""

import os
import sys

# Make the repo importable regardless of the CWD NSSM launches us with.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# MUST be set before `from main import run_dashboard` pulls in create_app, which
# reads this at call time to decide worker ownership.
os.environ["QC_DASHBOARD_ROLE"] = "1"

from main import load_config, run_dashboard          # noqa: E402
from src.db.store import Store                        # noqa: E402
from src.utils.logging_config import setup_logging    # noqa: E402


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "config.yaml")
    config = load_config(config_path)

    log_cfg = config.get("logging", {})
    setup_logging(
        log_dir=log_cfg.get("dir", os.path.join(script_dir, "logs")),
        level=log_cfg.get("level", "INFO"),
        max_bytes=log_cfg.get("max_file_size_mb", 10) * 1024 * 1024,
        backup_count=log_cfg.get("max_files", 5),
    )

    # Own Store in THIS process: sqlite3.Connection is unpicklable and
    # Store._WRITE_LOCK is per-process -- it must be constructed here, not
    # shared from the daemon.
    db_path = config.get("database", {}).get(
        "path", os.path.join(script_dir, "data", "monitor.db"))
    store = Store(db_path)

    # This process serves production traffic and no longer competes with the
    # daemon for the GIL, so prefer waitress's fixed thread pool over the
    # threaded dev server. Scoped to THIS process (not config.yaml) so the
    # current single-process daemon's behaviour is untouched. Falls back to the
    # dev server automatically if waitress isn't installed (_serve_waitress).
    dash_cfg = config.setdefault("dashboard", {})
    dash_cfg["use_waitress"] = True
    # 16, not the shared-process default of 8: this process no longer competes
    # with the daemon for the GIL, and 8 threads convoyed (observed waitress
    # queue depth 16). Override rather than setdefault so config's 8 doesn't win.
    dash_cfg["waitress_threads"] = 16

    # run_dashboard builds create_app(config, store) and serves it via waitress
    # (or the threaded dev server) exactly as the in-process dashboard did.
    run_dashboard(config, store)


if __name__ == "__main__":
    main()
