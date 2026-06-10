"""One-shot smoke: load config, point DB at temp copy, boot create_app
with the callback-wiring diagnostic on. Logs to stdout, exits non-zero
on any error.

Run from repo root:
    python tools/smoke_boot.py <path-to-smoke-db>
"""
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: smoke_boot.py <smoke-db-path>", file=sys.stderr)
        return 2
    smoke_db = sys.argv[1]
    if not Path(smoke_db).is_file():
        print(f"db not found: {smoke_db}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s %(name)-22s %(message)s",
    )
    os.environ["QC_MONITOR_CHECK_CALLBACKS"] = "1"

    import yaml
    cfg_path = REPO / "config" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    config["database"]["path"] = smoke_db
    config.setdefault("dashboard", {})["host"] = "127.0.0.1"
    config["dashboard"]["port"] = 0

    from src.db.store import Store
    store = Store(smoke_db)
    print(f"Store opened on {smoke_db}", flush=True)

    from src.dashboard.app import create_app
    app = create_app(config, store)
    n_cb = len(getattr(app, "_callback_list", []))
    print(f"create_app OK -- {n_cb} callbacks registered", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
