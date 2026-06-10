"""Force-render every tab's layout() function against the smoke DB.
Surfaces crashes the boot-time diagnostic silently swallows.

Run:
    python tools/smoke_layouts.py <smoke-db-path>
"""
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: smoke_layouts.py <smoke-db-path>", file=sys.stderr)
        return 2
    smoke_db = sys.argv[1]

    import yaml
    with open(REPO / "config" / "config.yaml", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config["database"]["path"] = smoke_db

    from src.db.store import Store
    store = Store(smoke_db)

    from src.dashboard.tabs import (
        alerts, signal_quality, stim, criticality,
        session_compare, activity_log, annotations, sessions,
        electrode_health, waveforms, evoked, settings,
        video, surgeries, maintenance, data_log_xref,
        lfp_browser, overview,
        review_status, event_verification,
    )

    tabs = [
        ("alerts", alerts.layout),
        ("signal_quality", signal_quality.layout),
        ("stim", stim.layout),
        ("criticality", criticality.layout),
        ("session_compare", session_compare.layout),
        ("activity_log", activity_log.layout),
        ("annotations", annotations.layout),
        ("sessions", sessions.layout),
        ("electrode_health", electrode_health.layout),
        ("waveforms", waveforms.layout),
        ("evoked", evoked.layout),
        ("settings", settings.layout),
        ("video", video.layout),
        ("surgeries", surgeries.layout),
        ("maintenance", maintenance.layout),
        ("data_log_xref", data_log_xref.layout),
        ("lfp_browser", lfp_browser.layout),
        ("overview", overview.layout),
        ("review_status", review_status.layout),
        ("event_verification", event_verification.layout),
    ]

    failures = []
    for name, fn in tabs:
        try:
            try:
                fn(store)
            except TypeError:
                fn(store, config)
            print(f"  [OK]   {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")

    if failures:
        print(f"\n{len(failures)} tab(s) crashed during layout():\n", file=sys.stderr)
        for name, exc in failures:
            print(f"--- {name} ---", file=sys.stderr)
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        return 1
    print(f"\nAll {len(tabs)} tabs rendered cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
