"""CLI: ``python -m src.evoked_shapes --animal BCH111 [--apply]``.

Dry-run by default (assemble + analyze, print the verdict and k range, write nothing).
``--apply`` writes the Stage 2 to 4 figures, report.md, and manifest.json under
data/derivatives/evoked_shapes/<animal>/.
"""

from __future__ import annotations

import argparse
import sys

from src.dashboard.data_helpers import load_config
from src.evoked_shapes import run as _run


def _log(msg: str) -> None:
    print("  ", msg, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--animal", default="BCH111")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--apply", action="store_true",
                    help="write figures + report (default: dry-run)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore any cached dataset")
    ap.add_argument("--days", type=int, default=None,
                    help="override the gather window (days of recordings)")
    a = ap.parse_args(argv)

    config = load_config()
    if a.days is not None:
        config.setdefault("evoked_shapes", {})["days"] = int(a.days)
    print(f"evoked_shapes: {a.animal} (apply={a.apply})", flush=True)
    summary = _run.build_animal(a.animal, config, apply=a.apply,
                                cache=not a.no_cache, log=_log)
    print("---", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    return 0 if summary.get("n_gathered", 0) >= 4 else 1


if __name__ == "__main__":
    sys.exit(main())
