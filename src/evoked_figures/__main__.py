"""CLI: python -m src.evoked_figures [--animal X] [--apply]

Dry-run (default) reports what would be rendered per animal; --apply writes the
PNGs + CSV + manifest under evoked_figures.out_root.
"""

from __future__ import annotations

import argparse
import time

import yaml

from src.evoked_figures import config as _cfg
from src.evoked_figures import run as _run
from src.utils.evoked_output import list_animals


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Render evoked-metric-over-time figures per animal.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", default=None, help="one animal; default = all")
    ap.add_argument("--apply", action="store_true",
                    help="write figures (default is a dry-run)")
    a = ap.parse_args(argv)

    config = {}
    try:
        config = yaml.safe_load(open(a.config, encoding="utf-8")) or {}
    except Exception as e:                             # noqa: BLE001
        print("config load failed:", e)

    evoked_dir = _cfg.evoked_dir(config)
    animals = [a.animal] if a.animal else list_animals(evoked_dir)
    metrics = _cfg.declared_metrics(config)
    print(f"{'APPLY' if a.apply else 'DRY-RUN'}: {len(animals)} animal(s) × "
          f"{len(metrics)} metrics × 2 plots -> {_cfg.out_root(config)}",
          flush=True)
    t0 = time.time()
    for i, an in enumerate(animals, 1):
        s = _run.build_animal(an, config, apply=a.apply)
        print(f"  [{i}/{len(animals)}] {an}: channel={s.get('channel')} "
              f"rendered={s.get('rendered')} missing={len(s.get('missing') or [])} "
              f"pruned={s.get('pruned', 0)} rows={s.get('n_rows', 0)}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    if not a.apply:
        print("(dry-run: no files written. re-run with --apply.)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
