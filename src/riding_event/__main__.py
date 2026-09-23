"""CLI: riding-event analysis (Prong A + Prong B) for one animal.

    python -m src.riding_event --animal BCH111                # auto-pick, both prongs
    python -m src.riding_event --animal BCH111 --mode residual
    python -m src.riding_event --recording <evoked_or_raw_path> --band 40 300

Reads ``config/config.yaml`` for the database path (which resolves the evoked
output dir + raw recording paths). Figures + CSVs + manifest land in
``data/derivatives/riding_event/<animal>/``. Progress prints throughout — the
raw-LFP read for Prong B is a multi-GB network read and takes a while.
"""

from __future__ import annotations

import argparse
import os


def _main(argv=None) -> int:
    import yaml

    from src.db.store import Store
    from src.riding_event import run as _run

    ap = argparse.ArgumentParser(description="Riding-event analysis (A + B)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", default="BCH111")
    ap.add_argument("--mode", choices=["both", "residual", "detect"],
                    default="both", help="which prong(s) to run")
    ap.add_argument("--recording", default=None,
                    help="force a specific evoked/raw path (skips auto-pick)")
    ap.add_argument("--band", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"),
                    help="detection band (Hz); default = the Prong-A excess band")
    ap.add_argument("--scan-files", type=int, default=20,
                    help="how many newest recordings to probe when auto-picking")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("opening database...", flush=True)
    store = Store(cfg["database"]["path"])
    out_dir = args.out or os.path.join("data", "derivatives", "riding_event",
                                       args.animal)
    summary = _run.build(store, args.animal, out_dir=out_dir, mode=args.mode,
                         recording=args.recording, band=args.band,
                         scan_files=args.scan_files,
                         progress=lambda m: print("  ", m, flush=True))
    if summary.get("reason"):
        print(f"{args.animal}: {summary['reason']}")
        return 1
    print(f"\n{args.animal}: {len(summary.get('figures', []))} figures -> {out_dir}")
    for key in ("prong_a", "prong_b"):
        if key in summary:
            print(f"  {key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
