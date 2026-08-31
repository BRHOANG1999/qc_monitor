"""CLI for the chronic-stim stability map.

    python -m src.chronic_stability --animal BCH111 [--probe]
        [--win-h 24 --gap-tol-h 6 --disp-k 4 --slope-k 6]
        [--out data/derivatives/chronic_stability] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

from src.dashboard.data_helpers import load_config
from src.db.store import Store
from src.preictal.isi import scored_seizures
from src.utils import evoked_output as eo


def _args(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.chronic_stability")
    p.add_argument("--animal", default="BCH111")
    p.add_argument("--win-h", type=float, default=24.0)
    p.add_argument("--gap-tol-h", type=float, default=6.0)
    p.add_argument("--disp-k", type=float, default=4.0)
    p.add_argument("--slope-k", type=float, default=6.0)
    p.add_argument("--spike-k", type=float, default=5.0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default="data/derivatives/chronic_stability")
    p.add_argument("--probe", action="store_true")
    return p.parse_args(argv)


def _probe(store, animal: str, evoked_dir: str) -> None:
    """Read-only sanity print: Rₐ keys/span, evoked files/tokens, seizures."""
    print(f"=== PROBE {animal} ===", flush=True)
    ser = store.impedance_series_by_channel()
    for k in sorted(x for x in ser if x[0] == animal):
        rows = ser[k]
        span = f"{rows[0]['chunk_datetime']} .. {rows[-1]['chunk_datetime']}"
        print(f"  Rₐ {k}: {len(rows)} pts  {span}")
    files = [f for f in eo.list_evoked_files(evoked_dir)
             if animal in os.path.basename(f)]
    toks = {}
    for f in files:
        toks[os.path.basename(f).split("__", 1)[0]] = \
            toks.get(os.path.basename(f).split("__", 1)[0], 0) + 1
    print(f"  evoked files: {len(files)}  tokens: {toks}")
    sz = scored_seizures(store, animal)
    if sz:
        a = _dt.datetime.fromtimestamp(sz[0].onset_epoch)
        b = _dt.datetime.fromtimestamp(sz[-1].onset_epoch)
        print(f"  seizures: {len(sz)}  {a:%Y-%m-%d %H:%M} .. {b:%Y-%m-%d %H:%M}")


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):     # allow Rₐ/µ in console output
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    a = _args(argv)
    cfg = load_config()
    store = Store(cfg["database"]["path"])
    evoked_dir = cfg["chronic_evoked"]["evoked_output_dir"]
    if a.probe:
        _probe(store, a.animal, evoked_dir)
        return 0
    from . import report
    flat = {"win_h": a.win_h, "gap_tol_h": a.gap_tol_h,
            "disp_k": a.disp_k, "slope_k": a.slope_k,
            "spike_k": a.spike_k}
    out_dir = os.path.join(a.out, a.animal)
    report.run(store, a.animal, evoked_dir, out_dir,
               flat_params=flat, n_boot=a.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
