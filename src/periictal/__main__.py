"""CLI: build the peri-ictal evoked matrix for one animal/protocol and render
the embedding, coloured by continuous time-to-onset and by the two confounds
(stim fingerprint, time-of-day) -- the first honest look at an animal before
any dashboard exists.

Usage:
    python -m src.periictal --animal BCH111 --protocol chronicStim --window-h 6
    python -m src.periictal --animal BCH111 --protocol chronicStim --method umap
    python -m src.periictal --animal BCH040 --protocol '' --window-h 6   # all
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                       # noqa: E402
from src.dashboard.data_helpers import load_config   # noqa: E402
from src.periictal import config as _cfg             # noqa: E402
from src.periictal import passive as _passive        # noqa: E402
from src.periictal import render                     # noqa: E402
from src.periictal.embed import confound_readout, embed  # noqa: E402
from src.periictal.matrix import build_matrix        # noqa: E402

_OUT_ROOT = os.path.join(_ROOT, "data", "derivatives", "periictal")


def _parse(argv):
    p = argparse.ArgumentParser(prog="python -m src.periictal")
    p.add_argument("--animal", required=True)
    p.add_argument("--protocol", default="chronicStim",
                   help="protocol-token substring; '' = all protocols")
    p.add_argument("--variant", default="evoked", choices=("evoked", "passive"))
    p.add_argument("--window-h", type=float, default=6.0)
    p.add_argument("--method", default="pca", choices=("pca", "umap"))
    p.add_argument("--out", default=_OUT_ROOT)
    return p.parse_args(argv)


def main(argv) -> int:
    args = _parse(argv)
    cfg = load_config()
    store = Store(cfg["database"]["path"])
    evoked_dir = cfg["chronic_evoked"]["evoked_output_dir"]
    protocol = args.protocol or None

    passive_cfg = _passive.passive_config() if args.variant == "passive" else None
    if passive_cfg is not None:
        print(f"Warming passive sidecars for {args.animal} "
              f"(window {passive_cfg.window_start_ms:.0f}.."
              f"{passive_cfg.window_end_ms:.1f} ms)...", flush=True)
        built, total = _passive.warm_passive(
            args.animal, evoked_dir, passive_cfg, protocol=protocol,
            progress=lambda d, n, fp: (d % 10 == 0 or d == n) and
            print(f"  passive {d}/{n}", flush=True))
        print(f"  passive warm: {built} built / {total} files", flush=True)

    print(f"Building {args.animal} / {protocol or 'ALL'} / {args.variant} "
          f"matrix (window {args.window_h} h)...", flush=True)
    t0 = time.time()
    df = build_matrix(store, args.animal, evoked_dir, protocol=protocol,
                      window_sec=args.window_h * 3600.0, variant=args.variant,
                      passive_cfg=passive_cfg)
    print(f"  {len(df)} lead-up stimuli, "
          f"{df['seizure_idx'].nunique() if len(df) else 0} seizures, "
          f"in {time.time() - t0:.1f}s", flush=True)
    if df.empty:
        print("  Nothing to embed -- <2 seizures or no fresh sidecars.")
        return 1

    print(f"Embedding ({args.method})...", flush=True)
    res = embed(df, method=args.method)
    emb, sub = res["emb"], df.loc[res["rows"]]
    print(f"  {res['meta'].get('n_points')} points, cols={len(res['cols'])}, "
          f"meta={res['meta']}", flush=True)

    _report_confounds(emb, sub)
    out_dir = os.path.join(args.out, args.animal, args.variant)
    os.makedirs(out_dir, exist_ok=True)
    _render_all(emb, sub, out_dir, args, res)
    print(f"Wrote figures to {out_dir}")
    return 0


def _report_confounds(emb, sub) -> None:
    ro = confound_readout(emb, sub)
    print("Confound readout (|Spearman rho| embedding-axis vs confound; "
          "high => the picture is that confound, not seizure proximity):")
    for k, v in ro.items():
        print(f"  {k:18} {'n/a' if v is None else f'{v:.2f}'}")


def _render_all(emb, sub, out_dir, args, res) -> None:
    import numpy as np
    stem = f"{args.animal}_{args.variant}_{res['method']}"
    hrs = sub["time_to_onset_sec"].to_numpy() / 3600.0
    render.scatter_continuous(
        emb, hrs, os.path.join(out_dir, f"{stem}_by_timetoonset.png"),
        title=f"{args.animal} {args.protocol or 'ALL'} {args.variant} "
              f"({res['method'].upper()}) — colour = hours to next seizure",
        clabel="hours to next onset", cmap="plasma_r")
    render.scatter_categorical(
        emb, sub["stim_key"].to_numpy(),
        os.path.join(out_dir, f"{stem}_by_stimfingerprint.png"),
        title=f"{args.animal} {args.variant} — colour = stim fingerprint "
              f"(confound check)", legend_title="stim key")
    render.scatter_continuous(
        emb, sub["hour_of_day"].to_numpy(),
        os.path.join(out_dir, f"{stem}_by_hourofday.png"),
        title=f"{args.animal} {args.variant} — colour = hour of day "
              f"(circadian confound check)", clabel="hour of day", cmap="twilight")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
