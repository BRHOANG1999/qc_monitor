"""CLI: the Chang et al. 2026 preictal-vs-interictal report for one animal.

Builds the evoked feature matrix, labels each lead-up stimulus preictal
(0-30 min before a seizure) vs interictal (60-90 min), then writes paper-style
figures (per-feature AUC screen, PDF/CDF for the top features, and the
prospective logistic-regression forecaster) plus a stats CSV to the
derivatives tree -- the headless twin of the "Preictal vs interictal" lens.

Usage:
    python -m src.periictal.forecast_report --animal BCH111
    python -m src.periictal.forecast_report --animal BCH111 --phases 6 --top 4
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                        # noqa: E402
from src.dashboard.data_helpers import load_config    # noqa: E402
from src.periictal import config as _cfg              # noqa: E402
from src.periictal import forecast as _fc             # noqa: E402
from src.periictal import render                      # noqa: E402
from src.periictal.matrix import build_matrix         # noqa: E402

_OUT_ROOT = os.path.join(_ROOT, "data", "derivatives", "periictal_forecast")


def _write_stats_csv(path: str, scan: list, res: dict, n_seizures: int) -> None:
    """Per-feature AUC/p/q + the forecaster's per-phase AUC + coefficients."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "auc_norm", "direction", "p", "q",
                    "n_pre", "n_inter"])
        for r in scan:
            w.writerow(["feature", r["feature"], f"{r['auc_norm']:.4f}",
                        r["direction"], f"{r['p']:.4g}", f"{r['q']:.4g}",
                        r["n_pre"], r["n_inter"]])
        w.writerow([])
        w.writerow(["forecaster", "mean_auc", f"{res.get('mean_auc'):.4f}",
                    "", "", "", "", ""])
        w.writerow(["forecaster", "phase_auc_slope",
                    f"{res.get('phase_auc_slope'):.4f}", "", "", "", "", ""])
        w.writerow(["forecaster", "n_phases", res.get("n_phases"), "", "", "",
                    "", ""])
        w.writerow(["forecaster", "n_seizures", n_seizures, "", "", "", "", ""])
        for ph in res.get("phases", []):
            w.writerow(["phase", f"P{ph['train_phase']}->P{ph['test_phase']}",
                        f"{ph['auc']:.4f}", "", "", "", ph["n_train"],
                        ph["n_test"]])
        for feat, c in (res.get("coefficients") or {}).items():
            w.writerow(["coefficient", feat, f"{c:.4f}", "", "", "", "", ""])


def run(store, animal: str, evoked_dir: str, out_dir: str, *,
        protocol: str | None = "chronicStim", window_h: float = 6.0,
        n_phases: int = 6, top: int = 4, n_perm: int = 500,
        warm_missing: bool = True) -> dict:
    """Build + label + render the report for *animal*. Returns a summary dict
    (also printed by the CLI). *warm_missing* recomputes stale/missing sidecars
    so a fresh checkout / a feature-version bump self-heals; pass False to read
    only the already-warm sidecars (fast, when the near-seizure files are
    known to be current)."""
    os.makedirs(out_dir, exist_ok=True)
    df = build_matrix(store, animal, evoked_dir, protocol=protocol or None,
                      window_sec=window_h * 3600.0, warm_missing=warm_missing)
    if df.empty:
        return {"animal": animal, "ok": False, "reason": "empty matrix"}
    lab = _fc.label_classes(df)
    n_pre = int((lab["class"] == _fc.CLASS_PREICTAL).sum())
    n_int = int((lab["class"] == _fc.CLASS_INTERICTAL).sum())
    n_sz = int(lab.loc[lab["class"] != "", "seizure_idx"].nunique())
    if n_pre == 0 or n_int == 0:
        return {"animal": animal, "ok": False,
                "reason": f"no class samples (pre={n_pre}, int={n_int})"}
    metrics = [m for m in _cfg.CHEAP_METRICS if m in lab.columns]
    scan = _fc.scan_features(lab, metrics, n_perm=n_perm)
    res = _fc.logistic_forecast(lab, n_phases=n_phases)

    stem = f"{animal}"
    render.feature_auc_bar(
        scan, os.path.join(out_dir, f"{stem}_feature_auc.png"),
        title=f"{animal} — preictal vs interictal AUC per feature")
    for r in scan[:max(1, top)]:
        pc = _fc.pdf_cdf(lab, r["feature"])
        render.pdf_cdf_panel(
            pc, os.path.join(out_dir, f"{stem}_pdfcdf_{r['feature']}.png"),
            feature=r["feature"],
            title=f"{animal} — {r['feature']} (AUC {r['auc_norm']:.3f}, "
                  f"q {r['q']:.3g})")
    render.forecast_panel(
        res, os.path.join(out_dir, f"{stem}_forecaster.png"),
        title=f"{animal} — prospective logistic forecaster "
              f"(mean AUC {res.get('mean_auc'):.3f}, {n_sz} seizures)")
    _write_stats_csv(os.path.join(out_dir, f"{stem}_stats.csv"), scan, res, n_sz)
    return {"animal": animal, "ok": True, "n_pre": n_pre, "n_inter": n_int,
            "n_seizures": n_sz, "top_feature": scan[0]["feature"],
            "top_auc": scan[0]["auc_norm"], "mean_forecast_auc": res.get("mean_auc"),
            "out_dir": out_dir}


def _parse(argv):
    p = argparse.ArgumentParser(prog="python -m src.periictal.forecast_report")
    p.add_argument("--animal", required=True)
    p.add_argument("--protocol", default="chronicStim",
                   help="protocol-token substring; '' = all protocols")
    p.add_argument("--window-h", type=float, default=6.0)
    p.add_argument("--phases", type=int, default=6)
    p.add_argument("--top", type=int, default=4,
                   help="render PDF/CDF for the top-N features by AUC")
    p.add_argument("--perm", type=int, default=500)
    p.add_argument("--no-warm", action="store_true",
                   help="read only already-warm sidecars (don't recompute "
                        "stale/missing ones)")
    p.add_argument("--out", default=_OUT_ROOT)
    return p.parse_args(argv)


def main(argv) -> int:
    args = _parse(argv)
    cfg = load_config()
    store = Store(cfg["database"]["path"])
    evoked_dir = cfg["chronic_evoked"]["evoked_output_dir"]
    out_dir = os.path.join(args.out, args.animal)
    print(f"Building preictal/interictal report for {args.animal} "
          f"({args.protocol or 'ALL'})...", flush=True)
    t0 = time.time()
    summary = run(store, args.animal, evoked_dir, out_dir,
                  protocol=args.protocol or None, window_h=args.window_h,
                  n_phases=args.phases, top=args.top, n_perm=args.perm,
                  warm_missing=not args.no_warm)
    print(f"  done in {time.time() - t0:.0f}s: {summary}", flush=True)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
