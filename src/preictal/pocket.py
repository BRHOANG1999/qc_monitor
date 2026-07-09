"""BIDS ``derivatives/`` pocket writer for pre-ictal runs.

Each run gets a self-describing pocket -- a human ``README`` + a machine
``dataset_description.json`` at every level, a ``config_snapshot.json``,
``seizures.csv`` (events + ISI + ceiling accounting) and ``scale_summary.csv``
(the deliverable mirror). Navigable by a researcher and an LLM alike; the SQLite
``preictal_scale_summary`` table stays the fast query surface. Writes are atomic
(tmp + os.replace).
"""

from __future__ import annotations

import csv
import json
import os

PIPELINE_NAME = "preictal_cwt"


def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_csv(path: str, columns: list[str], rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c, "") for c in columns])
    os.replace(tmp, path)


def ensure_root(root: str) -> None:
    """Create the top-level derivatives dir + BIDS dataset_description.json +
    orientation README once (idempotent)."""
    os.makedirs(root, exist_ok=True)
    desc = os.path.join(root, "dataset_description.json")
    if not os.path.exists(desc):
        _atomic_write(desc, json.dumps({
            "Name": "Pre-ictal CWT sweep",
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": PIPELINE_NAME}],
        }, indent=2))
    readme = os.path.join(root, "README")
    if not os.path.exists(readme):
        _atomic_write(readme,
            "Pre-ictal CWT sweep derivatives\n"
            "===============================\n\n"
            "Each runs/<run_id>_<scope>_<period>/ pocket holds one sweep: its\n"
            "config snapshot, the seizures it used (with ISI + lookback\n"
            "ceiling), and scale_summary.csv -- the discriminability statistic\n"
            "vs. CWT time-scale. The matching rows live in the SQLite\n"
            "preictal_scale_summary table (fast query surface).\n\n"
            "Score glossary: coeff_* = |CWT coefficient| summaries per scale;\n"
            "collapse_stat / null_* / forecast_* fill in at stages 2-3.\n")


def run_dir_name(run_id: int, scope: str, period_start: str | None,
                  period_end: str | None) -> str:
    def _d(s):
        return (s or "na").replace(":", "").replace("-", "")[:8] or "na"
    return f"{run_id:06d}_{scope}_{_d(period_start)}-{_d(period_end)}"


_SEIZURE_COLS = ["file_id", "animal_id", "chunk_datetime", "eo_sec", "bb_sec",
                 "racine", "seizure_type", "onset_epoch", "isi_sec",
                 "ceiling_sec", "used"]
_SCALE_COLS = ["feature", "channel_role", "scale_index", "scale",
               "pseudo_freq_hz", "coeff_mean", "coeff_std", "coeff_max",
               "n_seizures"]


def write_run_pocket(root: str, run_id: int, scope: str,
                      period_start: str | None, period_end: str | None,
                      run_meta: dict, scale_rows: list[dict],
                      seizure_rows: list[dict]) -> str:
    """Write one run's pocket. Returns its absolute path."""
    ensure_root(root)
    runs = os.path.join(root, "runs")
    os.makedirs(runs, exist_ok=True)
    pocket = os.path.join(runs, run_dir_name(run_id, scope, period_start,
                                             period_end))
    os.makedirs(os.path.join(pocket, "matrices"), exist_ok=True)
    # Machine manifest.
    _atomic_write(os.path.join(pocket, "dataset_description.json"),
                  json.dumps({
                      "Name": f"{PIPELINE_NAME} run {run_id}",
                      "DatasetType": "derivative",
                      "GeneratedBy": [{"Name": PIPELINE_NAME}],
                      "run_id": run_id, "scope": scope,
                      "period_start": period_start, "period_end": period_end,
                      **run_meta,
                      "artifacts": {"scale_summary": "scale_summary.csv",
                                    "seizures": "seizures.csv"},
                  }, indent=2, default=str))
    # Human README.
    n_sz = run_meta.get("n_seizures", len(seizure_rows))
    n_used = sum(1 for r in seizure_rows if r.get("used"))
    _atomic_write(os.path.join(pocket, "README"),
        f"Pre-ictal CWT run {run_id} ({scope})\n"
        f"period: {period_start} .. {period_end}\n"
        f"animals: {', '.join(run_meta.get('animals', []) or [])}\n\n"
        f"{n_sz} seizures in scope, {n_used} with a usable pre-ictal lead-up\n"
        f"(the rest dropped: first-of-animal, ISI too short, or missing data).\n"
        f"ISI-derived lookback ceiling (s): min={run_meta.get('ceiling_min')} "
        f"median={run_meta.get('ceiling_median')} "
        f"max={run_meta.get('ceiling_max')}\n\n"
        f"scale_summary.csv: |CWT coefficient| per (feature, channel_role, "
        f"scale). Higher pseudo_freq_hz = faster pre-ictal time-scale.\n"
        f"How to read: compare coeff_mean across scales for the headline "
        f"feature; the collapse/null/forecast columns arrive in stages 2-3.\n")
    _atomic_write(os.path.join(pocket, "config_snapshot.json"),
                  json.dumps(run_meta.get("config", {}), indent=2, default=str))
    _atomic_csv(os.path.join(pocket, "seizures.csv"), _SEIZURE_COLS,
                seizure_rows)
    _atomic_csv(os.path.join(pocket, "scale_summary.csv"), _SCALE_COLS,
                scale_rows)
    return pocket
