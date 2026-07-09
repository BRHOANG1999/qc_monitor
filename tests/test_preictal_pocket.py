"""Pre-ictal BIDS derivatives pocket: valid tree + manifests, atomic writes.

Run with: pytest tests/test_preictal_pocket.py -q
"""

from __future__ import annotations

import glob
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.preictal import pocket  # noqa: E402


def test_write_run_pocket_tree_and_manifest(tmp_path):
    root = str(tmp_path / "derivatives" / "preictal_cwt")
    scale_rows = [
        {"feature": "line_length", "channel_role": "eeg", "scale_index": 0,
         "scale": 2.0, "pseudo_freq_hz": 0.5, "coeff_mean": 1.1,
         "coeff_std": 0.2, "coeff_max": 1.9, "n_seizures": 4},
    ]
    seizure_rows = [
        {"file_id": 1, "animal_id": "BCH040", "chunk_datetime": "x",
         "eo_sec": 100.0, "bb_sec": 150.0, "racine": 3, "seizure_type": "LVF",
         "onset_epoch": 1.0, "isi_sec": None, "ceiling_sec": None, "used": 0},
        {"file_id": 2, "animal_id": "BCH040", "chunk_datetime": "y",
         "eo_sec": 50.0, "bb_sec": None, "racine": 4, "seizure_type": "LVF",
         "onset_epoch": 500.0, "isi_sec": 400.0, "ceiling_sec": 100.0,
         "used": 1},
    ]
    p = pocket.write_run_pocket(
        root, run_id=7, scope="daily", period_start="2026-07-08",
        period_end="2026-07-08",
        run_meta={"animals": ["BCH040"], "n_seizures": 2,
                  "ceiling_min": 100.0, "ceiling_median": 100.0,
                  "ceiling_max": 100.0, "config": {"preictal": {"x": 1}}},
        scale_rows=scale_rows, seizure_rows=seizure_rows)

    # Top-level BIDS description + README.
    assert os.path.exists(os.path.join(root, "dataset_description.json"))
    assert os.path.exists(os.path.join(root, "README"))
    top = json.load(open(os.path.join(root, "dataset_description.json")))
    assert top["DatasetType"] == "derivative"

    # Run pocket: manifest is valid JSON, CSVs present, matrices dir exists.
    man = json.load(open(os.path.join(p, "dataset_description.json")))
    assert man["run_id"] == 7 and man["scope"] == "daily"
    for f in ("README", "config_snapshot.json", "seizures.csv",
              "scale_summary.csv"):
        assert os.path.exists(os.path.join(p, f)), f
    assert os.path.isdir(os.path.join(p, "matrices"))

    # scale_summary.csv round-trips the deliverable columns.
    import csv
    with open(os.path.join(p, "scale_summary.csv"), newline="") as f:
        got = list(csv.DictReader(f))
    assert got[0]["feature"] == "line_length"
    assert float(got[0]["coeff_mean"]) == 1.1

    # No leftover atomic-write temp files anywhere in the tree.
    assert glob.glob(os.path.join(root, "**", "*.tmp"), recursive=True) == []


def test_ensure_root_idempotent(tmp_path):
    root = str(tmp_path / "d")
    pocket.ensure_root(root)
    pocket.ensure_root(root)                     # second call must not raise
    assert os.path.exists(os.path.join(root, "dataset_description.json"))
