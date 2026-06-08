"""Track D verification: bhz_csv.overwrite_day_csv + CsvDiff.

Each case feeds a known on-disk CSV state + a proposed
events_by_filename, then asserts the returned CsvDiff
matches expectations and the file on disk reflects the new
canonical set.

Run with: pytest tests/test_bhz_csv_overwrite.py -q
Or directly: python -m tests.test_bhz_csv_overwrite
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.utils.bhz_csv import (
    COLUMNS, CsvDiff, ONSET_LVF,
    build_file_meta, existing_filenames_in_csv,
    overwrite_day_csv, write_event_rows,
)


FS = 20000.0


def _meta(filename: str) -> dict:
    return build_file_meta(
        folder=r"U:\fixture\sess__BCH040SR_",
        filename=filename,
        fs=FS, cutoff=0.05, channel=2,
        peak_index=None, peak_stamp=None,
        peak_dt=datetime(2026, 2, 24, 2, 26, 48),
    )


def _ev(eo_sec: float, racine: int = 3) -> dict:
    return {
        "type": "LVF",
        "EO_sec": eo_sec, "LAS_sec": eo_sec + 10,
        "BO_sec": eo_sec + 5, "PID_sec": eo_sec + 30,
        "BB_sec": eo_sec + 50, "racine": racine,
        "onset_comment": "", "behavior_comment": "",
        "score_comment": "",
        "roomlight": "off", "video_quality": "Sufficient",
    }


class OverwriteEmptyCsv(unittest.TestCase):

    def test_added_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "20260224_BCH040.csv"
            diff = overwrite_day_csv(
                p,
                events_by_filename={
                    "a.mat": [_ev(300.0)],
                    "b.mat": [],
                },
                file_meta_by_filename={
                    "a.mat": _meta("a.mat"),
                    "b.mat": _meta("b.mat"),
                },
                fs=FS,
            )
            self.assertEqual(diff.added_filenames,
                              ["a.mat", "b.mat"])
            self.assertEqual(diff.removed_filenames, [])
            self.assertEqual(diff.modified_filenames, [])

            with p.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            by_fn = {r["filename"]: r for r in rows}
            self.assertEqual(by_fn["a.mat"]["Onset"], ONSET_LVF)
            self.assertEqual(by_fn["b.mat"]["Comment"],
                              "No events in file for current threshold")


class OverwriteRemoval(unittest.TestCase):

    def test_removed_filename_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "20260224_BCH040.csv"
            # Seed with two files.
            write_event_rows(p, _meta("keep.mat"),
                              [_ev(300.0)], FS)
            write_event_rows(p, _meta("drop.mat"),
                              [_ev(400.0)], FS)
            # Overwrite with only one of them.
            diff = overwrite_day_csv(
                p,
                events_by_filename={"keep.mat": [_ev(300.0)]},
                file_meta_by_filename={
                    "keep.mat": _meta("keep.mat"),
                },
                fs=FS,
            )
            self.assertEqual(diff.added_filenames, [])
            self.assertEqual(diff.removed_filenames,
                              ["drop.mat"])
            with p.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["filename"], "keep.mat")


class OverwriteNoop(unittest.TestCase):

    def test_idempotent_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            ev_dict = {"a.mat": [_ev(100.0)]}
            meta_dict = {"a.mat": _meta("a.mat")}
            d1 = overwrite_day_csv(p, ev_dict, meta_dict, FS)
            d2 = overwrite_day_csv(p, ev_dict, meta_dict, FS)
            self.assertEqual(d1.added_filenames, ["a.mat"])
            self.assertEqual(d2.added_filenames, [])
            self.assertEqual(d2.removed_filenames, [])
            self.assertEqual(d2.modified_filenames, [])


class OverwriteModification(unittest.TestCase):

    def test_modified_filename_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            # Seed with EO=300.
            overwrite_day_csv(
                p,
                events_by_filename={"a.mat": [_ev(300.0)]},
                file_meta_by_filename={
                    "a.mat": _meta("a.mat"),
                },
                fs=FS,
            )
            # Overwrite with EO=305.
            diff = overwrite_day_csv(
                p,
                events_by_filename={"a.mat": [_ev(305.0)]},
                file_meta_by_filename={
                    "a.mat": _meta("a.mat"),
                },
                fs=FS,
            )
            self.assertEqual(diff.added_filenames, [])
            self.assertEqual(diff.removed_filenames, [])
            self.assertEqual(diff.modified_filenames, ["a.mat"])


class ExistingFilenamesScan(unittest.TestCase):

    def test_groups_by_filename(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            overwrite_day_csv(
                p,
                events_by_filename={
                    "a.mat": [_ev(100.0), _ev(500.0)],
                    "b.mat": [_ev(200.0)],
                },
                file_meta_by_filename={
                    "a.mat": _meta("a.mat"),
                    "b.mat": _meta("b.mat"),
                },
                fs=FS,
            )
            groups = existing_filenames_in_csv(p)
            self.assertEqual(set(groups.keys()),
                              {"a.mat", "b.mat"})
            self.assertEqual(len(groups["a.mat"]), 2)
            self.assertEqual(len(groups["b.mat"]), 1)


if __name__ == "__main__":
    unittest.main()
