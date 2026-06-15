"""Golden test for the BHZ_DETECTOR-compatible CSV writer.

Verifies the header is byte-for-byte identical to the lab's
reference (``G:\\BHZ\\BHZ_CSV_Exports\\20260224_BCH040.csv``)
and that a round-trip of one LVF event + one "No events" file
produces rows pandas can re-read with the expected types.

Run with: pytest tests/test_bhz_csv.py -q
Or directly: python -m tests.test_bhz_csv
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.utils.bhz_csv import (
    COLUMNS, ONSET_LVF, build_file_meta, resolve_csv_path,
    write_event_rows,
)


REF_CSV = Path(r"G:\BHZ\BHZ_CSV_Exports\20260224_BCH040.csv")


def _ref_header() -> list[str]:
    """Read the live reference's header. Falls back to COLUMNS
    when the share isn't reachable so CI without the G: drive
    still passes."""
    if not REF_CSV.exists():
        return list(COLUMNS)
    with REF_CSV.open("r", encoding="utf-8") as f:
        return next(csv.reader(f))


class CsvHeader(unittest.TestCase):

    def test_header_matches_reference(self):
        """The first 44 columns must be the reference header
        verbatim. ``SoftwareVersion`` is a deliberate lab-local
        addition appended at the end (provenance per row), so it
        is excluded from the reference comparison."""
        ref = _ref_header()
        if ref and ref[-1] == "SoftwareVersion":
            ref = ref[:-1]  # fallback path returns our own COLUMNS
        expected = [c for c in COLUMNS if c != "SoftwareVersion"]
        self.assertEqual(expected, ref,
                         "BHZ CSV header drifted from the lab "
                         "reference. Update COLUMNS to match "
                         "G:\\BHZ\\BHZ_CSV_Exports.")
        self.assertEqual(COLUMNS[-1], "SoftwareVersion",
                         "SoftwareVersion must stay the trailing "
                         "column so MATLAB readtable is unaffected.")


class LvfRoundTrip(unittest.TestCase):

    def setUp(self):
        self.fs = 20000.0
        self.tmpdir = tempfile.mkdtemp(prefix="bhz_csv_")
        self.csv = Path(self.tmpdir) / "20260224_BCH040.csv"

    def tearDown(self):
        if self.csv.exists():
            self.csv.unlink()
        os.rmdir(self.tmpdir)

    def _file_meta(self) -> dict:
        return build_file_meta(
            folder=r"U:\FEBRUARY_2026\session__BCH040SR_",
            filename="session___2026_02_24__02_02_25.mat",
            fs=self.fs, cutoff=0.05, channel=2,
            peak_index=29163022,
            peak_stamp=740037.101947559,
            peak_dt=datetime(2026, 2, 24, 2, 26, 48),
        )

    def test_lvf_event_writes_one_row(self):
        events = [{
            "type": "LVF",
            "EO_sec":  28429513 / self.fs,
            "LAS_sec": 29013857 / self.fs,
            "BO_sec":  28770374 / self.fs,
            "PID_sec": 29226040 / self.fs,
            "BB_sec":  32028816 / self.fs,
            "racine":  3,
            "onset_comment": ("Brief fast, low amplitude waves "
                              "seen at the seizure onset."),
            "behavior_comment":
                "Marked by sudden body and head movement.",
            "score_comment": "Forelimb clonus observed.",
            "roomlight": "off",
            "video_quality": "Sufficient",
        }]
        n = write_event_rows(self.csv, self._file_meta(),
                              events, self.fs)
        self.assertEqual(n, 1)

        with self.csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        r = rows[0]

        # Header was the COLUMNS tuple.
        self.assertEqual(list(reader.fieldnames or []),
                         list(COLUMNS))

        # Sample-index conversions round-trip exactly when the
        # original was an integer.
        self.assertEqual(r["EventEO"], "28429513")
        self.assertEqual(r["EventLAS"], "29013857")
        self.assertEqual(r["EventBO"], "28770374")
        self.assertEqual(r["EventPID"], "29226040")
        self.assertEqual(r["EventBB"], "32028816")

        # Onset is the FULL string with parens.
        self.assertEqual(r["Onset"], ONSET_LVF)

        # Score is a small integer rendered without a decimal.
        self.assertEqual(r["Score"], "3")

        # Mode = manual for event rows; Target = LFP.
        self.assertEqual(r["Mode"], "manual")
        self.assertEqual(r["Target"], "LFP")

        # Empty Eventstart/Eventstop strings (not NaN, per the
        # reference's row encoding).
        self.assertEqual(r["Eventstart"], "")
        self.assertEqual(r["Eventstop"], "")

        # The lowercase 'scorecomment' (col 44) carries the
        # reviewer note; uppercase 'ScoreComment' (col 33)
        # stays empty.
        self.assertEqual(r["scorecomment"],
                         "Forelimb clonus observed.")
        self.assertEqual(r["ScoreComment"], "")

        # Quoted comma inside OnsetComment round-trips through
        # csv.QUOTE_MINIMAL.
        self.assertEqual(
            r["OnsetComment"],
            "Brief fast, low amplitude waves seen at "
            "the seizure onset.")


class NoEventsFile(unittest.TestCase):

    def test_no_events_row(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "20260224_BCH040.csv"
            fm = build_file_meta(
                folder=r"U:\some\folder",
                filename="no_events__2026_02_24.mat",
                fs=20000.0, cutoff=0.05, channel=2,
                peak_index=None, peak_stamp=None, peak_dt=None,
            )
            n = write_event_rows(p, fm, [], 20000.0)
            self.assertEqual(n, 1)

            with p.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            r = rows[0]

            # Numeric Event* cols read NaN. The reference treats
            # Eventstart / Eventstop as STRING columns (empty
            # cells, not NaN) -- they aren't real sample-index
            # fields. The bhz_csv module's _NUMERIC_COLS mirrors
            # this distinction.
            string_event_cols = {"Eventstart", "Eventstop"}
            for c in COLUMNS:
                if not c.startswith("Event"):
                    continue
                if c in string_event_cols:
                    self.assertEqual(r[c], "",
                                     f"{c} should be empty string")
                else:
                    self.assertEqual(r[c], "NaN",
                                     f"{c} should be NaN")

            # Comment is the canonical 'No events' string,
            # Mode is empty (not 'manual'), Target=LFP.
            self.assertEqual(
                r["Comment"],
                "No events in file for current threshold")
            self.assertEqual(r["Mode"], "")
            self.assertEqual(r["Target"], "LFP")


class Idempotency(unittest.TestCase):

    def test_repeat_save_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            fm = build_file_meta(
                folder=r"U:\f", filename="x.mat",
                fs=20000.0, cutoff=0.05, channel=2,
                peak_index=1, peak_stamp=1.0,
                peak_dt=datetime(2026, 1, 1),
            )
            ev = [{"type": "HYP", "EO_sec": 1.0,
                    "BO_sec": 2.0, "PID_sec": 3.0,
                    "BB_sec": 4.0, "racine": 2,
                    "onset_comment": "", "behavior_comment": "",
                    "score_comment": "", "roomlight": "",
                    "video_quality": ""}]
            n1 = write_event_rows(p, fm, ev, 20000.0)
            n2 = write_event_rows(p, fm, ev, 20000.0)
            self.assertEqual(n1, 1)
            self.assertEqual(n2, 0,
                             "second save must dedupe by EventEO")
            with p.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)


class FilenameResolver(unittest.TestCase):

    def test_resolve_csv_path(self):
        from datetime import date
        p = resolve_csv_path(
            r"G:\BHZ\BHZ_CSV_Exports",
            "{date}_{animal}.csv",
            date(2026, 2, 24), "BCH040",
        )
        self.assertEqual(
            str(p),
            r"G:\BHZ\BHZ_CSV_Exports\20260224_BCH040.csv",
        )


if __name__ == "__main__":
    unittest.main()
