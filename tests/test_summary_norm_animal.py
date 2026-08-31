"""summary_sheet._norm_animal: zero-pad canonicalization so surgery-sheet id
variants (BCH60 / BCH0100) reconcile with the DB's zero-padded ids (BCH060 /
BCH100). Regression for BCH060's blank surgery columns.

Run: pytest tests/test_summary_norm_animal.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.summary_sheet import _norm_animal  # noqa: E402


def test_padding_variants_canonicalize_equal():
    for v in ("60", "060", "BCH60", "BCH060", "bch060", " BCH 60 "):
        assert _norm_animal(v) == "BCH060", v


def test_over_and_under_padding():
    assert _norm_animal("BCH0100") == "BCH100"      # over-padded -> BCH100
    assert _norm_animal("100") == "BCH100"
    assert _norm_animal("BCH63") == "BCH063"
    assert _norm_animal("BCH111") == "BCH111"       # already canonical
    assert _norm_animal("39") == "BCH039"


def test_four_plus_digit_ids_not_truncated():
    assert _norm_animal("BCH1000") == "BCH1000"     # zfill(3) never truncates


def test_non_bch_and_empty_pass_through():
    assert _norm_animal("Randles") == "RANDLES"
    assert _norm_animal("") == ""
    assert _norm_animal(None) == ""


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
