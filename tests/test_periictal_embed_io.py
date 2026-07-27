"""Save / load a peri-ictal embedding bundle (export / import).

Covers the serialisation behind the Explorer's Export/Import: a round-trip
preserves the selection + the render result (DataFrames + arrays), and a
wrong/corrupt/foreign file is rejected with a clear error before it can reach a
lens.

Run: pytest tests/test_periictal_embed_io.py -q
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import embed_io as EIO  # noqa: E402


def _selection():
    return {"animal": "BCH111", "protocol": "chronicStim-5nC-5nC",
            "variant": "evoked", "window_h": 6, "method": "umap",
            "winmode": "full"}


def _result():
    return {"emb": np.arange(6.0).reshape(3, 2),
            "cols": ["m1", "m2"], "meta": {"n_neighbors": 15},
            "method": "umap", "n_seizures": 2, "empty": False,
            "sub": pd.DataFrame({"seizure_idx": [0, 0, 1]}),
            "full": pd.DataFrame({"phase": ["pre", "post"]})}


# --------------------------------------------------------- round trip --- #

def test_round_trip_preserves_selection_and_result():
    sel, res = _selection(), _result()
    sel2, res2 = EIO.loads(EIO.dumps(sel, res))
    assert sel2 == sel
    assert np.array_equal(res2["emb"], res["emb"])
    assert res2["cols"] == res["cols"] and res2["meta"] == res["meta"]
    assert res2["sub"].equals(res["sub"])       # DataFrames survive intact
    assert res2["full"].equals(res["full"])


# ------------------------------------------------------- rejections --- #

def test_wrong_version_rejected():
    blob = pickle.dumps({"version": 999, "selection": {}, "result": {}})
    with pytest.raises(ValueError, match="version"):
        EIO.loads(blob)


def test_non_bundle_object_rejected():
    with pytest.raises(ValueError):
        EIO.loads(pickle.dumps([1, 2, 3]))


def test_result_missing_embedding_rejected():
    blob = pickle.dumps({"version": EIO._VERSION,
                         "selection": _selection(),
                         "result": {"cols": [], "meta": {}, "method": "pca"}})
    with pytest.raises(ValueError, match="emb"):
        EIO.loads(blob)


def test_garbage_bytes_rejected_cleanly():
    with pytest.raises(ValueError, match="readable"):
        EIO.loads(b"not a pickle at all")


# --------------------------------------------------------- naming --- #

def test_suggested_name_is_filesystem_safe():
    name = EIO.suggested_name(_selection())
    assert name == "periictal_BCH111_chronicStim-5nC-5nC_umap.pkl"
    weird = EIO.suggested_name({"animal": "A/B\\C", "protocol": None,
                                "method": "pca"})
    assert "/" not in weird and "\\" not in weird and weird.endswith(".pkl")


def test_dumps_requires_dicts():
    with pytest.raises(AssertionError):
        EIO.dumps("nope", {})
    with pytest.raises(AssertionError):
        EIO.dumps({}, "nope")
