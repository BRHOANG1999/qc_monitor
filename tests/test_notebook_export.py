"""Peri-ictal notebook export: ZIP structure + frozen-matrix round-trip.

(End-to-end kernel execution of every generated notebook against the frozen data
was verified separately with nbconvert; this suite keeps the fast, deterministic
structural guarantees.)
"""

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.periictal import notebook_export as nx
from src.utils.evoked_features import CHEAP_COLUMNS

_LENSES = ["periictal_slidingauc", "periictal_pdfcdf",
           "periictal_trend", "periictal_embedding"]
_PARAMS = {
    "periictal_slidingauc": {"variant": "evoked", "n_windows": 12,
                             "band_lo": 3600.0, "band_hi": 21600.0},
    "periictal_pdfcdf": {"variant": "evoked", "nphases": 6},
    "periictal_trend": {"variant": "evoked", "feature": CHEAP_COLUMNS[0]},
    "periictal_embedding": {"variant": "evoked", "method": "pca"},
}


def _full():
    rng = np.random.default_rng(0)
    frames = []
    for s in range(3):
        onset = 1e6 + s * 30000.0
        tto = np.arange(10.0, 22000.0, 10.0)
        n = tto.size
        d = {"phase": "pre", "time_to_onset_sec": tto, "seizure_idx": s,
             "seizure_onset_epoch": onset, "seizure_racine": 2.0, "channel": "C1",
             "session": "chronicStim-5nC", "rec": "r", "stim_key": "k",
             "stim_status": "stim", "hour_of_day": rng.uniform(0, 24, n),
             "lead_bin": 0, "t_epoch": onset - tto, "abs_dt": None}
        for c in CHEAP_COLUMNS:
            d[c] = rng.normal(0, 1, n)
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("lens", _LENSES)
def test_export_zip_structure(lens):
    full = _full()
    blob, fname = nx.build_export(lens, full, {"animal": "BCH111",
                                               "variant": "evoked"}, _PARAMS[lens])
    assert fname.startswith(lens) and fname.endswith(".zip")
    z = zipfile.ZipFile(io.BytesIO(blob))
    assert {"analysis.ipynb", nx.DATA_FILE, "README.md"} <= set(z.namelist())
    nb = json.loads(z.read("analysis.ipynb"))               # valid notebook JSON
    assert nb["nbformat"] == 4 and nb["cells"]
    assert all("id" in c for c in nb["cells"])              # nbformat 4.5 cell ids
    # every notebook re-runs the pure analysis modules (not a copy of the code)
    srcs = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "src.periictal" in srcs and nx.DATA_FILE in srcs


def test_frozen_matrix_round_trips():
    full = _full()
    blob, _ = nx.build_export("periictal_slidingauc", full,
                              {"animal": "BCH111"}, _PARAMS["periictal_slidingauc"])
    z = zipfile.ZipFile(io.BytesIO(blob))
    got = pd.read_pickle(io.BytesIO(z.read(nx.DATA_FILE)), compression="gzip")
    assert got.shape == full.shape
    assert list(got.columns) == list(full.columns)
    pd.testing.assert_frame_equal(got, full)


def test_unsupported_lens_raises():
    with pytest.raises(ValueError):
        nx.build_export("periictal_waveform", _full(), {}, {})
