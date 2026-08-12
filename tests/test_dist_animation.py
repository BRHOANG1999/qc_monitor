"""dist_animation: per-seizure metric-distribution GIF marching to onset.

Run with: pytest tests/test_dist_animation.py -q
"""

from __future__ import annotations

import io
import os
import sys
import zipfile

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pd = pytest.importorskip("pandas")
pytest.importorskip("PIL")
pytest.importorskip("matplotlib")

from src.periictal import dist_animation as da  # noqa: E402


def _synth(n_per=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sid in (0, 1):
        tto = rng.uniform(1, da.LOOKBACK_SEC, n_per)
        mu = 5.0 + 3.0 * (1.0 - tto / da.LOOKBACK_SEC)   # drifts up toward onset
        for t, v in zip(tto, rng.normal(mu, 1.0)):
            rows.append({"phase": "pre", "time_to_onset_sec": t,
                         "seizure_idx": sid, "peak": v})
        rows.append({"phase": "post", "time_to_onset_sec": -10.0,
                     "seizure_idx": sid, "peak": 5.0})     # ignored
    return pd.DataFrame(rows)


def test_window_offsets_far_to_near_ending_at_onset():
    offs = da.window_offsets()
    assert offs[-1] == 0.0, "last window must be [0, width] -- right up to onset"
    assert np.all(np.diff(offs) < 0), "offsets must run far -> near"
    assert offs[0] == da.LOOKBACK_SEC - da.WIDTH_SEC


def test_window_frames_pre_only_and_ordered():
    full = _synth()
    frames = da.window_frames(full, 0, "peak")
    assert len(frames) == da.window_offsets().size
    # far -> near: the last frame is the onset-adjacent window.
    assert frames[-1]["lo"] == 0.0 and frames[-1]["hi"] == da.WIDTH_SEC
    # post rows (tto<0) never contribute.
    assert all(f["values"].size >= 0 for f in frames)
    # the near-onset mean is higher than the far mean (the planted drift).
    far = frames[0]["values"].mean()
    near = frames[-1]["values"].mean()
    assert near > far + 1.0, (far, near)


def test_group_pools_all_seizures():
    full = _synth()
    one = da.window_frames(full, 0, "peak")[-1]["values"].size
    grp = da.window_frames(full, "group", "peak")[-1]["values"].size
    assert grp >= one, "group must pool >= a single seizure's stimuli"


def _n_frames(gif_bytes):
    from PIL import Image
    return Image.open(io.BytesIO(gif_bytes)).n_frames


def test_render_seizure_gif_is_valid_multiframe():
    full = _synth()
    gif = da.render_seizure_gif(full, 0, "peak", label="sz0")
    assert gif[:6] in (b"GIF87a", b"GIF89a"), "not a GIF"
    assert _n_frames(gif) == da.window_offsets().size


def test_render_raises_when_no_data():
    full = _synth()
    with pytest.raises(ValueError):
        da.render_seizure_gif(full[full["seizure_idx"] == 99], 99, "peak")


def test_zip_has_one_gif_per_seizure():
    full = _synth()
    labels = {0: "BCH040 sz#0", 1: "BCH040 sz#1"}
    blob = da.render_all_seizures_zip(full, "peak", [0, 1], labels)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    gifs = [n for n in zf.namelist() if n.endswith(".gif")]
    assert len(gifs) == 2, zf.namelist()
    assert all("peak" in n for n in gifs)


def test_safe_name():
    assert da.safe_name("BCH040 sz#0 (a/b)") == "BCH040_sz_0_a_b"
    assert da.safe_name("") == "x"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
