"""Pre-ictal trajectory: robust-z, decimated windowed features, and cross-chunk
lead-up concatenation (with a too-large-gap skip).

Run with: pytest tests/test_preictal_trajectory.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.preictal import trajectory as traj  # noqa: E402
from src.preictal.types import FeatureSpec, Seizure  # noqa: E402
from src.utils.mat_loader import ChunkData  # noqa: E402


def test_robust_z_median_mad():
    z = traj.robust_z(np.array([1., 2., 3., 4., 5.]))
    assert abs(z[2]) < 1e-9                        # median -> 0
    assert abs(z[0] + z[4]) < 1e-9                 # symmetric
    # Flat input -> zeros (MAD 0 -> scale 1).
    assert np.allclose(traj.robust_z(np.array([7., 7., 7.])), 0.0)


def test_feature_trajectory_windows():
    mean_feat = FeatureSpec(name="_mean", fn=lambda w, fs: float(np.mean(w)))
    sig = np.concatenate([np.full(10, 1.0), np.full(10, 2.0), np.full(10, 3.0)])
    t = traj.feature_trajectory(sig, fs=10.0, feature=mean_feat,
                                step_sec=1.0, target_fs=500.0)
    assert np.allclose(t, [1.0, 2.0, 3.0])         # one mean per 1 s window


_T0 = datetime(2026, 1, 1, 0, 0, 0).timestamp()


def _seed_two_chunks(store, gap_sec=0.0):
    """File 1 at t0 (100 s), file 2 at t0+100+gap (100 s)."""
    f2_start = datetime.fromtimestamp(_T0 + 100.0 + gap_sec).strftime(
        "%Y_%m_%d__%H_%M_%S")
    with store.connection() as conn:
        conn.execute("INSERT INTO processed_files (id, file_path, session_dir, "
                     "chunk_datetime, duration_sec) VALUES "
                     "(1,'/sess/1.mat','/sess','2026_01_01__00_00_00',100.0)")
        conn.execute("INSERT INTO processed_files (id, file_path, session_dir, "
                     "chunk_datetime, duration_sec) VALUES "
                     "(2,'/sess/2.mat','/sess',?,100.0)", (f2_start,))
        conn.commit()


def _fake_get_chunk(monkeypatch, fs=10.0):
    def gc(path):
        val = 1.0 if path.endswith("1.mat") else 2.0
        n = int(100 * fs)
        sig = np.full((n, 1), val)
        return ChunkData(signal=sig, fs=fs, num_channels=1, num_samples=n,
                         duration_sec=100.0, source_path=path)
    monkeypatch.setattr("src.utils.chunk_cache.get_chunk", gc)


def _seizure():
    # Onset 50 s into file 2 -> absolute t0+150.
    return Seizure(animal_id="BCH040", file_id=2, file_path="/sess/2.mat",
                   session_dir="/sess", chunk_datetime="2026_01_01__00_01_40",
                   eo_sec=50.0, bb_sec=None, racine=3, seizure_type="LVF",
                   onset_epoch=_T0 + 150.0)


def test_gather_leadup_straddles_chunks(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "m.db"))
    _seed_two_chunks(store, gap_sec=0.0)
    _fake_get_chunk(monkeypatch)
    # ceiling 120 s -> lead-up [t0+30, t0+150]: 70 s of file 1 + 50 s of file 2.
    sig, fs = traj.gather_leadup_signal(store, _seizure(), ceiling_sec=120.0,
                                        channel_index=0)
    assert fs == 10.0
    assert sig.size == int(120 * 10)               # 1200 samples
    assert np.allclose(sig[:700], 1.0)             # file 1 portion
    assert np.allclose(sig[700:], 2.0)             # file 2 portion


def test_gather_leadup_skips_on_large_gap(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "m.db"))
    _seed_two_chunks(store, gap_sec=600.0)         # 10-min hole between chunks
    _fake_get_chunk(monkeypatch)
    sz = Seizure(animal_id="BCH040", file_id=2, file_path="/sess/2.mat",
                 session_dir="/sess",
                 chunk_datetime=datetime.fromtimestamp(_T0 + 700.0).strftime(
                     "%Y_%m_%d__%H_%M_%S"),
                 eo_sec=50.0, bb_sec=None, racine=3, seizure_type="LVF",
                 onset_epoch=_T0 + 750.0)
    # Lead-up reaches back into file 1, but the 600 s gap > max_gap_sec=120.
    sig, fs = traj.gather_leadup_signal(store, sz, ceiling_sec=740.0,
                                        channel_index=0, max_gap_sec=120.0)
    assert sig is None and fs is None
