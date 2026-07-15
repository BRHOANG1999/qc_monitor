"""Peri-ictal matrix: the query-time join of evoked stimuli to the seizure
timeline. Covers continuous time-to-next-onset, nearest-UPCOMING assignment,
post-ictal / first-seizure drops, and the retroactive relabel (scoring a new
seizure rewrites preceding stimuli's lead-times). Plus embed's stratified
subsample + PCA default.

Run with: pytest tests/test_periictal_matrix.py -q
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                        # noqa: E402
from src.periictal import embed as pe                 # noqa: E402
from src.periictal.matrix import build_matrix         # noqa: E402
from src.utils.evoked_output import write_feature_sidecar  # noqa: E402

_ANIMAL = "BCH040"


def _epoch(iso: str, val: float = 1.0) -> dict:
    """A minimal sidecar row at absolute time *iso* with a couple finite
    metrics (the matrix keeps NaNs; two metrics keep embed happy)."""
    return {"channel": f"{_ANIMAL}SR", "electrode": "SR", "session": "sess",
            "abs_dt": iso, "stim_time_sec": 0.0, "peak": None, "trough": None,
            "line_length": val, "rms_amplitude": val * 2.0}


def _seed_seizure(store, fid: int, chunk_dt: str, eo: float, rac: int = 3):
    with store.connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed_files (id, file_path, session_dir, "
            "chunk_datetime) VALUES (?,?,?,?)",
            (fid, f"/sess/{fid}.mat", "/sess", chunk_dt))
        conn.execute(
            "INSERT INTO review_state (file_id, user_email, status, animal_id, "
            "markers_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (fid, "p@x", "pi_approved", _ANIMAL,
             json.dumps([{"EO_sec": eo, "racine": rac, "type": "LVF"}]),
             "2026-01-01", "2026-01-01"))
        conn.commit()


def _seed_sidecar(evoked_dir: str, rows: list) -> None:
    os.makedirs(evoked_dir, exist_ok=True)
    mat = os.path.join(
        evoked_dir, f"sess__stimCopy_{_ANIMAL}SR___2026_03_02__00_00_00_evoked.mat")
    with open(mat, "wb") as f:
        f.write(b"\x00")            # dummy .mat so the sidecar mtime-stamp matches
    write_feature_sidecar(mat, _ANIMAL, rows)


def _build(tmp_path, seizures, rows, **kw):
    store = Store(str(tmp_path / "data" / "m.db"))
    for fid, cd, eo in seizures:
        _seed_seizure(store, fid, cd, eo)
    evoked_dir = str(tmp_path / "evoked")
    _seed_sidecar(evoked_dir, rows)
    return build_matrix(store, _ANIMAL, evoked_dir, attach_fingerprint=False,
                        window_sec=6 * 3600.0, **kw)


def test_time_to_onset_and_first_seizure_and_postictal_drops(tmp_path):
    # Seizure A onset 03-01 01:00, seizure B onset 03-02 01:00 (ISI 24h).
    seizures = [(1, "2026_03_01__00_00_00", 3600.0),
                (2, "2026_03_02__00_00_00", 3600.0)]
    rows = [
        _epoch("2026-03-02T00:59:00"),   # 60 s before B  -> pre, tto=60
        _epoch("2026-03-02T00:00:00"),   # 3600 s before B -> pre
        _epoch("2026-03-01T00:30:00"),   # before FIRST seizure A -> dropped
        _epoch("2026-03-01T01:10:00"),   # 10 min after A -> now a phase=post row
    ]
    df = _build(tmp_path, seizures, rows)
    pre = df[df["phase"] == "pre"]
    assert len(pre) == 2                          # only the two valid lead-ups
    ttos = sorted(pre["time_to_onset_sec"].round().tolist())
    assert ttos == [60.0, 3600.0]
    assert set(pre["seizure_idx"]) == {1}         # both precede seizure B (idx 1)
    assert (pre["time_to_onset_sec"] > 0).all()
    # the post-ictal stimulus (10 min after A) is now a phase="post" row of A.
    post = df[df["phase"] == "post"]
    assert len(post) == 1
    assert post["seizure_idx"].iloc[0] == 0                    # after seizure A
    assert round(post["time_to_onset_sec"].iloc[0]) == -600.0  # negative = past


def test_nearest_upcoming_not_nearest_absolute(tmp_path):
    # A stimulus 10 min after A and 50 min before B must bind to B (upcoming) for
    # its pre row, never to the closer past seizure A.
    seizures = [(1, "2026_03_02__00_00_00", 0.0),      # A onset 03-02 00:00
                (2, "2026_03_02__01_00_00", 0.0)]      # B onset 03-02 01:00
    rows = [_epoch("2026-03-02T00:10:00")]             # +10m from A, -50m from B
    df = _build(tmp_path, seizures, rows,
                post_ictal_buffer_sec=60.0)            # small buffer so it's kept
    pre = df[df["phase"] == "pre"]
    assert len(pre) == 1
    assert round(pre["time_to_onset_sec"].iloc[0]) == 3000.0   # 50 min to B
    assert pre["seizure_idx"].iloc[0] == 1
    # the same stimulus is ALSO a post row of A (10 min after) — overlap is fine.
    post = df[df["phase"] == "post"]
    assert len(post) == 1 and post["seizure_idx"].iloc[0] == 0
    assert round(post["time_to_onset_sec"].iloc[0]) == -600.0


def test_symmetric_post_onset_window(tmp_path):
    # Symmetric peri-onset window: stimuli after an onset (within window_sec) are
    # kept as phase="post" with NEGATIVE tto, tied to the PRECEDING seizure; a
    # stimulus beyond the window is dropped.
    seizures = [(1, "2026_03_01__00_00_00", 3600.0),   # A onset 03-01 01:00
                (2, "2026_03_02__00_00_00", 3600.0)]   # B onset 03-02 01:00
    rows = [
        _epoch("2026-03-01T01:30:00"),   # 30 min after A -> post of A (in 6 h win)
        _epoch("2026-03-02T02:00:00"),   # 60 min after B -> post of B
        _epoch("2026-03-01T08:00:00"),   # 7 h after A -> beyond window -> dropped
    ]
    df = _build(tmp_path, seizures, rows)
    post = df[df["phase"] == "post"].sort_values("time_to_onset_sec")
    assert len(post) == 2
    assert (post["time_to_onset_sec"] < 0).all()
    assert sorted(post["seizure_idx"].tolist()) == [0, 1]      # A and B each once
    assert "phase" in df.columns and set(df["phase"]) <= {"pre", "post"}


def test_retroactive_relabel_when_new_seizure_scored(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    _seed_seizure(store, 1, "2026_03_01__00_00_00", 3600.0)   # A onset 03-01 01:00
    _seed_seizure(store, 2, "2026_03_02__00_00_00", 3600.0)   # B onset 03-02 01:00
    evoked_dir = str(tmp_path / "evoked")
    _seed_sidecar(evoked_dir, [_epoch("2026-03-02T00:59:00")])  # 60 s before B
    df1 = build_matrix(store, _ANIMAL, evoked_dir, attach_fingerprint=False,
                       window_sec=6 * 3600.0)
    assert round(df1["time_to_onset_sec"].iloc[0]) == 60.0
    # Score a NEW seizure 30 s after that stimulus -> its lead-time must change.
    _seed_seizure(store, 3, "2026_03_02__00_59_30", 0.0)       # onset 00:59:30
    df2 = build_matrix(store, _ANIMAL, evoked_dir, attach_fingerprint=False,
                       window_sec=6 * 3600.0)
    assert round(df2["time_to_onset_sec"].iloc[0]) == 30.0     # relabelled


# ------------------------------------------------------------------ #
#  embed: PCA default + stratified subsample balances seizure x lead_bin
# ------------------------------------------------------------------ #

def test_embed_pca_default_and_shape():
    import pandas as pd
    n = 300
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "seizure_idx": rng.integers(0, 3, n),
        "lead_bin": rng.integers(0, 4, n),
        "hour_of_day": rng.uniform(0, 24, n),
        "stim_key": "2nC",
        "line_length": rng.normal(size=n),
        "rms_amplitude": rng.normal(size=n),
    })
    res = pe.embed(df, metrics=["line_length", "rms_amplitude"], method="pca")
    assert res["method"] == "pca"
    assert res["emb"].shape == (n, 2)
    assert res["rows"].shape[0] == n


def test_embed_stratified_subsample_balances_cells():
    import pandas as pd
    # One dominant (seizure 0, bin 0) cell of 1000 vs many small cells.
    big = pd.DataFrame({"seizure_idx": 0, "lead_bin": 0,
                        "line_length": np.zeros(1000)})
    small = pd.DataFrame({"seizure_idx": np.repeat([1, 2], 10),
                          "lead_bin": np.tile([1, 2], 10),
                          "line_length": np.ones(20)})
    df = pd.concat([big, small], ignore_index=True)
    rows = pe._stratified_subsample(df, cap=60, seed=0)
    sub = df.loc[rows]
    # The dominant cell must NOT swamp: its share is bounded near the per-cell cap.
    dom = ((sub["seizure_idx"] == 0) & (sub["lead_bin"] == 0)).sum()
    assert dom <= 30                       # not the ~1000 it would be uniformly
    assert len(rows) <= 60


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
