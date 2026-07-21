"""Background sidecar warmer + sidecar version compatibility.

Regression cover for the incident where a schema bump made every existing
sidecar "stale": the peri-ictal build then either returned an empty matrix or,
with warm_missing, tried to recompute hundreds of multi-GB recordings inline
and appeared to hang forever on "joining stimuli to seizures".
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import evoked_output as EO  # noqa: E402
from src.utils import sidecar_warm as SW   # noqa: E402


def _write_sidecar(mat: Path, animal: str, version: str, rows=None):
    sp = Path(EO.feature_sidecar_path(str(mat), animal))
    sp.write_text(json.dumps({
        "version": version, "animal": animal, "variant": "evoked",
        "config_sig": None, "source": mat.name,
        "source_mtime": os.path.getmtime(mat),
        "columns": [], "rows": rows if rows is not None else [{"channel": "A"}],
    }), encoding="utf-8")
    return sp


# ----------------------------------------------------- version tolerance --- #

def test_older_schema_sidecar_is_still_readable(tmp_path):
    """A bump that only ADDS columns must not invalidate existing sidecars."""
    mat = tmp_path / "rec_evoked.mat"
    mat.write_bytes(b"x")
    _write_sidecar(mat, "BCH111", "2")
    rows = EO.read_feature_sidecar(str(mat), "BCH111")
    assert rows, "an older-but-compatible sidecar must still read"


def test_unknown_schema_version_is_rejected(tmp_path):
    mat = tmp_path / "rec_evoked.mat"
    mat.write_bytes(b"x")
    _write_sidecar(mat, "BCH111", "999")
    assert EO.read_feature_sidecar(str(mat), "BCH111") is None


def test_sidecar_is_current_is_stricter_than_readable(tmp_path):
    """Readable (v2) but NOT current -- that's what the warmer upgrades."""
    mat = tmp_path / "rec_evoked.mat"
    mat.write_bytes(b"x")
    _write_sidecar(mat, "BCH111", "2")
    assert EO.read_feature_sidecar(str(mat), "BCH111")
    assert EO.sidecar_is_current(str(mat), "BCH111") is False
    _write_sidecar(mat, "BCH111", EO._FEATURE_SIDECAR_VERSION)
    assert EO.sidecar_is_current(str(mat), "BCH111") is True


def test_sidecar_is_current_false_when_source_changed(tmp_path):
    mat = tmp_path / "rec_evoked.mat"
    mat.write_bytes(b"x")
    _write_sidecar(mat, "BCH111", EO._FEATURE_SIDECAR_VERSION)
    os.utime(mat, (0, 0))                     # source changed under us
    assert EO.sidecar_is_current(str(mat), "BCH111") is False


def test_sidecar_is_current_false_when_absent(tmp_path):
    mat = tmp_path / "nope_evoked.mat"
    mat.write_bytes(b"x")
    assert EO.sidecar_is_current(str(mat), "BCH111") is False


# ------------------------------------------------------------- settings --- #

def _cfg(**over):
    w = {"enabled": True, "interval_sec": 300, "batch": 4,
         "upgrade_outdated": False}
    w.update(over)
    return {"chronic_evoked": {"evoked_output_dir": "", "sidecar_warm": w}}


def test_settings_defaults_and_clamps():
    assert SW._settings(_cfg())[:3] == (True, 300.0, 4)
    assert SW._settings(_cfg(interval_sec=1))[1] == SW._MIN_INTERVAL_SEC
    assert SW._settings(_cfg(batch=0))[2] == 1
    assert SW._settings(_cfg(interval_sec="junk"))[1] == SW.DEFAULT_INTERVAL_SEC
    assert SW._settings(_cfg(enabled=False))[0] is False
    assert SW._settings({})[0] is True          # enabled by default


def test_start_worker_respects_disabled():
    assert SW.start_worker(object(), _cfg(enabled=False)) is False


def test_find_pending_empty_without_evoked_dir():
    assert SW.find_pending(_cfg()) == []
    assert SW.find_pending({}) == []


def test_warm_batch_never_raises(monkeypatch):
    monkeypatch.setattr(SW, "find_pending",
                        lambda *a, **k: [("bad.mat", "BCH111")])
    monkeypatch.setattr(EO, "read_or_compute_sidecar",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    out = SW.warm_batch(_cfg(), batch=1)
    assert out["failed"] == 1 and out["built"] == 0
