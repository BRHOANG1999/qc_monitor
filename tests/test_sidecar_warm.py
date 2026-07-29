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

def test_listed_compatible_version_is_readable(tmp_path, monkeypatch):
    """The anti-cliff mechanism: a version listed as compatible must READ.

    An ADDITIVE bump belongs in that set -- readers pull columns by name, so an
    older sidecar is good data minus the new columns. Asserted through the set
    rather than a hardcoded "2", because which versions are additive changes.
    """
    monkeypatch.setattr(EO, "_COMPATIBLE_SIDECAR_VERSIONS",
                        set(EO._COMPATIBLE_SIDECAR_VERSIONS) | {"3"})
    mat = tmp_path / "rec_evoked.mat"
    mat.write_bytes(b"x")
    _write_sidecar(mat, "BCH111", "3")
    assert EO.read_feature_sidecar(str(mat), "BCH111")


def test_non_additive_bump_rejects_older_sidecars(tmp_path):
    """v4 computes every feature on a 5-trial sliding average, so pre-v4 rows
    measure a different quantity. Mixing them would put two noise regimes in one
    matrix, split by WHEN each file was warmed -- so they must not read."""
    mat = tmp_path / "rec_evoked.mat"
    mat.write_bytes(b"x")
    for stale in ("2", "3"):
        _write_sidecar(mat, "BCH111", stale)
        assert EO.read_feature_sidecar(str(mat), "BCH111") is None, stale


def test_unknown_schema_version_is_rejected(tmp_path):
    mat = tmp_path / "rec_evoked.mat"
    mat.write_bytes(b"x")
    _write_sidecar(mat, "BCH111", "999")
    assert EO.read_feature_sidecar(str(mat), "BCH111") is None


def test_sidecar_is_current_is_stricter_than_readable(tmp_path, monkeypatch):
    """Readable but NOT current -- that is what the warmer upgrades."""
    monkeypatch.setattr(EO, "_COMPATIBLE_SIDECAR_VERSIONS",
                        set(EO._COMPATIBLE_SIDECAR_VERSIONS) | {"3"})
    mat = tmp_path / "rec_evoked.mat"
    mat.write_bytes(b"x")
    _write_sidecar(mat, "BCH111", "3")
    assert EO.read_feature_sidecar(str(mat), "BCH111")
    assert EO.sidecar_is_current(str(mat), "BCH111") is False
    _write_sidecar(mat, "BCH111", EO._FEATURE_SIDECAR_VERSION)
    assert EO.sidecar_is_current(str(mat), "BCH111") is True


def test_wavelet_salvage_refuses_a_stale_version(tmp_path):
    """The incremental upgrade must not splice v2/v3 wavelet columns (computed
    per single epoch) into an otherwise trial-averaged v4 row."""
    mat = tmp_path / "rec_evoked.mat"
    mat.write_bytes(b"x")
    _write_sidecar(mat, "BCH111", "3")
    assert EO._read_raw_sidecar_rows(str(mat), "BCH111") is None
    _write_sidecar(mat, "BCH111", EO._FEATURE_SIDECAR_VERSION)
    assert EO._read_raw_sidecar_rows(str(mat), "BCH111")


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


# ------------------------------------------------ find_pending re-warm --- #

def _evoked_dir_cfg(d):
    return {"chronic_evoked": {"evoked_output_dir": str(d)}}


def test_find_pending_rebuilds_rejected_version_without_the_flag(tmp_path):
    """The bug: after a NON-additive bump the old sidecar still EXISTS, so
    find_pending (upgrade_outdated=False) skipped it and the corpus never
    re-warmed -- every build stayed empty. A build-REJECTED sidecar must be
    pending regardless of the flag."""
    mat = tmp_path / "sess__stimCopy_BCH111SR___2026_03_02__00_00_00_evoked.mat"
    mat.write_bytes(b"x")
    _write_sidecar(mat, "BCH111", "3")                 # old, v5 rejects it
    pend = SW.find_pending(_evoked_dir_cfg(tmp_path), limit=10,
                           upgrade_outdated=False)
    assert (str(mat), "BCH111") in pend


def test_find_pending_skips_a_current_sidecar(tmp_path):
    mat = tmp_path / "sess__stimCopy_BCH111SR___2026_03_02__00_00_00_evoked.mat"
    mat.write_bytes(b"x")
    _write_sidecar(mat, "BCH111", EO._FEATURE_SIDECAR_VERSION)   # v5, mtime-fresh
    pend = SW.find_pending(_evoked_dir_cfg(tmp_path), limit=10)
    assert (str(mat), "BCH111") not in pend


def test_find_pending_reports_a_missing_sidecar(tmp_path):
    mat = tmp_path / "sess__stimCopy_BCH111SR___2026_03_02__00_00_00_evoked.mat"
    mat.write_bytes(b"x")                               # no sidecar written
    pend = SW.find_pending(_evoked_dir_cfg(tmp_path), limit=10)
    assert (str(mat), "BCH111") in pend


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


# ------------------------------------------------ locality-aware cadence --- #

def _cadence_cfg(evoked_dir, **warm):
    return {"chronic_evoked": {"evoked_output_dir": evoked_dir,
                               "sidecar_warm": {"enabled": True, **warm}}}


def test_local_dir_gets_the_brisk_default_cadence(monkeypatch):
    """A local fixed disk drops the multi-minute share-recovery idle."""
    monkeypatch.setattr(SW, "_path_is_local", lambda p: True)
    _e, interval, batch, _u = SW._settings(_cadence_cfg("D:/x/evokedOutput"))
    assert (interval, batch) == (SW.LOCAL_INTERVAL_SEC, SW.LOCAL_BATCH)
    assert SW._default_cadence(_cadence_cfg("D:/x"))[0] == "local"


def test_remote_dir_keeps_the_gentle_default_cadence(monkeypatch):
    monkeypatch.setattr(SW, "_path_is_local", lambda p: False)
    _e, interval, batch, _u = SW._settings(
        _cadence_cfg("//100.106.104.22/db/evokedOutput"))
    assert (interval, batch) == (SW.DEFAULT_INTERVAL_SEC, SW.DEFAULT_BATCH)
    assert SW._default_cadence(_cadence_cfg("//srv/s"))[0] == "remote"


def test_explicit_cadence_overrides_locality(monkeypatch):
    """A pinned interval/batch wins over the auto choice, even on a local dir."""
    monkeypatch.setattr(SW, "_path_is_local", lambda p: True)
    _e, interval, batch, _u = SW._settings(
        _cadence_cfg("D:/x", interval_sec=300, batch=4))
    assert (interval, batch) == (300.0, 4)


def test_path_is_local_rejects_unc_and_empty():
    assert SW._path_is_local("//100.106.104.22/database/evokedOutput") is False
    assert SW._path_is_local(r"\\100.106.104.22\database") is False
    assert SW._path_is_local("") is False
    assert SW._path_is_local(None) is False


def test_path_is_local_accepts_a_fixed_drive():
    """The project's own drive is a local fixed disk on the Windows lab box."""
    if sys.platform != "win32":
        return
    assert SW._path_is_local(os.path.abspath(".")) is True


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
