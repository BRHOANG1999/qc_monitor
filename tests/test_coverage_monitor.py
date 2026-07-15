"""Coverage monitor: gap classes + ack/allowlist + stale-gating + the
non-circular heartbeat + the healthy-day-silent digest.

Run with: pytest tests/test_coverage_monitor.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                              # noqa: E402
from src.utils.evoked_output import write_feature_sidecar    # noqa: E402
from src.notifications import coverage_monitor as cm         # noqa: E402

_NOW = datetime(2026, 7, 15, 12, 0, 0)


def _mat(evoked_dir, animal, dt, with_sidecar):
    name = f"sess__stimCopy_{animal}SR_{dt:%Y_%m_%d__%H_%M_%S}_evoked.mat"
    p = os.path.join(evoked_dir, name)
    with open(p, "wb") as f:
        f.write(b"\x00")
    if with_sidecar:
        write_feature_sidecar(p, animal, [{"channel": f"{animal}SR",
                                           "abs_dt": dt.isoformat(),
                                           "line_length": 1.0}])
    return p


def _config(evoked_dir, **cov):
    c = {"chronic_evoked": {"evoked_output_dir": str(evoked_dir)},
         "notifications": {"coverage_digest": {"enabled": True,
                                               "stale_hours": 48,
                                               "recipients": ["pi@lab"],
                                               **cov}}}
    return c


# ------------------------------------------------------------------ #
#  store: acks + pipeline gaps + heartbeat
# ------------------------------------------------------------------ #

def test_coverage_ack_roundtrip(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    store.coverage_ack("/f/x.mat", "errored", reason="known dead file",
                       acked_by="pi@lab")
    assert ("/f/x.mat", "errored") in store.coverage_acked_set()
    store.coverage_unack("/f/x.mat", "errored")
    assert ("/f/x.mat", "errored") not in store.coverage_acked_set()


def test_pipeline_gaps_counts(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    with store.connection() as conn:
        conn.execute("INSERT INTO processed_files (id, file_path, status) "
                     "VALUES (1,'/a.mat','pending')")
        conn.execute("INSERT INTO processed_files (id, file_path, status) "
                     "VALUES (2,'/b.mat','error')")
        conn.execute("INSERT INTO processed_files (id, file_path, status, "
                     "duration_sec) VALUES (3,'/c.mat','done',NULL)")
        conn.commit()
    g = store.coverage_pipeline_gaps()
    assert g["stuck_pending"]["count"] == 1 and g["stuck_pending"]["paths"] == ["/a.mat"]
    assert g["errored"]["count"] == 1
    assert g["missing_duration"]["count"] == 1


def test_heartbeat_tick_then_success(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    store.heartbeat_tick("coverage")
    hb = store.heartbeat_get("coverage")
    assert hb["tick_fired_at"] and hb["last_success_at"] is None
    store.heartbeat_success("coverage", note="ok")
    hb2 = store.heartbeat_get("coverage")
    assert hb2["last_success_at"] and hb2["tick_fired_at"] == hb["tick_fired_at"]


# ------------------------------------------------------------------ #
#  evoked-sidecar coverage + stale gating
# ------------------------------------------------------------------ #

def test_evoked_coverage_stale_gating(tmp_path):
    ev = tmp_path / "evoked"; ev.mkdir()
    fresh = _NOW - timedelta(hours=1)  # healthy backlog
    # distinct datetimes so the filenames don't collide
    _mat(str(ev), "BCH062", _NOW - timedelta(days=5), with_sidecar=True)   # analyzed
    _mat(str(ev), "BCH062", _NOW - timedelta(days=6), with_sidecar=False)  # STALE missing
    _mat(str(ev), "BCH062", fresh, with_sidecar=False)   # fresh missing (ok)
    _mat(str(ev), "BCH060", _NOW - timedelta(days=5), with_sidecar=False)  # stale, other

    cov = cm.evoked_sidecar_coverage(_config(ev), now=_NOW)
    assert cov["BCH062"]["total"] == 3 and cov["BCH062"]["have"] == 1
    assert cov["BCH062"]["missing"] == 2
    assert cov["BCH062"]["stale_missing"] == 1            # only the old one
    assert cov["BCH060"]["stale_missing"] == 1

    # allowlist BCH060 -> it drops out entirely
    cov2 = cm.evoked_sidecar_coverage(_config(ev, allowlist=["BCH060"]), now=_NOW)
    assert "BCH060" not in cov2


# ------------------------------------------------------------------ #
#  evaluate: healthy vs gap, ack applied
# ------------------------------------------------------------------ #

def test_evaluate_healthy_when_only_fresh_missing(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    ev = tmp_path / "evoked"; ev.mkdir()
    _mat(str(ev), "BCH062", _NOW - timedelta(hours=2), with_sidecar=False)  # fresh
    res = cm.evaluate(store, _config(ev), now=_NOW)
    assert res["healthy"] is True
    assert res["evoked_missing"] == 1 and res["evoked_stale"] == 0
    # heartbeat recorded by the scan
    assert store.heartbeat_get("coverage")["last_success_at"]


def test_evaluate_flags_stale_and_ack_clears_pipeline(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    ev = tmp_path / "evoked"; ev.mkdir()
    _mat(str(ev), "BCH062", _NOW - timedelta(days=5), with_sidecar=False)  # stale
    with store.connection() as conn:
        conn.execute("INSERT INTO processed_files (id, file_path, status) "
                     "VALUES (1,'/bad.mat','error')")
        conn.commit()

    res = cm.evaluate(store, _config(ev), now=_NOW)
    assert res["healthy"] is False
    assert res["evoked_stale"] == 1
    assert res["pipeline"]["errored"]["count"] == 1

    # ack the errored file -> it drops out; still unhealthy from the stale evoked
    store.coverage_ack("/bad.mat", "errored")
    res2 = cm.evaluate(store, _config(ev), now=_NOW)
    assert res2["pipeline"]["errored"]["count"] == 0
    assert res2["pipeline_total"] == 0
    assert res2["evoked_stale"] == 1 and res2["healthy"] is False


# ------------------------------------------------------------------ #
#  digest: silent on healthy, sends on gap, guards
# ------------------------------------------------------------------ #

class _Emailer:
    def __init__(self):
        self.calls = []

    def send(self, subject, body, body_html=None, recipients=None,
             subject_prefix=True, severity="info"):
        self.calls.append({"subject": subject, "body": body})
        return True


def test_digest_silent_when_healthy(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    ev = tmp_path / "evoked"; ev.mkdir()
    _mat(str(ev), "BCH062", _NOW - timedelta(hours=1), with_sidecar=True)
    em = _Emailer()
    out = cm.send_coverage_digest(_NOW.date(), _config(ev), store, em)
    assert out["sent"] is False and out["reason"] == "healthy"
    assert em.calls == []


def test_digest_sends_on_stale_gap(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    ev = tmp_path / "evoked"; ev.mkdir()
    _mat(str(ev), "BCH062", _NOW - timedelta(days=6), with_sidecar=False)
    em = _Emailer()
    out = cm.send_coverage_digest(_NOW.date(), _config(ev), store, em)
    assert out["sent"] is True and len(em.calls) == 1
    assert "BCH062" in em.calls[0]["body"]


def test_digest_guards(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    ev = tmp_path / "evoked"; ev.mkdir()
    assert cm.send_coverage_digest(
        _NOW.date(),
        {"notifications": {"coverage_digest": {"enabled": False}}},
        store, _Emailer())["reason"] == "disabled"
    assert cm.send_coverage_digest(
        _NOW.date(),
        {"chronic_evoked": {"evoked_output_dir": str(ev)},
         "notifications": {"coverage_digest": {"enabled": True}}},
        store, _Emailer())["reason"] == "no recipients"
