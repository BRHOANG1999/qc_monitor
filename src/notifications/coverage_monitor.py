"""Coverage monitor -- surface recordings that fell through the cracks.

Two axes:
  * EVOKED-SIDECAR coverage: for every ``*_evoked.mat`` in the toolkit's
    evokedOutput dir, per animal, is a feature sidecar present? This is the gap
    that silently starved the evoked-metric figures -- 5 of 7 animals sat at 0%
    and nobody knew until asked.
  * processed_files PIPELINE gaps: stuck_pending / stale_processing / errored /
    missing_duration (currently zero, but a stalled ingest would show here).

Trust hinges on a near-zero false-positive rate, so the digest reports only:
  - gaps NOT acknowledged (store.coverage_ack, scoped to (path, gap_class)) and
    NOT allow-listed (config, whole-class/animal), and
  - evoked recordings that are STALE -- older than ``stale_hours``. A recording
    that arrived this morning is a healthy backlog, not a failure; only one
    that's been sitting unanalyzed for days is a real gap. (Work-gated, like the
    dead-man's-switch: don't cry wolf on fresh, expected silence.)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from src.utils.evoked_output import (DEFAULT_EVOKED_DIR, animals_in_filename,
                                     feature_sidecar_path, list_evoked_files,
                                     parse_recording_dt)

logger = logging.getLogger("qc_monitor.notifications.coverage_monitor")

EVOKED_GAP = "evoked_sidecar_missing"
HEARTBEAT_JOB = "coverage"


def _cfg(config: dict) -> dict:
    return (((config or {}).get("notifications", {}) or {})
            .get("coverage_digest", {}) or {})


def _allowlist(config: dict) -> list[str]:
    return [str(x).lower() for x in (_cfg(config).get("allowlist") or [])]


def _allowed(text: str, allow: list[str]) -> bool:
    t = (text or "").lower()
    return any(a in t for a in allow)


def _evoked_dir(config: dict) -> str:
    ce = (config or {}).get("chronic_evoked", {}) or {}
    return ce.get("evoked_output_dir") or DEFAULT_EVOKED_DIR


def evoked_sidecar_coverage(config: dict, now: datetime | None = None) -> dict:
    """Per-animal evoked feature-sidecar coverage from the evokedOutput dir:
    ``{animal: {total, have, missing, stale_missing, newest_stale_iso}}``.
    Allow-listed animals are skipped. *stale_missing* counts recordings older
    than ``stale_hours`` (default 48) that still have no sidecar."""
    now = now or datetime.now()
    stale_before = now - timedelta(
        hours=float(_cfg(config).get("stale_hours", 48)))
    allow = _allowlist(config)
    files = list_evoked_files(_evoked_dir(config))
    per: dict = {}
    for i, fp in enumerate(files):
        assert i < 1_000_000, "evoked coverage scan runaway"
        for a in animals_in_filename(fp):
            if _allowed(a, allow):
                continue
            slot = per.setdefault(a, {"total": 0, "have": 0, "missing": 0,
                                      "stale_missing": 0,
                                      "newest_stale_iso": ""})
            slot["total"] += 1
            if os.path.exists(feature_sidecar_path(fp, a)):
                slot["have"] += 1
                continue
            slot["missing"] += 1
            dt = parse_recording_dt(fp)
            if dt is not None and dt < stale_before:      # persistently missing
                slot["stale_missing"] += 1
                if dt.isoformat() > slot["newest_stale_iso"]:
                    slot["newest_stale_iso"] = dt.isoformat()
    return per


def evaluate(store, config: dict, now: datetime | None = None) -> dict:
    """Full coverage picture with acks + allowlist + stale-gating applied.
    Records this scan's heartbeat (tick before, success after) so the scan's own
    liveness is queryable and the dead-man's-switch never reports from inside
    the run it measures."""
    store.heartbeat_tick(HEARTBEAT_JOB)
    acked = store.coverage_acked_set()
    allow = _allowlist(config)

    raw = store.coverage_pipeline_gaps(cap=1000)
    pipeline: dict = {}
    pipeline_total = 0
    for cls, d in raw.items():
        unacked = [p for p in d["paths"]
                   if (p, cls) not in acked and not _allowed(p, allow)]
        # d["paths"] is capped; if the class exceeds the cap we can't ack-filter
        # the tail, so report the raw count and note the (rare) truncation.
        n = len(unacked) if d["count"] <= len(d["paths"]) else d["count"]
        pipeline[cls] = {"count": n, "sample": unacked[:8],
                         "truncated": d["count"] > len(d["paths"])}
        pipeline_total += n

    evoked = evoked_sidecar_coverage(config, now=now)
    evoked_stale = sum(v["stale_missing"] for v in evoked.values())

    result = {
        "pipeline": pipeline,
        "pipeline_total": pipeline_total,
        "evoked": evoked,
        "evoked_stale": evoked_stale,
        "evoked_missing": sum(v["missing"] for v in evoked.values()),
        "n_acked": len(acked),
        "healthy": pipeline_total == 0 and evoked_stale == 0,
    }
    store.heartbeat_success(
        HEARTBEAT_JOB,
        note=f"pipe={pipeline_total} evoked_stale={evoked_stale}")
    return result


# --------------------------------------------------------------------------- #
#  Digest
# --------------------------------------------------------------------------- #

def _lines(res: dict) -> list[str]:
    ls: list[str] = []
    if res["healthy"]:
        ls.append("Coverage OK — no unacknowledged pipeline gaps and no "
                  "persistently-unanalyzed evoked recordings.")
    pt = res["pipeline_total"]
    if pt:
        ls.append(f"Pipeline gaps ({pt} file(s)):")
        for cls, d in res["pipeline"].items():
            if d["count"]:
                more = " (+more, truncated)" if d.get("truncated") else ""
                ls.append(f"  {cls}: {d['count']}{more}")
                for p in d["sample"]:
                    ls.append(f"      {p}")
    if res["evoked_stale"]:
        ls.append(f"Evoked recordings unanalyzed >stale window "
                  f"({res['evoked_stale']} total):")
        for a, v in sorted(res["evoked"].items()):
            if v["stale_missing"]:
                ls.append(f"  {a}: {v['stale_missing']} stale of "
                          f"{v['missing']} missing / {v['total']} recordings "
                          f"(newest stale {v['newest_stale_iso'] or '?'})")
    if res["n_acked"]:
        ls.append(f"({res['n_acked']} gap(s) acknowledged and hidden.)")
    return ls


def send_coverage_digest(today, config: dict, store, emailer) -> dict:
    """Build + send the coverage digest. Sends ONLY when there's something to
    report (unacked pipeline gaps or stale-unanalyzed evoked recordings), so a
    healthy day is silent and the digest stays trusted. Returns a status dict."""
    cfg = _cfg(config)
    if not cfg.get("enabled", False):
        return {"sent": False, "reason": "disabled"}
    if store is None:
        return {"sent": False, "reason": "no store"}
    recipients = list(
        cfg.get("recipients")
        or ((config.get("alerting", {}) or {}).get("smtp", {})
            .get("recipients", []) or []))
    if not recipients:
        return {"sent": False, "reason": "no recipients"}

    res = evaluate(store, config)
    if res["healthy"]:
        return {"sent": False, "reason": "healthy", **_counts(res)}

    lines = _lines(res)
    text = "\n".join(lines)
    html = "<h3>Coverage monitor</h3><pre>" + text + "</pre>"
    subject = (f"Coverage: {res['pipeline_total']} pipeline gap(s), "
               f"{res['evoked_stale']} stale evoked "
               f"({today.isoformat() if hasattr(today, 'isoformat') else today})")
    try:
        ok = emailer.send(subject=subject, body=text, body_html=html,
                          recipients=recipients, subject_prefix=False,
                          severity="warning")
    except Exception as e:  # noqa: BLE001
        logger.error("coverage digest send failed: %s", e)
        return {"sent": False, "reason": f"send error: {e}"}
    return {"sent": bool(ok), "recipients": len(recipients), **_counts(res)}


def _counts(res: dict) -> dict:
    return {"pipeline_total": res["pipeline_total"],
            "evoked_stale": res["evoked_stale"],
            "evoked_missing": res["evoked_missing"]}
