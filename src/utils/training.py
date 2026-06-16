"""Training-tab grading: score a student's blind re-score against the
PI-validated answer.

Pure functions only -- no Dash, no DB -- so the agreement logic is fully
unit-testable (the same pattern used for ``apply_event_types`` and
``_next_in_pool``). The Training tab and the store handle the UI and
persistence; everything pedagogically load-bearing lives here.

Validated and student events share the ``blank_event()`` shape from
``src/dashboard/tabs/video_events.py`` -- landmark times are in SECONDS
(``EO_sec`` ...), which is the natural comparison unit (no CSV / sample-
index conversion).

Curriculum stages:
  1 Detection      -- is there a BHZ? (events present yes/no)
  2 Type + Racine  -- per event: EEG onset type (LVF/HYP) + Racine,
                      located by EO onset
  3 Full           -- per event: type + Racine + all five landmarks
                      (EO/LAS/BO/PID/BB)
"""

from __future__ import annotations

from typing import Any

# The five onset landmarks, in fill order. LAS is LVF-only; a validated
# event that lacks a landmark (None) is simply skipped from that event's
# denominator, so HYP events aren't penalised for having no LAS.
LANDMARKS: tuple[str, ...] = ("EO", "LAS", "BO", "PID", "BB")

STAGES: dict[int, dict[str, Any]] = {
    1: {"label": "Detection", "scored": "Is there a BHZ? (yes / no)"},
    2: {"label": "Type + Racine",
        "scored": "Per event: onset type (LVF/HYP) + Racine, by EO onset"},
    3: {"label": "Full scoring",
        "scored": "Per event: type + Racine + all five landmarks"},
}


def _real_events(events: Any) -> list[dict]:
    """Keep only event dicts that carry a usable EO onset or a type --
    drops blank rows so neither side is graded against scaffolding."""
    out: list[dict] = []
    for e in (events or []):
        if not isinstance(e, dict):
            continue
        if e.get("EO_sec") is not None or (e.get("type") or ""):
            out.append(e)
    return out


def match_events(validated: list[dict], student: list[dict],
                  onset_tol_s: float) -> tuple[list[tuple[int, int]], set]:
    """Greedily pair student events to validated events by EO onset.

    For each validated event (in list order) take the nearest unused
    student event whose EO is within *onset_tol_s* seconds. Returns
    ``(pairs, used_student_idx)`` where *pairs* is ``[(v_idx, s_idx)]``.
    Events with no numeric EO can't be matched.
    """
    assert onset_tol_s >= 0, "onset_tol_s must be >= 0"
    assert isinstance(validated, list) and isinstance(student, list), \
        "validated/student must be lists"
    pairs: list[tuple[int, int]] = []
    used: set = set()
    for vi, v in enumerate(validated):
        v_eo = v.get("EO_sec")
        if v_eo is None:
            continue
        best_si, best_d = None, None
        for si, s in enumerate(student):
            if si in used:
                continue
            s_eo = s.get("EO_sec")
            if s_eo is None:
                continue
            d = abs(float(s_eo) - float(v_eo))
            if d <= onset_tol_s and (best_d is None or d < best_d):
                best_si, best_d = si, d
        if best_si is not None:
            used.add(best_si)
            pairs.append((vi, best_si))
    return pairs, used


def _racine_agrees(v_racine: Any, s_racine: Any, racine_tol: int) -> bool:
    if v_racine is None and s_racine is None:
        return True
    if v_racine is None or s_racine is None:
        return False
    try:
        return abs(int(v_racine) - int(s_racine)) <= racine_tol
    except (TypeError, ValueError):
        return False


def _pair_credit(v: dict, s: dict, stage: int,
                  onset_tol_s: float, racine_tol: int) -> float:
    """Fraction of a matched pair's components the student got right."""
    comps: list[float] = [
        1.0 if (v.get("type") or "") == (s.get("type") or "") else 0.0,
        1.0 if _racine_agrees(v.get("racine"), s.get("racine"),
                               racine_tol) else 0.0,
    ]
    if stage == 3:
        for lm in LANDMARKS:
            v_t = v.get(f"{lm}_sec")
            if v_t is None:
                continue  # validated has no such landmark -> not scored
            s_t = s.get(f"{lm}_sec")
            ok = (s_t is not None
                  and abs(float(s_t) - float(v_t)) <= onset_tol_s)
            comps.append(1.0 if ok else 0.0)
    return sum(comps) / len(comps) if comps else 0.0


def grade_attempt(stage: int, validated_events: Any, student_answer: Any,
                   *, onset_tol_s: float = 10.0,
                   racine_tol: int = 1) -> dict:
    """Score a student's submission against the validated answer.

    *student_answer* is ``{"events_present": bool, "events": [..]}``.
    Returns ``{"score": 0..1, "breakdown": {...}}``. Stage 1 is a binary
    detection match. Stages 2-3 match events by EO onset and average each
    matched pair's components, dividing total credit by
    ``len(validated) + false_positives`` so misses AND false alarms cost;
    empty-vs-empty scores 1.0.
    """
    assert stage in (1, 2, 3), "stage must be 1, 2 or 3"
    assert isinstance(student_answer, dict), "student_answer must be a dict"

    validated = _real_events(validated_events)
    validated_has = len(validated) > 0

    if stage == 1:
        student_has = bool(student_answer.get("events_present"))
        return {
            "score": 1.0 if student_has == validated_has else 0.0,
            "breakdown": {"validated_has": validated_has,
                          "student_has": student_has},
        }

    student = _real_events(student_answer.get("events"))
    # An explicit "no events" with none added is a clean no-events call.
    if not validated and not student:
        return {"score": 1.0,
                "breakdown": {"matched": 0, "missed": 0,
                              "false_pos": 0, "validated": 0}}

    pairs, used = match_events(validated, student, onset_tol_s)
    total_credit = sum(
        _pair_credit(validated[vi], student[si], stage,
                     onset_tol_s, racine_tol)
        for vi, si in pairs)
    false_pos = len(student) - len(used)
    denom = len(validated) + false_pos
    score = (total_credit / denom) if denom > 0 else 1.0
    return {
        "score": max(0.0, min(1.0, score)),
        "breakdown": {
            "matched": len(pairs),
            "missed": len(validated) - len(pairs),
            "false_pos": false_pos,
            "validated": len(validated),
            "credit": round(total_credit, 3),
        },
    }


def rolling_ready(scores: list, pass_pct: float, window: int) -> bool:
    """True when the student is ready to advance: at least *window*
    graded attempts AND the mean of the most recent *window* scores
    (each 0..1) is >= ``pass_pct/100``. *scores* is chronological."""
    assert window >= 1, "window must be >= 1"
    assert 0 <= pass_pct <= 100, "pass_pct must be a percentage"
    vals = [float(s) for s in (scores or [])]
    if len(vals) < window:
        return False
    recent = vals[-window:]
    return (sum(recent) / len(recent)) >= (pass_pct / 100.0)
