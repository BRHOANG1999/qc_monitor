"""CONSORT-style trial + seizure accounting for the peri-ictal analysis.

The first thing a committee asks, and the thing that resolves "the UMAP shows 4
seizures but the paired test used 3": make every count explicit --
seizures scored -> dropped (outcome-blind bad-stim, with the drop reason) ->
seizures with BOTH windows available -> per-seizure per-class trial counts (+
session). Exclusion is whole-seizure, so it removes both windows of a dropped
seizure equally and cannot create a within-kept-seizure class asymmetry (a
residual per-class count gap is benign: the further-back interictal window more
often runs off the start of the record). See docs/periictal_registered_analyses.md.

Pure + Dash-free: ``build_ledger`` takes the seizure list + labelled matrix and
returns a plain dict; ``accounting`` is the thin Store-backed wrapper.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.periictal.forecast import CLASS_INTERICTAL, CLASS_PREICTAL

_MAX_SEIZURES = 1_000_000


def _abs_dt(epoch) -> str:
    try:
        return datetime.fromtimestamp(float(epoch)).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        return ""


def _int_or(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def build_ledger(scored: list, excl_keys, key_fn, df_labeled, *,
                 reason: str = "excluded by reviewer (bad-stim)") -> dict:
    """Pure accounting. *scored* = chronological Seizure list; *excl_keys* = set of
    exclusion keys; *key_fn(seizure)->key*; *df_labeled* = the built matrix AFTER
    ``label_classes`` (columns seizure_idx, class, seizure_onset_epoch, session).
    Returns the counts dict. ``included`` preserves ``scored`` order, so its index
    matches the matrix ``seizure_idx``."""
    assert key_fn is not None, "key_fn required"
    excl_keys = set(excl_keys or ())
    assert len(scored) < _MAX_SEIZURES, "seizure count runaway"
    excluded, included = [], []
    for s in scored:
        rec = {"file_id": _int_or(getattr(s, "file_id", None)),
               "onset_epoch": float(getattr(s, "onset_epoch", float("nan"))),
               "abs_dt": _abs_dt(getattr(s, "onset_epoch", None)),
               "racine": _int_or(getattr(s, "racine", None))}
        if key_fn(s) in excl_keys:
            excluded.append({**rec, "reason": reason})
        else:
            included.append(rec)

    per_seizure = _per_seizure_counts(df_labeled, included)
    n_both = sum(1 for p in per_seizure if p["both_windows"])
    return {"n_scored": len(scored), "n_excluded": len(excluded),
            "n_included": len(included), "excluded": excluded,
            "per_seizure": per_seizure,
            "totals": {"n_pre": int(sum(p["n_pre"] for p in per_seizure)),
                       "n_inter": int(sum(p["n_inter"] for p in per_seizure)),
                       "n_seizures_both_windows": n_both},
            "reconciles": len(scored) - len(excluded) == len(included)}


def _per_seizure_counts(df_labeled, included: list) -> list:
    """Per-seizure preictal/interictal trial counts + session, from the labelled
    matrix, joined to the included seizure by matrix ``seizure_idx``."""
    if df_labeled is None or len(df_labeled) == 0 or "class" not in df_labeled:
        return []
    sid = df_labeled["seizure_idx"].to_numpy()
    cls = df_labeled["class"].to_numpy()
    onset = df_labeled["seizure_onset_epoch"].to_numpy(dtype=float)
    sess = (df_labeled["session"].to_numpy() if "session" in df_labeled
            else np.array([""] * len(df_labeled), dtype=object))
    out: list = []
    uniq = np.unique(sid)
    assert uniq.size < _MAX_SEIZURES, "seizure_idx runaway"
    for s in uniq:
        m = sid == s
        n_pre = int(np.sum(m & (cls == CLASS_PREICTAL)))
        n_inter = int(np.sum(m & (cls == CLASS_INTERICTAL)))
        idx = _int_or(s)
        oe = float(onset[m][0]) if m.any() else float("nan")
        out.append({"seizure_idx": idx,
                    "onset_epoch": oe, "abs_dt": _abs_dt(oe),
                    "session": str(sess[m][0]) if m.any() else "",
                    "racine": (included[idx]["racine"]
                               if idx is not None and 0 <= idx < len(included)
                               else None),
                    "n_pre": n_pre, "n_inter": n_inter,
                    "both_windows": bool(n_pre > 0 and n_inter > 0)})
    return out


def accounting(store, animal_id: str, df_labeled) -> dict:
    """Store-backed wrapper: pull scored seizures + exclusion keys, then
    ``build_ledger``. Never raises on a missing exclusion table."""
    from src.preictal.isi import scored_seizures
    assert animal_id, "animal_id required"
    scored = scored_seizures(store, animal_id)
    try:
        excl = store.excluded_seizure_keys(animal_id)
    except Exception:                                  # noqa: BLE001
        excl = set()
    out = build_ledger(scored, excl,
                        lambda s: store.seizure_excl_key(s.file_id, s.eo_sec),
                        df_labeled)
    out["animal"] = animal_id
    return out
