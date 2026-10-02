"""Continuous raw-LFP envelope around each seizure, for the animation's
"full LFP" panel — so you can see the actual LFP over the −120→0 min pre-ictal
window and the seizure discharge at onset.

Wraps riding_event.seizure_lfp.gather_seizure_lfp (reads the raw multi-GB
continuous recording via the chunk cache, blanks stim, decimates to a min-max
envelope). Expensive (raw read per seizure) → cached incrementally per onset.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from . import config as C

_CACHE = C.CACHE_DIR + "/BCH111_lfp_envelopes.pkl"
_KEYS = ("env_t", "env_lo", "env_hi", "z_t", "z_lo", "z_hi", "phfo_min",
         "onset_epoch")


def _log(m):
    print(f"[preictal_biomarker.continuous_lfp] {m}", flush=True)


def _key(o) -> str:
    return str(int(round(float(o))))


def build_lfp_envelopes(store, onsets, *, pre_h=2.0, post_h=0.35,
                        force=False) -> dict:
    """{onset-key: envelope dict} for each onset; raw read only for uncached ones."""
    from src.riding_event.seizure_lfp import gather_seizure_lfp
    from src.riding_event.periictal import ensure_template
    cached = {}
    if not force and os.path.exists(_CACHE):
        with open(_CACHE, "rb") as f:
            cached = pickle.load(f)
    tmpl = None
    todo = [o for o in onsets if _key(o) not in cached]
    _log(f"{len(onsets)} onsets; {len(todo)} need a raw-LFP read "
         f"(pre {pre_h} h, post {post_h} h) ...")
    for i, o in enumerate(todo):
        if tmpl is None:
            tmpl = ensure_template(store, C.ANIMAL)
        _log(f"  [{i+1}/{len(todo)}] reading raw LFP around onset {_key(o)} ...")
        try:
            g = gather_seizure_lfp(store, C.ANIMAL, float(o), tmpl,
                                   pre_h=pre_h, post_h=post_h)
            cached[_key(o)] = {k: g.get(k) for k in _KEYS}
        except Exception as e:                            # noqa: BLE001
            _log(f"    skip {_key(o)}: {type(e).__name__}: {e}")
            cached[_key(o)] = None
        os.makedirs(C.CACHE_DIR, exist_ok=True)
        with open(_CACHE, "wb") as f:                     # incremental / resumable
            pickle.dump(cached, f)
    return {_key(o): cached.get(_key(o)) for o in onsets}
