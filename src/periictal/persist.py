"""Disk cache for the event x metric matrix.

Building the matrix reads + JSON-parses every sidecar for an animal (~20 s for
BCH111's 300 files, minutes for BCH040's 1450), so an interactive tab can't do
it on every build. This caches the built DataFrame keyed by a signature that
captures BOTH the feature inputs (sidecar paths + mtimes, via
``evoked_figures.run.inputs_signature``) AND the seizure timeline (onset epochs)
-- so scoring a new seizure, which retroactively rewrites every preceding
stimulus's lead-time, correctly invalidates the cache.
"""

from __future__ import annotations

import hashlib
import os
import pickle

import pandas as pd

from src.evoked_figures.run import inputs_signature
from src.periictal import matrix as _matrix
from src.periictal import passive as _passive
from src.preictal.isi import included_seizures
from src.utils.evoked_output import (animals_in_filename, feature_sidecar_path,
                                     list_evoked_files)

_MAX_FILES = 2_000_000

# Bump when the matrix column schema changes, so old cached pickles (which lack
# the new columns) are invalidated rather than silently loaded. v2 added `rec`;
# v3 added `phase` (pre/post) + the symmetric post-onset rows; v4 added the Chang
# et al. 2026 feature columns (via the widened CHEAP_METRICS).
_SCHEMA_VERSION = "4"


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name))


def _dir_mtimes(evoked_dir: str) -> dict:
    """``{basename: st_mtime}`` for every entry in *evoked_dir*, from ONE
    directory enumeration.

    A ``getmtime()`` per sidecar is a separate round-trip on a network share
    (~10 ms each -> ~5.5 s for BCH111's 568 sidecars, paid on EVERY build,
    cache hit included). ``os.scandir`` returns each entry's mtime as part of
    the single directory read (~20 ms total), so the whole stat walk collapses
    to one call. Sidecars live beside the ``.mat`` in *evoked_dir*."""
    out: dict = {}
    try:
        with os.scandir(evoked_dir) as it:
            for e in it:
                try:
                    out[e.name] = e.stat().st_mtime
                except OSError:
                    pass
    except OSError:
        pass
    return out


def _sidecar_inputs(animal: str, evoked_dir: str, sidecar_variant: str) -> list:
    """(sidecar_path, mtime) for *animal*'s *sidecar_variant* sidecars -- a cheap
    stat walk (no JSON read) so the signature is fast to compute. A window/config
    change rewarms the sidecar (new mtime), which flows into this signature and
    invalidates the cached matrix."""
    mtimes = _dir_mtimes(evoked_dir)
    out: list = []
    for i, fp in enumerate(list_evoked_files(evoked_dir)):
        assert i < _MAX_FILES, "sidecar stat walk runaway"
        if animal not in animals_in_filename(fp):
            continue
        sp = feature_sidecar_path(fp, animal, sidecar_variant)
        mt = mtimes.get(os.path.basename(sp))
        if mt is not None:
            out.append((sp, mt))
    return out


def _seizure_sig(store, animal: str) -> str:
    # included_seizures (not scored_seizures): excluding a bad-data seizure
    # changes the onset set here too, so the cached matrix invalidates.
    h = hashlib.sha256()
    for s in included_seizures(store, animal):
        h.update(f"{s.onset_epoch:.1f}|".encode("ascii"))
    return h.hexdigest()


def _signature(store, animal, evoked_dir, sidecar_variant, protocol,
               window_sec, extra: str, cfg_sig: str) -> str:
    parts = [inputs_signature(_sidecar_inputs(animal, evoked_dir, sidecar_variant)),
             _seizure_sig(store, animal),
             f"{protocol}|{window_sec}|{sidecar_variant}|{cfg_sig}|{extra}"
             f"|v{_SCHEMA_VERSION}"]
    return hashlib.sha256("::".join(parts).encode()).hexdigest()[:16]


def _prune(cache_dir: str, prefix: str, keep_name: str) -> None:
    """Drop stale cache files for the same selection (older signatures)."""
    try:
        for name in os.listdir(cache_dir):
            if name.startswith(prefix) and name != keep_name:
                try:
                    os.remove(os.path.join(cache_dir, name))
                except OSError:
                    pass
    except OSError:
        pass


def build_matrix_cached(store, animal: str, evoked_dir: str, cache_dir: str, *,
                        protocol: str | None = None,
                        window_sec: float = 6 * 3600.0,
                        variant: str = "evoked", feature_cfg=None,
                        sidecar_variant: str | None = None, passive_cfg=None,
                        **kw) -> pd.DataFrame:
    """``matrix.build_matrix`` with a signature-keyed pickle cache. On a
    signature hit the DataFrame is loaded from disk (fast); otherwise it is
    built and cached atomically, and stale cache files for this selection are
    pruned. The feature window (via *feature_cfg* / *sidecar_variant*) is part of
    the signature, so different windows cache separately."""
    assert cache_dir, "cache_dir required"
    os.makedirs(cache_dir, exist_ok=True)
    cfg = feature_cfg if feature_cfg is not None else passive_cfg
    sv = sidecar_variant or ("passive" if variant == "passive" else "evoked")
    cfg_sig = _passive.config_sig(cfg) if cfg is not None else ""
    sig = _signature(store, animal, evoked_dir, sv, protocol,
                     window_sec, repr(sorted(kw.items())), cfg_sig)
    prefix = f"{_safe(animal)}.{sv}.{_safe(protocol or 'all')}.w{int(window_sec)}."
    name = f"{prefix}{sig}.pkl"
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except Exception:                                # noqa: BLE001
            pass                                         # corrupt -> rebuild
    df = _matrix.build_matrix(store, animal, evoked_dir, protocol=protocol,
                              window_sec=window_sec, variant=variant,
                              feature_cfg=cfg, sidecar_variant=sv, **kw)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        _prune(cache_dir, prefix, name)
    except OSError:
        pass
    return df
