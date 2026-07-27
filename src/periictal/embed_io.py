"""Save / load a built peri-ictal embedding so the exact context can be
re-opened without rebuilding the matrix + re-running UMAP.

A saved file bundles the *selection* (animal / protocol / variant / window /
method that produced it) with the *result* dict the Explorer's lenses render
from (``emb``, ``sub``, ``full``, ``cols``, ``meta``, ``readout``, ``method``,
``n_seizures``). Because that result holds pandas DataFrames + numpy arrays, the
container is a pickle.

SECURITY / TRUST: unpickling executes code, so ``loads`` will happily run
whatever is in the blob. This is meant for a local operator re-importing THEIR
OWN exports (and same-lab copies). Do not import a file from an untrusted
source. If cross-machine sharing ever needs a hardened format, swap the body for
a zip of parquet (DataFrames) + npy (arrays) + json (meta) -- the public
``dumps``/``loads`` signature can stay the same.
"""

from __future__ import annotations

import pickle

# Bump if the bundle shape changes so an old/foreign file is rejected cleanly
# rather than half-loaded into a mismatched renderer.
_VERSION = 1

# Result keys the Explorer lenses need; a bundle missing these isn't loadable.
_REQUIRED_RESULT_KEYS = ("emb", "cols", "meta", "method")


def dumps(selection: dict, result: dict) -> bytes:
    """Serialise a ``(selection, result)`` pair to bytes for download."""
    assert isinstance(selection, dict), "selection must be a dict"
    assert isinstance(result, dict), "result must be a dict"
    payload = {"version": _VERSION, "selection": selection, "result": result}
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def loads(blob: bytes) -> tuple[dict, dict]:
    """``(selection, result)`` from a bundle, or ``ValueError`` if it isn't one.

    Validates the version + shape BEFORE handing the result to the UI, so a
    wrong/corrupt/foreign file fails with a clear message instead of rendering
    garbage or raising deep in a lens."""
    try:
        obj = pickle.loads(blob)
    except Exception as e:  # noqa: BLE001 -- unpickle can raise many types
        raise ValueError(f"not a readable embedding file ({e})") from e
    if not isinstance(obj, dict) or obj.get("version") != _VERSION:
        got = obj.get("version") if isinstance(obj, dict) else type(obj).__name__
        raise ValueError(
            f"not a peri-ictal embedding export (version {got!r}, "
            f"expected {_VERSION})")
    selection, result = obj.get("selection"), obj.get("result")
    if not isinstance(selection, dict) or not isinstance(result, dict):
        raise ValueError("corrupt embedding export (missing selection/result)")
    missing = [k for k in _REQUIRED_RESULT_KEYS if k not in result]
    if missing:
        raise ValueError(
            f"corrupt embedding export (result missing {', '.join(missing)})")
    return selection, result


def suggested_name(selection: dict) -> str:
    """A filesystem-safe download name from the selection scope."""
    def _safe(v) -> str:
        return "".join(c if (c.isalnum() or c in "-.") else "-"
                       for c in str(v)) or "x"
    animal = _safe((selection or {}).get("animal") or "animal")
    proto = _safe((selection or {}).get("protocol") or "all")
    method = _safe((selection or {}).get("method") or "pca")
    return f"periictal_{animal}_{proto}_{method}.pkl"
