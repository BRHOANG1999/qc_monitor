"""Process-local LRU cache around load_mat for the dashboard.

The LFP Browser and Video Review re-query the same .mat file every time
the user zooms or seeks. load_mat is slow (full SMB read of hundreds of
MB), so cache the most-recently used ChunkData objects keyed by file
path. The dispatcher does NOT route through here -- it has its own
load_mat calls on a separate thread and shouldn't share a working set
with the dashboard.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import BoundedSemaphore, RLock

from src.utils.mat_loader import ChunkData, load_mat


_MAX_ENTRIES = 3
# Entry count alone does NOT bound memory: one recording's signal matrix runs
# 0.3-4.7 GB (measured across the production processed_files rows), so three
# "bounded" entries reached ~14 GB. Cap the resident BYTES too -- evict oldest
# until the total fits, always keeping the most recent (the caller needs it).
_MAX_BYTES = 4_000_000_000
_od: "OrderedDict[str, ChunkData]" = OrderedDict()
_lock = RLock()

# load_mat runs OUTSIDE _lock (so a slow SMB read doesn't block cache hits),
# which means every concurrent caller can hold its own multi-GB recording at
# the same time -- the cache budget bounds what is RETAINED, not what is
# IN FLIGHT. With the mass_analyze scan pool (4 workers) plus dashboard
# threads that is tens of GB of transient peak. This semaphore bounds the
# number of simultaneous loads; waiters re-check the cache on wake, so a
# queued thread usually gets a hit instead of a second read of the same file.
_MAX_CONCURRENT_LOADS = 2
_load_sem = BoundedSemaphore(_MAX_CONCURRENT_LOADS)


def _nbytes(chunk: ChunkData) -> int:
    """Resident size of a cached chunk (the signal matrix dominates)."""
    sig = getattr(chunk, "signal", None)
    try:
        return int(sig.nbytes) if sig is not None else 0
    except AttributeError:
        return 0


def _evict_locked() -> None:
    """Enforce BOTH the entry cap and the byte budget. Caller holds _lock."""
    while len(_od) > _MAX_ENTRIES:
        _od.popitem(last=False)
    total = sum(_nbytes(c) for c in _od.values())
    while total > _MAX_BYTES and len(_od) > 1:
        _k, victim = _od.popitem(last=False)
        total -= _nbytes(victim)


def get_chunk(file_path: str) -> ChunkData:
    """Return the ChunkData for *file_path*, loading and caching it on
    miss. Cached entries are shared by reference; treat the returned
    ChunkData as read-only."""
    assert isinstance(file_path, str) and file_path, "file_path required"

    with _lock:
        hit = _od.pop(file_path, None)
        if hit is not None:
            _od[file_path] = hit
            return hit

    # Bound how many multi-GB reads are in flight at once (see _load_sem).
    with _load_sem:
        # Re-check under the lock: while we waited for a slot another thread
        # may have loaded this very file, so we return its copy instead of
        # reading (and holding) a second one.
        with _lock:
            hit = _od.pop(file_path, None)
            if hit is not None:
                _od[file_path] = hit
                return hit
        chunk = load_mat(file_path)
    assert chunk.signal.ndim == 2, "signal must be 2-D"

    with _lock:
        _od[file_path] = chunk
        _evict_locked()
    return chunk


def evict(file_path: str) -> None:
    """Drop a cached entry, e.g. after the dispatcher rewrites the file."""
    assert isinstance(file_path, str), "file_path must be a string"
    with _lock:
        _od.pop(file_path, None)


def cache_info() -> dict:
    """Debug helper: paths currently cached, in LRU order (oldest first),
    plus the resident bytes vs the budget (the thing that actually matters)."""
    with _lock:
        return {"entries": list(_od.keys()), "max_entries": _MAX_ENTRIES,
                "bytes": sum(_nbytes(c) for c in _od.values()),
                "max_bytes": _MAX_BYTES}
