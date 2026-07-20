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
# Sized against the real corpus: the average recording's signal matrix is
# ~1.9 GB and the largest ~4.7 GB, so a budget of only 4 GB held barely ONE
# recording -- the background sweep then evicted the interactive user's
# recording on every load and the Video tab re-read it from SMB every time.
_MAX_BYTES = 8_000_000_000
_od: "OrderedDict[str, ChunkData]" = OrderedDict()
_lock = RLock()

# load_mat runs OUTSIDE _lock (so a slow SMB read doesn't block cache hits),
# which means every concurrent caller can hold its own multi-GB recording at
# the same time -- the cache budget bounds what is RETAINED, not what is
# IN FLIGHT. With the mass_analyze scan pool (4 workers) plus dashboard
# threads that is tens of GB of transient peak. This semaphore bounds the
# number of simultaneous loads; waiters re-check the cache on wake, so a
# queued thread usually gets a hit instead of a second read of the same file.
# Interactive callers (dashboard) and background sweeps get SEPARATE budgets.
# With one shared semaphore the auto_filter sweep -- which streams hundreds of
# files back-to-back -- held every slot, so opening a tab meant queueing behind
# multi-GB SMB reads. A dedicated interactive budget keeps the UI responsive
# while still bounding total in-flight memory (2 + 1 recordings).
_MAX_CONCURRENT_LOADS = 2          # interactive / dashboard
_MAX_CONCURRENT_SCAN_LOADS = 1     # background sweeps
_load_sem = BoundedSemaphore(_MAX_CONCURRENT_LOADS)
_scan_sem = BoundedSemaphore(_MAX_CONCURRENT_SCAN_LOADS)


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


def get_chunk(file_path: str, *, transient: bool = False) -> ChunkData:
    """Return the ChunkData for *file_path*, loading and caching it on
    miss. Cached entries are shared by reference; treat the returned
    ChunkData as read-only.

    *transient* marks a BACKGROUND sweep read (mass_analyze screening
    hundreds of files once). Such reads use their own small load budget so
    they can't starve the dashboard, and they are cached at the LRU-OLDEST
    position so they are evicted first -- a one-shot sweep must not push out
    the recording an operator is actively looking at. They are still cached,
    because the sweep screens several CHANNELS of the same file in a row.
    """
    assert isinstance(file_path, str) and file_path, "file_path required"

    with _lock:
        hit = _od.pop(file_path, None)
        if hit is not None:
            # A hit is promoted to newest ONLY for interactive callers, so a
            # sweep re-touching a file can't promote it over live UI data.
            if transient:
                _od[file_path] = hit
                _od.move_to_end(file_path, last=False)
            else:
                _od[file_path] = hit
            return hit

    # Bound how many multi-GB reads are in flight (separate budgets so a
    # sweep can never occupy every slot -- see _load_sem / _scan_sem).
    with (_scan_sem if transient else _load_sem):
        # Re-check under the lock: while we waited for a slot another thread
        # may have loaded this very file, so we return its copy instead of
        # reading (and holding) a second one.
        with _lock:
            hit = _od.pop(file_path, None)
            if hit is not None:
                _od[file_path] = hit
                if transient:
                    _od.move_to_end(file_path, last=False)
                return hit
        chunk = load_mat(file_path)
    assert chunk.signal.ndim == 2, "signal must be 2-D"

    with _lock:
        _od[file_path] = chunk
        if transient:                     # evict me before any UI entry
            _od.move_to_end(file_path, last=False)
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
