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
from threading import RLock

from src.utils.mat_loader import ChunkData, load_mat


_MAX_ENTRIES = 3
_od: "OrderedDict[str, ChunkData]" = OrderedDict()
_lock = RLock()


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

    chunk = load_mat(file_path)
    assert chunk.signal.ndim == 2, "signal must be 2-D"

    with _lock:
        _od[file_path] = chunk
        while len(_od) > _MAX_ENTRIES:
            _od.popitem(last=False)
    return chunk


def evict(file_path: str) -> None:
    """Drop a cached entry, e.g. after the dispatcher rewrites the file."""
    assert isinstance(file_path, str), "file_path must be a string"
    with _lock:
        _od.pop(file_path, None)


def cache_info() -> dict:
    """Debug helper: paths currently cached, in LRU order (oldest first)."""
    with _lock:
        return {"entries": list(_od.keys()), "max_entries": _MAX_ENTRIES}
