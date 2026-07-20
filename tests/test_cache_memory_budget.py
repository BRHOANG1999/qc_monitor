"""Memory budgets on the caches that drove the ~59 GB RSS incident.

The caps in these caches bounded how MANY results were memoized, not how BIG
they were -- and one result can be a whole animal's history (~2.5M row dicts,
~8.5 GB) or a whole recording's signal matrix (up to ~4.7 GB). Six/three
"bounded" entries were therefore tens of GB. These tests pin the byte budgets.
"""

import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.tabs import chronic_evoked as CE  # noqa: E402
from src.utils import chunk_cache as CC              # noqa: E402


# ---------------------------------------------------------------- rows ---- #

def test_oversized_result_is_not_memoized():
    """A single result bigger than the whole budget must not be cached."""
    assert CE._cache_worth_keeping([{}] * 10) is True
    assert CE._cache_worth_keeping([]) is False
    assert CE._cache_worth_keeping([{}] * (CE._MAX_CACHED_ROWS + 1)) is False


def test_evict_to_budget_enforces_rows_not_just_entries():
    cache = OrderedDict()
    per = CE._MAX_CACHED_ROWS // 2
    # Three entries, each half the budget -> entry cap (6) is NOT hit, but the
    # row budget is exceeded. Before the fix this kept all three.
    for i in range(3):
        cache[f"k{i}"] = [{}] * per
    CE._evict_to_budget(cache, 6)
    total = sum(len(v) for v in cache.values())
    assert total <= CE._MAX_CACHED_ROWS, "row budget not enforced"
    assert len(cache) < 3, "nothing was evicted"
    assert "k2" in cache, "the newest entry (needed by the caller) was evicted"


def test_evict_to_budget_keeps_at_least_the_newest():
    cache = OrderedDict()
    cache["only"] = [{}] * (CE._MAX_CACHED_ROWS * 3)   # oversized on its own
    CE._evict_to_budget(cache, 6)
    assert "only" in cache, "must never evict the entry the caller is using"


def test_entry_cap_still_applies():
    cache = OrderedDict()
    for i in range(10):
        cache[f"k{i}"] = [{}] * 5
    CE._evict_to_budget(cache, 6)
    assert len(cache) <= 6


def test_budget_is_sane():
    # ~1.5 GB at the measured ~3.3 KB/row.
    assert 100_000 <= CE._MAX_CACHED_ROWS <= 2_000_000
    assert CE._ROW_BYTES_EST >= 1000


# --------------------------------------------------------------- chunks ---- #

def _chunk(mb: int):
    n = (mb * 1_000_000) // 8
    return SimpleNamespace(signal=np.zeros((n, 1), dtype=np.float64), fs=20000.0)


def test_chunk_cache_evicts_on_bytes_not_just_entries(monkeypatch):
    monkeypatch.setattr(CC, "_MAX_BYTES", 300_000_000)      # 300 MB budget
    monkeypatch.setattr(CC, "_od", OrderedDict())
    with CC._lock:
        for i in range(3):
            CC._od[f"f{i}"] = _chunk(200)                   # 200 MB each
        CC._evict_locked()
    info = CC.cache_info()
    assert info["bytes"] <= CC._MAX_BYTES, "byte budget not enforced"
    assert "f2" in info["entries"], "newest chunk was evicted"
    assert len(info["entries"]) < 3


def test_chunk_cache_keeps_single_oversized_entry(monkeypatch):
    monkeypatch.setattr(CC, "_MAX_BYTES", 10_000_000)
    monkeypatch.setattr(CC, "_od", OrderedDict())
    with CC._lock:
        CC._od["big"] = _chunk(200)
        CC._evict_locked()
    assert "big" in CC.cache_info()["entries"]


def test_chunk_cache_info_reports_bytes(monkeypatch):
    monkeypatch.setattr(CC, "_od", OrderedDict())
    with CC._lock:
        CC._od["a"] = _chunk(8)
    info = CC.cache_info()
    assert info["bytes"] > 0 and info["max_bytes"] > 0
    assert set(("entries", "max_entries", "bytes", "max_bytes")) <= set(info)


def test_nbytes_tolerates_missing_signal():
    assert CC._nbytes(SimpleNamespace()) == 0


# ------------------------------------------------- in-flight peak memory --- #

def test_concurrent_loads_are_bounded(monkeypatch):
    """load_mat runs outside the lock, so without a semaphore every scan
    thread holds its own multi-GB recording at once."""
    import threading
    import time
    live = {"now": 0, "peak": 0}
    lk = threading.Lock()

    def fake_load(path):
        with lk:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.15)
        with lk:
            live["now"] -= 1
        return SimpleNamespace(signal=np.zeros((10, 1)), fs=1.0)

    monkeypatch.setattr(CC, "load_mat", fake_load)
    monkeypatch.setattr(CC, "_od", OrderedDict())
    threads = [threading.Thread(target=CC.get_chunk, args=(f"f{i}",))
               for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert live["peak"] <= CC._MAX_CONCURRENT_LOADS


def test_scan_pool_does_not_leak_threads():
    """Repeated sweeps must not stack abandoned pools (the 216-thread bug)."""
    import threading
    import time
    from src.utils import mass_analyze as MA

    def work(f):
        time.sleep(0.05)
        return f

    before = threading.active_count()
    for _ in range(3):
        MA._run_file_pool(list(range(8)), work, lambda: False,
                          lambda f, r: None, workers=4)
    time.sleep(0.3)
    assert threading.active_count() <= before + 1


def test_scan_pool_joins_even_when_cancelled():
    import threading
    import time
    from src.utils import mass_analyze as MA

    def work(f):
        time.sleep(0.05)
        return f

    before = threading.active_count()
    assert MA._run_file_pool(list(range(20)), work, lambda: True,
                             lambda f, r: None, workers=4) is True
    time.sleep(0.3)
    assert threading.active_count() <= before + 1
