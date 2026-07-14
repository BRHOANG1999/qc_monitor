"""flagged_pool_counts_per_animal: the single source of truth for the
auto-filter FLAG backlog, shared by the Overview card + Video Review queue.

Run with: pytest tests/test_flagged_pool_counts.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402


def _seed(store):
    with store.connection() as conn:
        for fid in (1, 2, 3, 4):
            conn.execute(
                "INSERT INTO processed_files (id, file_path, session_dir, "
                "has_video) VALUES (?,?,?,?)",
                (fid, f"/f{fid}.mat", "/s", 0 if fid == 4 else 1))

        def flag(fid, animal):
            conn.execute(
                "INSERT INTO review_event_log (file_id, user_email, action, "
                "payload_json, animal_id, at) VALUES (?,?,?,?,?,?)",
                (fid, "auto", "auto_filter_flag", "{}", animal, "2026-01-01"))
        flag(1, "BCH062")
        flag(2, "BCH062")
        flag(3, "BCH061")
        flag(4, "BCH062")            # file 4 has NO video -> excluded
        conn.commit()


def test_counts_flagged_minus_reviewed(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    _seed(store)

    c = store.flagged_pool_counts_per_animal()
    assert c.get("BCH062") == 2 and c.get("BCH061") == 1   # file 4 (no video) out

    # finalising file 1 (has_events) drops BCH062 to 1
    store.mark_review(1, "u@x", "has_events",
                      markers=[{"EO_sec": 1.0, "racine": 3, "type": "LVF"}],
                      animal_id="BCH062")
    assert store.flagged_pool_counts_per_animal().get("BCH062") == 1

    # pi_flagged (PI sent it back for re-review) is NOT finalised -> still counts
    store.mark_review(2, "pi@x", "pi_flagged", animal_id="BCH062")
    assert store.flagged_pool_counts_per_animal().get("BCH062") == 1


def test_overview_card_uses_flag_pool(tmp_path):
    """behavioral_seizure_status_per_animal.n_flag_pool == the flagged pool
    (the Overview card + Video Review now read the same number)."""
    store = Store(str(tmp_path / "data" / "m.db"))
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO session_config (session_dir, channel_names, "
            "eeg_channels, discovered_at) VALUES ('/s', ?, ?, 't')",
            ('["stimCopy","BCH062SR"]', "[1]"))
        conn.execute("INSERT INTO processed_files (id, file_path, session_dir, "
                     "chunk_datetime, has_video) VALUES "
                     "(1,'/f1.mat','/s','2026_03_01__00_00_00',1)")
        conn.execute(
            "INSERT INTO review_event_log (file_id, user_email, action, "
            "payload_json, animal_id, at) VALUES "
            "(1,'auto','auto_filter_flag','{}','BCH062','2026-01-01')")
        conn.commit()
    pool = store.flagged_pool_counts_per_animal()
    rows = store.behavioral_seizure_status_per_animal(days=7)
    bymatch = {r["animal_id"]: r for r in rows}
    assert bymatch["BCH062"]["n_flag_pool"] == pool["BCH062"] == 1
