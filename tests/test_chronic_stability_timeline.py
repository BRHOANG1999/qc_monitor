"""Unit tests for src/chronic_stability/config_timeline.py (pure parts)."""

import numpy as np

from src.chronic_stability import config_timeline as ct


def _sess(start_h, rec, stim, charge):
    return {"session_dir": f"/x/{rec}_{start_h}", "token": "tok",
            "record_site": rec, "stim_site": stim, "record_channel": f"A{rec}",
            "start_epoch": float(start_h * 3600), "charge_nC": charge,
            "pulse_width_us": 150.0, "neg_pulse_ratio": 3.0,
            "frequency_hz": 0.5, "gain": 100.0, "fp_status": "mapped",
            "report_channel": 2}


def test_intervals_merge_and_swap():
    sessions = [_sess(0, "SR", "SLM", 2.0), _sess(1, "SR", "SLM", 5.0),
                _sess(10, "SLM", "SR", 5.0), _sess(20, "SLM", "SR", 5.0)]
    ivs = ct.config_intervals(sessions)
    assert len(ivs) == 2
    assert ivs[0]["record_site"] == "SR" and ivs[0]["stim_site"] == "SLM"
    assert ivs[0]["end_epoch"] == 10 * 3600
    assert ivs[1]["record_site"] == "SLM" and ivs[1]["end_epoch"] == float("inf")


def test_events_site_swap_and_charge_change():
    sessions = [_sess(0, "SR", "SLM", 5.0), _sess(1, "SR", "SLM", 2.0),
                _sess(10, "SLM", "SR", 5.0)]
    ev = ct.detect_config_events(sessions)
    kinds = [e["event_type"] for e in ev]
    assert kinds == ["charge_change", "site_swap"]
    assert ev[1]["at_epoch"] == 10 * 3600
    assert ev[1]["provenance"]["table"] == "session_config"


def test_assign_epochs_half_open():
    sessions = [_sess(0, "SR", "SLM", 5.0), _sess(10, "SLM", "SR", 5.0)]
    ivs = ct.config_intervals(sessions)
    ep = np.array([-1, 0, 5, 10, 100]) * 3600.0
    idx = ct.assign_epochs_to_configs(ep, ivs)
    assert idx.tolist() == [-1, 0, 0, 1, 1]     # start inclusive, end exclusive


def test_modal_charge():
    sessions = [_sess(0, "SLM", "SR", 5.0), _sess(1, "SLM", "SR", 5.0),
                _sess(2, "SLM", "SR", 2.0)]
    ivs = ct.config_intervals(sessions)
    assert ivs[0]["modal_charge_nC"] == 5.0
    assert ivs[0]["charges_seen"] == [2.0, 5.0]
