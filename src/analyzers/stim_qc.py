"""Tier 1/2: Stimulation QC from STIM_REPORT.txt files."""

import os
from src.utils.stim_parser import parse_stim_report_file


def analyze_stim_report(mat_path: str) -> list[dict] | None:
    """Find and parse the STIM_REPORT.txt adjacent to a .mat file.

    Returns list of per-channel stim metrics, or None if no report found.
    """
    # STIM_REPORT is named: <basename>_STIM_REPORT.txt
    base = mat_path.rsplit(".", 1)[0]
    report_path = base + "_STIM_REPORT.txt"

    if not os.path.exists(report_path):
        return None

    report = parse_stim_report_file(report_path)
    results = []

    for ch in report.channels:
        expected_pulses = 0
        if ch.frequency_hz > 0 and report.duration_sec > 0:
            expected_pulses = int(ch.frequency_hz * report.duration_sec)

        delivery_pct = 0.0
        if expected_pulses > 0:
            delivery_pct = 100.0 * ch.total_pulses / expected_pulses

        results.append({
            "stim_channel": ch.channel_num,
            "charge_nC": ch.charge_nC,
            "pulse_width_us": ch.pulse_width_us,
            "neg_pulse_ratio": ch.neg_pulse_ratio,
            "frequency_hz": ch.frequency_hz,
            "gain": ch.gain,
            "total_pulses": ch.total_pulses,
            "total_charge_nC": ch.total_charge_nC,
            "duration_sec": report.duration_sec,
            "expected_pulses": expected_pulses,
            "delivery_pct": delivery_pct,
        })

    return results
