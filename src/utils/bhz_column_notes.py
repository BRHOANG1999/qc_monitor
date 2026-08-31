"""Per-column "how it was computed" notes for the BHZ Analysis Log.

Single source of truth for the hover-tooltip NOTE attached to each auto-computed
column header across the workbook's three tab types (per-animal ``<X> Daily`` +
``<X> Events`` tabs, and the ``Summary`` tab). The writers apply these
idempotently on each sync (so new animal tabs get them automatically), and
``annotate_all`` / the ``--apply`` CLI back-fills every existing tab.

Keys are the exact header strings; matching to the live header is
case/punctuation-insensitive (``sheets_write._norm``), so minor spacing
differences don't matter. Headers with no entry (hand-filled columns like
Comments/Notes) are left untouched.
"""

from __future__ import annotations

import logging

from src.utils import sheets_write as _sw

logger = logging.getLogger("qc_monitor.utils.bhz_column_notes")

# --- <animal> Daily tab (writer: event_verification._push_scored_to_sheet) --- #
DAILY_NOTES = {
    "Date": "Recording day (one row per day for this animal), from the recording "
            "file's timestamp. This is the row key.",
    "MouseID": "Animal ID from the review record. Legacy whole-file rows with no "
               "animal fall back to the session's first animal channel.",
    "CSV File Name": "The per-animal-day seizure CSV for this row "
                     "(YYYYMMDD_<animal>.csv); blank if no CSV base dir is set.",
    "Number of Behavioral Events": "Confirmed seizures this day = PI-approved "
        "onsets PLUS onsets still awaiting landmark placement (see 'Events "
        "Needing Onsets'). Each is a scored event with an EEG onset.",
    "Events Needing Onsets": "Subset of 'Number of Behavioral Events' that are "
        "scored (have an EEG onset) but still need landmarks placed "
        "(status = needs_scoring). Not a separate bucket.",
    "Max Racine": "Highest Racine motor score among the day's events "
                  "(approved + pending); 0 if none.",
    "Who Completed Analysis": "Distinct reviewer email(s) on the day's "
        "PI-approved recordings. Pending (needs_scoring) reviewers are excluded.",
    "Recording Location": "Electrode location(s) recorded for this animal in the "
                          "session (e.g. SR, SLM), from the electrode metadata.",
    "Channel(s)": "Recording channel number(s), shown 1-based (the database "
        "stores 0-based). Kept as text so Sheets doesn't misread '2, 3' as a date.",
    "Type of Recording": "Protocol name(s) for the day, joined with ' + ' "
        "(e.g. 'baseline + stim'). The experimental protocol, not a stim/baseline "
        "flag.",
    "More Settings": "Commanded stimulation for this animal's electrode from the "
        "STIM_REPORT: charge (nC), pulse width (us), frequency (Hz). Blank on days "
        "with no stim file. (Not from session_config, which carries a placeholder.)",
    # alias-only columns (present on some legacy tabs)
    "Files": "Number of PI-approved recording files for the animal that day.",
    "Files With Events": "Approved files that contained at least one seizure.",
    "During Stim": "Approved seizures that occurred on a stim file that day.",
    "No Event Files": "Approved files with zero seizures that day.",
    "Exported At": "When this row was last written by the sync.",
}

# --- <animal> Events tab (writer: event_verification, upsert_event_rows) --- #
EVENTS_NOTES = {
    "Date": "Recording date of the seizure (YYYY-MM-DD), from the file timestamp.",
    "Onset (clock)": "Absolute wall-clock time of the EEG onset = recording start "
        "+ the onset offset (EO_sec), to the millisecond. Row key (with 'File').",
    "Racine": "Racine motor score for this seizure.",
    "Type": "Seizure type (e.g. LVF, HYP).",
    "Light": "Light condition recorded for the seizure.",
    "File": "Recording file (name) the seizure is in. Row key (with 'Onset').",
    "Status": "Review state: 'Approved' (PI-approved) or 'Needs more onsets' "
              "(scored, still awaiting landmarks).",
}

# --- Summary tab (writer: summary_sheet.write_summary) --- #
_GAP = ("Percent of the span between first and last recording that has data: "
        "100 * (span - total gap) / span")
SUMMARY_NOTES = {
    "KA location": "Kainic-acid injection target (the 'Target 1' cell) from the "
                   "surgery BCH Log sheet.",
    "Surgeon": "Surgeon from the surgery BCH Log sheet.",
    "Time from KA to start of recording": "Days from the KA injection date to the "
                                          "animal's first recording.",
    "# of overt Bh Sz in first 2 weeks": "Scored seizures in the first 14 days of "
        "RECORDING (measured from the first recording, not from KA).",
    "Total recording time": "Total recording hours = sum of all file durations "
        "for this animal (files with no measured duration are estimated at 1 hour "
        "per chunk).",
    "Total overt bh sz during that time": "All scored seizures across the animal's "
        "recording days. A scored seizure has BOTH an EEG onset and a Racine score.",
    "Overall Bh SZ rate": "Seizures per recording-day = total scored seizures / "
        "(total recording hours / 24). Based on recording time, not calendar days.",
    "Electrode(s) recorded": "Electrode(s) the animal was recorded on, from the "
                             "recording channel names.",
    "Earliest recording": "Timestamp of the animal's first recording.",
    "Latest recording": "Timestamp of the animal's most recent recording.",
    "Animal coverage %": f"{_GAP} (whole-animal timeline; a recording counts once "
                         "regardless of how many electrodes it carried).",
    "Animal # gaps (>2h)": "Number of gaps longer than 2 hours between consecutive "
                           "recordings.",
    "Animal total gap (h)": "Total gap hours (each gap = elapsed time minus one "
        "nominal 1-hour chunk, so a contiguous hourly run counts as ~0).",
    "Animal largest gap (h)": "Longest single gap between consecutive recordings "
                              "(hours).",
    "Animal smallest gap (h)": "Shortest counted gap (>2 h), in hours.",
    "SLM coverage %": f"{_GAP}, restricted to the SLM-electrode recording timeline.",
    "SLM # gaps (>2h)": "Number of >2 h gaps on the SLM-only recording timeline.",
    "SLM total gap (h)": "Total gap hours on the SLM-only recording timeline.",
    "SR coverage %": f"{_GAP}, restricted to the SR-electrode recording timeline.",
    "SR # gaps (>2h)": "Number of >2 h gaps on the SR-only recording timeline.",
    "SR total gap (h)": "Total gap hours on the SR-only recording timeline.",
    "Earliest file": "Filename of the animal's first recording.",
    "Latest file": "Filename of the animal's most recent recording.",
    "Last updated": "When this row was auto-filled by the summary job.",
    "QC Monitor version": "The QC Monitor version that wrote this row.",
    "Reliability": "'Check: ...' when the auto numbers are less trustworthy (no "
        "recording, or coverage < 80% = gappy); otherwise 'OK'.",
}


def apply_daily(svc, sheet_id: str, tab: str) -> int:
    return _safe(svc, sheet_id, tab, DAILY_NOTES)


def apply_events(svc, sheet_id: str, tab: str) -> int:
    return _safe(svc, sheet_id, tab, EVENTS_NOTES)


def apply_summary(svc, sheet_id: str, tab: str) -> int:
    return _safe(svc, sheet_id, tab, SUMMARY_NOTES)


def _safe(svc, sheet_id: str, tab: str, notes: dict) -> int:
    """set_header_notes, but never raise -- header notes are cosmetic and must
    not break the sync that calls them."""
    try:
        return _sw.set_header_notes(svc, sheet_id, tab, notes)
    except Exception as e:                                # noqa: BLE001
        logger.warning("header notes failed for tab %r: %s", tab, e)
        return 0


def annotate_all(svc, sheet_id: str, *, summary_suffix: str = " Daily",
                 events_suffix: str = " Events",
                 summary_tab: str = "Summary") -> dict:
    """Back-fill notes on EVERY relevant tab in the workbook: each ``<X> Daily``,
    each ``<X> Events``, and the ``Summary`` tab. Returns {tab: n_headers_noted}."""
    out: dict = {}
    for title in sorted(_sw.sheet_titles(svc, sheet_id)):
        if title == summary_tab:
            out[title] = apply_summary(svc, sheet_id, title)
        elif events_suffix and title.endswith(events_suffix):
            out[title] = apply_events(svc, sheet_id, title)
        elif summary_suffix and title.endswith(summary_suffix):
            out[title] = apply_daily(svc, sheet_id, title)
    return out


def _main(argv=None) -> int:
    import argparse

    import yaml

    from src.utils.sheets import _resolve_sa_path

    ap = argparse.ArgumentParser(description="Annotate BHZ log column headers")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--apply", action="store_true",
                    help="actually write notes (default: list the tabs found)")
    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gs = cfg.get("google_sheets", {}) or {}
    sa = _resolve_sa_path(cfg, gs.get("service_account_file"))
    sid = gs.get("spreadsheet_id")
    svc = _sw._sheets_api_rw(sa)
    daily = gs.get("summary_tab_suffix", " Daily")
    events = gs.get("events_tab_suffix", " Events")
    if not args.apply:
        titles = sorted(_sw.sheet_titles(svc, sid))
        tabs = [t for t in titles
                if t == "Summary" or t.endswith(daily) or t.endswith(events)]
        print(f"Would annotate {len(tabs)} tab(s): {tabs}")
        print("Re-run with --apply to write the notes.")
        return 0
    res = annotate_all(svc, sid, summary_suffix=daily, events_suffix=events)
    for tab, n in sorted(res.items()):
        print(f"  {tab}: {n} headers noted")
    print(f"Annotated {len(res)} tab(s), {sum(res.values())} headers total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
