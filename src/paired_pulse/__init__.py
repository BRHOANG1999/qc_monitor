"""Paired-pulse (PPR) downstream analysis for BCH111.

The acquisition side (KMRecorder) delivers two identical pulses 50 ms apart, the
pair repeating at 0.5 Hz. The qc_monitor evoked pipeline dedups the 2nd pulse
(min_stimulus_distance_sec=0.1 > 0.05) so each 50 ms pair becomes ONE evokedOutput
epoch whose +/-500 ms window holds BOTH pulses (anchor at t=0, partner at +/-50 ms;
the dedup keeps the higher-amplitude pulse, so the partner side varies per epoch).

This package measures the LFP response to pulse 1 (S1) and pulse 2 (S2) -- each
windowed 1-49 ms at its OWN onset -- and tracks the paired-pulse ratio
PPR = feature(S2)/feature(S1) over time and vs seizure proximity. PPR<1 = paired-
pulse depression (intact inhibition); PPR>1 = facilitation (reduced inhibition /
hyperexcitability). A more targeted seizure biomarker than the single-pulse evoked
response.
"""
