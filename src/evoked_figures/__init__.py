"""Evoked-metric-over-time figure export.

Per animal, on its primary channel, render two PNGs per evoked metric — a
scatter-only and a percentile-shaded plot — plus a colocated CSV of the data
and a provenance manifest, under
``<out_root>/<animal>/<metric>/``. Pure Python (matplotlib Agg); the data path
mirrors the chronic-evoked tab (feature sidecars + rolling percentiles) so the
figures match what the dashboard shows.
"""
