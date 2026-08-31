"""Chronic-stim stability mapping (read-only analysis).

Maps a subject's stimulus-delivery stability history across its
``(stim site, record site)`` configurations, gated on access resistance (Rₐ),
with per-trial artifact metrics (polarity, amplitude, shape), seizure counts,
and distribution comparisons. Mapping only -- no experimental recommendations.

Convention (subject wiring, per the operator): in a channel ``BCH111<AREA>``
the NAMED ``<AREA>`` is the RECORD site; the UNNAMED site is the STIM target.
The codebase's ``stim_map``/``impedance_refresh`` positional rule (electrode
after a stimCopy = "stimulated") is used only to DETECT the named electrode and
swaps; the ``(stim, record)`` labels are then applied per this convention.
"""

from __future__ import annotations

__all__ = [
    "ra_flatrun",
    "config_timeline",
    "metrics",
    "compare",
    "render",
    "report",
]
