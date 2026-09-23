"""Riding-event analysis: the oscillatory event that rides on the evoked response
(and its spontaneous, non-stim cousins).

Two prongs share one core:

* **Prong A** (``residual``) subtracts a per-recording robust-median *average
  evoked response* from each epoch so the extra event stands out from the
  stereotyped waveform, flags event-carrying epochs by residual energy, and
  compares the power spectrum of event windows vs clean responses (``spectra``).
* **Prong B** (``detect``) is stim-agnostic: detect candidate events on the
  continuous LFP, align them by their rising edge, build a median template, and
  detect further events with a normalized-correlation (matched-filter) threshold.

``primitives`` holds the pure numpy/scipy core (unit-tested plant-and-recover);
``render`` draws the dark-themed figures; ``run`` / ``__main__`` auto-pick the
strongest recording and orchestrate both prongs from the CLI.
"""
