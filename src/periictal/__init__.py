"""Peri-ictal evoked explorer.

Joins the per-stimulus evoked feature sidecars (``src.utils.evoked_output`` /
``src.evoked_figures.data``) to the scored-seizure timeline
(``src.preictal.isi``) into a tidy ``event x metric`` matrix keyed by each
stimulus's CONTINUOUS time-to-next-seizure-onset, and embeds it (PCA by
default, UMAP opt-in) so the evoked-response feature space can be explored
against seizure proximity.

This is an EXPLORATION surface, not an inferential test: a low-dimensional
embedding can show apparent structure from noise or from confounds (stim
settings, circadian phase, post-ictal tails of clustered seizures). The
package therefore keeps the confounds first-class (group/colour by the
STIM_REPORT hardware fingerprint and by time-of-day) so a contaminated
embedding is visible rather than hidden. The supervised leave-one-seizure-out
test that would turn a picture into a claim reuses ``src.preictal.validation``
+ ``src.preictal.scoring`` and is a deliberate future module (not built here).
"""

from __future__ import annotations
