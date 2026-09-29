"""Evoked-response SHAPE analysis: discover recurring response shapes (not sizes),
decide whether they are discrete modes or a continuum, and quantify how trustworthy
that structure is.

A previous attempt (PCA on baseline-subtracted traces) failed because PC1 was
amplitude, and the peri-ictal feature embedding turned out to be circadian rather
than shape-driven. This package works in SHAPE space: every trial is amplitude-
normalized first (the norm divisor is kept as the per-trial gain), so structure
reflects morphology, not scaling.

Scope (this pass, Stages 0 to 4):

  Stage 0  synth.py       synthetic traces with KNOWN structure, the test oracle.
  Stage 1  data.py        assemble X (n_trials x n_samples), fs, meta from the
                          MATLAB evokedOutput .mat files; preprocess.py normalizes.
  Stage 2  structure.py   discrete vs continuous: pairwise-correlation histogram,
                          Hartigan dip test, diffusion-map coordinate, densities.
  Stage 3  cluster.py     correlation / spherical k-means (+ reuse of the GMM-by-BIC
                          and NMF views from src.evoked_typology).
  Stage 4  select_k.py    BOUND k (not point-optimize): BIC + held-out log-likelihood,
                          template-correlation ceiling, null-corrected bootstrap
                          ARI / Jaccard stability, split-half reproducibility.

Deferred (follow-ups): Stage 5 shape-vs-seizure-risk, Stage 6 conv autoencoder,
the BPC (Miller/Hermes 2021) NNMF pipeline.

Reuses tested repo machinery rather than re-implementing it: src.evoked_typology
(gather + normalize + GMM + NMF), src.utils.evoked_features (preprocess), src.periictal
(matrix time-to-onset, embed PCA/UMAP, cluster_features, resample block bootstrap),
src.preictal.isi (seizure timeline + leaders + ISI ceilings).

Run: ``python -m src.evoked_shapes --animal BCH111 --apply``.
"""

from __future__ import annotations

__all__ = [
    "config",
    "synth",
    "preprocess",
    "structure",
    "cluster",
    "select_k",
    "data",
    "render",
    "report",
    "run",
]
