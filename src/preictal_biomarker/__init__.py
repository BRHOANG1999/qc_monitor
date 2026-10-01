"""BCH111 pre-ictal evoked state-space biomarker — reproducible pipeline.

Rebuilds (as committed, versioned code) the analysis whose scratch generators
were lost, keeping only the surviving Fig-01 backbone (``evoked_stim_corr``) and
the ``periictal`` primitives. The companion notebook
``notebooks/bch111_preictal_biomarker.ipynb`` runs the whole thing cell by cell
so a reviewer can reproduce every figure and null by hand.

Pipeline (see each submodule):
  features  -> per-epoch 2-50 ms / LP500, Z_ss-detrended 8-feature matrix
  states    -> PCA (6 PC, 95% var) + k-means (k=4) state labels + characterisation
  occupancy -> pre-ictal vs baseline state occupancy + circular-shift onset null
  trajectory-> per-seizure 10-min dominant-state trajectory
  timeline  -> continuous window-dominant state timeline (24 h / 2 d / 1 wk)
  figures   -> dark-theme static figures (incl. the null-method intermediaries)
  anim      -> shared-PCA-space pre-ictal trajectory animations (lead / all)
"""
