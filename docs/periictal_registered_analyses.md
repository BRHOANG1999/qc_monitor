# Peri-ictal registered analyses

**Why this file exists.** With 3–4 seizures as the unit of inference and ~40 features
× several lenses, "scan everything and keep what looks consistent" is the path of least
resistance and the exact forking path the inference-hardening plan is built to close. So
**each analysis is pre-specified here — the null, the statistic, the unit, the split, and
the ONE primary feature — before its output is looked at.** One entry per analysis. Fill
the *pre-specified* block first; record the result verbatim afterward; never edit a
pre-specification after seeing the result (add a new dated entry instead).

Evidence for every method named here: `docs/periictal_methods_evidence.md`.

## Standing conventions (apply to every entry unless overridden)

- **Unit of inference:** the **seizure**, never the stimulus (pseudoreplication — B1). Pooled
  per-stimulus statistics are descriptive only, reported with the composition caveat (B2).
- **Small-k floor:** the sign-flip / signed-rank two-sided p floors at `1/2^(k-1)`
  (`trendtest.sign_flip_floor`; C1/C2): k=3→0.25, k=5→0.0625, **k≥6 for <0.05**. Every
  seizure-level verdict prints this next to it, so `p=1.000` at k=3 reads as "directions
  disagree", not "no effect". **Target: ≥6 clean seizures before a seizure-level claim is
  testable.**
- **Graded null (when a p is wanted at small k):** the **circular-shift surrogate**
  (`trendtest.circular_shift_surrogate_p`), which tests *alignment to onset* and preserves
  within-seizure autocorrelation (E3). It is **not** a power upgrade — report `n_eff_shifts`.
- **CIs:** within-seizure **block bootstrap** (`resample`, E1/E2), never trial-level; report
  the effective block count. If block count < ~20, flag the CI as noisy.
- **Standardization:** interictal-referenced within-seizure (`forecast.standardize_to_interictal`)
  for the scale-dependent views (pooled AUC / PDF / logistic). The per-seizure AUC forest is
  rank-invariant to it (A2/L3) and is reported on raw features.
- **Multiple features:** the all-feature scan is **exploratory**, BH-FDR-controlled (D2), and
  labelled as such; it does **not** carry the primary claim.
- **Exclusions:** whole-seizure, outcome-blind (bad-stim). The CONSORT ledger (plan item 0)
  reports seizures in → dropped (reason) → both-windows-available → per-seizure per-class
  counts, before any test is read.

## Primary-feature declaration — REQUIRES THE ANALYST'S SIGN-OFF

The primary statistic must rest on **one** feature named in advance. Picking the feature
*because it topped an exploratory scan* re-opens the forking path, so this cannot be chosen
from the data. Options, in decreasing defensibility:

1. **A theory-named feature fixed a priori** — e.g. a critical-slowing measure (lag-1
   `autocorrelation`, `variance`; H1/H2) or a Chang-paper feature. Most defensible.
2. **The Chang et al. 2026 best-5 set** (`config.PAPER_BEST5`) as a *pre-registered composite*
   (the multivariable model score), with the individual five as pre-specified secondaries.
   Uses external prior information, not this dataset's scan.
3. **A single feature frozen from a DIFFERENT animal/cohort's scan** (out-of-sample), never
   from the animal under test.

> **Decision (fill in and date before running):** primary feature = `__________`;
> justification = `__________`; secondaries = `__________`. Until this is signed off,
> all per-feature results are exploratory.

---

## Entry template (copy per analysis)

```
### <analysis name> — <date>
PRE-SPECIFIED (before looking):
- Question:
- Primary feature:                 (from the declaration above)
- Unit / n seizures:
- Statistic:                       (e.g. per-seizure AUC forest; per-seizure slope forest)
- Null / p:                        (circular-shift surrogate, n_surrogates; or "forest only, no p at n<6")
- LOSO split:                      (leave-one-seizure-out; how folds are formed)
- CI method:                       (within-seizure block bootstrap, n_boot)
- Decision rule stated in advance: (what result would count as support / against)
- Exclusions (outcome-blind):

RESULT (verbatim, after):
- Ledger (in → dropped → used, per class per seizure):
- Per-seizure forest (each seizure's estimate + CI):
- Aggregate / surrogate p + n_eff_shifts + sign-flip floor:
- Interpretation (bounded by k; "underpowered by construction / directions (in)consistent"):
```

---

## Registered entries

_(none yet — add the first entry here, pre-specified, before running the hardened BCH111
chronicStim analysis.)_
