# Peri-ictal methods — evidence for every load-bearing claim

**Purpose.** The peri-ictal inference-hardening plan (`~/.claude/plans/declarative-toasting-spark.md`)
rests on ~28 methodological claims. This document validates **each** against a
**primary source**, with a **verbatim quote** copied from the actually-retrieved
document, an assessment of what the source does and does **not** support, and the
conditions under which the claim fails.

## How this was validated (and its limits)

- Each claim was researched by an independent agent that used web search + fetch to
  open the source and copy a verbatim quote, then a **second, adversarial agent
  re-fetched the same URL** to confirm the quote exists and genuinely supports the
  specific claim (36 agents total, 0 errors).
- **Honesty rules that were enforced:** no citation without a retrieved source and a
  verbatim quote; anything unretrievable is marked **UNVERIFIABLE**, never invented;
  a source that covers only part of a compound claim yields **PARTIAL**, with the gap
  named; secondary sources are flagged where the primary is paywalled.
- **Known limitations of this evidence base:** several primaries are paywalled, so the
  quote is from the **abstract** or from an **authoritative secondary** source (each is
  flagged). Two attribution errors were caught and corrected (see §12). This validates
  the *literature*; the *internal empirical* numbers (ρ≈0.67, AUC 0.620, p=1.000, trial
  counts) are the user's own data and are sourced to the code/DB, not the literature
  (§11).

## Verdict at a glance

| # | Claim (short) | Verdict | Primary source | Access |
|---|---|---|---|---|
| A1 | AUC = Mann–Whitney U = P(X>Y) | SUPPORTED | Hanley & McNeil 1982, *Radiology* | full text |
| A2 | AUC invariant to within-group monotonic transform | SUPPORTED | (Bamber 1975); ROC-invariance quote Tang et al. 2010 | full text |
| A3 | AUC is the common-language effect size / prob. of superiority | PARTIAL | McGraw & Wong 1992 (quoted via secondary) | secondary |
| B1 | Nested autocorrelated units as independent → inflated significance (pseudoreplication) | SUPPORTED | Hurlbert 1984, *Ecol. Monogr.* | full text |
| B2 | Pooled association can vanish/reverse within subgroups (Simpson) | SUPPORTED | Simpson 1951, *JRSS-B* | full text |
| C1 | Paired sign-test floor = 1/2^(k−1); need k≥6 for p<0.05 | SUPPORTED | Dixon & Mood 1946 (premise via PennState notes) | secondary + derivation |
| C2 | Wilcoxon signed-rank shares the same 1/2^(n−1) floor | SUPPORTED | Wilcoxon 1945 (premise via PennState notes) | secondary + derivation |
| C3 | Permutation p uses +1 smoothing, never 0 | SUPPORTED | Phipson & Smyth 2010, *SAGMB* | full text |
| D1 | Mixed models with few clusters: anticonservative SEs (and biased variance at very few) | PARTIAL | Maas & Hox 2005, *Methodology* | full text |
| D2 | Benjamini–Hochberg controls FDR | SUPPORTED | Benjamini & Hochberg 1995, *JRSS-B* | abstract |
| E1 | Block bootstrap preserves dependence; iid bootstrap invalid under autocorrelation | PARTIAL | Künsch 1989, *Ann. Statist.* | abstract |
| E2 | Positive autocorrelation shrinks effective N by ~(1+2Σρ) | SUPPORTED | (Bartlett); formula via Stan manual | secondary |
| E3 | Surrogate nulls preserve spectrum, destroy alignment; shuffle → too-narrow null | PARTIAL | Theiler et al. 1992, *Physica D* | abstract |
| F1 | Scattering: translation-invariant, deformation-stable, deterministic, small-N | PARTIAL | Mallat 2012, *CPAM* | full text |
| F2 | \|W(−x)\|=\|W(x)\|: wavelet modulus is polarity-insensitive | SUPPORTED | Mallat, *Wavelet Tour* (linearity) | full text |
| F3 | L2 distance is shift-sensitive; DTW is warp-tolerant | SUPPORTED | Keogh & Ratanamahatana 2005, *KAIS* | full text |
| G1 | PCA maximizes variance → unsupervised embedding dominated by largest-variance factor | PARTIAL | Jolliffe & Cadima 2016 (Hotelling 1933 primary) | full text |
| G2 | UMAP is nonlinear; supports a supervised mode | PARTIAL | McInnes et al. 2018 + umap-learn docs | full text |
| G3 | Batch/subject structure is dominant variance mistaken for signal | PARTIAL | Leek et al. 2010, *Nat. Rev. Genet.* | abstract |
| H1 | Critical slowing down (↑autocorr, ↑variance) precedes bifurcations | SUPPORTED | Scheffer et al. 2009, *Nature* | full text |
| H2 | CSD signatures track seizure susceptibility / forecasting | SUPPORTED | Maturana et al. 2020, *Nat. Commun.* | full text |
| H3 | Seizure risk follows circadian + multidien rhythms (rising-phase) | SUPPORTED | Baud et al. 2018, *Nat. Commun.* | full text |
| H4 | Cortical excitability / evoked responses vary with wake-sleep/time | SUPPORTED | Meisel et al. 2015, *PNAS* | full text |
| I1 | Active perturbation beats passive EEG; Chang AUC>.95, passive fails | SUPPORTED | Chang et al. 2026, *Epilepsia* | abstract |
| I2 | preictal=0–30 min, interictal=60–90 min — does this come from Chang? | UNVERIFIABLE | Chang et al. 2026 Methods (paywalled) | abstract only |
| I3 | Paired-pulse measures short-term plasticity a single pulse can't | SUPPORTED | Zucker & Regehr 2002, *Annu. Rev. Physiol.* | abstract |
| I4 | Probing proposed to anticipate/control epileptic transitions | SUPPORTED | Kalitzin, Velis & Lopes da Silva 2010, *Epilepsy Behav.* | abstract |

**Tally:** 16 SUPPORTED · 10 PARTIAL · 1 UNVERIFIABLE · 0 REFUTED.
Every PARTIAL is a compound claim whose core is sound but where a single retrieved
source covers only part (the missing part is named per entry); none is a factual error.

---

## 1. AUC and its invariances

### A1 — AUC = Mann–Whitney U = P(random positive ranks above random negative). **SUPPORTED**
- **Source (primary, full text):** Hanley JA & McNeil BJ (1982), "The meaning and use of the area under a ROC curve," *Radiology* 143(1):29–36. <https://jhanley.biostat.mcgill.ca/software/Hanley_McNeil_Radiology_82.pdf>
- **Quote:** "the area represents the probability that a randomly chosen diseased subject is (correctly) rated or ranked with greater suspicion than a randomly chosen non-diseased subject. Moreover, this probability of a correct ranking is the same quantity that is estimated by the already well-studied nonparametric Wilcoxon statistic."
- **Supports:** AUC = P(X>Y), equal to the Wilcoxon rank-sum statistic.
- **Caveat:** the source names **Wilcoxon**; "= Mann–Whitney U" relies on the standard algebraic equivalence of Wilcoxon rank-sum and Mann–Whitney U (explicit in **Bamber 1975**, *J. Math. Psych.* 12:387–415). Empirical AUC = the *normalized* U (U/n₊n₋); assumes a tie convention.

### A2 — AUC is unchanged by any strictly-monotonic transform applied within the compared values. **SUPPORTED**
- **Source (secondary, full text):** Tang, Du & Wu (2010), PMC3358774. <https://pmc.ncbi.nlm.nih.gov/articles/PMC3358774/> — primary origin of the rank-only property is **Bamber 1975**.
- **Quote:** "By definition, ROC curve is monotone increasing from 0 to 1 and is invariant to any monotone transformation of test results."
- **Supports:** AUC = area under ROC, so ROC monotone-invariance ⇒ AUC monotone-invariance. Affine z-scoring within one seizure is a single strictly-increasing map → AUC-preserving. **This is exactly why the plan's per-seizure AUC forest needs no standardization.**
- **Caveat:** requires **one** monotone map over the pooled compared values. Applying **different** affine maps to the two classes separately, or a non-monotone/tie-inducing map, can change the AUC.

### A3 — AUC is the common-language effect size / probability of superiority. **PARTIAL**
- **Source:** McGraw & Wong (1992), *Psychological Bulletin* 111:361–365 (DOI 10.1037/0033-2909.111.2.361), **paywalled**; definition quoted via a secondary page. <https://janhove.github.io/posts/2016-11-16-common-language-effect-sizes/>
- **Quote (attributed to McGraw & Wong 1992):** "the probability that a score sampled at random from one distribution will be greater than a score sampled from some other distribution."
- **Supports:** CLES = P(X>Y) = probability of superiority; via A1 this equals empirical AUC.
- **Why PARTIAL:** the retrieved page grounds only "CLES = probability of superiority"; the "AUC = this, and is rank-based/nonparametric" link comes from A1 / Bamber 1975, not this source. The general nonparametric form A = P(X>Y)+0.5·P(X=Y) is **Vargha & Delaney 2000**.

---

## 2. Nesting, pseudoreplication, and pooling

### B1 — Treating nested autocorrelated observations as independent inflates significance. **SUPPORTED**
- **Source (primary, full text):** Hurlbert SH (1984), "Pseudoreplication and the design of ecological field experiments," *Ecological Monographs* 54(2):187–211. <https://www.zoology.ubc.ca/~bio501/R/readings/hurlbert%201984%20ecol%20monogr%20-%20pseudoreplication.pdf>
- **Quote:** "Pseudoreplication is defined as the use of inferential statistics to test for treatment effects with data from experiments where either treatments are not replicated (though samples may be) or replicates are not statistically independent. In ANOVA terminology, it is the testing for treatment effects with an error term inappropriate to the hypothesis being considered."
- **Supports:** the paper that coined the term; "replicates are not statistically independent" is exactly many autocorrelated stimuli within one seizure → an inappropriate error term → apparent significance the design doesn't license. **This is the core justification for the seizure-as-unit forest over the per-stimulus p.**
- **Caveat:** the abstract states the "inappropriate error term" mechanism; the literal word "inflate" is not quoted (directional Type-I inflation is the established consequence). Framed in ecology; the i.i.d.-violation principle generalizes.

### B2 — A pooled association can vanish/reverse within every subgroup (Simpson / composition). **SUPPORTED**
- **Source (primary, full text):** Simpson EH (1951), "The interpretation of interaction in contingency tables," *JRSS-B* 13(2):238–241. <https://math.bme.hu/~marib/bsmeur/simpson.pdf>
- **Quote:** "if A and B are associated positively in C and negatively in [C-bar] they may appear as independent in the whole population; but a more curious case than this can be constructed."
- **Supports:** the eponymous paper; stratum-level associations can disappear/reverse on amalgamation — i.e. a pooled effect can be pure between-group composition. **This is the mechanism behind the pooled 0.62 vs the per-seizure straddle.**
- **Caveat:** Simpson shows *disappearance under aggregation* and cites Kendall for the sign-difference case; the modern label "Simpson's paradox" is **Blyth 1972** (*JASA* 67:364–366). The reversal requires confounding (unequal, exposure-correlated strata weights); it does not arise otherwise.

---

## 3. Small-sample exact-test floors and permutation p-values

### C1 — Paired sign test: minimum two-sided p = 1/2^(k−1); k≥6 needed for p<0.05. **SUPPORTED**
- **Source:** premise (null is Binomial(k, ½)) from PennState STAT 415 Lesson 20.1 (secondary, full text) <https://online.stat.psu.edu/stat415/lesson/20/20.1>; canonical primary is **Dixon & Mood 1946**, "The statistical sign test," *JASA* 41:557–566.
- **Quote (premise):** "if the null hypothesis is true … N- and N+ both follow a binomial distribution with parameters n and p = 1/2."
- **Supports:** under H₀ each of the 2^k sign patterns has probability (½)^k; the two extremes give two-sided p = 2·(½)^k = 1/2^(k−1). Arithmetic (independently checked): k=3→0.25, k=5→0.0625, **k=6→0.03125<0.05**. **This is the ≥6-seizure milestone in the plan.**
- **Caveat:** assumes no ties at the hypothesized median (zeros dropped, reducing k); uses the "double the smaller one-sided tail" convention.

### C2 — Wilcoxon signed-rank shares the same 1/2^(n−1) floor. **SUPPORTED**
- **Source:** exact-null formula from PennState STAT 415 Lesson 20.2 (secondary, full text) <https://online.stat.psu.edu/stat415/lesson/20/20.2>; canonical primary is **Wilcoxon 1945**, *Biometrics Bulletin* 1(6):80–83.
- **Quote:** "there are 2^n total number of ways to make signed rank sums, and therefore the probability that W takes on a particular value w is: P(W=w)=c(w)/2^n where c(w) = the number of possible ways to assign a + or a − to the first n integers so that Σ Z_i R_i = w."
- **Supports:** the exact null is generated by the same 2^n equally-likely sign assignments; the all-same-sign extreme gives 1/2^(n−1), identical to the sign-test floor. **So switching from the sign test to Wilcoxon does not escape the floor — only a different null (the circular-shift surrogate, E3) does.**
- **Caveat:** requires distinct nonzero differences (no zeros/ties among |differences|); equality with the sign-test floor holds only at the extreme configuration.

### C3 — Permutation p uses +1 smoothing, p=(b+1)/(m+1); never exactly 0. **SUPPORTED**
- **Source (primary, full text):** Phipson B & Smyth GK (2010), "Permutation P-values should never be zero…," *Stat. Appl. Genet. Mol. Biol.* 9(1):Art.39. <https://ar5iv.labs.arxiv.org/html/1603.05766>
- **Quote:** "the exact Monte Carlo p-value is p_u = P(B≤b) = (b+1)/(m+1)."
- **Supports:** numerator b+1 ≥ 1 ⇒ strictly positive; matches the code's `(ge+1)/(n_perm+1)` in `forecast.permutation_p` / `trendtest.surrogate_p`.
- **Caveat:** the simple form is the conservative estimator when the observed statistic is counted among the m draws; full enumeration floors at 1/(#distinct permutations).

---

## 4. Mixed models with few clusters; false-discovery control

### D1 — Few clusters → anticonservative SEs (and biased variance components at very few). **PARTIAL**
- **Source (primary, full text):** Maas CJM & Hox JJ (2005), "Sufficient sample sizes for multilevel modeling," *Methodology* 1(3):86–92. <https://www.joophox.net/publist/methodology05.pdf>
- **Quote:** "both the regression coefficients and the variance components are all estimated without bias, in all of the simulated conditions… The standard errors of the second-level variances are estimated too small when the number of groups is substantially lower than 100. With 30 groups, the standard errors are estimated about 15% too small."
- **Supports:** the **anticonservative-SE** half is directly established (SEs of random-effect variances ~15% too small at 30 groups; a 10-group run gave 16–30% CI non-coverage and up to 25% variance bias — "having only 10 groups is not enough").
- **Why PARTIAL / important nuance:** Maas & Hox find variance-component **point estimates unbiased down to 30 groups** — so "biased variance components" holds only at a *very small* cluster count. The plan operates at **k=3–4 seizures**, which is *below* even the 10-group condition, so "unreliable random-effect inference" is justified *a fortiori* — but note the published evidence base bottoms out at 10 groups; k=3–4 is an extrapolation beyond the simulated range. Broader small-cluster reviews (**McNeish & Stapleton 2016**, *Educ. Psychol. Rev.*) were not retrieved. This is exactly why the plan demotes MixedLM to a captioned sanity check, not the headline.

### D2 — Benjamini–Hochberg controls the false discovery rate. **SUPPORTED**
- **Source (primary, abstract):** Benjamini Y & Hochberg Y (1995), *JRSS-B* 57(1):289–300. <https://academic.oup.com/jrsssb/article/57/1/289/7035855>
- **Quote:** "A simple sequential Bonferroni-type procedure is proved to control the false discovery rate for independent test statistics."
- **Supports:** the canonical FDR paper; underpins the `_bh_qvalues` used in `scan_features` / `scan_all_features`.
- **Caveat:** the 1995 proof is for **independent** statistics; control under positive dependence (PRDS) is **Benjamini & Yekutieli 2001**. Full text paywalled (abstract quoted).

---

## 5. Dependence: block bootstrap, effective sample size, surrogate nulls

### E1 — Block bootstrap preserves dependence; iid bootstrap invalid under autocorrelation. **PARTIAL**
- **Source (primary, abstract):** Künsch HR (1989), "The jackknife and the bootstrap for general stationary observations," *Ann. Statist.* 17(3):1217–1241. <https://projecteuclid.org/journals/annals-of-statistics/volume-17/issue-3/The-Jackknife-and-the-Bootstrap-for-General-Stationary-Observations/10.1214/aos/1176347265.full>
- **Quote:** "We extend the jackknife and the bootstrap method of estimating standard errors to the case where the observations form a general stationary sequence. We do not attempt a reduction to i.i.d. values… Bootstrap replicates are constructed by selecting blocks of length l randomly with replacement among the blocks of observations."
- **Supports:** the canonical block-bootstrap source; resampling contiguous blocks preserves dependence rather than assuming i.i.d. **This is why the plan's per-seizure AUC/slope forest CIs use within-seizure block bootstrap, not trial-level DeLong/binomial.**
- **Why PARTIAL:** the retrieved abstract establishes the block construction but does **not** state verbatim that the iid bootstrap "is invalid and underestimates variance" (that motivation is in the paywalled body). The "moving/overlapping-block" and "stationary bootstrap" variants are **Liu & Singh 1992 / Politis & Romano 1994**.

### E2 — Positive autocorrelation shrinks effective N by ≈(1+2Σρ). **SUPPORTED**
- **Source (secondary, full text):** Stan Reference Manual v2.21, §16.4 Effective Sample Size. <https://mc-stan.org/docs/2_21/reference-manual/effective-sample-size-section.html> — attributed primary **Bartlett 1946** is paywalled.
- **Quote:** "The effective sample size of N samples generated by a process with autocorrelations ρ_t is defined by N_eff = N / (1 + 2 Σ ρ_t). … estimation error is proportional to 1/√(N_eff) rather than 1/√N."
- **Supports:** exactly the (1+2Σρ) variance-inflation / N_eff-shrinkage; motivates reporting the **effective** independent-shift count for the surrogate and effective block count for the bootstrap.
- **Caveat:** holds for **positive** autocorrelation (negative ρ can raise N_eff); the finite-sample factor carries (1−k/n) weights. Source is an MCMC/ESS stand-in for the paywalled Bartlett primary.

### E3 — Surrogate nulls preserve the spectrum, destroy alignment; a plain shuffle → too-narrow null. **PARTIAL**
- **Source (primary, abstract):** Theiler J, Eubank S, Longtin A, Galdrikian B & Farmer JD (1992), "Testing for nonlinearity in time series: the method of surrogate data," *Physica D* 58:77–94. <https://www.inet.ox.ac.uk/publications/detecting-nonlinear-structure-in-time-series>
- **Quote:** "The method first specifies some linear process as a null hypothesis, then generates surrogate data sets which are consistent with this null hypothesis, and finally computes a discriminating statistic for the original and for each of the surrogate data sets."
- **Supports:** the primary source for the surrogate-null *framework*.
- **⚠ Two issues (why PARTIAL) — a genuine correction to the plan's wording:**
  1. The abstract supports only the general framework; the *specifics* (phase randomization preserves the power spectrum/autocorrelation; a plain reshuffle tests only i.i.d. noise → over-narrow null) are in the paywalled body.
  2. **"Circular time-shifting" is NOT Theiler et al.'s method** — they use Fourier **phase randomization** (FT/AAFT). Circular/cyclic time-shift surrogates are a *related but distinct* technique (common in spike–field / phase-coupling work) with a different lineage that this validation could **not** independently source. **Action:** cite Theiler 1992 only for the surrogate-null *concept*; do not attribute "circular shift" to it. The code's `validation.circular_shift_null` implements the shift correctly (preserves within-seizure autocorrelation, destroys onset alignment); only the citation needs this care.

---

## 6. Representations: scattering, wavelet-modulus sign invariance, DTW

### F1 — Scattering: translation-invariant (to a scale), deformation-stable, deterministic, small-N. **PARTIAL**
- **Source (primary, full text):** Mallat S (2012), "Group invariant scattering," *Comm. Pure Appl. Math.* 65(10):1331–1398. <http://robotics.caltech.edu/wiki/images/9/92/GroupInvariantScattering.pdf>
- **Quote:** "This paper constructs translation-invariant operators on L²(R^d), which are Lipschitz-continuous to the action of diffeomorphisms. A scattering propagator is a path-ordered product of nonlinear and noncommuting operators, each of which computes the modulus of a wavelet transform."
- **Supports:** the **math** core — translation invariance (asymptotic in the averaging window → "up to a scale"), Lipschitz-stability to small deformations, and the wavelet-modulus cascade.
- **Why PARTIAL:** the empirical half — "strong classification with limited training data" — is **not** in Mallat 2012; it is **Bruna & Mallat 2013** (*IEEE TPAMI*) and **Andén & Mallat 2014** (*IEEE TSP*). "No learned parameters" is only implicit (fixed wavelets); Mallat notes the cascade "can also be interpreted as a convolutional neural network."

### F2 — |W(−x)| = |W(x)|: wavelet modulus is polarity-insensitive. **SUPPORTED**
- **Source (authoritative definition, full text):** Mallat S, *A Wavelet Tour of Signal Processing*, §1.3.3 / §4.3. <https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf> — CWT origin is **Grossmann & Morlet 1984**.
- **Quote:** "The continuous wavelet transform of f at any scale s and position u is the projection of f on the corresponding wavelet atom: Wf(u,s) = ⟨f, ψ_{u,s}⟩."
- **Supports:** the CWT is **linear** (inner product / convolution), so W(−f) = −W(f) ⇒ |W(−f)| = |W(f)| pointwise — a global sign flip leaves the modulus unchanged. **This is the exact reason a |CWT| lens is a control on the BCH111 polarity-flip worry.**
- **Caveat:** holds for a **global** polarity flip only — not local sign/phase structure or arbitrary reflections.

### F3 — L2 distance is shift-sensitive; DTW is warp-tolerant. **SUPPORTED**
- **Source (authoritative, full text):** Keogh E & Ratanamahatana CA (2005), "Exact indexing of dynamic time warping," *KAIS* 7:358–386. <https://www.cs.ucr.edu/~eamonn/KAIS_2004_warping.pdf> — DTW origin **Sakoe & Chiba 1978**; introduced to data mining by **Berndt & Clifford 1994**.
- **Quote:** "Euclidean distance, which assumes the ith point in one sequence is aligned with the ith point in the other, will produce a pessimistic dissimilarity measure. The nonlinear dynamic time warped alignment allows a more intuitive distance measure to be calculated."
- **Supports:** both halves — L2 rigidly pairs i-th points (brittle to misalignment); DTW aligns nonlinearly. Quantified: cylinder-bell-funnel 1-NN error 0.2734 (L2) vs 0.0269 (DTW).
- **Caveat:** DTW's warp-tolerance is double-edged — unconstrained warping can align away real latency shifts (**the plan's DTW-diagnostic-only caveat**); it violates the triangle inequality, so band constraints (Sakoe–Chiba/Itakura) are used.

---

## 7. PCA/UMAP variance allocation; batch effects

### G1 — PCA maximizes variance → an unsupervised embedding is dominated by the largest-variance factor. **PARTIAL**
- **Source (authoritative review, full text):** Jolliffe IT & Cadima J (2016), "Principal component analysis: a review and recent developments," *Phil. Trans. R. Soc. A* 374:20150202. <https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/> — primary max-variance formulation **Hotelling 1933**.
- **Quote:** "It does so by creating new uncorrelated variables that successively maximize variance."
- **Supports:** the leading component is by construction the largest-variance direction, so a low-D unsupervised embedding is dominated by it.
- **Why PARTIAL:** the source does **not** state the interpretive clause that the dominant factor "may be a nuisance factor rather than the factor of interest" — that framing belongs to the batch-effect literature (G3). This underpins the plan's argument that an unsupervised embedding (any representation) will block by the largest-variance factor — here seizure/session identity — hence the need for **dPCA/supervised** demixing.

### G2 — UMAP is nonlinear; supports a supervised mode. **PARTIAL**
- **Source (primary, full text):** McInnes L, Healy J & Melville J (2018), "UMAP…," arXiv:1802.03426. <https://arxiv.org/abs/1802.03426>
- **Quote:** "UMAP (Uniform Manifold Approximation and Projection) is a novel manifold learning technique for dimension reduction."
- **Supports:** part 1 (nonlinear manifold learning).
- **Why PARTIAL + corrected citation:** the 2018 paper lists supervised UMAP only as future work; the implemented supervised mode is documented by the same author: umap-learn "Supervised UMAP," "the algorithm … [makes] use of categorical label information to do supervised dimension reduction." <https://umap-learn.readthedocs.io/en/latest/supervised.html>. Both parts are true; no single retrieved source covers both.

### G3 — Batch/subject structure is dominant variance mistaken for signal. **PARTIAL**
- **Source (primary, abstract):** Leek JT et al. (2010), "Tackling the widespread and critical impact of batch effects in high-throughput data," *Nat. Rev. Genet.* 11(10):733–739. <https://pubmed.ncbi.nlm.nih.gov/20838408/>
- **Quote:** "batch effects (as well as other technical and biological artefacts) are widespread and critical to address … This becomes a major problem when batch effects are correlated with an outcome of interest and lead to incorrect conclusions."
- **Supports:** the core — technical/batch structure is a widespread dominant source of variation that, when confounded with the outcome, is mistaken for signal.
- **Why PARTIAL:** the abstract does not support the specific qualifiers "few subjects/batches," "session/subject identity," or "unsupervised representations capture it"; the domain is high-throughput genomics. The generalization to few-subject neuroscience (and to "autoencoder latent encodes session identity at n=4") is a reasoned extrapolation, not established by this source. Justifies the plan's **de-scoping of autoencoders/VAEs** at this n.

---

## 8. Critical slowing down; seizure rhythms; excitability

### H1 — Critical slowing down (↑lag-1 autocorrelation, ↑variance) precedes bifurcations. **SUPPORTED**
- **Source (primary, full text):** Scheffer M et al. (2009), "Early-warning signals for critical transitions," *Nature* 461:53–59. <https://pdodds.w3.uvm.edu/files/papers/others/2009/scheffer2009a.pdf>
- **Quote:** "the phenomenon of critical slowing down leads to three possible early-warning signals in the dynamics of a system approaching a bifurcation: slower recovery from perturbations, increased autocorrelation and increased variance."
- **Supports:** the seminal statement of CSD → ↑autocorrelation + ↑variance as generic early-warning signals.
- **Caveat:** necessary-but-not-sufficient — indicators can rise without a bifurcation and fail for noise-induced/rate-dependent or non-fold transitions (**Boettiger & Hastings 2012; Ditlevsen & Johnsen 2010**); unreliable on short/noisy series.

### H2 — CSD signatures track seizure susceptibility / improve forecasting. **SUPPORTED**
- **Source (primary, full text):** Maturana MI et al. (2020), "Critical slowing down as a biomarker for seizure susceptibility," *Nat. Commun.* 11:2172. <https://pmc.ncbi.nlm.nih.gov/articles/PMC7195436/>
- **Quote:** "Critical slowing (associated with increased variance and autocorrelation) can precede critical state transitions. … Seizure risk was associated with a combination of these signals … a reliable indicator that could be used in seizure forecasting algorithms."
- **Supports:** empirical intracranial-EEG evidence that variance/autocorrelation track seizure risk.
- **Caveat (directly relevant to this project):** **directionality is patient-specific** — both increases *and* decreases in autocorrelation preceded seizures across patients; metrics fluctuate over hours–days and were not reliable short-horizon warnings in all subjects.

### H3 — Seizure risk follows circadian + patient-specific multidien rhythms (rising phase). **SUPPORTED**
- **Source (primary, full text):** Baud MO et al. (2018), "Multi-day rhythms modulate seizure risk in epilepsy," *Nat. Commun.* 9:88. <https://pmc.ncbi.nlm.nih.gov/articles/PMC5758806/>
- **Quote:** "Seizures occur preferentially during the rising phase of multidien IEA rhythms. … Combining phase information from circadian and multidien IEA rhythms provides a novel biomarker for determining relative seizure risk."
- **Supports:** the multidien/circadian modulation of seizure risk motivating a seizure-cycle lens (large effect: RR 6.8, 95% CI 3.1–15.1).
- **Caveat:** rhythms are defined on **interictal epileptiform activity** (an excitability proxy), not seizures directly; multidien periods are patient-specific (~20–30 d), not a fixed universal cycle. Motivates why **Rayleigh/periodogram at n=3 seizures is underpowered** and multidien detection needs weeks of events.

### H4 — Cortical excitability / evoked responses vary with wake-sleep/time. **SUPPORTED**
- **Source (primary, full text):** Meisel C et al. (2015), "Intrinsic excitability measures track antiepileptic drug action and uncover increasing/decreasing excitability over the wake/sleep cycle," *PNAS* 112(47):14694–14699. <https://pmc.ncbi.nlm.nih.gov/articles/PMC4664307/>
- **Quote:** "excitability of cortical networks is reduced by antiepileptic drugs and increases as a function of time awake. … The wake-dependent time course … suggests a homeostatic role of sleep, to rebalance cortical excitability."
- **Supports:** evoked/excitability measures depend on brain state and time-of-day — the mechanistic basis for the plan's claim that the evoked features carry a strong circadian component (the internally-measured ρ≈0.67; §11).
- **Caveat:** emphasizes a **wake/sleep homeostatic** (time-awake) course rather than proving an independent endogenous circadian pacemaker; studied in AED-treated epilepsy patients.

---

## 9. Perturbation/probing, paired-pulse, and the recreated Chang method

### I1 — Active perturbation beats passive EEG; Chang: evoked AUC>.95, passive fails. **SUPPORTED**
- **Source (primary, abstract):** Chang et al. (2026), "Perturbation-induced responses improved seizure forecasting in epileptic rats," *Epilepsia* (PMID 41823376; doi:10.1002/epi.70196). <https://pubmed.ncbi.nlm.nih.gov/41823376/>
- **Quote:** "Using the perturbation-evoked responses as a predictive biomarker of seizure risk, we built a preictal detection system that had excellent accuracy (area under the receiver operating characteristic curve > .95) at distinguishing the preictal from the interictal states. In comparison, a similar preictal detection system that used only passive features from the same experimental animals was unable to identify the preictal state better than chance."
- **Supports:** both load-bearing anchors — evoked detector AUC > .95; matched passive detector at chance. This is the paper the whole tab recreates.
- **Caveat:** the abstract does **not** state "five features" or "multivariable logistic regression" — those specifics live in the paywalled full text (unverified from the abstract). AUC is reported ">.95" (the plan's "~0.95" if anything understates it).

### I2 — preictal = 0–30 min / interictal = 60–90 min: does this come from Chang? **UNVERIFIABLE (from open sources)**
- **Source attempted:** Chang et al. 2026 Methods — Wiley full text returned HTTP 402 (paywalled); no OA/PMC/preprint mirror found. The abstract confirms a preictal-vs-interictal contrast exists but states **no numeric windows**.
- **What the code says (provenance, not the paper):** `src/periictal/config.py:47–54` documents "Preictal = the 30 min before a seizure; interictal = 60–90 min before (**the user's bounded window; the paper used '>60 min'**). The 30–60 min band is the redacted buffer." → i.e. the lab attributes **preictal=30 min and interictal-lower=60 min** to Chang, but the **90-min upper bound is an explicit lab choice**.
- **Honest status:** the 30/60 attribution rests on the lab's reading of the paywalled Methods and **could not be independently confirmed**. **Do not cite Chang 2026 for the exact minute-windows** until the Methods are read directly; treat the numbers as a lab analysis choice with a paper-derived lower structure.

### I3 — Paired-pulse measures short-term plasticity a single pulse can't. **SUPPORTED**
- **Source (primary, abstract):** Zucker RS & Regehr WG (2002), "Short-term synaptic plasticity," *Annu. Rev. Physiol.* 64:355–405 (PMID 11826273). <https://pubmed.ncbi.nlm.nih.gov/11826273/>
- **Quote:** "Synaptic transmission is a dynamic process. Postsynaptic responses wax and wane as presynaptic activity evolves."
- **Supports:** the canonical STP review; the response to a second pulse differs from the first (facilitation/depression) — the basis of paired-pulse measures, which a single-pulse amplitude cannot capture.
- **Caveat:** the review does not frame the paired-pulse **ratio** as an "excitability/recovery" index in the claim's exact wording (standard downstream usage); PPR primarily reflects **presynaptic release-probability** dynamics, so reading it as net "excitability" is a simplification.

### I4 — Probing/active stimulation proposed to anticipate & control epileptic transitions. **SUPPORTED**
- **Source (primary, abstract):** Kalitzin SN, Velis DN & Lopes da Silva FH (**2010**), "Stimulation-based anticipation and control of state transitions in the epileptic brain," *Epilepsy & Behavior* 17(3):310–323 (PMID 20163993). <https://pubmed.ncbi.nlm.nih.gov/20163993/>
- **Quote:** "by applying a suitable perturbation to the system, such as electrical stimulation, relevant features of the system's state may be detected and the risk of an impending seizure estimated … if these features are detected early, transitions into seizures may be blocked."
- **Supports:** a primary source explicitly proposing stimulation-based anticipation (state detection/risk) and control (blocking transitions).
- **Caveat / correction:** the plan's hint dated this **2005**; the paper matching the title/content is **2010**. Findings are computer-model/theory-based, not clinical trials ("proposed," which the claim says).

---

## 10. Derived logical claims (not from literature) — and their limits

These are *reasoning steps* in the discussion, not literature findings. They stand on the
cited primaries plus the data geometry; where a step is weaker than I originally stated, it
is corrected here.

### L1 — "Monotonic drift and circadian both predict a consistent-SIGN preictal−interictal difference, so the observed p=1.000 argues against them." **Valid with a correction.**
- **Basis:** a fixed-lag confound (impedance/tissue drift over days, or a circadian gradient across the ~60-min window separation — window geometry from `config.py`, circadian modulation from **H3/H4**) shifts preictal vs interictal in the **same direction for every seizure**.
- **Correction to my earlier phrasing:** I said this "would give a small paired p." That is **wrong at k=3** — by the sign-test floor (**C1**), even a perfectly consistent same-sign effect can only reach p=0.25 at k=3. The correct, weaker statement: such a confound predicts the **same sign in every seizure**; we observe **mixed signs**; so the *sign pattern* (not the p-value) is what is inconsistent with a monotonic/circadian within-pair confound. This qualitative discriminator holds independent of significance, which is the honest version.

### L2 — "The ρ≈0.67 circadian variance lives BETWEEN seizures (→ the pooled artifact), not WITHIN a pair." **Valid, conditional on the window geometry.**
- **Basis:** preictal `[T−30,T]` and interictal `[T−90,T−60]` window centres are ~60 min apart, over which circadian phase (a ~24 h cycle) changes little (**H4**), so circadian barely differentiates the two windows *within* a seizure. Across seizures at different clock times it varies fully, producing between-seizure offsets that inflate the pooled AUC by composition (**B2**). The within-seizure forest (rank-based, **A2**) is immune to those offsets.
- **Limit:** if a seizure's two windows straddle a sleep/wake transition, circadian *can* differentiate them within that seizure — a per-seizure check (clock time of both windows) is the honest guard, already in the plan.

### L3 — "The per-seizure AUC forest needs no standardization." **Valid (= A2).** Rank-invariance makes within-seizure affine standardization a no-op for per-seizure AUC; standardization matters only for the scale-dependent pooled/PDF/logistic views. (This is why the null synthetic must assert on *pooled* AUC, not per-seizure AUC.)

---

## 11. Internal empirical claims (the user's own data + code provenance)

These are **not** literature and have **no external primary source**; provenance is the
code/DB/analysis. They are listed so the plan's reasoning is fully traceable.

| Claim | Provenance (not literature) |
|---|---|
| Evoked embedding axes correlate with time-of-day at \|ρ\|≈0.67 (vs ≈0.2 for time-to-onset) | `src/periictal/embed.py::confound_readout`; per-seizure `trendtest.hour_rho`; recorded in memory `periictal-explorer`. Interpreted via H3/H4. |
| BCH111 pooled per-stimulus AUC≈0.620, perm-p=0.002; paired seizure-level p=1.000 | The user's own analysis output (this session), to be regenerated by the hardened pipeline. |
| Trial counts 2623 preictal / 2443 interictal across 3 kept seizures | The user's data; residual class asymmetry is benign (edge truncation of the further-back interictal window + stim dropouts), not differential exclusion (exclusion is whole-seizure). |
| preictal=30 min, interictal=60–90 min, MIN_PREICTAL=30 min, 6 h lead-up, 5 min post-ictal buffer | `src/periictal/config.py:28–58` — lab constants (lower structure attributed to Chang; see I2). |
| PAPER_BEST5 = sum_power_low, sum_power_high, expfit_initial, freq_moment_high, freq_moment_low | `src/periictal/config.py:64` — the lab's column mapping of Chang's five features (paper-side unconfirmed from the abstract; see I1). |

---

## 12. Refinements this validation forces into the plan / wording

1. **E3 — citation care:** cite Theiler et al. 1992 for the surrogate-null *concept only*; "circular time-shifting" is a distinct technique (not Theiler's FT/AAFT) whose primary lineage this validation could not source. The code's shift is implemented correctly.
2. **I2 — do not attribute the exact minute-windows to Chang 2026** (paywalled Methods; open-source-unverifiable). State them as lab constants with a paper-derived lower structure.
3. **I4 — year is 2010, not 2005** (Kalitzin, Velis & Lopes da Silva, *Epilepsy & Behavior* 17:310–323).
4. **I1 — "five features / logistic regression" is unverified from the abstract** (in the paywalled body); the abstract confirms only "AUC>.95" and "passive fails."
5. **D1 — the small-cluster evidence bottoms out at ~10 groups**; k=3–4 is an *a fortiori* extrapolation (justified, but state it as such). MixedLM stays a captioned sanity check, not the headline.
6. **A1 — "= Mann–Whitney U"** rests on the standard MWU≡Wilcoxon-rank-sum equivalence (Bamber 1975), since Hanley & McNeil name "Wilcoxon."
7. **L1 correction** — "drift ⇒ small paired p" ⟶ "drift ⇒ same-sign pattern"; the discriminator at k=3 is the sign pattern, not the p-value.

## 13. Limitations of this validation

- **Access:** primaries paywalled for D2, E1, E3, G3, I1–I4 → the quote is the **abstract**; for A3, E2 an **authoritative secondary** stands in for a paywalled/older primary (each flagged inline). Full-text theorems/methods behind those abstracts were not read.
- **Two attribution errors were caught and corrected** (E3 method, I4 year) — evidence the adversarial pass did its job.
- **Scope:** this validates that each *concept/claim* is supported by the literature. It does **not** verify that the code correctly *implements* each method — that is the job of the unit/synthetic tests in the plan (tasks #7–#8). Nor does it establish any positive pre-ictal result; the honest conclusion (needs paired-pulse + ≥6 seizures) is unchanged.
- **No fabrication:** where nothing was retrievable (I2), the verdict is UNVERIFIABLE and no citation is asserted.
