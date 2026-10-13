# WineMood: Experimental Results (Phase 3)

All numbers come from the generated [`results-tables.md`](results-tables.md),
which `winemood exp report` recomputes from stored predictions. The setup is:

- **Intervals:** 95 % cluster-bootstrap intervals over split groups (2,000
  resamples).
- **Test sets:** `grouped_v1` has 23,991 reviews. `taster_holdout_v1` and
  `taster_random_v1` have 19,554 each.
- **Test access:** each test set was used once, for the pre-declared scope
  only (19 logged accesses on 2026-10-13, after the selection was frozen in
  commit `15bc2a6`).
- **Rules:** all hypotheses use the pre-registered rules in
  [`experimental-protocol.md`](experimental-protocol.md) §6, §9 and §10.

## 1. Headline (RQ1)

| Model (T1, test) | MAE [95% CI] | RMSE | R² | Within ±2 points |
|---|---|---|---|---|
| B0: training mean | 2.541 [2.518, 2.563] | 3.099 | 0.000 | 46.3 % |
| B1: length only | 2.012 [1.993, 2.032] | 2.534 | 0.331 | 57.4 % |
| B2: metadata (not text) | 1.862 [1.844, 1.881] | 2.349 | 0.426 | 61.1 % |
| **B3\*: word TF-IDF + Ridge (selected)** | **1.302 [1.289, 1.315]** | 1.655 | 0.715 | 78.1 % |
| N2: frozen MiniLM + Ridge | 1.694 [1.678, 1.711] | 2.132 | 0.526 | 65.6 % |
| N1: BiLSTM (mean of 5 seeds; SD 0.007) | 1.286 [1.274, 1.299] | 1.640 | 0.720 | 78.6 % |

Tasting-note text predicts the critic's score with a mean absolute error of
about **1.3 points** on the 80–100 scale. That is roughly half the error of
predicting the mean (2.54), and it explains about 72 % of the score variance.
For comparison, distinct reviews of the *same wine and vintage* in the corpus
differ by **1.69 points** on average (1,508 pairs). The model's error is
therefore of the same order as the disagreement between repeated expert
reviews. That comparison is descriptive: the repeat reviews may involve
different tasters and tasting occasions.

## 2. Pre-registered hypotheses

| | Hypothesis | Result (test) | Verdict |
|---|---|---|---|
| H1 | Text beats trivial cues | B3 − B0 = −1.239 [−1.262, −1.216]; B3 − length = −0.709 [−0.729, −0.690] | **Supported** |
| H2 | Sequence models beat bag-of-words | BiLSTM − B3 = −0.016 [−0.026, −0.007], which exceeds the seed SD (0.007). Frozen encoder − B3 = +0.392 [+0.376, +0.407] | **Supported for the BiLSTM only, with a very small effect**. The frozen encoder is clearly worse. |
| H3 | Part of the signal is reviewer-specific | Unseen-taster MAE − size-matched random MAE: TF-IDF +0.091 [+0.071, +0.112]; BiLSTM +0.052 [+0.033, +0.072] (exploratory) | **Supported**, with confounds (§4) |
| H4 | Drinking-window years act as a shortcut | Masking 19xx/20xx: +0.008 [+0.006, +0.010] MAE | Effect size reported (exploratory). The specific year values matter little (§5). |
| H5 | Keeping negations does not hurt | Keep-negations − legacy: +0.0028 [+0.0013, +0.0042]. The CI excludes 0, but the effect is below the seed SD (≈ 0.008). | **Supported under the §6 rule** (no detectable harm). The point estimate is a 0.2 % MAE increase. |

**How to read H2 (RQ2).** The BiLSTM's test advantage of 0.016 points
(1.2 %) passes the pre-registered bar. It is statistically detectable but
practically negligible. On validation the same comparison was +0.004
[−0.009, +0.018], with an interval that included 0, so the model-selection
rule correctly kept the simpler TF-IDF model. We do **not** revise the
selection after seeing test results, because that would be test-set tuning.
The frozen general-purpose sentence encoder underperforms a domain-fitted
bag-of-words model by a wide margin. Its selected regularisation (α = 0.01)
sat at the edge of the pre-declared grid, so it may be slightly under-tuned.
Fine-tuning was not attempted (see §7).

## 3. Model selection

The rule is in [`model-selection.md`](model-selection.md). On validation, B3\*
had MAE 1.2906, N2 1.6843, and N1 1.2862 ± 0.0082. N1's improvement over B3\*
was 0.0044, with a CI of [−0.0091, +0.0180] that includes 0, so the
**selected model is B3\* (E04)**:
- word 1–2-grams, min_df 5, `norm_min_v1`, Ridge α = 1.0;
- 81,413 features;
- about 2 minutes to train the whole grid on CPU.

It is also the most interpretable and cheapest candidate.

## 4. Reviewer generalisation (RQ4, H3)

- **Test MAE.** On reviews by **tasters never seen in training**, TF-IDF
  test MAE rises from 1.321 (size-matched random split, same rows overall)
  to 1.412. R² falls from 0.692 to 0.571.
- **Re-tuning.** Re-tuning on the holdout's unseen-taster validation set chose
  a much simpler vectoriser: unigrams, min_df 10, legacy preprocessing,
  α = 0.3. The random split chose B3\*'s settings. This is consistent with
  bigram-level, taster-specific phrasing being less transferable.
- **Per-taster error.** Error varies strongly by unseen taster. One held-out
  taster (1,021 test reviews, mean score 86.6) has MAE 2.29, with
  predictions 1.95 points too high on average. The two large held-out
  tasters have MAE 1.27 and 1.47.
- **Exploratory, not pre-registered.** On the unseen-taster test set the
  BiLSTM beats TF-IDF by 0.049 [0.040, 0.058] MAE. On the random split the
  gap is only 0.010 [−0.020, +0.001]. Sequence models may transfer across
  reviewers slightly better, but this is a hypothesis for future work, not
  a finding.

**Confounds.**
- The holdout test set consists of only three tasters, chosen at random by
  the protocol.
- Its score distribution differs from the random test set (mean 88.87,
  SD 2.73 vs 88.63, SD 3.01).
- Tasters differ in average score (85.78–90.63 in the corpus), so part of
  the gap is calibration, not language understanding.

The difference shows that **performance does not transfer fully to new
critics**. It does not isolate *why*.

## 5. Shortcuts and ablations (RQ5)

| Variant (validation unless stated) | MAE | Δ vs B3\* [95% CI] |
|---|---|---|
| Length only (test) | 2.012 | +0.71 (test) |
| B3\* + length features | 1.2665 | −0.024 [−0.031, −0.017] |
| Years masked (test, H4) | 1.310 | +0.008 [+0.006, +0.010] |
| Years masked | 1.3005 | +0.010 [+0.007, +0.013] |
| All digits masked | 1.3030 | +0.012 [+0.009, +0.016] |
| Character n-grams instead of words | 1.2910 | +0.000 [−0.012, +0.013] |
| Word + character n-grams | 1.2517 | **−0.039 [−0.046, −0.032]** |
| Legacy normalisation (best in the E04 grid) | 1.3177 | +0.027 |
| Legacy normalisation, keeping negations | 1.3244 | +0.034 [+0.022, +0.045] |

- **Length** is a strong standalone cue (R² 0.33). The text model already
  captures most of it, but adding explicit length features still helps a
  little (−0.024 on validation). Residuals drift with length: short reviews
  are over-predicted (+0.32 points in the shortest decile), and long reviews
  are under-predicted (−0.31 in the longest).
- **Numbers.** Masking year *values* costs only about 0.01 MAE. However, the
  placeholder itself gets one of the largest positive coefficients (`__year__`
  +8.40; `now __year__` +5.66). In the unmasked model, five year tokens rank
  among the twenty strongest positive terms (2030 is sixth; 2025, 2022, 2020,
  2024 and 2023 follow).
  Interpretation: **the presence of a drinking-window recommendation, which
  signals an age-worthy wine, is a strong cue, while the specific years add
  little**. This is a form of shortcut, since it is about the critic's
  framing and not the sensory description, but it cannot be removed by
  masking values.
- **Character features** add complementary signal when combined with words
  (−0.039 on validation). This variant was a validation-only ablation and is
  outside the frozen test scope, so no test claim is made. It is the most
  promising direction for future work.
- **Normalisation.** Removing punctuation and stopwords (the legacy
  pipeline) hurts the bag-of-words model.

## 6. Secondary tasks (separate from T1)

| Task | Model (test) | Macro-F1 [95% CI] | Balanced accuracy | QWK |
|---|---|---|---|---|
| T2: 5 WE categories | Majority dummy | 0.103 [0.102, 0.104] | 0.200 | 0.000 |
| | Multinomial LR (balanced) | 0.571 [0.562, 0.580] | 0.591 | 0.742 |
| | Ordinal LR (Frank & Hall) | 0.575 [0.564, 0.586] | 0.537 | 0.755 |
| | B3\* regression, binned | 0.560 [0.549, 0.571] | 0.522 | 0.760 |
| T3: ≥ 90 points | Majority dummy | 0.383 | 0.500 | – |
| | Logistic regression | 0.842 [0.837, 0.846] | 0.839 | – (Brier 0.106) |
| | B3\* regression, thresholded | 0.837 [0.833, 0.842] | 0.832 | – |

- The **extreme categories are hard**. Under the ordinal model, Acceptable
  and Superb/Classic recall are both 0.32, and most errors are adjacent-class
  confusions (see the confusion matrix).
- Class-balanced LR trades precision for recall on the rare classes:
  Acceptable recall 0.55, Superb recall 0.52.
- **Binning the regression model** gives nearly the same ordinal agreement
  (QWK 0.760) and T3 macro-F1 (0.837) as dedicated classifiers. A single
  score model can therefore serve all three tasks.

## 7. Completed, deferred and not-run experiments

| Item | Status |
|---|---|
| All experiments in `configs/experiments.toml` (23) | Completed on validation |
| Test scope (17 + 2 derived) | Completed once, logged |
| Transformer **fine-tuning** | **Not run.** The project owner chose the frozen-encoder design. Native-Windows TensorFlow has no GPU support, and GPU fine-tuning was estimated at 6–9 h. |
| Winery-grouped robustness split | Deferred (protocol §10) |
| Word + character model on test | Not in the frozen scope. Validation-only result. |
| Qualitative reading of 50 validation errors | Completed by the AI assistant; see [`error-analysis.md`](error-analysis.md) |

**Compute:** all 23 validation experiments took 57 minutes of CPU time
(BiLSTM about 25 minutes in total, about 20 s per epoch). The LSTM selected
on validation stopped at epoch 1–2 in every seed. Validation error rose after
the first epoch while training error kept falling, so the network overfits
quickly at this data size.
