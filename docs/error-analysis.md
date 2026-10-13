# WineMood: Error Analysis (Phase 3)

The quantitative tables are generated in
[`results-tables.md`](results-tables.md) (section "Error analysis") from the
selected model's **test** predictions (B3\*, E04), keyed by `review_id`. The
qualitative review uses **validation** errors only, as the protocol requires.
Critic scores are one expert's rating on one occasion; they are not an
objective measure of wine quality.

## 1. Error distribution

- **Spread.** Test absolute error has a median of 1.08 points; the 90th
  percentile is 2.72 and the 99th is 4.39. 78.1 % of predictions fall within
  ±2 points.
- **Bias.** Overall bias is small (+0.045 points, 95 % CI [+0.024, +0.066]).
- **Regression towards the mean.** This is the dominant pattern:
  - the slope of prediction on true score is 0.707, and the prediction SD is
    2.59 against a true SD of 3.10;
  - the mean residual decreases monotonically with the true score, from
    +3.04 at 80 points to −4.80 at 100 points (figure:
    `docs/figures/results/residual_by_true_score.png`);
  - every model shows the same shape, and the length-only and encoder models
    show it more strongly.
- **Boundary error.** Errors concentrate at the ends of the scale. MAE is
  1.97 for scores 80–83 (n = 1,181), 1.22 for 84–93, and 2.11 for 94–100
  (n = 1,203). The ten test reviews scoring 99–100 are under-predicted by
  4.2–4.8 points on average. This is expected for a squared-loss model on a
  unimodal target, and it matters for any use that cares about the extremes.

## 2. Error by subgroup (test, descriptive)

- **Length.** MAE rises from about 1.24 in the shortest deciles to 1.39 in
  the longest. Residuals drift from +0.32 (short reviews over-predicted) to
  −0.31 (long reviews under-predicted). Length is predictive, and the model
  only partly captures it (see the length ablation in
  [`experimental-results.md`](experimental-results.md) §5).
- **Taster.**
  - Among tasters with at least 100 test reviews, MAE ranges from 0.91 to
    1.55.
  - Two tasters with fewer reviews are systematically over-predicted by the
    model (mean residual +0.66 and +0.77), meaning their scores are lower
    than their language suggests.
  - On the taster-holdout split, one unseen taster has MAE 2.29, with
    predictions 1.95 points too high. That is consistent with a critic who
    scores lower than the corpus average for similar language. Tasters are
    anonymised by review count.
- **Price.**
  - Reviews of cheap wines (≤ $15) are over-predicted (+0.70), and expensive
    wines (> $75) are under-predicted (−1.04, MAE 1.61).
  - Price is not a model input, so the model cannot know it. The pattern
    says that critics score expensive wines higher than their tasting-note
    language alone implies, which is an association and not a causal claim.
- **Country.** MAE ranges from 1.08 (Italy) to 1.42 (France).
- **Year mention.** Reviews with and without a year have almost identical
  MAE (1.29 vs 1.31).

## 3. Target noise

The target itself is noisy, and this bounds achievable accuracy:
- 23 duplicate-text groups carry different scores, with a mean range of 1.78
  points.
- 1,508 pairs of *distinct* reviews of the same wine and vintage differ by
  1.69 points on average (median 1.0).

These pairs may involve different tasters or tasting occasions, so the
figures are a descriptive reference, not a formal noise ceiling. They show
that an MAE of about 1.3 is already of the same order as the disagreement
between repeated expert assessments.

## 4. Model disagreement

- **Correlation.** The test predictions of B3\* and the BiLSTM correlate at
  0.943.
- **Where they disagree.** The largest disagreements (up to about 5 points)
  occur on reviews whose praise or criticism is carried by a few strong
  phrases. Examples: "fall in love with this 100% Cabernet" (true 97: TF-IDF
  97.2, BiLSTM 91.9) and "a hard, too-toasty wine … completely dominated by
  smoky wood" (true 83: TF-IDF 88.1, BiLSTM 84.4).
- **What it suggests.** The bag-of-words model weights isolated evaluative
  words heavily. The sequence model sometimes captures the overall framing
  better and sometimes worse, so neither dominates.

## 5. Qualitative review of 50 validation errors

**Sample.**
- 50 reviews drawn at random (seed 42) from the 1,485 validation reviews
  where B3\* errs by ≥ 2.5 points.
- This is a deliberate focus on large errors, a refinement of the protocol's
  "randomly sampled validation errors" that is noted here.
- The AI assistant that built the pipeline read and categorised them (protocol
  §10). This is not an independent human annotation.
- Each review has one primary category.

| Category | n | Typical pattern |
|---|---|---|
| A. Praise vocabulary, moderate score (over-prediction) | 15 | "delicious", "elegant", "pristine", "great intensity" in reviews scored 84–88; positive words are common at every level |
| B. Understated or plain language, high score (under-prediction) | 13 | Short, matter-of-fact notes scored 89–93 ("It's not a big wine, but it offers a mouthful of pleasure", 89) |
| C. Exceptional scores compressed (≥ 94) | 5 | 94–98-point reviews predicted 90–92; superlatives are present but rare |
| D. Clearly negative cues, prediction still too high | 8 | "stewed", "grating acidity", "lacks the spine", "industrial cleaner" in 80–84-point reviews; the model moves in the right direction but not far enough |
| E. Producer, vintage or context narrative instead of sensory description | 5 | Winery history and site descriptions with little evaluative language |
| G. Age-worthiness framing ("needs time", "hard now", "lay down") | 3 | Language about youth and structure is ambiguous: it can signal a great wine that is still closed, or a flawed one |
| F. Score stated in the text | 1 | "92-94. Barrel sample." (see §6) |

**Interpretation.**
- Categories A and B together account for 28 of 50 errors. Tasting notes use
  an overwhelmingly positive register at every score, so the *degree* of
  quality is carried by subtle intensity cues, by length and by critic
  habits, rather than by polarity.
- Category D reflects regression to the mean more than a failure to read the
  language.

These observations motivate future work on ordinal or quantile losses for
the extremes, and on per-critic calibration.

## 6. New data-quality finding: score ranges inside the text

During the qualitative review, one description turned out to begin with the
critic's own score range ("92-94. Barrel sample.").

- **Prevalence.** A corpus-wide check found **628 unique descriptions (0.5 %)
  that begin with a score range** such as "86-88 Barrel sample". The true
  score lies inside the stated range in 99.8 % of them.
- **Why it matters.** This is direct target information in the input text,
  so it is a leakage source that the Phase 1–2 audits missed. It is
  recorded as audit finding A-31.
- **Measured effect (post hoc, no retraining).**
  - The 125 affected test reviews have B3\* MAE 1.26, against 1.30 for the
    rest.
  - Excluding them changes the B3\* test MAE from 1.3023 to 1.3025.
  - For the seed-ensemble BiLSTM predictions, it changes from 1.2533 to
    1.2537.
  - The models barely exploit the prefix, and **the reported results are
    materially unaffected**.
- **Remediation.** The remediation is proposed for the next data version
  (`roadmap.md`): strip the leading range before modelling, and report
  results under that version. It is not applied retroactively, because that
  would change the frozen Phase 3 protocol after test access.

> **Phase 4 correction.** The counts above recognised hyphen and en-dash
> ranges only. Including em-dash ranges, **669 unique descriptions (680 raw
> rows)** are affected, 135 of them in the test set. Recomputed on the same
> stored predictions, E04 test MAE is 1.3023 with them and 1.3024 without, so
> the conclusion is unchanged. See
> [`a31-remediation.md`](a31-remediation.md), which also describes the
> isolated `norm_min_v2` correction.

## 7. Limitations of this analysis

- Subgroup tables are descriptive. They do not adjust for confounding (for
  example, taster × country × price) and carry no causal interpretation.
- Subgroup metrics with small n (such as the 99–100-point reviews) are
  unstable.
- The qualitative categories reflect one annotator's judgement, made by an AI
  system.
