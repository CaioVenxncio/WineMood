# WineMood: Problem Formulation

**Status:** decision record, Phase 2 (2026-10-12).
**Decision:** primary formulation **B (score regression)**; secondary
formulation **A (ordinal score-category classification)**; formulation
**C (genuine sentiment)** rejected for this dataset and recorded as future work.
The decision was made by the project owner at the Phase 2 scientific gate,
on the recommendation below. It was checked against the real score
distribution before any modelling (see §4).

All dataset figures quoted here come from the code-generated
[`docs/data-report.md`](data-report.md) (dataset SHA-256 `52af2643…67ac9`).

## 1. What went wrong originally

The original project labelled reviews *Positive* (≥ 90 points), *Neutral*
(80–89) and *Negative* (< 80) and called the task sentiment analysis. Two facts
make that formulation untenable:

1. **The Negative class cannot occur.** Wine Enthusiast does not publish
   reviews of wines scoring below 80, and the dataset confirms it: 0 of
   129,971 reviews score below 80 (audit A-01).
2. **The labels are not sentiment.** The score and the text are produced by the
   same critic for the same wine. A score of 85 is "Good" on the publisher's
   own scale, so calling it "neutral" mislabels both the construct and the
   value (audit A-02).

The real question the data can answer is: **how much of a critic's numeric
judgement can be recovered from the text of their tasting note?**

## 2. Evidence from the data

| Property | Value | Implication |
|---|---|---|
| Score range | 80–100, integers, all 21 values present | Bounded and ordinal; regression is well posed |
| Mean / SD | 88.44 / 3.09 (deduplicated) | Narrow range: an error of 2 points is substantial |
| Wine Enthusiast categories | 2.4 / 24.8 / 34.8 / 33.0 / 4.9 / 0.11 % | "Classic" (129 reviews) is too sparse to learn as its own class |
| ≥ 90 vs < 90 | 38.0 % / 62.0 % | A balanced, simple binary task is available |
| Length vs score | Spearman ρ = 0.51 (train; 95 % CI 0.506–0.516) | Strong shortcut; baselines must control for it |
| Price vs score | Spearman ρ = 0.61 | Text that mentions price or prestige carries signal |
| Sentiment annotations | None | Formulation C has no ground truth |

## 3. Candidate formulations

### Option A: score-band classification

Predict a category derived from the score. Thresholds are taken from **Wine
Enthusiast's published rating scale**, not invented: 80–82 Acceptable, 83–86
Good, 87–89 Very Good, 90–93 Excellent, 94–97 Superb, 98–100 Classic. The scale
was verified on 2026-10-12 via the Wine Enthusiast buying guide and secondary
reproductions (see the research proposal, §11). Variants:

- `we_category_v1`: all six published categories.
- `we_category_merged_v1`: Classic merged into Superb by a rule declared
  before the data was inspected. Adjacent sparse classes are merged until each
  holds ≥ 1 % of the training split.
- `points_binary_90_v1`: ≥ 90 ("Excellent" or better) vs < 90.

### Option B: score regression

Predict the integer score directly (`points_regression_v1`). An ordinal
variant treats the 21 values, or the categories, as ordered classes (e.g. the
cumulative binary decomposition of Frank & Hall, 2001).

### Option C: genuine sentiment classification

Predict the polarity a reader perceives in the text. This requires human
polarity annotations, which the dataset does not contain. Off-the-shelf
sentiment lexicons or models could produce *scores*, but those are model
outputs, not ground truth. Using them as targets would replace one invalid
proxy with another.

## 4. Comparison

| Criterion | A: Category classification | B: Score regression | C: Genuine sentiment |
|---|---|---|---|
| **Scientific validity** | Valid if named "score category". Publisher thresholds avoid arbitrariness | Highest. The target is the measured quantity itself | Valid construct, but **no ground truth** in this data |
| **Dataset suitability** | Good. 6-class version has a 0.11 % class, 5-class merged version is workable, binary is balanced | Excellent. 21 integer values, unimodal, every value present | Unsuitable without new annotation |
| **Interpretability** | High for non-experts ("Excellent vs Very Good") | High: error in points (MAE). Categories can be derived from predictions | High, if annotated |
| **Experimental complexity** | Low–moderate (imbalance handling for the 5/6-class versions) | Low (standard regression metrics) | High: annotation guidelines, multiple annotators, agreement study |
| **Potential biases** | Edge effects at thresholds (89 vs 90 are adjacent, yet different classes). Minority classes | Score compression (SD 3.1). Regression to the mean at the tails | Annotator bias. Mismatch between expert and lay register |
| **Reproducibility** | Fully reproducible (deterministic rules) | Fully reproducible | Depends on annotation process |
| **Practical value** | Matches how scores are communicated to consumers | Finest-grained, and supports any downstream thresholding | Would enable genuine opinion mining |
| **Portfolio / graduate value** | Good, but well-trodden as the only task | Strong: well-posed, honest, rich error analysis (residuals by score, taster, length) | Strong *if* done properly, but out of scope for one person in this phase |

**Shared confounds** (apply to A and B): review length (ρ = 0.51), taster
identity (19 tasters, whose mean scores range from 85.78 to 90.63), drinking-window years
("2020", "2022" are among the terms most associated with ≥ 90 points) and
price-related language. These are not reasons to reject A or B. They are
reasons for the controls in the experimental protocol: a length-only baseline,
a taster-held-out split, and residual analysis.

## 5. Decision and rationale

**Primary: B, regression on `points` (`points_regression_v1`).** It uses the
full information in the target, requires no thresholds, and yields an error
measure in the score's own units. Classification metrics can still be derived
by binning its predictions with the published categories, which makes A and B
directly comparable.

**Secondary: A, ordinal classification with `we_category_merged_v1`, plus
`points_binary_90_v1` as an experimental candidate.** Both are reported
because they correspond to how scores are actually used. Neither is presented
as superior before experiments are run.

**Rejected for this dataset: C.** No polarity annotation exists. A
human-annotated subset with measured inter-annotator agreement is listed as
future work (research proposal §10). Score-derived labels will never be
described as sentiment.

**Retained for history only:** `points_band_v0` (flagged `research_use=False`).

## 6. Consequences

- Phase 3 reports regression metrics as the headline, and classification metrics
  for the secondary targets. See [`experimental-protocol.md`](experimental-protocol.md).
- The Streamlit UI still serves the legacy model and its `points_band_v0`
  labels, with a disclaimer. It will be aligned with this formulation in
  Phase 4.
