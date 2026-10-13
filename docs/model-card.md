# Model Card: WineMood B3\* (E04, word TF-IDF + Ridge)

## Model details

- **What it predicts.** A Wine Enthusiast critic's numeric score (80–100)
  from the text of the tasting note.
- **Architecture.**
  - Lower-casing (`norm_min_v1`).
  - Word 1–2-gram TF-IDF (sublinear tf, min_df 5, 81,413 features, fitted on
    the training split only).
  - Ridge regression (α = 1.0); predictions clipped to [80, 100].
- **Version.** Experiment `E04_tfidf_word`, configuration hash
  `results/validation/E04_tfidf_word.json → config_hash`. It was selected on
  validation and frozen on 2026-10-13 (commit `15bc2a6`).
- **Artifact.**
  - `experiments/grouped_v1/E04_tfidf_word/model/`, stored as JSON + `.npy`
    (no pickle). It is not committed and can be regenerated with
    `winemood exp run E04_tfidf_word`.
  - It loads with `winemood.inference.regression.ScorePredictor`, which also
    returns the derived Wine Enthusiast category.
- **Developed by.** The WineMood project (research/portfolio code; MIT
  licence for the code).

## Intended use

- Research on how much of an expert's rating is recoverable from tasting-note
  language. It is also a demonstration model for the WineMood application.
- The output is best read as *"the score a Wine Enthusiast critic would
  likely have assigned to a note written like this"*.

## Out-of-scope uses

- **Not** a measure of wine quality, value or consumer preference.
- **Not** sentiment analysis. The training targets are critic scores, and the
  data contains no sentiment annotations.
- Not for consumer reviews, other publications, other languages, or notes
  written in a different register. No evidence supports transfer to them.
- Not for any commercial use. The training data is CC BY-NC-SA 4.0 (see
  *Licensing*).
- Not for decisions about individual producers or critics.

## Training data

- Kaggle *Wine Reviews* (`zynicide/wine-reviews`, v4): WineEnthusiast reviews
  scraped in 2017. SHA-256 `52af2643…67ac9`.
- 129,971 rows, of which 119,954 remain after removing exact duplicates.
  Training uses the `grouped_v1` train split (83,967 reviews), which is
  grouped by near-duplicate clusters and by wine identity, and stratified by
  category.
- The coverage is skewed: the US accounts for 42 % of reviews, 19 named
  critics wrote them, the largest critic wrote 19.6 %, and wines below 80
  points are never published.

## Evaluation

- **Data.** The untouched `grouped_v1` test split (23,991 reviews), evaluated
  once.
- **Intervals.** 95 % cluster-bootstrap intervals over split groups.

| Metric (test) | Value |
|---|---|
| MAE | 1.302 [1.289, 1.315] points |
| RMSE | 1.655 [1.639, 1.671] |
| R² | 0.715 [0.709, 0.720] |
| Median absolute error | 1.08 |
| Within ±2 points | 78.1 % |
| Bias (pred − true) | +0.045 [+0.024, +0.066] |
| Derived 5-category macro-F1 / QWK | 0.560 / 0.760 |
| Derived ≥ 90 macro-F1 | 0.837 |

**Reference points (same test set):**
- predicting the mean: MAE 2.541;
- length only: 2.012;
- BiLSTM (5 seeds): 1.286 (0.016 better; statistically detectable, but
  practically negligible);
- frozen MiniLM encoder + Ridge: 1.694.

**Unseen critics.** When the model is re-tuned and evaluated on critics it
never saw, MAE rises to 1.412, against 1.321 on a size-matched random split.

## Limitations and known biases

- **Regression to the mean.** Scores of 80–83 are over-predicted, and scores
  of 94–100 are under-predicted by about 2–5 points. Extreme ratings are the
  least reliable.
- **Length and framing cues.** Longer notes and notes containing a
  drinking-window recommendation ("drink 2020–2030") receive higher
  predictions. Part of the signal is the critic's framing rather than
  sensory description.
- **Critic dependence.**
  - Accuracy varies by critic (MAE 0.91–1.55 across known critics).
  - It degrades for critics whose scoring calibration is unusual: up to
    MAE 2.29 for one unseen critic.
- **Price association.** Notes on expensive wines are under-predicted, and
  notes on cheap wines over-predicted. The model never sees price.
- **Target noise.** Repeat reviews of the same wine differ by about 1.7
  points on average, so precision beyond about ±1 point should not be
  expected.
- **Data artifacts.** 0.5 % of descriptions begin with the critic's score
  range (barrel samples; audit A-31). The measured effect on the reported
  metrics is negligible (Δ MAE 0.0002), but it is a leakage source that a
  future data version will remove.
- **Era and scope.** The data covers a single publication, English only,
  and 2017-era reviews.

## Ethical considerations and licensing

- **Licence.** The training data is CC BY-NC-SA 4.0. The model is a derived
  work, so it is for non-commercial use, it must attribute the source, and
  it must be shared under the same licence. Trained weights are not
  committed; they are regenerated from the data.
- **Named critics.** Critics are real, named people. Evaluation reports
  critics anonymously and in aggregate, and the model must not be used to
  characterise individuals.
- **Copyright.** Published examples are limited to short excerpts.

## Reproduction

See [`reproducibility.md`](reproducibility.md):

```bash
winemood data download && winemood data split --check
winemood exp run E04_tfidf_word
```
