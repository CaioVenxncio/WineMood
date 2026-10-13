# WineMood: Experimental Protocol (pre-registration for Phase 3)

**Status:** fixed before any Phase 3 model is trained (2026-10-12). Changes
after Phase 3 begins must be recorded in §10 with date and reason. **No model
results exist yet**, and nothing below claims that any model is better than
another.

Targets are defined in [`problem-formulation.md`](problem-formulation.md); data
facts come from [`data-report.md`](data-report.md).

## 1. Tasks

| ID | Target | Type | Role |
|---|---|---|---|
| T1 | `points_regression_v1` | Regression on integer points 80–100 | **Primary** |
| T2 | `we_category_merged_v1` | 5-class ordinal classification | Secondary |
| T3 | `points_binary_90_v1` | Binary classification | Secondary (experimental candidate) |

`points_band_v0` is never used as a research target.

## 2. Data and splits

**Pipeline** (`winemood data download → verify → split`), deterministic and
implemented in `src/winemood/data/`:

1. **Integrity:** the raw CSV must match the pinned SHA-256
   (`configs/default.toml`).
2. **Exact deduplication** on normalised text (lower-case, collapsed
   whitespace), keeping the lowest `review_id`. On the real data this removes
   10,017 rows. 23 duplicate groups carried conflicting scores; this is
   reported, not resolved.
3. **Grouping:** rows are linked by union-find when they share
   - a near-duplicate cluster (MinHash/LSH on word 3-grams, verified Jaccard
     ≥ 0.8), or
   - a normalised `title`, i.e. the same wine and vintage reviewed more than
     once.

   Linked rows can never be split apart. On the real data: 26 near-duplicate
   pairs, 2,099 rows sharing a title, and 963 multi-row groups.
4. **Primary split `grouped_v1`:** `StratifiedGroupKFold` (10 folds, shuffled,
   seed 42), stratified on `we_category_v1`. Folds 0–1 form the test set,
   fold 2 the validation set, and folds 3–9 the training set. Result: 83,967 /
   11,996 / 23,991 rows (70.0 / 10.0 / 20.0 %), with category proportions
   matching to within 0.01 percentage points.
5. **Secondary split `taster_holdout_v1`** (for RQ3): whole tasters are held
   out. Result: 57,631 / 17,795 / 19,554 rows. 24,974 rows are excluded:
   rows without a taster, plus 62 rows in wine groups reviewed by tasters in
   different partitions.

**Rationale and tradeoffs**

- **70/10/20 proportions.** The test set is large enough for tight bootstrap
  intervals on the rarest merged class (Acceptable, about 2.4 %, so about 580
  test reviews). The validation set (about 12k) is ample for model selection.
- **Grouping by title** is conservative. It costs nothing measurable in
  stratification balance and prevents a model from being scored on a wine
  it has already seen.
- **Winery-level grouping is not applied.** It would remove producer-level
  memorisation but would distort the split heavily, since some wineries have
  hundreds of reviews. It is listed as a robustness analysis (§8).
- **Taster holdout is imbalanced by construction**, because tasters
  contribute very different numbers of reviews. Its numbers are only compared
  within that split, never with `grouped_v1`.

**Fingerprints.** `data/split_fingerprints.json` records the dataset hash,
config hash, counts and SHA-256 of the sorted ids per split.
`winemood data split --check` recomputes the splits and fails on any mismatch.
This check passed on 2026-10-12.

## 3. Leakage rules

1. Every learned transformation is fitted on **train only**. This covers
   tokenizers, vocabularies, TF-IDF statistics, scalers, embeddings fine-tuned
   on task data, and class-merge rules.
2. The **validation** set is used for model selection, early stopping and
   threshold or calibration fitting. The **test** set is used **once per final
   model configuration**, after all choices are frozen. Each test evaluation
   is logged with its timestamp and config hash.
3. Preprocessing is shared between training and inference
   (`winemood.preprocessing.TextPreprocessor`).
4. Metadata columns (price, taster, variety, country, winery, title) are
   **not** model inputs for the text-only models. A metadata-only baseline is
   reported separately to quantify how much signal is non-textual.

## 4. Models

Each model must beat the simpler ones before its cost is justified.

| # | Model | T1 (regression) | T2/T3 (classification) |
|---|---|---|---|
| B0 | Dummy | `DummyRegressor(mean)` | `DummyClassifier(most_frequent)` and `(stratified)` |
| B1 | Length only | Linear regression on word count | Logistic regression on word count |
| B2 | Metadata only (reference, not text) | Ridge on one-hot taster, country, variety + log price | Logistic regression, same features |
| B3 | TF-IDF (word 1–2-grams; char 3–5-grams as a variant) | Ridge | Logistic regression (multinomial). For T2, also a cumulative-binary ordinal decomposition (Frank & Hall, 2001) |
| N1 | Embedding + (Bi)LSTM (Hochreiter & Schmidhuber, 1997), reproduced under this protocol | Linear output, MSE loss | Softmax, cross-entropy |
| N2 | *Conditional:* pretrained Transformer encoder (e.g. a BERT-family model; Devlin et al., 2019), fine-tuned | Regression head | Classification head |

N2 runs only if (a) a GPU budget is available (§7) and (b) N1 has been
evaluated. Its purpose is to test RQ2, not to maximise leaderboard numbers.

**Ablations** (pre-declared):
- negation-preserving preprocessing (`keep_negations`);
- masking 4-digit numbers, to test the drinking-window shortcut seen in EDA;
- B3 with length added as a feature.

## 5. Metrics

**T1, regression**
- MAE (primary), RMSE, R².
- Error distribution: residual histogram; residuals by true score and by
  length decile. Calibration-in-the-large (mean residual).
- Derived classification: predictions are rounded and binned with the
  published categories, then scored with the T2 metrics. This makes T1 and
  T2 models comparable on the same scale.

**T2/T3, classification**
- Macro-F1 (primary), weighted-F1, balanced accuracy.
- Per-class precision, recall and F1; confusion matrix.
- For T2 also MAE over class indices and quadratic-weighted Cohen's κ, which
  account for ordinal distance.
- Probability quality for T3: Brier score and a reliability diagram.

## 6. Model selection, seeds and statistics

- **Hyperparameter search:** a grid or random search on the **validation**
  set with a fixed budget, declared per model before running.
  - B3: regularisation strength over 7 log-spaced values; n-gram range over
    2 options; vocabulary min_df over 3 options.
  - N1: at most 20 configurations.
  - N2: at most 6 configurations.

  The selection metric is the task's primary metric on validation. After
  selection, the model is refit on train only; validation is never merged in,
  so the selection evidence stays clean.
- **Seeds:** all randomness goes through `set_global_seed`. Linear models are
  deterministic given the seed. Neural models are trained with **5 seeds**,
  and mean ± SD is reported.
- **Uncertainty:** 95 % bootstrap confidence intervals over test examples
  (1,000 resamples). Each model comparison uses the **paired bootstrap** of
  the metric difference on the same resamples, following the test-selection
  guidance of Dror et al. (2018).
- **Decision rule:** model X is called *better* than Y only if the 95 %
  paired-bootstrap interval of the difference excludes 0. For neural models,
  the difference must also exceed the seed-to-seed SD. Otherwise the result
  is reported as "no detectable difference".
- **Multiple comparisons:** the comparisons that bear on hypotheses are
  pre-listed (§9). Any further comparisons are labelled exploratory.

## 7. Resources and runtime

- The data pipeline runs on CPU in about 3 minutes. B0–B3 are expected to
  train in minutes on CPU.
- N1 runs on CPU or a single GPU. N2 requires a GPU (e.g. a free-tier
  notebook GPU). If no GPU is available, N2 is skipped and the skip is
  reported. Runtime and hardware are logged for every run.

## 8. Error analysis

- Look up errors by `review_id`, never by array position. The original
  notebook's indexing bug is audit A-03.
- Stratify residuals and confusions by true score, WE category, review
  length, taster (anonymised), and the presence of drinking-window years.
- **Robustness:**
  - score the `taster_holdout_v1` split (RQ3);
  - a winery-grouped split as an additional check;
  - performance on the 23 conflicting-duplicate groups, reported separately.
- **Qualitative review:** about 50 randomly sampled validation errors (never
  test), read and categorised by hand.

## 9. Hypotheses and planned comparisons

These refine research proposal §4.

| ID | Hypothesis | Comparison | Supported if |
|---|---|---|---|
| H1 | Text predicts score beyond trivial cues | B3 vs B0 and B3 vs B1 (T1, MAE) | Paired-bootstrap CI of MAE difference excludes 0 in B3's favour, for both |
| H2 | Sequence models add value over bag-of-words | N1 (and N2) vs B3 | Same rule, and the gain exceeds seed SD |
| H3 | Part of the signal is taster-specific | B3 on `taster_holdout_v1` vs B3 on a size-matched random split of the same rows | Holdout MAE is higher, with CI excluding 0 |
| H4 | Drinking-window numbers are a shortcut | B3 vs B3 with numbers masked | Report the effect size. Exploratory, no directional claim |
| H5 | Removing negations does not help | N1 with vs without `keep_negations` | Report the effect size with CI |

## 10. Deviations log

All amendments below were recorded on 2026-10-12, **before any Phase 3
model was trained and before any test-set access**. The project owner
approved them at the Phase 3 planning gate.

| Date | Change | Reason |
|---|---|---|
| 2026-10-12 | **Cluster bootstrap.** All confidence intervals resample whole split groups (same wine / near-duplicate clusters), not single rows. 2,000 resamples. Comparisons on the same test set are paired. | Rows inside a group are not independent. Groups are almost all singletons, so intervals widen only slightly, but the resampling unit now matches the split unit. |
| 2026-10-12 | **H3 uses an unpaired comparison.** `taster_random_v1` is a grouped random assignment of exactly the rows assigned in `taster_holdout_v1`, with partition sizes matched to within one group. The two test sets contain different rows, so the H3 difference is estimated by independent cluster bootstraps of each test set. | Pairing is impossible across different test sets. |
| 2026-10-12 | **N2 is a frozen encoder.** `sentence-transformers/all-MiniLM-L6-v2` mean-pooled embeddings (pinned model revision recorded per run) feed a Ridge head fitted on train, on CPU. There is no fine-tuning. | Native-Windows TensorFlow has no GPU support. The project owner chose the frozen-encoder design over GPU fine-tuning. Results are reported as "frozen encoder", never as fine-tuned Transformer results. |
| 2026-10-12 | **Preprocessing versions declared.** `norm_min_v1` (lower-case; the tokenizer handles the rest) is the default for new models. `norm_legacy_v0` is the Phase 1 pipeline (punctuation and NLTK stopwords removed, which drops negations). `keep_negations` is the legacy pipeline without removing negations (H5). `mask_numbers` is one of `none`, `years` (19xx/20xx → `<year>`, the H4 treatment) or `all` (every digit run → `<num>`, a diagnostic only). All regression predictions are clipped to [80, 100]. | Makes the H4/H5 treatments and the default pipeline explicit before running. |
| 2026-10-12 | **Test-set scope fixed.** Only these runs are evaluated on test. On `grouped_v1`: T1 = B0, B1, B2, B3\*, B3\* with years masked (H4), N1\* (5 seeds), N1\* keep-negations (5 seeds, H5), N2\*; T2 = dummy, LR\*, ordinal LR\*, best T1 model binned; T3 = dummy, LR\*, best T1 model thresholded. On `taster_holdout_v1` and `taster_random_v1`: B3 and N1 (3 seeds), each re-tuned on its own split's validation (H3). \* is the configuration selected on validation and frozen in `results/selection.json`. Every other variant (word vs char features, normalisation variants, length-augmented B3, all-number masking, hyperparameter grids) is a **validation-only** diagnostic. | Bounds and documents test-set multiplicity before any access. |
| 2026-10-12 | **Hyperparameter budgets.** B3: n-gram {1, 1–2} × min_df {2, 5, 10} × α (7 log-spaced) × preprocessing {`norm_min_v1`, `norm_legacy_v0`}. T2/T3 classifiers reuse the T1-selected vectoriser settings and tune C (and class weighting). N1: 8 configurations × 1 seed, then 5 seeds of the best. | Within the §6 limits ("at most 20" for N1). |
| 2026-10-12 | **Winery-grouped robustness split deferred**, and the 23 conflicting-duplicate groups are analysed as target noise. | Deduplication keeps one copy of each conflicting group, so per-copy performance cannot be measured. Instead, the score disagreement within those groups, and between repeat reviews of the same wine, is reported as an empirical noise floor. Winery grouping is left for future work. |
| 2026-10-12 | **Final-model selection rule** (recorded before `exp select` and before any test access). Candidates are B3\* (E04), N2\* (E12) and N1\* (E10), in order of complexity. Start from the simplest. A more complex candidate replaces the current choice only if its validation MAE is lower AND the 95 % paired cluster-bootstrap CI of MAE(current) − MAE(candidate) on validation excludes 0. | A negligible gain must not buy complexity. The rule is implemented in `Runner.select`. |
| 2026-10-12 | **N1 on the taster splits is not re-tuned over the full grid.** E41/E43 reuse the architecture selected on `grouped_v1` (E10) and retrain from scratch on each split's training set, with early stopping on that split's own validation set (3 seeds). B3 (E40/E42) is fully re-tuned. | Compute budget. The LSTM grid would triple the CPU time. Hyperparameters only are transferred, never weights or checkpoints. |
| 2026-10-12 | **Qualitative error review.** The 50 randomly sampled validation errors are categorised by the AI assistant that implemented the pipeline, and are labelled as such. | Transparency about who did the annotation. It is not an independent human annotation. |
