# WineMood: Research Proposal

**Status:** draft for Phase 2. **No experiments have been run under this
proposal.** Every quantitative statement below is either a property of
the dataset that still needs verification (marked *verify*) or a hypothesis.

## 1. Background and motivation

The original WineMood project trained an LSTM to classify wine reviews as
positive, neutral or negative, deriving those labels from critic scores. The
audit (`docs/audit.md`, A-01 and A-02) shows that these labels are score bands,
not sentiment. The "negative" band is also empty in the data.

The underlying question is still worth asking, once it is stated
correctly. Professional tasting notes form a specialised register: they are
dense with sensory descriptors ("graphite", "brambly", "taut") and evaluative
hedges, and they are written by a small group of experts who also assign a
numeric score. How much of an expert's numeric judgement is recoverable from
their prose, and *which* language carries it, is a well-posed problem in applied
NLP. It touches several areas:

- **Text regression and ordinal classification:** predicting a numeric or
  ordered quantity from text.
- **Domain-specific evaluative language:** how a specialist register
  expresses quality without overt polarity words.
- **Author confounds:** separating the signal in the language from the
  identity of the writer.

## 2. Distinguishing the task from sentiment analysis

| | Sentiment analysis | WineMood's actual task |
|---|---|---|
| Target | The polarity or emotion a writer expresses, usually annotated by human readers | A numeric quality score assigned by the writer themselves |
| Annotation | Independent annotators, with measurable agreement | None. The score is metadata produced with the text |
| Typical text | Consumer reviews that contain overt praise or complaint | Expert tasting notes, descriptive and rarely negative in tone |
| Valid claim | "The model detects negative opinions" | "The model predicts whether the critic scored the wine ≥ 90" |

WineMood will use the second framing. The word "sentiment" will only be
used if a human-annotated polarity label is introduced. If that happens, it
will be a new, separately named label scheme (see §5).

## 3. Research problem

> To what extent can a wine critic's numeric score be predicted from the text
> of their tasting note, and which linguistic properties of the note carry
> this information beyond trivial cues such as length and author identity?

## 4. Research questions and hypotheses

Each hypothesis states the comparison that would support or refute it.
Thresholds such as "meaningfully above" will be set before Phase 3 runs begin,
using bootstrap confidence intervals on the held-out test set.

**RQ1: Predictability.** Can score bands be predicted from tasting-note text
above trivial baselines?
- *H1:* A TF-IDF + logistic regression model achieves higher macro-F1 than
  both the majority-class baseline and a description-length-only baseline,
  with non-overlapping 95 % bootstrap confidence intervals.

**RQ2: Model capacity.** Do sequence models add value over bag-of-words
models in this register?
- *H2:* A sequence model (the reproduced LSTM or a pre-trained transformer
  encoder) improves on TF-IDF + LR by a margin larger than seed-to-seed
  variance (≥ 3 seeds).
- *H2-null:* The improvement falls within seed variance. This outcome is
  publishable as a negative result and is consistent with lexical cues
  dominating.

**RQ3: Author confound.** How much of the predictive signal is author-specific?
- *H3:* Performance on a split grouped by `taster_name` (unseen tasters at test
  time) is lower than on a random deduplicated split.

**RQ4: Linguistic carriers.** Which lexical features are most associated with
high scores, and are they stable across tasters?
- *H4 (exploratory):* A subset of evaluative descriptors (e.g. intensity and
  length-of-finish terms) is positively associated with score across
  most tasters. This is analysed with model coefficients and per-taster
  consistency, not asserted in advance.

**RQ5: Preprocessing.** Does removing stopwords, and negations in particular,
affect performance?
- *H5:* Keeping negation tokens does not reduce macro-F1, and it may improve
  it. This is tested as a controlled ablation (`keep_negations`).

## 5. Dataset

**Source.** Kaggle, *Wine Reviews* by zynicide
(<https://www.kaggle.com/datasets/zynicide/wine-reviews>), file
`winemag-data-130k-v2.csv`. The data was scraped from WineEnthusiast.

**Relevant fields.** `description` (text), `points` (integer score),
`taster_name`, `price`, `variety`, `country`, `winery`, `title`.

**Known or expected properties** (all *verify* in Phase 2 with
`winemood.data.schema.validate_reviews`):

- About 130k rows.
- `points` in [80, 100].
- Duplicate descriptions are present.
- `taster_name` is missing for a substantial share of rows.
- Scores are concentrated in the mid-to-high 80s.

**Licence and terms.** These must be confirmed on the Kaggle page and recorded in
`data/README.md` before any data or derived artifact is redistributed. The raw
data is never committed to this repository.

**Limitations.**

- The data covers a single publication and a few dozen critics.
- It is English-only.
- It is a snapshot from the time of scraping.
- Wines scoring below 80 are not published, so the low end is censored.
- The scores are not independent of the text, because the same author
  produces both.

### Candidate target definitions (decision required in Phase 2)

Each target becomes a named, versioned scheme in `winemood.data.labels`. The
legacy scheme `points_band_v0` is kept unchanged for reference only.

| Scheme | Definition | Rationale | Caveat |
|---|---|---|---|
| `points_binary_90_v1` | 1 if points ≥ 90, else 0 | 90 is a widely used "outstanding" marketing threshold, and both classes have adequate support | Hard boundary between 89 and 90 |
| `points_ordinal_v1` | e.g. 80–84 / 85–89 / 90–94 / 95–100 | Retains more information and supports ordinal metrics | The top band is small |
| `points_regression_v1` | Raw points as a continuous target | No threshold artefacts, and MAE is interpretable | Different model heads and metrics |

The legacy three-way scheme will **not** be carried forward, because its
"negative" class is empty. That decision, and the choice among the
candidates above, is recorded in Phase 2 before any model is trained.

## 6. Evaluation methodology

**Splits.**

- *Primary:* deduplicate on normalised text, then stratified random
  train/validation/test at 70/10/20. The fixed seed is in `configs/`. Implemented
  in `winemood.data.split`.
- *Grouped:* a split by `taster_name` for RQ3, and optionally by `winery` to
  test memorisation of producers.
- The test set is evaluated **once per final model**. Validation data
  drives all model selection and early stopping.

**Leakage controls.**

- Every tokenizer, vectoriser and scaler is fitted on the training split only.
- Split disjointness is asserted in code.
- Preprocessing is one shared implementation for training and inference.

**Metrics.**

- *Classification:* macro-F1 (primary), balanced accuracy, per-class
  precision/recall/F1, confusion matrix.
- *Probability quality:* expected calibration error and Brier score.
- *Ordinal and regression:* MAE, Spearman ρ, quadratic-weighted κ for ordinal
  bands.
- *Uncertainty:* 95 % bootstrap confidence intervals over test examples, plus
  mean ± std over ≥ 3 training seeds for neural models.

**Error analysis.** Misclassified examples are retrieved by their preserved
row index (fixing A-03) and stratified by taster and length.

## 7. Baselines and models

In order of increasing complexity. Each one must justify its cost against the
one before it.

1. Majority class.
2. Length-only logistic regression (number of tokens).
3. TF-IDF (word and character n-grams) + logistic regression or linear SVM.
4. Legacy architecture reproduced correctly: Embedding + LSTM, trained under
   the leakage-free protocol.
5. *(Optional, if compute allows)* Fine-tuned pre-trained transformer
   encoder. The model choice will be justified in Phase 3.

## 8. Reproducibility requirements

- Pinned dependencies (`pyproject.toml`).
- All randomness seeded through `winemood.seeding.set_global_seed`.
- Every run is driven by a config file. The config, the git commit hash and
  the metrics are saved next to the artifact.
- Data checksum (SHA-256 of the raw CSV) recorded with each run.
- No result is reported in documentation unless a stored run produced it.

## 9. Threats to validity

- **Construct validity.** Score bands proxy "perceived quality", not opinion
  polarity (§2). Thresholds are arbitrary.
- **Internal validity.** There are author confounds (RQ3), duplicates (now
  removed), and length or metadata shortcuts (baseline 2 targets the length
  shortcut).
- **External validity.** The data comes from a single publication and a
  specialised register, so results may not transfer to consumer reviews or
  other critics.
- **Statistical conclusion validity.** Single-seed neural results are
  unreliable, so multiple seeds and confidence intervals are required.
- **Measurement.** Critic scores are themselves noisy, and inter-critic
  agreement is unknown in this dataset.

## 10. Future directions

- Per-taster calibration models (does a "92" mean the same for every critic?).
- Cross-publication transfer if a second, legally usable corpus is identified.
- A small, human-annotated polarity subset. This would enable genuine
  sentiment analysis as a *separate* task, with reported inter-annotator
  agreement.
- Interpretability: comparing linear-model coefficients with attribution
  methods for neural models.

## 11. References

Citations will be added in Phase 2, after checking each source directly.
Following the project rule against fabrication, no reference is listed here
until its bibliographic details have been checked against the original
publication.
