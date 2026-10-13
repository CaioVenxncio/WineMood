# WineMood

**How much of a wine critic's numerical rating can be predicted from the
language of their review?**

WineMood is an applied NLP research project and interactive application. It
studies the relationship between professional tasting-note language and expert
scores (80–100) using about 120,000 Wine Enthusiast reviews. The project
combines a pre-registered evaluation protocol, leakage-controlled data splits
and reproducible experiments with a research application for exploring the
results.

> WineMood predicts **critic ratings**. It does not measure wine quality, and
> it is not sentiment analysis: the training targets are critics' scores, not
> annotated emotions.

## Findings (held-out test set, 23,991 reviews; 95 % cluster-bootstrap CIs)

| Model | MAE (points) | R² |
|---|---|---|
| Predict the training mean | 2.541 [2.518, 2.563] | 0.00 |
| Review length only | 2.012 [1.993, 2.032] | 0.33 |
| Frozen MiniLM sentence encoder + Ridge | 1.694 [1.678, 1.711] | 0.53 |
| **Word TF-IDF + Ridge (selected)** | **1.302 [1.289, 1.315]** | **0.71** |
| BiLSTM (mean of 5 seeds) | 1.286 [1.274, 1.299] | 0.72 |

- **Text predicts the score well.** The error is about 1.3 points, the same
  order as the disagreement between repeat reviews of the same wine (about 1.7
  points).
- **The model was selected before the test set was touched.** The BiLSTM's
  0.016-point advantage was not detectable on validation, so the simpler,
  interpretable model was chosen. The test result does not change that
  decision.
- **Accuracy degrades for critics unseen in training**: MAE 1.41 against 1.32
  on a matched random split.
- **Predictions regress toward the middle of the scale**, and part of the
  signal reflects framing (length, drinking-window years) rather than
  sensory description.

Details:
- [experimental results](docs/experimental-results.md)
- [error analysis](docs/error-analysis.md)
- [model card](docs/model-card.md)
- [pre-registered protocol](docs/experimental-protocol.md)

## The application

```bash
pip install -e ".[dev]"
winemood model install path/to/E04_tfidf_word.zip   # or: winemood exp run E04_tfidf_word
streamlit run app.py
```

- **Predict.** Estimate a critic score for one review, or a CSV of up to
  1,000 reviews, with an exact breakdown of how the model reached the number.
- **Research.** Model comparison with confidence intervals, the
  pre-registered hypotheses, generalisation to unseen critics, ablations and
  error structure. Everything is read from the committed result records.
- **Explore.** The model's learned word and phrase associations, per-review
  attributions, and dataset composition and biases.

The application runs on the lightweight core dependencies: no TensorFlow or
PyTorch. It starts without a model artifact and explains how to obtain one.
See [docs/application.md](docs/application.md) and
[docs/deployment.md](docs/deployment.md).

## Research pipeline

```bash
winemood data download && winemood data split --check   # checksummed data, fingerprinted splits
winemood exp run E04_tfidf_word                         # train + validate (never touches test)
winemood exp report                                     # statistics and generated tables
```

The full sequence and environment are in
[docs/reproducibility.md](docs/reproducibility.md). Neural experiments need
the optional extras `[neural]` (TensorFlow) and `[transformer]` (PyTorch).

## Repository

| Path | Contents |
|---|---|
| `src/winemood/` | `data/` (acquisition, validation, splits), `features/`, `models/`, `experiments/`, `evaluation/`, `inference/`, `app/` |
| `configs/` | Run configuration and the experiment matrix |
| `results/` | Committed result records: validation, test, selection, test-access log, report |
| `docs/` | Research design, results, model card, application and deployment guides |
| `tests/` | pytest suite on synthetic data. Ruff; GitHub Actions workflow with light and neural jobs |

## Data and licences

- **Dataset.** [*Wine Reviews*](https://www.kaggle.com/datasets/zynicide/wine-reviews)
  (Kaggle, v4; WineEnthusiast, 2017) is licensed **CC BY-NC-SA 4.0**. The
  data is not included in this repository.
- **Trained models.** These are derived works under the same licence, for
  non-commercial use only. They are distributed separately, not committed.
- **Code.** MIT (see [`LICENSE`](LICENSE)).
- **Fonts.** SIL OFL 1.1 (see `static/fonts/`).
