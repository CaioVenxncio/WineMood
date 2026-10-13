# WineMood

**WineMood** is an NLP research project on the language of professional wine
tasting notes. It asks how accurately a critic's numeric score (80–100) can be
predicted from the text they wrote, and which language carries that signal.

It began in 2025 as a coursework "sentiment classifier". An audit showed that
the labels were score thresholds, not sentiment, and that no results had ever
been recorded. The project has since been rebuilt as a reproducible research
codebase in approval-gated phases (see [`docs/roadmap.md`](docs/roadmap.md)).
**Status:** Phases 1–3 are complete and awaiting review; Phase 4 (application)
has not started.

## Key results (held-out test set, 23,991 reviews)

| Model | MAE (points) [95% CI] | R² |
|---|---|---|
| Predict the training mean | 2.541 [2.518, 2.563] | 0.00 |
| Review length only | 2.012 [1.993, 2.032] | 0.33 |
| **Word TF-IDF + Ridge (selected)** | **1.302 [1.289, 1.315]** | **0.72** |
| BiLSTM (mean of 5 seeds) | 1.286 [1.274, 1.299] | 0.72 |
| Frozen MiniLM sentence encoder + Ridge | 1.694 [1.678, 1.711] | 0.53 |

- Text predicts the critic's score with an error of about 1.3 points, roughly
  the size of the disagreement between repeat reviews of the same wine (about
  1.7 points).
- The BiLSTM's advantage over TF-IDF is statistically detectable but
  practically negligible, so the simpler model was selected on validation
  before the test set was touched.
- Error grows for critics unseen in training: MAE 1.41 against 1.32 on a
  size-matched random split.
- Predictions regress toward the mean at the extremes of the scale.

Full results, pre-registered hypothesis tests and limitations:
- [`docs/experimental-results.md`](docs/experimental-results.md)
- [`docs/error-analysis.md`](docs/error-analysis.md)
- [`docs/model-card.md`](docs/model-card.md)

This is **not** sentiment analysis. The dataset has no sentiment annotations,
and the old "Negative" class (< 80 points) cannot occur (see
[`docs/problem-formulation.md`](docs/problem-formulation.md)).

## What is implemented

| Area | Contents |
|---|---|
| Research design | [Audit](docs/audit.md), [problem formulation](docs/problem-formulation.md), [proposal](docs/research-proposal.md), [pre-registered protocol](docs/experimental-protocol.md) |
| Data | Checksummed download, validation, deduplication, near-duplicate and same-wine grouping, three fingerprinted splits ([data report](docs/data-report.md)) |
| Experiments | 23 configured experiments: baselines, TF-IDF linear models, BiLSTM, frozen encoder, classification tasks, reviewer holdout, ablations ([development](docs/model-development.md), [selection](docs/model-selection.md)) |
| Evaluation | Cluster-bootstrap CIs, paired comparisons, error analysis, interpretation, generated [results tables](docs/results-tables.md), test-access log |
| Engineering | `winemood` package and CLI, pickle-free artifacts, pytest on synthetic data, Ruff, GitHub Actions workflow (not yet run on GitHub) |
| Application | Streamlit UI that serves the *legacy* model if you supply its files. Integrating the new models is Phase 4. |

## Quick start

Requires Python 3.11–3.13.

```bash
pip install -e ".[dev]"
pytest                                   # synthetic data; no dataset needed

winemood data download && winemood data split --check
winemood exp run E04_tfidf_word          # train + validate the selected model
```

[`docs/reproducibility.md`](docs/reproducibility.md) gives the full sequence
(all experiments, selection, the single test evaluation, the report) and the
environment used.

## Data and licence

[*Wine Reviews*](https://www.kaggle.com/datasets/zynicide/wine-reviews)
(Kaggle, v4; WineEnthusiast reviews scraped in 2017) is licensed
**CC BY-NC-SA 4.0**:
- the data is not included in this repository;
- trained models are derived works and are not committed;
- use is non-commercial only.

The code is MIT-licensed (see [`LICENSE`](LICENSE)).
