# WineMood

**WineMood** is an NLP research project on the language of professional wine
tasting notes. It asks how much of a critic's numeric score can be recovered
from the text they wrote, and which language carries that signal.

The project started in 2025 as a coursework "wine review sentiment classifier".
A 2026 audit found that its labels were critic-score thresholds rather than
sentiment, that one of its three classes cannot occur in the data, and that no
results had ever been recorded. The project is being rebuilt as a reproducible
research codebase in five phases (see [`docs/roadmap.md`](docs/roadmap.md)).

> **Current status (Phase 1 complete, pending review):** the engineering
> foundations, audit and research design are in place. **No trained model is
> distributed and no performance results exist yet.** Model training and
> evaluation are scheduled for Phase 3.

## What is implemented

| Component | Status |
|---|---|
| Technical & scientific audit | [`docs/audit.md`](docs/audit.md) |
| Research proposal (questions, hypotheses, evaluation plan) | [`docs/research-proposal.md`](docs/research-proposal.md) |
| Architecture & decisions | [`docs/architecture.md`](docs/architecture.md) |
| `winemood` package | Config, structured logging, seeding, data validation, versioned label schemes, deduplicating split-before-fit utilities, shared preprocessing, inference interface |
| Streamlit app | English UI that serves a legacy LSTM *if* you supply artifacts. Without them, it explains what is missing |
| Tests | pytest suite on synthetic data, including headless UI tests |
| CI | GitHub Actions workflow (Ruff + pytest). Not yet run on GitHub |

**Not yet implemented:**
- the training pipeline;
- the evaluation pipeline;
- baselines;
- reproducible model artifacts;
- a data report on the real dataset.

## Important caveat about labels

The legacy model predicts three **score bands** derived from WineEnthusiast
`points`, preserved as the label scheme `points_band_v0`:

| Legacy name | Rule |
|---|---|
| Positive | points ≥ 90 |
| Neutral | 80 ≤ points ≤ 89 |
| Negative | points < 80 (**does not occur in the dataset**) |

These are not human-annotated sentiment labels. The replacement target
definition is a Phase 2 decision (see the research proposal, §5).

## Quick start

Requires Python 3.11–3.13.

```bash
git clone https://github.com/CaioVenxncio/WineMood.git
cd WineMood
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

pytest                 # run the test suite
ruff check . && ruff format --check .
streamlit run app.py   # launch the demo UI
```

The app starts without a model and shows how to supply one. See
[`artifacts/README.md`](artifacts/README.md) for the artifact layout and
[`data/README.md`](data/README.md) for dataset acquisition.

## Dataset

[*Wine Reviews*](https://www.kaggle.com/datasets/zynicide/wine-reviews)
(Kaggle, zynicide): about 130k WineEnthusiast reviews with `description`,
`points`, `taster_name`, price, variety and origin. The data is not included in
this repository. Its licence and terms will be confirmed and recorded in
Phase 2.

## Repository layout

```
app.py                  Streamlit entry point
configs/default.toml    Run configuration
src/winemood/           Package (data, preprocessing, inference, ui)
tests/                  pytest suite (synthetic fixtures)
docs/                   Audit, research proposal, architecture, roadmap
notebooks/legacy/       Original 2025 Colab notebook (unmodified, superseded)
data/, artifacts/       Instructions only; contents are git-ignored
```

## Technology

Python, TensorFlow 2.20 + tf-keras (legacy LSTM), scikit-learn, pandas, NLTK,
Streamlit, pytest, Ruff, GitHub Actions. The rationale for each is in
[`docs/architecture.md`](docs/architecture.md).

## License

MIT. See [`LICENSE`](LICENSE). The dataset is subject to its own terms.
