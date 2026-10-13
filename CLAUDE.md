# CLAUDE.md

Guidance for AI coding assistants working in this repository.

## Project

WineMood is an NLP research project that predicts a wine critic's score
(80–100) from the text of the tasting note. The primary task is regression;
the secondary tasks are Wine Enthusiast category and ≥ 90 classification.
It is **not** sentiment analysis (see `docs/problem-formulation.md`). The
project is developed in approval-gated phases (`docs/roadmap.md`). Do not
start work belonging to a later phase without explicit approval.

## Commands

```bash
pip install -e ".[dev]"     # light install: app + classical pipeline (Python 3.11-3.13)
pip install -e ".[neural]"  # TensorFlow, for BiLSTM experiments / the legacy loader only
pytest                      # tests: synthetic fixtures, no dataset/network needed
ruff check . && ruff format --check .

winemood data download      # fetch + verify the pinned SHA-256 (no credentials)
winemood data validate      # -> reports/validation.json
winemood data split         # manifests (git-ignored) + data/split_fingerprints.json
winemood data split --check # must print OK before any modelling
winemood data eda --publish # regenerates docs/data-report.md and docs/figures/

winemood exp list           # experiment matrix (configs/experiments.toml) and status
winemood exp run <ID ...>   # train + validate; never touches test
winemood exp select         # freeze results/selection.json
winemood exp final          # single, logged test evaluation of the frozen scope
winemood exp report         # regenerates results/report.json + docs/results-tables.md

winemood model package      # zip E04 + SHA-256 manifest + licence -> dist/models/
winemood model install X.zip  # verify + extract -> models/e04 (git-ignored)

streamlit run app.py        # Predict / Research / Explore (docs/application.md)
```

On Windows, set `TF_CPP_MIN_LOG_LEVEL=2` to silence TensorFlow logging.

## Layout

- `src/winemood/`: the package. `app.py` is only a shim.
- `src/winemood/data/protocol.py`: the research split protocol. Use it, not
  the Phase 1 `split.split_dataset`, for any experiment.
- `configs/default.toml`: the single source for paths, checksum, seed,
  targets, near-duplicate and split parameters.
- `docs/data-report.md` and `docs/results-tables.md`: **auto-generated, never
  edit by hand**.
- `results/`: committed, small results (validation/test JSON, selection, test
  access log). `experiments/`: git-ignored models and predictions.
- **The Phase 3 test set has been used.** `results/test_access_log.jsonl` is the
  record. Do not re-run `exp final` or tune anything against test results. Any
  new model work needs a new, pre-registered evaluation plan.
- `docs/experimental-protocol.md`: pre-registered Phase 3 protocol. Record any
  deviation in its §10.
- `notebooks/legacy/`: the original notebook. **Do not edit it**, because the
  audit cites its cells.
- `src/winemood/app/`: the application. `services/` must stay Streamlit-free
  and unit-tested. The UI reads **only** committed `results/` aggregates and
  the installed model, never test predictions, raw data or `load_split`. A
  guard test enforces this.

## Application rules

- The served model is E04 with `norm_min_v1`. Never apply `norm_min_v2` (the
  A-31 fix) to it; the model service refuses mismatched preprocessing.
- Every number in the UI comes from `results/`. Never hard-code metrics in
  UI code.
- Never show per-prediction confidence, probabilities, intervals or
  "sentiment" wording. Show aggregate performance only, labelled as such.
- Explanations must reconstruct the model output exactly
  (`app/services/explain.py`).
- Keep the app TensorFlow-free. The light CI job runs without it.

## Non-negotiable rules

1. **Never fabricate results.** No metric, table or claim goes into the docs or
   UI unless a stored, reproducible run produced it. Never state that tests
   passed without running them.
2. **Labels are explicit and versioned.** Add new schemes in
   `winemood/data/labels.py` under new names, and never change an existing
   scheme's rule. Never call score-derived labels "sentiment".
   `points_band_v0` is legacy-only (`research_use=False`).
3. **Split before fit.** Use the `grouped_v1` manifests. Fit any tokenizer,
   vectoriser, scaler or merge rule on train only. Validation is for model
   selection; the test set is used once per final model and is never read
   during EDA or tuning.
4. **One preprocessing implementation.** Training and inference both use
   `winemood.preprocessing.TextPreprocessor` with the same config.
5. **Seed everything** with `winemood.seeding.set_global_seed`.
6. **Fail gracefully.** Missing data or artifacts raise typed errors
   (`DatasetNotFoundError`, `AcquisitionError`, `ArtifactError` subclasses).
   Never return `None` to signal failure.
7. **Tests use synthetic data.** Do not add tests that download the dataset.
8. **No new technology without justification** recorded in
   `docs/architecture.md`. Avoid services, databases and heavy frameworks.
9. **Data licence is CC BY-NC-SA 4.0.** Do not commit data (`data/raw/`),
   manifests (`data/processed/`) or artifacts (`*.h5`, `*.pickle`). Report
   tasters only as anonymised aggregates. Do not push to GitHub unless asked.

## Conventions

- Type annotations on public functions. Frozen dataclasses for config and results.
- Logging through `winemood.logging_utils.get_logger`, with structured fields
  passed via `extra={...}`. Do not use reserved `LogRecord` keys such as `name`.
- Keep source files ASCII. Write non-ASCII characters in regexes as escapes
  in raw strings, because pandas 3 may route string regexes to PyArrow/RE2.
- Ruff (line length 100) is the formatter and linter.
- All documentation is in English.
