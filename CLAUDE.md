# CLAUDE.md

Guidance for AI coding assistants working in this repository.

## Project

WineMood is an NLP research project. It predicts critic score bands from
wine tasting notes, and it is **not** sentiment analysis (see
`docs/research-proposal.md` §2). It is developed in approval-gated phases
(`docs/roadmap.md`). Do not start work belonging to a later phase without
explicit approval.

## Commands

```bash
pip install -e ".[dev]"     # install (Python 3.11-3.13)
pytest                      # tests: synthetic fixtures, no dataset/network needed
ruff check . && ruff format --check .
streamlit run app.py
```

On Windows, set `TF_CPP_MIN_LOG_LEVEL=2` to silence TensorFlow logging.

## Layout

- `src/winemood/`: the package. `app.py` is only a shim.
- `configs/default.toml`: the single source for paths, seed, label scheme,
  split and preprocessing parameters.
- `notebooks/legacy/`: the original notebook. **Do not edit it**, because the
  audit cites its cells.
- `docs/`: audit, research proposal, architecture, roadmap.

## Non-negotiable rules

1. **Never fabricate results.** No metric, table or claim goes into the docs or
   UI unless a stored, reproducible run produced it. Never state that tests
   passed without running them.
2. **Labels are explicit and versioned.** Add new `LabelScheme`s in
   `winemood/data/labels.py` under new names. Never change an existing
   scheme's rule.
3. **Split before fit.** Use `winemood.data.split.split_dataset`. Any
   tokenizer, vectoriser or scaler is fitted on the training split only.
   The test set is not used for model selection.
4. **One preprocessing implementation.** Training and inference both use
   `winemood.preprocessing.TextPreprocessor` with the same config.
5. **Seed everything** with `winemood.seeding.set_global_seed`.
6. **Fail gracefully.** Missing data or artifacts raise typed errors
   (`DatasetNotFoundError`, `ArtifactError` subclasses) and the UI explains
   them. Never return `None` to signal failure.
7. **Tests use synthetic data.** Do not add tests that download the dataset.
8. **No new technology without justification** recorded in
   `docs/architecture.md`. Avoid services, databases and heavy frameworks.
9. Do not commit data (`data/raw/`) or artifacts (`*.h5`, `*.pickle`). Do not
   push to GitHub unless asked.

## Conventions

- Type annotations on public functions. Frozen dataclasses for config and results.
- Logging through `winemood.logging_utils.get_logger`, with structured fields
  passed via `extra={...}`.
- Ruff (line length 100) is the formatter and linter.
- All documentation is in English.
