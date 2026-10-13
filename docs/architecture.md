# WineMood: Architecture

This document describes the architecture introduced in Phase 1, the reasons for
each decision and the tradeoffs accepted. It describes **what exists now**.
Components planned for later phases are marked *(planned)*.

## 1. Goals and non-goals

**Goals**

- One source of truth for every stage: data rules, label definitions,
  preprocessing and configuration are each defined once and shared by
  training and inference.
- Make the methodological safeguards (explicit labels, deduplication,
  split-before-fit) part of the code rather than conventions.
- Testable without the real dataset or trained models.
- Fail gracefully and informatively when data or artifacts are missing.

**Non-goals (deliberately excluded)**

- Microservices, REST APIs, databases, message queues, containers and
  experiment-tracking servers. The project is a single-researcher codebase
  with a demo UI, and none of these would pay for their operational cost.

## 2. Layout

```
configs/default.toml        Run configuration (paths, seed, labels, split, preprocessing)
app.py                      Streamlit entry point (thin shim)
src/winemood/
  config.py                 Typed, frozen dataclasses loaded from TOML
  logging_utils.py          key=value structured logging (stdlib)
  seeding.py                set_global_seed(): Python, NumPy, TensorFlow
  preprocessing.py          TextPreprocessor (shared by training and inference)
  data/
    ingest.py               load_reviews(): CSV loading, acquisition hint on absence
    schema.py               validate_reviews() -> ValidationReport
    labels.py               Versioned LabelScheme registry (points_band_v0)
    split.py                deduplicate(), split_dataset(), assert_no_text_overlap()
  inference/
    predictor.py            Predictor protocol, PredictionResult, artifact errors
    legacy_lstm.py          LegacyLSTMPredictor + restricted tokenizer unpickler
  ui/streamlit_app.py       Streamlit UI
  training/                 (planned, Phase 3)
  evaluation/               (planned, Phase 3)
tests/                      pytest suite on synthetic fixtures
notebooks/legacy/           Original Colab notebook, preserved unmodified
docs/                       Audit, research proposal, architecture, roadmap
```

## 3. Data flow

```
raw CSV ──ingest.load_reviews──► DataFrame
          │
          ▼
  schema.validate_reviews ──► ValidationReport (errors stop the pipeline;
          │                                     warnings are logged)
          ▼
  labels.assign_labels(scheme) ──► class_support() warns on empty classes
          │
          ▼
  split.split_dataset: deduplicate ─► stratified train/val/test ─► disjointness assertion
          │
          ▼  (only `train` may be used to fit anything below)
  preprocessing.TextPreprocessor ─► tokenizer/vectoriser fit   (planned, Phase 3)
          │
          ▼
  model training ─► evaluation ─► versioned artifacts           (planned, Phase 3)
          │
          ▼
  inference.Predictor ─► ui.streamlit_app
```

Today only the inference half runs end-to-end, and it needs externally
supplied legacy artifacts. The data half is implemented and tested on
synthetic data, but has not yet been run on the real dataset (Phase 2).

## 4. Key decisions

### 4.1 One installable package (`src/` layout)

*Decision:* A single `winemood` package installed with `pip install -e .`.
`app.py` is a five-line shim.
*Why:* The `src/` layout means tests run against the installed package, not
loose files in the working directory, so import errors show up in CI.
*Tradeoff:* One extra install step. `app.py` adds `src/` to `sys.path` so that
`streamlit run app.py` also works from a fresh clone.

### 4.2 Versioned label schemes, never edited in place

*Decision:* Each target definition is a named, immutable `LabelScheme`. The
legacy rule is preserved byte-for-byte as `points_band_v0`, with its known
defect documented in the scheme itself.
*Why:* The audit's central finding is about labels (A-01, A-02). Naming
schemes makes every result traceable to its labelling rule and rules out
silent redefinition.
*Tradeoff:* The UI still shows the legacy "Negative/Neutral/Positive" names,
together with their score thresholds and an explicit disclaimer. Renaming
them is deferred to the Phase 2 label decision.

### 4.3 Split before fit, enforced in code

*Decision:* `split_dataset` deduplicates on normalised text, splits stratified
DataFrames (preserving the original row index), then asserts that no text
appears in two splits.
*Why:* This addresses A-03 (row identity is kept for error analysis), A-07 and
A-08.
*Tradeoff:* Exact and whitespace/case duplicates are caught. Near-duplicates
(minor edits) are not, and that remains a Phase 2 question.

### 4.4 Framework-agnostic `Predictor` protocol

*Decision:* The UI depends only on `Predictor.predict(texts) -> list[PredictionResult]`.
*Why:* Phase 3 will add baselines (e.g. scikit-learn) and possibly other
architectures. They can plug in without UI changes, and tests can use fakes.
*Tradeoff:* A small amount of indirection.

### 4.5 TensorFlow kept as a core dependency, legacy artifacts through `tf-keras`

*Decision:* `tensorflow` and `tf-keras` are core dependencies, pinned to the same
minor version (2.20, verified together on Python 3.13). Legacy artifacts are
loaded with `tf_keras`.
*Why:* Keeping TensorFlow core was the project owner's decision, which
preserves the existing LSTM path. Recent TensorFlow ships Keras 3, so `tf-keras`
is needed for Keras 2 HDF5 models and tokenizers (audit A-16).
*Tradeoff:* A heavy install (hundreds of MB) and a slower CI. The `tf_keras`
compatibility package will eventually be retired. Phase 3 should save new
artifacts in a non-pickle format (`tokenizer.json`, native Keras format).

### 4.6 Restricted unpickling and JSON-first tokenizer loading

*Decision:* The loader prefers `tokenizer.json`. Pickles go through an
`Unpickler` that only materialises the Keras `Tokenizer` and plain containers,
remapping old Keras module paths.
*Why:* This addresses A-24 (arbitrary code execution) and A-16 (moved
classes).
*Tradeoff:* It reduces the risk but does not remove it. Artifacts must still
come from trusted sources.

### 4.7 Configuration: TOML + frozen dataclasses (stdlib only)

*Decision:* `configs/default.toml` is parsed with `tomllib` into frozen
dataclasses. Unknown keys and invalid split fractions are rejected.
*Why:* It is typed, validated and dependency-free. Hydra or Pydantic would add
dependencies without a current need.
*Tradeoff:* Fewer features (no CLI overrides, no config composition). Revisit
if Phase 3 needs experiment sweeps.

### 4.8 Logging: stdlib with a key=value formatter

*Decision:* `logging` with a formatter that appends `extra=` fields as
`key=value` pairs.
*Why:* Structured, greppable logs with no new dependency.

### 4.9 Errors are typed, and the UI degrades gracefully

*Decision:*
- Artifact problems raise `ArtifactNotFoundError` or `ArtifactIncompatibleError`.
- A missing dataset raises `DatasetNotFoundError` with acquisition instructions.
- Config problems raise `ConfigError`.
- The UI catches artifact errors, explains them and disables prediction.

*Why:* This addresses A-09. The original app returned `None` and crashed later
with an `AttributeError` (reproduced, audit §4).

### 4.10 The legacy notebook is preserved, not edited

*Decision:* The notebook was moved to `notebooks/legacy/` and left
unchanged.
*Why:* The audit cites its cells as evidence, so editing it would destroy the
record. The package is now the single source of the application code.

## 5. Testing strategy

- **Unit tests** cover config, schema validation, label schemes,
  deduplication and splitting, preprocessing, and the artifact error paths.
  All of them use synthetic DataFrames (`tests/conftest.py`) and a fixed
  stopword list, so the suite needs neither the dataset nor NLTK downloads.
- **Integration tests** build tiny TensorFlow models on the fly, save them in
  the legacy formats (HDF5 + pickle/JSON tokenizer) and load them through the
  production loader.
- **UI tests** use Streamlit's headless `AppTest` and check both the
  "no artifacts" state and an end-to-end prediction.

## 6. Known architectural gaps

- No training or evaluation modules yet (Phase 3).
- No artifact metadata (config, commit hash, data checksum) is written yet,
  because no artifacts are produced yet (Phase 3).
- No type checker in CI. Annotations are present, and adding `mypy` is
  deferred until the API stabilises.
