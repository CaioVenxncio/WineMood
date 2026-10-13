# WineMood: Reproducibility Guide

These are the exact steps to reproduce every Phase 3 number from a clean
clone. All numbers in [`results-tables.md`](results-tables.md) are regenerated
from stored predictions by `winemood exp report`.

> **Commit history note (audit A-32).** The Phase 3 commits `15bc2a6` and
> `4f6d77a` do not contain `src/winemood/experiments/`, because of an
> over-broad `.gitignore` rule. The package was committed in `a5f2313`, and
> history was intentionally not rewritten. **Reproduce Phase 3 from `a5f2313`
> or later.** The committed package is the code that executed the frozen
> selection and the test evaluation (see `docs/audit.md`, A-32). All result
> records are unchanged.

## 1. Environment

| Item | Version used for the reported results |
|---|---|
| OS | Windows 10 (native, CPU only) |
| Python | 3.13 |
| TensorFlow / Keras | 2.20.0 / 3.15.1 (CPU; native-Windows TensorFlow has no GPU support) |
| scikit-learn / NumPy / pandas / SciPy | 1.9.1 / 2.5.4 / 3.0.6 / 1.18.1 |
| PyTorch / transformers (N2 only) | 2.14.1+cpu / 5.19.0 |
| Hardware | Intel i5-9400F (6 cores), 16 GB RAM |

Each run records its own package versions, hardware, git commit and dirty
flag in `results/validation/<id>.json` → `meta`.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# Optional, for the frozen-encoder experiment (E12):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[transformer]"
```

## 2. Data and splits

```bash
winemood data download         # verifies SHA-256 52af2643…67ac9
winemood data split            # grouped_v1, taster_holdout_v1, taster_random_v1 (+ groups.csv)
winemood data split --check    # must print "OK: splits reproduce the committed fingerprints"
```

Every experiment re-verifies the dataset hash and the manifest fingerprints
before it loads any data.

## 3. Experiments (train + validation only)

```bash
winemood exp list                                   # matrix, status, test-scope flag
winemood exp run E01_dummy_mean E02_length E03_metadata E04_tfidf_word
winemood exp run E05_tfidf_char E06_tfidf_word_char E07_tfidf_plus_length \
                 E08_tfidf_mask_years E09_tfidf_mask_all E13_tfidf_keep_negations
winemood exp run E20_t2_majority E20b_t2_stratified E21_t2_logistic E22_t2_ordinal \
                 E30_t3_majority E31_t3_logistic E40_holdout_tfidf E42_random_tfidf
winemood exp run E12_encoder                        # needs the [transformer] extra
winemood exp run E10_lstm --pilot                   # optional: time one epoch
winemood exp run E10_lstm E11_lstm_keep_negations E41_holdout_lstm E43_random_lstm
```

Derived experiments (`base = ...`) need their base experiment to be run first.
The order above respects this.

**Seeds and determinism**
- The global seed is 42 (`configs/default.toml`).
- LSTM seeds are listed per experiment: 42–46 for E10 and E11, and 42–44
  for E41 and E43.
- Classical models are deterministic, and a test asserts that rerunning
  reproduces the same validation MAE exactly.
- The LSTM uses TensorFlow op determinism on CPU, and a test asserts
  identical predictions for identical seeds on the same machine. Results on
  other hardware or library versions may differ slightly.

## 4. Selection and the single test evaluation

```bash
winemood exp select     # freezes results/selection.json (rule: docs/model-selection.md)
winemood exp final      # evaluates ONLY the frozen test scope; appends results/test_access_log.jsonl
winemood exp report     # bootstrap CIs, hypotheses, error analysis, figures, docs/results-tables.md
```

`exp final` refuses to run without a frozen selection, and it refuses to
re-evaluate an already-logged configuration unless `--allow-repeat` is passed,
which is itself logged. The committed access log shows when the test set was
used.

## 5. Artifacts

| Location | Content | Committed |
|---|---|---|
| `results/validation/*.json` | selected configs, grids, validation metrics, run metadata | yes |
| `results/test/*.json`, `results/report.json`, `results/selection.json`, `results/test_access_log.jsonl` | test metrics, statistics, selection, access log | yes |
| `experiments/<split>/<id>/` | models (`.keras`, JSON + `.npy`), predictions by `review_id`, histories | no (git-ignored) |
| `experiments/cache/encoder/` | cached sentence embeddings | no |

Trained models are not committed. They are derived from CC BY-NC-SA data, and
they can be regenerated with the commands above. All artifacts load without
pickle: `.keras` files with `safe_mode=True`, and linear models from JSON +
`.npy` with `allow_pickle=False`.

## 6. Runtime (measured on the hardware above)

The per-experiment training durations are recorded in each
`results/validation/<id>.json` → `meta.seconds`, and are summarised in
[`experimental-results.md`](experimental-results.md).
