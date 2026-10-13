# WineMood: Model Development

This document describes how each Phase 3 model is built, trained and
persisted. The experiment matrix lives in `configs/experiments.toml`, the
pre-registered rules in [`experimental-protocol.md`](experimental-protocol.md)
(including the §10 amendments), and the measured results in
[`experimental-results.md`](experimental-results.md).

## 1. Common pipeline

```
manifest ids ──► deduplicated corpus rows (fingerprints verified on every load)
      │
      ▼
TextPreprocessor (declared version; nothing learned)
      │
      ▼
features fitted on TRAIN only ──► head fitted on TRAIN ──► scored on VALIDATION
      │
      ▼
best configuration by the task's selection metric ──► saved artifact + predictions
```

- **Data access.** `winemood.experiments.data.load_split` returns train and
  validation frames. Before returning anything, it re-checks the dataset
  SHA-256 and each manifest's id fingerprint. It refuses to return the test
  partition unless it is called from the final-evaluation path.
- **Preprocessing versions** (`winemood.preprocessing.preprocessing_config`):

  | Version | Steps |
  |---|---|
  | `norm_min_v1` | Lower-case only. Tokenisation is left to the vectoriser. |
  | `norm_legacy_v0` | Lower-case, punctuation removed, NLTK English stopwords removed (which drops negations). This is the original notebook's pipeline. |
  | `keep_negations` | `norm_legacy_v0` without removing *no, not, nor, never, none, nothing, neither*. |

  Number masking runs before the steps above. `years` replaces
  19xx/20xx with the token `__year__`, and `all` replaces every digit run
  with `__num__`. The placeholders consist only of word characters, so
  they survive punctuation stripping and cannot collide with real words. The
  protocol text writes them as `<year>` and `<num>`.
- **Targets.**
  - T1 is the integer score.
  - T2 is `we_category_merged_v1` (five ordered classes).
  - T3 is `points_binary_90_v1`.
- **Clipping.** All regression predictions are clipped to [80, 100].

## 2. Classical models

| ID | Features | Head | Tuned on validation |
|---|---|---|---|
| B0 (E01) | none | constant = training mean | n/a |
| B1 (E02) | word count, character count (standardised) | OLS | n/a |
| B2 (E03) | one-hot taster (all 19 + missing), top-20 country, top-50 variety (+ "other"), log price with training-median imputation + missing flag | Ridge | α |
| B3 (E04) | word TF-IDF (sublinear tf, L2, smooth idf) | Ridge | preprocessing, n-gram range, min_df, α (84 configurations) |
| T2 (E21) | B3\* vectoriser | multinomial logistic regression | C, class weighting |
| T2 (E22) | B3\* vectoriser | Frank & Hall ordinal decomposition: one logistic model per threshold P(y > k); class probabilities from monotone cumulative differences | C |
| T3 (E31) | B3\* vectoriser | logistic regression | C, class weighting |

**Persistence without pickle.**
- `SafeTfidf` stores the vocabulary, idf weights and parameters as JSON and
  `.npy`. It reloads them through a fixed-vocabulary `CountVectorizer` plus
  explicit idf weighting and L2 normalisation. A test asserts the transform is
  numerically identical to scikit-learn's `TfidfVectorizer`.
- Heads store their coefficients as `.npy` and predict with NumPy.
- Metadata encoders store their category lists as JSON.
- `np.load` is always called with `allow_pickle=False`.

## 3. Neural models

### N1: Embedding + BiLSTM (E10)

| Component | Setting |
|---|---|
| Preprocessing | `norm_legacy_v0`, the original notebook's normalisation, as required by the "corrected reproduction" framing and the H5 contrast |
| Tokenisation | `\w+` tokens. A Keras `TextVectorization` layer (vocabulary ≤ 20,000) is adapted on the **training texts only**. |
| Sequence length | 95th percentile of training token counts (37 tokens on `grouped_v1`); post-padding, truncation |
| Network | Embedding (masking) → Bidirectional LSTM → Dropout → Dense(32, ReLU) → Dense(1) |
| Target | Standardised with the training mean and SD. The exported model ends in a `Rescaling` layer, so it outputs points. |
| Loss / optimiser | MSE / Adam, learning rate 1e-3, batch size 128 |
| Early stopping | Validation MAE, patience 2, best weights restored, at most 12 epochs |
| Grid (validation, seed 42) | embedding {64, 128} × LSTM units {32, 64} × dropout {0.2, 0.4} = 8 configurations |
| Seeds | 42–46 for the selected configuration. Reported as mean ± SD over seeds. |
| Determinism | `keras.utils.set_random_seed` plus TensorFlow op determinism on CPU. A test asserts identical predictions for identical seeds. |
| Persistence | Vectoriser → network → rescaling, saved as a native `.keras` file and loaded with `safe_mode=True` (no Lambda layers, no pickle) |

The original notebook's mistakes are avoided:
- the vocabulary is fitted after the split and on train only;
- there is a single model and a single selection criterion;
- the test set is never used for validation.

### N2: frozen sentence encoder (E12)

`sentence-transformers/all-MiniLM-L6-v2`, pinned to the model-repository
commit recorded in the run metadata, with these settings:
- weights loaded from `safetensors` only;
- inputs truncated to 128 tokens;
- mean pooling over the attention mask, then L2 normalisation;
- input text only number-masked (casing and punctuation kept, as the encoder
  expects).

A Ridge head is fitted on the training embeddings, with α tuned on
validation. The encoder is **never fine-tuned**; this was the project owner's
decision at the Phase 3 gate. Embeddings are cached on disk, keyed by model,
revision and input text.

## 4. Experiment tracking

Every run writes three things:
- `results/validation/<id>.json` (committed), containing:
  - the selected configuration and its hash;
  - the full validation grid;
  - validation metrics;
  - per-seed results;
  - metadata: run id, git commit and dirty flag, package versions, hardware,
    dataset hash, split id hashes, timing.
- `experiments/<split>/<id>/` (git-ignored), containing the model, the
  validation predictions keyed by `review_id`, `grid.csv` and the training
  histories.
- A row in `experiments/registry.csv`.

`winemood exp final` adds `results/test/<id>.json` and a line in the committed
`results/test_access_log.jsonl`. It refuses to run without a frozen
`results/selection.json`, and it refuses to re-evaluate a logged
configuration unless `--allow-repeat` is passed, which is itself logged.

## 5. Inference

`winemood.inference.regression.ScorePredictor` loads a saved T1 artifact,
either linear or LSTM. It returns a clipped score and the derived Wine
Enthusiast category. This is the interface the Phase 4 application will use.
