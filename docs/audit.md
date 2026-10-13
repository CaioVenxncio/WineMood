# WineMood: Technical and Scientific Audit

**Scope:** the repository as of commit `e80f649` (2026-02-07): `app.py`,
`colab_notebook.ipynb`, `README.md`, `requirements.txt`, `packages.txt`.
**Date:** 2026-10-12.
**Method:** I read every file, inspected the git history, and checked claims
against the code. Where a claim depends on the external dataset or third-party
library behaviour, the evidence column says so, and items marked
*verify* were not executed during this audit.

**Severity scale**

| Level | Meaning |
|---|---|
| Critical | Invalidates the scientific claims or prevents the system from working. |
| High | Biases results materially or creates serious reliability/security risk. |
| Medium | Degrades maintainability, reproducibility or correctness in edge cases. |
| Low | Cosmetic, stylistic or minor efficiency issues. |

"Notebook cell" references use the cell IDs stored in the `.ipynb` file
(the notebook now lives at `notebooks/legacy/colab_notebook.ipynb`).

---

## 1. Summary

The original WineMood project presents itself as a three-class *sentiment*
classifier. The audit finds that:

1. The target labels are thresholds on critic scores, not sentiment (A-02).
2. One of the three classes cannot occur in the dataset (A-01).
3. The repository contains **no executed notebook outputs and no trained
   artifacts**, so no performance figure has ever been recorded (A-05, A-06).
4. The training pipeline contains several sources of leakage (A-04, A-07,
   A-08).
5. The error-analysis code prints texts that do not correspond to the
   misclassified samples (A-03).
6. As shipped, the Streamlit app cannot produce a prediction (A-06, A-09).

None of these findings implies the project idea is unsound. They mean the
existing code cannot support the claims made in the README, and that every
quantitative result must be produced afresh under a corrected methodology.

## 2. Findings

### Scientific validity and methodology

| ID | Severity | Finding | Evidence | Impact | Recommendation | Phase |
|---|---|---|---|---|---|---|
| A-01 | Critical | **The "Negative" class is structurally empty.** The rule labels `points < 80` as negative, but WineEnthusiast only publishes wines scoring 80 or above. | Notebook cell-13 (`create_sentiment_label`). The README itself states "points: wine score (from 80 to 100)". The exact minimum must be confirmed on the downloaded CSV in Phase 2 (`winemood.data.schema.validate_reviews` reports it). | The model has a 3-way softmax, but in practice it is trained on 2 classes. Any "Negative" prediction is an artefact of the untrained output unit. The confusion matrix and classification report would show an empty row. | Do not use this scheme for claims. Define new, explicitly versioned label schemes in Phase 2. The legacy scheme is preserved as `points_band_v0`, and `class_support()` warns about empty classes. | 2 |
| A-02 | Critical | **The labels measure critic score bands, not sentiment.** The `points` score and the `description` are written by the same taster for the same wine. The task is therefore to predict the critic's own score band from their tasting note. | Notebook cells 12–13; README §"Label Creation". | Calling outputs "sentiment" is unsupported. Tasting notes are descriptive and almost uniformly non-negative in register. An 85-point wine is described as "good", not "neutral". Neutral/negative polarity labels have no human annotation behind them. | Reframe the task as *score-band prediction from tasting-note language* (see `docs/research-proposal.md`). Reserve the word "sentiment" for human-annotated polarity, if that is ever collected. | 2 |
| A-03 | Critical | **The error analysis is indexed incorrectly.** `df_filtered.index.isin(X_test[sample_indices])` tests whether DataFrame row labels appear among the *padded token IDs* of the selected test rows. | Notebook cell-30. | The "misclassified examples" printed are arbitrary rows whose index happens to equal a token ID. They are unrelated to the misclassifications, and the true/predicted labels shown beside them are mismatched. Qualitative conclusions drawn from this output are invalid. | Carry the original row index through the split (split DataFrames, not arrays, as `winemood.data.split.split_dataset` does) and look texts up by that index. | 3 |
| A-04 | Critical | **The last notebook cells overwrite the model with an inconsistent one trained against the test set.** Cell `171d5195` builds a new model with `Embedding(input_dim=5000)` while the tokenizer uses `num_words=10000`, trains it with `validation_data=(X_test, y_test)`, and cell `19efd130` saves it over `model.h5`. | Notebook cells `171d5195`, `19efd130`. | (a) Token IDs 5000–9999 are out of range for the embedding. On CPU, TensorFlow raises an error. On some GPU kernels, out-of-range lookups are known to return zeros silently *(verify)*. (b) If it runs, the shipped `model.h5` is not the model evaluated in cell-26. (c) Using the test set for validation turns it into a selection set. | Delete these cells' behaviour from any future pipeline: one training entry point, one artifact, and the test set touched once. | 3 |
| A-05 | Critical | **No result is recorded anywhere.** All 37 code cells have `execution_count: null` and empty outputs. The README's "Accuracy above 80%" appears under "Expected Results" and is not backed by any run. The notebook conclusion asserts "O modelo alcançou uma boa acurácia" ("the model achieved good accuracy") without a number. | `colab_notebook.ipynb` (no outputs). README §"Expected Results". Notebook cell-41. | Any performance claim about WineMood is currently unverifiable. | Remove unsupported claims (done in the README). Phase 3 must produce metrics from a logged, reproducible run. | 1 (docs), 3 |
| A-07 | High | **The tokenizer is fitted on the full corpus before the split.** | Notebook cell-18 (`fit_on_texts(df_filtered['processed_text'])`) precedes cell-20 (`train_test_split`). | Test-set vocabulary and frequency ranks leak into the feature space. The leak is mild for a frequency-capped vocabulary, but it is a methodological error. | Split first, then fit any tokenizer or vectorizer on the training split only. `split_dataset` now runs before any fitting by design. | 3 |
| A-08 | High | **No deduplication before splitting.** The dataset is known to contain repeated descriptions *(measured in Phase 2: 10,016 exact duplicate descriptions, 7.7 % of rows)*. | Notebook cells 12–20 contain no `drop_duplicates`. | Identical texts can sit in both train and test, which inflates test scores. | Deduplicate on normalised text before splitting, and assert disjointness. Implemented in `winemood.data.split` (`deduplicate`, `assert_no_text_overlap`). | 2 |
| A-10 | High | **There is no baseline, and accuracy is the only headline metric.** The resulting two-class problem is imbalanced *(measured in Phase 2: 62.0 % vs 38.0 % after deduplication)*. | Notebook cells 22–28. | Without a majority-class or linear baseline, no accuracy figure is interpretable. A model predicting only "Neutral" may score highly. | Report a majority baseline, TF-IDF + logistic regression, macro-F1, balanced accuracy and per-class metrics. | 3 |
| A-11 | High | **Stopword removal deletes negations.** NLTK's English list contains `no`, `not`, `nor`, plus contraction fragments. | Notebook cell-16. `app.py:107`. | "not balanced" and "balanced" become identical inputs, which removes polarity cues that any sentiment-like task depends on. | Treat negation handling as an explicit, tested experimental factor. `PreprocessingConfig.keep_negations` exists but defaults to the legacy behaviour, so model inputs are not changed silently. | 3 |
| A-12 | High | **No random seeds are set** for NumPy, Python or TensorFlow. Only `train_test_split` uses `random_state=42`. The error-analysis sample uses unseeded `np.random.choice`. | Notebook cells 3, 20, 22–24, 30. | Runs are not reproducible. | Seed everything through `winemood.seeding.set_global_seed`. | 1 (tooling), 3 |
| A-13 | Medium | **Inconsistent model-selection criteria.** `EarlyStopping` monitors `val_loss` and `ModelCheckpoint` monitors `val_accuracy`. | Notebook cell-23. | The restored "best" weights and the saved checkpoint can come from different epochs. | Use one selection criterion, chosen before training and recorded in the config. | 3 |
| A-14 | Medium | **The validation split is unstratified.** `validation_split=0.1` takes the last 10 % of the (already shuffled) training arrays without stratification. | Notebook cell-24. | Minor: the class balance of the validation set may drift. | Create an explicit stratified validation split (implemented in `split_dataset`). | 3 |
| A-15 | Medium | **Hand-crafted "test examples" are out of distribution.** The examples in cell-35 (e.g. "Terrible wine, tastes like vinegar") differ in register from professional tasting notes and would be scored < 80, a band absent from training. | Notebook cell-35. | They show nothing about generalisation, yet they look like evidence. | Clearly label such probes as qualitative. Collect them as a small, separate challenge set if the topic is studied (Phase 3). | 3 |

### Software reliability

| ID | Severity | Finding | Evidence | Impact | Recommendation | Phase |
|---|---|---|---|---|---|---|
| A-06 | Critical | **Model artifacts are missing.** `model.h5` and `tokenizer.pickle` are not in the repository, and git history shows they were never committed. | `git log --all --name-status` lists only `.ipynb`, `.md` and `.txt` files. `app.py:82-86`. | The app cannot make predictions from a fresh clone or a Streamlit Cloud deployment. | Document artifact acquisition (`artifacts/README.md`) and fail gracefully (done). Produce versioned artifacts from a reproducible pipeline (Phase 3). | 1, 3 |
| A-09 | Critical | **The app crashes when artifacts fail to load.** `load_model_and_tokenizer` returns `(None, None)` on any exception, and the app keeps going and calls methods on `None`. | `app.py:89-91`, `app.py:111`. **Reproduced** (§4): `AttributeError: 'NoneType' object has no attribute 'texts_to_sequences'`. | The user sees a traceback instead of a diagnosis. | Typed artifact errors (`ArtifactNotFoundError`, `ArtifactIncompatibleError`) and a UI state that disables prediction and explains the fix (done). | 1 |
| A-16 | Medium | **Unbounded TensorFlow version breaks legacy artifact loading.** `tensorflow>=2.15.0` resolves to the newest TensorFlow, which ships Keras 3. The `app.py:6` import itself **still works**: under TF 2.20 / Keras 3.15, `tensorflow.keras.preprocessing.text.Tokenizer` resolves to `keras.src.legacy.preprocessing.text`. However, the module `keras.preprocessing.text`, under which Keras 2-era tokenizers were pickled, no longer exists, so `pickle.load` of such a tokenizer fails. | `requirements.txt:2`. `app.py:6-7, 85-86`. Verified (§4). The test `test_keras2_era_tokenizer_pickle_is_remapped` reproduces the pickle failure. | A tokenizer produced by the original Colab notebook (TF 2.12 era) would not load in a fresh install. | Load legacy artifacts through `tf-keras` (the Keras 2 compatibility package), remap the legacy class path in a restricted unpickler, and pin mutually compatible versions (done). | 1 |
| A-17 | Medium | **Dependency drift.** The notebook pins `streamlit==1.22.0`, `tensorflow==2.12.0`, `numpy==1.23.5` (cell-37). Cell `19efd130` then overwrites `requirements.txt` with unpinned names. The repository pins only `streamlit==1.45.1`. `seaborn` is imported (cell-3) but declared nowhere. | `requirements.txt`. Notebook cells 3, 37, `19efd130`. | Training and serving environments cannot be reconstructed. | One authoritative dependency declaration (`pyproject.toml`) with tested ranges (done). | 1 |
| A-18 | Medium | **`packages.txt` contains `python3.10`.** On Streamlit Community Cloud, `packages.txt` lists *apt* packages. It does not select the Python version. | `packages.txt`. | The file does not do what it appears to do. The Python version is set in the Cloud app settings. | Remove the file (done). Document the Python requirement in `pyproject.toml`. | 1 |
| A-19 | Medium | **Training/serving preprocessing is duplicated.** The cleaning function is defined twice (notebook cell-16 and `app.py:94`), and `max_length=100` is hardcoded separately. | `app.py:94-114`. Notebook cells 16, 18. | Silent train/serve skew if either copy changes. | One `TextPreprocessor` and one config value used by every stage (done for inference; training adopts it in Phase 3). | 1 |
| A-20 | Medium | **The notebook embeds a full copy of `app.py`** (`%%writefile app.py`, cell-40). | Notebook cell-40. | Two diverging sources of the application. | Treat the notebook as a historical record (moved to `notebooks/legacy/`). The package is the single source. | 1 |
| A-21 | Low | **Inefficient inference.** The app calls the NLTK resource lookup and rebuilds the stopword set on every prediction, and calls `model.predict` once per input line. | `app.py:102-108`, `app.py:208-212`. | Slow batch classification. | Cache stopwords once and predict in a single batch (done). | 1 |
| A-22 | Low | **Duplicated plotting code and heavy inline CSS/HTML.** Two copies of the bar-chart code and `unsafe_allow_html=True` throughout. | `app.py:146-177`, `app.py:244-274`, `app.py:25-75`. | Maintainability. | A single chart helper with native Streamlit components (done). | 1 |
| A-23 | Low | **`recurrent_dropout` disables cuDNN kernels** in TF2 LSTM layers, which slows GPU training substantially. | Notebook cell-22. | Performance only. | Revisit during Phase 3 model design. | 3 |

### Security

| ID | Severity | Finding | Evidence | Impact | Recommendation | Phase |
|---|---|---|---|---|---|---|
| A-24 | Medium | **Unrestricted `pickle.load` of the tokenizer.** | `app.py:85-86`. | Loading an untrusted pickle executes arbitrary code. | A restricted unpickler that only materialises the Keras `Tokenizer` and plain containers. Prefer `tokenizer.json` (done). Document that artifacts must come from trusted sources. | 1 |
| A-25 | Low | **Kaggle API credentials are handled inside the notebook** (`files.upload()` of `kaggle.json`). | Notebook cell-5. | Risk of credentials being committed or shared alongside the notebook. | Document credential handling outside the repository. `kaggle.json` is now in `.gitignore`. | 1 |
| A-26 | Low | **`unsafe_allow_html=True`** is used for static content. User input is not interpolated, so there is no injection path today, but the pattern is fragile. | `app.py:139-143`. | Future XSS risk if user text is ever interpolated. | Use native components (done). | 1 |

### Documentation and provenance

| ID | Severity | Finding | Evidence | Impact | Recommendation | Phase |
|---|---|---|---|---|---|---|
| A-27 | High | **The README makes unsupported or inaccurate claims:** "Accuracy above 80%", "Fully functional project", "Ready for commercial platform integration", "classifies emotions". | README §§"Expected Results", "Project Highlights", "Objectives". | Misleading to readers and reviewers. | Rewrite the README to describe only what is implemented (done). | 1 |
| A-28 | Medium | **The license is claimed but absent.** The README refers to an MIT `LICENSE` file that does not exist. | README §"License". | Legal ambiguity for reuse. | Add `LICENSE` (done, MIT, as the README stated). | 1 |
| A-29 | Medium | **Dataset provenance and terms are undocumented.** The Kaggle dataset was scraped from WineEnthusiast. Its license and terms of use, and the redistribution status of the reviews, are not recorded. | README §"Dataset". | Possible restrictions on redistributing data or derived artifacts. | Record source, version, licence and access date in `data/README.md`. Confirm the licence on the Kaggle page before any redistribution *(Phase 2: licence confirmed as CC BY-NC-SA 4.0 from Kaggle metadata)*. | 2 |
| A-30 | Low | **Language inconsistency.** The UI and notebook are in Portuguese and the README is in English. | `app.py`, notebook. | Inconsistent presentation. | UI converted to English (done). The notebook is left unchanged as a historical record. | 1 |

## 3. Confounds and threats specific to this dataset

These are not defects in the code. Phase 2 must address them in the research design.

- **Taster identity.** A small number of tasters wrote most reviews. Vocabulary
  and score calibration are taster-specific, so a model may learn to identify
  the author rather than evaluative language. Mitigations: grouped splits by
  `taster_name` and per-taster analysis.
- **Length and score correlation.** Higher-scored wines tend to receive longer
  descriptions *(confirmed in Phase 2: Spearman rho = 0.51 on the training split)*. A length-only baseline is required to rule out a
  trivial shortcut.
- **Metadata leakage in text.** Descriptions sometimes mention price, variety
  or region, which correlate with score. This is legitimate signal for the
  task, but it must be acknowledged when interpreting "linguistic" findings.
- **Threshold artefacts.** Scores cluster around 85–90, and a hard cut at 90
  places near-identical wines (89 vs 90) in different classes. Ordinal or
  regression formulations should be compared.

## 4. Verification log

The following checks were executed during the Phase 1 implementation:

| Check | Result |
|---|---|
| Artifacts in git history (`git log --all --name-status`) | None found. Only `.ipynb`, `.md` and `.txt` files were ever committed. |
| Executed outputs in the notebook | None. All code cells have `execution_count: null`. |
| `from tensorflow.keras.preprocessing.text import Tokenizer` with tensorflow 2.20.0 / keras 3.15.1 / Python 3.13.14 (A-16) | **Succeeds.** The class resolves to `keras.src.legacy.preprocessing.text`. The planning-stage expectation that this import fails was **wrong** and has been corrected. `from keras.preprocessing import text` fails with `ImportError`. |
| Plain `pickle.loads` of a tokenizer whose class path is `keras.preprocessing.text.Tokenizer` (A-16) | Fails under Keras 3, and loads through `_TokenizerUnpickler` (automated test). |
| Original `app.py` (git `HEAD`) run headless with Streamlit `AppTest`, no artifacts present (A-06, A-09) | On load: `st.error` "Erro ao carregar o modelo: [Errno 2] Unable to synchronously open file …". After clicking *Classify*: uncaught `AttributeError: 'NoneType' object has no attribute 'texts_to_sequences'`. |

## 5. Resolution status

Findings resolved in Phase 1: A-05 (documentation part), A-06 (graceful
handling), A-09, A-16, A-17, A-18, A-19 (inference side), A-20, A-21, A-22,
A-24, A-25 (ignore rule), A-26, A-27, A-28, A-30.

Findings with tooling in place, awaiting application in later phases: A-01
(`class_support` warning), A-07/A-08/A-14 (`split_dataset`), A-11
(`keep_negations` flag), A-12 (`set_global_seed`).

**Phase 2 (2026-10-12)**

- **Resolved:**
  - A-01: confirmed by Wine Enthusiast's policy and by the data (0 reviews
    below 80). The scheme is retired as a research target.
  - A-02: reformulated as score regression and score-category
    classification; see `docs/problem-formulation.md`.
  - A-08: exact deduplication, near-duplicate and same-wine grouping, and an
    isolation assertion, run on the real data.
  - A-29: licence and provenance recorded.
- **Methodology fixed, implementation in Phase 3:** A-03 (error analysis by
  `review_id`), A-04 and A-13 (single training entry point and selection
  criterion), A-10 (baselines and metrics), A-15 (qualitative probes). See
  `docs/experimental-protocol.md`.
- **Open:** A-23 (LSTM design, Phase 3).

## 6. Phase 2 verification log

| Check | Result |
|---|---|
| Kaggle metadata (public API) | Licence CC BY-NC-SA 4.0, dataset version 4, last updated 2017-11-27, three files. |
| Unauthenticated download | Works: the endpoint redirects to a signed storage URL. 17,180,972-byte zip, 52,908,706-byte CSV, SHA-256 `52af2643…67ac9` (pinned). |
| `points` below 80 (A-01) | 0 of 129,971. Range 80–100, all integers. |
| Exact duplicate descriptions (A-08) | 10,016 (10,017 after case and whitespace normalisation). 23 duplicate groups carry conflicting scores. |
| Near-duplicates | 26 verified pairs (Jaccard ≥ 0.8) after exact deduplication. Detector recall on planted near-duplicates of real reviews: 0.99 at one word replaced, 0.93 at three. No false merges. |
| Split isolation on real data | 0 review IDs, 0 normalised texts, 0 titles and 0 groups shared across train, validation and test. This was checked by the package and again by independent pandas code. |
| Split reproducibility | `winemood data split --check` in a fresh process reproduced the committed fingerprints. |
