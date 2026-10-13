# WineMood: Modernisation Roadmap

There are five sequential phases. A phase begins only after the previous phase's
acceptance criteria are met and the project owner has approved it. Each
criterion is designed to be checked objectively.

---

## Phase 1: Repository audit and engineering foundations ✅ (committed locally on branch `modernization`)

**Deliverables**

- `docs/audit.md`: severity-rated findings with evidence and a verification log.
- `docs/research-proposal.md`, `docs/architecture.md`, `docs/roadmap.md`,
  `README.md`, `CLAUDE.md`.
- The `winemood` package: config, logging, seeding, data validation, label
  schemes, leakage-safe split, preprocessing, inference interface, English
  Streamlit UI.
- `pyproject.toml`, Ruff, pytest suite on synthetic fixtures, and a GitHub
  Actions CI workflow.
- Dataset and artifact acquisition instructions.

**Dependencies:** none.

**Risks**

- TensorFlow and tf-keras compatibility may drift in future releases.
  Mitigated by bounded version ranges.

**Acceptance criteria**

- [x] `ruff check .` and `ruff format --check .` pass locally.
- [x] `pytest` passes locally without the dataset or network access.
- [x] The app starts with no artifacts and shows guidance instead of a
      traceback (automated `AppTest`).
- [x] The README contains no performance claim that is not backed by a stored
      run.
- [ ] The CI workflow passes on GitHub. It has not been run yet, because
      nothing has been pushed.

---

## Phase 2: Dataset validation and scientific methodology ✅ (implemented, awaiting review)

**Outcome:**
- Primary target is score regression; secondary targets are Wine Enthusiast
  category and binary ≥ 90 classification. See `docs/problem-formulation.md`.
- Data facts are in `docs/data-report.md`.
- The Phase 3 protocol is pre-registered in `docs/experimental-protocol.md`.

**Deliverables (as planned)**

- A reproducible dataset acquisition script, with a recorded SHA-256 checksum
  and licence/terms documented in `data/README.md`.
- A data report produced by code, never typed by hand:
  - row counts, missing values, exact and near-duplicate counts;
  - score distribution;
  - taster coverage;
  - length statistics.
- A label decision record: the chosen scheme(s) from the research proposal
  §5, implemented as new versioned `LabelScheme`s with tests. The legacy
  scheme is retained for reference only.
- A frozen split specification: seed, fractions, and grouping for the random
  and taster-grouped splits, stored as row-ID manifests.
- A pre-registration note listing the metrics, baselines, confidence-interval
  method and decision thresholds for H1–H5, written **before** any model is
  trained.
- References verified against original sources.

**Dependencies:** Phase 1. Kaggle access for the dataset.

**Risks**

- The licence may restrict redistribution of the data or derived artifacts.
- Near-duplicate detection may be costly at 130k rows.
- `taster_name` missingness may weaken the grouped split.

**Acceptance criteria**

- [x] `validate_reviews` runs on the real CSV. Its results are committed in
      the generated `docs/data-report.md`; the full JSON is regenerated into
      the git-ignored `reports/`.
- [x] Split manifests are reproducible. A test covers synthetic data, and
      `winemood data split --check` reproduced the committed fingerprints on
      the real data.
- [x] Train, validation and test share no texts, titles or groups. This is
      asserted on real data by the package and independently re-checked.
- [x] Every label class has non-zero support in every split. The weakest
      case is Classic in `taster_holdout_v1` validation, with 1 review.
      `we_category_merged_v1` removes that weakness for modelling.
- [x] The pre-registration (`docs/experimental-protocol.md`) was written
      before any Phase 3 work.
- [ ] Phase 2 is merged after review (pending the project owner).

---

## Phase 3: Model development, baselines and rigorous evaluation ✅ (implemented, awaiting review)

**Outcome:**
- **Selected model.** B3\* (word TF-IDF + Ridge) has test MAE 1.302
  [1.289, 1.315].
- **Neural models.** The BiLSTM is 0.016 MAE better, which is statistically
  detectable but negligible. The frozen encoder is clearly worse.
- **Hypotheses.** H1, H2 (BiLSTM only), H3 and H5 are supported; H4 is
  reported as an effect size.
- **New finding.** Audit A-31 (score ranges in the text) was found and
  quantified.

Details are in `docs/experimental-results.md`.

**Deliverables (as planned)**

- `winemood.training` and `winemood.evaluation` modules with a config-driven
  CLI.
- Baselines: majority class, length-only, TF-IDF + linear model.
- A corrected reproduction of the legacy LSTM. Optionally, a pre-trained
  transformer if justified.
- An evaluation suite:
  - macro-F1, balanced accuracy, per-class metrics, calibration;
  - bootstrap confidence intervals;
  - ≥ 3 seeds per neural model.
- Error analysis indexed by preserved row IDs, which fixes audit A-03.
- Ablations: negation handling (H5) and the taster-grouped split (H3).
- Versioned artifacts stored with their config, git commit, data checksum and
  metrics (`tokenizer.json` / native Keras format, not pickle).

**Dependencies:** Phase 2 label decision, split manifests and pre-registration.

**Risks**

- Compute limits for transformers.
- Results may not support the hypotheses. That is acceptable, and negative
  results are reported as such.

**Acceptance criteria**

- [x] Each reported number can be regenerated by documented commands
      (`docs/reproducibility.md`). Classical reruns reproduce exactly (tested),
      and LSTM runs are deterministic per seed on CPU (tested).
- [x] Every model is compared against the baselines on the same split, with
      cluster-bootstrap confidence intervals.
- [x] The test set was used exactly once per frozen configuration (19 logged
      accesses, `results/test_access_log.jsonl`).
- [x] The results tables in the docs are generated (`docs/results-tables.md`).
- [ ] Phase 3 is merged after review (pending the project owner). CI on
      GitHub has not yet run.

---

## Phase 4: Professional research dashboard and user experience

**Deliverables**

- A Streamlit app with:
  - model selection across trained artifacts;
  - per-prediction probabilities and calibration context;
  - token-level or coefficient-based explanations, where methodologically sound;
  - a dataset explorer;
  - a results page that reads stored evaluation outputs.
- Label names and copy aligned with the Phase 2 label decision.
- Serve the selected score model through `winemood.inference.regression.ScorePredictor`.
- Data version 2: strip the leading score ranges found in audit A-31, then
  re-run the pipeline under a new, versioned preprocessing step.
- Accessibility basics: colour-blind-safe palette and text alternatives
  for charts.

**Dependencies:** Phase 3 artifacts and evaluation outputs.

**Risks**

- Explanations may be over-interpreted. Mitigated by caveats and by limiting
  explanations to methods appropriate for each model.

**Acceptance criteria**

- Every number shown in the UI traces back to a stored run file.
- `AppTest` coverage for each page, including missing-artifact states.
- The UI makes no claims beyond those in the evaluation report.

---

## Phase 5: Reproducibility, documentation, deployment and final QA

**Deliverables**

- A one-command reproduction path (e.g. `make reproduce` or a documented
  script sequence) from raw data to results tables.
- A technical report (paper-style) with methods, results, limitations and
  ethics/licensing statement.
- A deployed demo (Streamlit Community Cloud or equivalent), with artifacts
  fetched from a versioned release.
- Final dependency lock, CI badge, `CITATION.cff`, tagged release.

**Dependencies:** Phases 1–4.

**Risks**

- Model artifact size may exceed hosting limits.
- Dataset licence may prevent public hosting of derived artifacts.

**Acceptance criteria**

- A fresh clone on a clean machine reproduces the reported metrics within
  stated tolerance by following the README alone. This is checked by someone
  other than the author, or in a clean CI job.
- CI is green on the release tag.
- Every claim in the README and the report is linked to evidence.
