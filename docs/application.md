# WineMood: Research Application

`streamlit run app.py` starts a three-page research application built on the
model selected in Phase 3 (**E04: word TF-IDF + Ridge**). The application
predicts a *critic's score*. It does not measure wine quality, and it does
not analyse sentiment.

## Pages

| Page | Purpose | Data source |
|---|---|---|
| **Predict** | Estimate the score (80–100) for one review or a CSV of up to 1,000 reviews. Shows the Wine Enthusiast category, the model's aggregate accuracy, an exact "why this number" breakdown and the limitations. | The installed model artifact; `results/` for the aggregate accuracy |
| **Research** | Selected model and selection rule; test-set comparison of all models with 95 % CIs; pre-registered hypotheses H1–H5; generalisation to unseen critics; validation-only ablations; error structure; secondary classification tasks; limitations | `results/report.json`, `results/test/`, `results/validation/`, `results/selection.json` |
| **Explore** | Exact per-review attributions; all learned coefficients, with search and filters; numerical proxies (drinking-window years, A-31 score tokens); dataset composition and biases | The model artifact; `results/eda/summary.json` |

Navigation uses Streamlit's top navigation bar, which collapses to a menu on
narrow screens.

## Display policies (scientific integrity)

- **Score.** The model's raw output is clipped to Wine Enthusiast's published
  80–100 scale and shown with one decimal. When clipping occurs, the raw value
  and the policy are displayed.
- **Category.** The category is derived from the estimate rounded half-to-even
  to an integer, using the bands of `we_category_merged_v1`: Acceptable 80–82,
  Good 83–86, Very Good 87–89, Excellent 90–93, Superb or Classic 94–100. The
  descriptions paraphrase the publisher's scale.
- **No per-prediction uncertainty.** There is no confidence percentage, no
  probability and no per-review interval, because none has been calibrated.
  The page shows the model's *aggregate* test performance and says explicitly
  that it is not an uncertainty range for the individual estimate.
- **No sentiment framing.** A UI test checks that the rendered pages never use
  terms such as "sentiment score", "confidence score" or "probability that".
- **A-31.** A review that starts with a score range (for example "92-94") gets
  a notice that the model learned to read such ranges. The input is *not*
  altered, and the model keeps its exact training preprocessing
  (`norm_min_v1`). See [`a31-remediation.md`](a31-remediation.md).
- **Numbers come from records.** Every metric is read from the committed
  `results/` files. The UI code holds no copies of metrics, and a test
  compares the displayed values with the JSON.
- **Test data stays out.** The application reads published aggregates only.
  A guard test fails if application code references test predictions, the
  raw dataset or `load_split`. The predicted-vs-actual and residual charts use
  binned aggregates (`test_prediction_aggregates` in `report.json`) computed
  once from the stored Phase 3 predictions.

## Explanations

For the linear model, an estimate equals the intercept plus Σ (TF-IDF weight
× coefficient) over the features present in the review. The *Why this number*
expander and the *Explain a review* tab show these exact terms:

- `winemood.app.services.explain` recomputes the sum in float64 and refuses to
  display an explanation that does not reconstruct the model's own output
  (tolerance 1e-6 points against the float64 recomputation, 1e-3 against the
  float32 prediction path).
- n-grams are listed as features. Words are not highlighted, because
  overlapping unigrams and bigrams have no faithful per-token decomposition.
- Captions state that contributions are the model's arithmetic, not causal
  effects.

## Architecture

```
app.py                          shim -> winemood.app.main.run()
src/winemood/app/
  main.py                       page config, chart theme, CSS, st.navigation
  services/                     no Streamlit imports; unit-tested
    model_service.py            locate -> verify manifest -> load -> compatibility checks
    results.py                  typed parsing of results/ (labels, protocols, CIs)
    explain.py                  exact linear attributions + reconstruction check
    batch.py                    CSV validation (size, encoding, schema, length) and export
  ui/
    theme.py                    design tokens, scoped CSS (wm-* classes), Altair theme
    components.py               header, section, score, metric row, note, empty state, chart
    content.py                  fixed explanatory copy (no metrics)
    state.py                    cached model / results access (st.cache_resource)
    pages/predict.py, research.py, explore.py
```

**Model resolution.**
1. If `WINEMOOD_MODEL_DIR` is set, it is used **exclusively**: a broken
   explicit path is never silently replaced by another model.
2. Otherwise, if `WINEMOOD_MODEL_URL` is set (hosted deployments), it is also
   used exclusively. The release archive is downloaded once, accepted only if
   it matches `WINEMOOD_MODEL_SHA256`, and installed in
   `models/release-<first 16 hex digits of the checksum>/`. A later start
   reuses the verified install. A damaged install is downloaded again.
3. Otherwise, the application tries `models/e04` (an installed release
   archive).
4. Then it tries `experiments/grouped_v1/E04_tfidf_word` (a local Phase 3
   build).

A directory with a `manifest.json` must pass SHA-256 verification. The model
must be E04, linear, with `norm_min_v1` and no masking. When a check fails,
the pages show the reason and the setup commands; they do not show a
traceback. A failed load is not cached: the next interaction tries again, so a
temporary network error during the first download does not need an app
restart.

## Design system

- **Colour.**
  - Burgundy accent `#7A1F3D` on a warm off-white `#FAF8F5`, with charcoal
    text `#1F1F1F` and muted text `#5E5852`.
  - "Raises" and "lowers" use slate `#2E5E6E` and rose `#9C3A4E`. They are
    always paired with sign, position or labels, never conveyed by colour
    alone.
  - Every colour used for text reaches WCAG AA contrast (≥ 4.5:1): text
    15.6:1, accent 9.5:1, muted 6.6:1.
- **Type.**
  - Source Serif 4 for headings (registered as "WineMood Serif", because
    Streamlit drops font-face family names that contain a digit).
  - Inter for body text.
  - JetBrains Mono for metrics, with tabular numerals.
  - All three are self-hosted OFL fonts in `static/fonts/`, falling back to
    system fonts.
- **Implementation.**
  - The theme is set in `.streamlit/config.toml` using native Streamlit
    options.
  - Custom CSS styles only WineMood's own `wm-*` classes rendered through
    `st.html`. It never targets Streamlit's internal DOM.
  - All dynamic text is HTML-escaped.
- **Accessibility.**
  - Every chart has a text description, also attached as its Vega description,
    plus a "Show the numbers behind this chart" table.
  - Inputs have visible labels.
  - The score has an ARIA status label.
  - All controls are native Streamlit widgets, which are keyboard-operable.

## UI testing

```bash
pytest tests/test_app_ui.py tests/test_app_services.py tests/test_a31.py
```

The tests use a synthetic TF-IDF + Ridge model built in a temporary
directory, with the same artifact layout, plus the real committed `results/`.
They cover:
- page rendering with and without the model and the results;
- the prediction flow, empty input and the A-31 notice;
- the forbidden-terms check;
- explanation reconstruction and the clipping policy;
- CSV limits and export;
- artifact tampering and preprocessing mismatch;
- the absence of TensorFlow and PyTorch imports (checked in a subprocess);
- the guard against test-data access.

Visual review was done in headless Chromium at 1400 px and 390 px widths.
