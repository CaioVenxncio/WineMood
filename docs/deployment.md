# WineMood: Deployment Guide

**Status:** the application and the release archive are prepared and verified
for a public deployment. **Nothing has been released or deployed yet**, and
nothing deploys automatically. The steps in §6 and §7 are for the project
owner to run.

## 1. Recommended host

**Streamlit Community Cloud (free tier)** is the recommended host. It runs
`app.py` directly from the GitHub repository and installs `requirements.txt`,
which installs this package with the **light core dependencies**: no
TensorFlow, PyTorch or Transformers. The app is CPU-only and stateless, and
it needs a single model archive of about 1 MB, downloaded from a GitHub
Release and verified by SHA-256.

Two alternatives also work: any container or VM host that runs
`streamlit run app.py`, or a local machine. Paid services are not required.
Check the current resource limits in Streamlit's documentation before
deploying. The memory measured below (§8) is well under typical free-tier
limits.

## 2. Requirements

| Item | Value |
|---|---|
| Python | **3.13** (select it in the Cloud app's advanced settings). `requires-python` is `>=3.11,<3.14`, so a newer default Python would fail to install. |
| Install | `requirements.txt`: the package (`.`) plus the direct runtime dependencies **pinned** to the verified versions (streamlit 1.65.0, altair 6.3.0, pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, nltk 3.10.3, matplotlib 3.11.2). |
| Entry point | `app.py` (repository root) |
| Config | `.streamlit/config.toml`: theme, self-hosted fonts (static file serving), 1 MB upload limit, viewer toolbar, no usage statistics |
| Data at runtime | Committed `results/` records only. The dataset is **not** needed. |
| Model | The E04 release archive (§3), fetched via `WINEMOOD_MODEL_URL` (§4) |

The CI job `deploy-check` reproduces this environment on every pull request:
a non-editable install of `requirements.txt` on Python 3.13, a check that no
neural framework is present, the application tests, and a live server health
check.

## 3. Model artifact

The served model is the frozen Phase 3 selection, **E04 word TF-IDF + Ridge**
(`results/selection.json`), preprocessing `norm_min_v1`, no number masking.
It is not retrained for the release. Trained models are derived works of the
CC BY-NC-SA 4.0 dataset. They are not committed to git; they are distributed
as a release archive.

**Packaging** (already done):

```bash
winemood model package E04_tfidf_word      # -> dist/models/E04_tfidf_word.zip
```

The archive contains:
- `run.json`;
- `model/` (JSON + `.npy`; no pickle);
- `manifest.json` (provenance + SHA-256 of every file);
- `LICENSE-MODEL.md` (CC BY-NC-SA 4.0 notice and dataset attribution).

| Archive | Size | SHA-256 |
|---|---|---|
| `dist/models/E04_tfidf_word.zip` (packaged 2026-10-13) | 1,100,871 bytes | `3b68051435d8f2c899bf883d63b1cec1ed552941e94b494ae16eba6a3dd2b564` |

**Verified for the release** (light environment, no TensorFlow or PyTorch):
- The manifest matches the frozen records: experiment, config hash
  `70969891f05d`, dataset SHA-256, and test metrics (MAE 1.3023, R² 0.7148,
  n = 23,991).
- Every model file is byte-identical to the Phase 3 build.
- It downloads over HTTP, passes the archive checksum and per-file
  verification, and loads.
- Single predictions, exact explanations and a 1,000-row batch CSV all work.
- A wrong checksum and an unreachable URL are both refused with a readable
  message.

Re-packaging embeds a new timestamp, which changes the archive checksum
(the per-file checksums stay the same). **Upload this exact file** and do not
re-package it. If it is ever re-packaged, record the new checksum everywhere.

## 4. How the application finds the model

In order:

1. `WINEMOOD_MODEL_DIR=/path/to/installed/model`. Used exclusively; the app
   never falls back to another model.
2. **Hosted:** `WINEMOOD_MODEL_URL` + `WINEMOOD_MODEL_SHA256`. Also used
   exclusively.
   - On first start the app downloads the archive and **refuses it unless the
     archive checksum matches**.
   - It then extracts the archive safely and verifies every file against the
     manifest.
   - The install goes to `models/release-<first 16 hex digits of the
     checksum>/`, so changing the configured checksum always installs exactly
     that release, never a stale one.
   - A verified install is reused; a damaged one is downloaded again.
   - A failed attempt (for example, a network error) is not cached. The next
     page interaction retries.
3. `models/e04/`, created by `winemood model install <archive>`.
4. `experiments/grouped_v1/E04_tfidf_word/`, a local Phase 3 build.

If no model is available, the Predict page and the Explore model views show
the reason and setup instructions. The Research page and the dataset view
work without a model.

| Variable | Purpose |
|---|---|
| `WINEMOOD_MODEL_DIR` | Explicit model directory (exclusive) |
| `WINEMOOD_MODEL_URL` / `WINEMOOD_MODEL_SHA256` | Hosted download of the release archive (both required; the checksum must be 64 hex characters) |
| `WINEMOOD_RESULTS_DIR` | Alternative location of the `results/` records (tests use this) |
| `WINEMOOD_ROOT` | Override the project root, if the package is installed elsewhere |

The model URL and checksum are public values, not credentials. Streamlit
Community Cloud's *Secrets* field is simply where it accepts environment
variables. No secret is needed to run the application.

## 5. Release plan (proposed; not yet published)

| Item | Proposed value |
|---|---|
| Tag | `v1.0.0` (on the merge commit of the Phase 5 pull request; package version `1.0.0`) |
| Release title | `WineMood v1.0.0 - E04 TF-IDF + Ridge critic-score model` |
| Asset file name | `winemood-e04-tfidf-ridge-v1.0.0.zip` (the archive in §3, renamed; renaming does not change the checksum) |
| Asset SHA-256 | `3b68051435d8f2c899bf883d63b1cec1ed552941e94b494ae16eba6a3dd2b564` |
| Asset URL | `https://github.com/CaioVenxncio/WineMood/releases/download/v1.0.0/winemood-e04-tfidf-ridge-v1.0.0.zip` |
| Release notes | Must include: what the model predicts and does not predict; the test metrics with the record they come from; the SHA-256; the dataset attribution and the CC BY-NC-SA 4.0 notice for the asset (the code is MIT) |

**Licensing requirements for the asset** (CC BY-NC-SA 4.0, inherited from the
dataset):
- **Attribution.** Name the dataset (*Wine Reviews*, Kaggle
  `zynicide/wine-reviews`), link the licence, and state the change: a
  regression model was trained on deduplicated review text. The archive's
  `LICENSE-MODEL.md` names the dataset, links the licence and states that the
  model is a derived work trained on it; the release notes carry the full
  attribution.
- **Non-commercial.** No paid access, advertising or other monetisation of the
  model or the demo.
- **Share-alike.** The asset and any adaptations stay under CC BY-NC-SA 4.0,
  and no additional restrictions may be added.
- **Upstream provenance.** The reviews were written by WineEnthusiast critics
  and scraped in 2017. The Kaggle licence is the one the dataset was published
  under. The model contains n-gram vocabulary and weights, not review texts.

## 6. Publication procedure (manual; for the project owner)

1. Merge the Phase 5 pull request after CI is green.
2. Create the release:
   - **Tag:** `v1.0.0` on the merge commit.
   - **Asset:** upload `dist/models/E04_tfidf_word.zip`, renamed to
     `winemood-e04-tfidf-ridge-v1.0.0.zip`.
   - **Notes:** paste the release notes.
3. Download the published asset and check its checksum:

   ```bash
   curl -fsSLO https://github.com/CaioVenxncio/WineMood/releases/download/v1.0.0/winemood-e04-tfidf-ridge-v1.0.0.zip
   sha256sum winemood-e04-tfidf-ridge-v1.0.0.zip   # must print 3b680514...b2564
   ```

## 7. Streamlit Community Cloud settings

| Setting | Value |
|---|---|
| Repository | `CaioVenxncio/WineMood` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version (advanced settings) | `3.13` |
| Secrets (advanced settings, TOML) | see below |

```toml
WINEMOOD_MODEL_URL = "https://github.com/CaioVenxncio/WineMood/releases/download/v1.0.0/winemood-e04-tfidf-ridge-v1.0.0.zip"
WINEMOOD_MODEL_SHA256 = "3b68051435d8f2c899bf883d63b1cec1ed552941e94b494ae16eba6a3dd2b564"
```

Root-level secrets are exposed to the app as environment variables. Do not
commit a `.streamlit/secrets.toml`; it is git-ignored.

**After deploying, check:**
- The Predict page shows a score for the example review. The model is
  accepted only after the archive checksum and the manifest checks pass, so a
  score means the verified release is being served.
- The Research and Explore pages render.

If the model cannot be fetched or verified, the Predict page shows "The score
model is not available" with the refusal reason (for example, a checksum
mismatch) instead of a traceback.

## 8. Measured resource use

Measured on 2026-10-13: Windows 10, Intel i5-9400F, Python 3.13, in a fresh
**light** virtual environment without TensorFlow, Keras, PyTorch or
Transformers, using the installed and verified release artifact.

| Measurement | Result |
|---|---|
| Light environment install | 58 s; 557 MB site-packages (full research environment: 2,750 MB) |
| Server start → health check OK | 2.4 s |
| First page fully rendered in headless Chromium (includes model load) | 5.8 s |
| Package import / verified model load | 1.93 s / 0.125 s |
| Single prediction | 15.7 ms median, 18.4 ms p95 (in-process) |
| Prediction round trip in the browser | 0.23 s |
| Exact explanation of one review | 16.1 ms median |
| 1,000-review CSV (validate + predict) | 0.14 s |
| Research page load | 2.7 s |
| Server memory (process tree), idle → after Predict + Research | 86 MB → 249 MB (peak 253 MB) |
| Heavy neural modules imported by the app | none |

## 9. Known limitations

- The deployed model is the Phase 3 E04 model. It has learned the A-31
  score-range tokens; the application flags such input (see
  `a31-remediation.md`).
- Estimates are critic-score predictions for English, Wine Enthusiast-style
  tasting notes. They are not sentiment and not a measure of wine quality,
  and they carry no per-prediction uncertainty.
- Licence: non-commercial use only (CC BY-NC-SA 4.0). A public demo must
  attribute the dataset (the About menu and the Explore page do) and must not
  be monetised.
- Deep links (`/research`, `/explore`) trigger two harmless 404 health probes
  in Streamlit's client. This is Streamlit behaviour and does not affect the
  app.
