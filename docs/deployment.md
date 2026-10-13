# WineMood: Deployment Guide

**Status:** the application is ready for deployment review. Nothing has been
deployed, and nothing deploys automatically.

## 1. Recommended host

**Streamlit Community Cloud (free tier)** is the recommended host. It runs
`app.py` directly from the GitHub repository and installs `requirements.txt`,
which installs this package with the **light core dependencies**: no
TensorFlow, PyTorch or Transformers. The app is CPU-only and stateless, and
it needs a single model artifact of about 1 MB.

Two alternatives also work: any container or VM host that runs
`streamlit run app.py`, or a local machine. Paid services are not required.
Check the current resource limits in Streamlit's documentation before
deploying. The memory measured below is well under typical free-tier limits.

## 2. Requirements

| Item | Value |
|---|---|
| Python | 3.11–3.13 (set it in the host's settings; `requires-python` is `>=3.11,<3.14`) |
| Install | `pip install .` via `requirements.txt` (light core: streamlit, altair, pandas, numpy, scikit-learn, scipy, nltk, matplotlib) |
| Start command | `streamlit run app.py` |
| Config | `.streamlit/config.toml`: theme, self-hosted fonts (static file serving), 1 MB upload limit, viewer toolbar |
| Data at runtime | Committed `results/` records only. The dataset is **not** needed. |
| Model | E04 release archive (see §3) |

## 3. Model artifact

Trained models are derived works of the CC BY-NC-SA 4.0 dataset. They are not
committed to git; they are distributed as a release archive.

**Packaging** (already done locally, ready to upload):

```bash
winemood model package E04_tfidf_word      # -> dist/models/E04_tfidf_word.zip
```

The archive contains `run.json`, `model/` (JSON + `.npy`; no pickle),
`manifest.json` (provenance + SHA-256 of every file) and `LICENSE-MODEL.md`.

| Archive | Size | SHA-256 |
|---|---|---|
| `dist/models/E04_tfidf_word.zip` (built 2026-10-13) | 1,100,871 bytes | `3b68051435d8f2c899bf883d63b1cec1ed552941e94b494ae16eba6a3dd2b564` |

Re-packaging embeds a new timestamp, which changes the archive checksum
(the per-file checksums stay the same). Always record the checksum of the
exact file you upload.

**Providing it to the application**, in the order the app checks:

1. `WINEMOOD_MODEL_DIR=/path/to/installed/model`. This is used exclusively,
   so the app never falls back to another model.
2. `models/e04/`, created by `winemood model install dist/models/E04_tfidf_word.zip`.
3. `experiments/grouped_v1/E04_tfidf_word/`, a local Phase 3 build.
4. **Hosted:** set `WINEMOOD_MODEL_URL` (for example, the GitHub Release asset
   URL) and `WINEMOOD_MODEL_SHA256` (the archive checksum above). On first
   start the app downloads the archive, **refuses it unless the archive
   checksum matches**, extracts it safely, and verifies every file against the
   manifest. On Streamlit Community Cloud, put both values in the app's
   *Secrets* as root-level keys; they are exposed as environment variables.

If no model is available, the Predict and Explore model views show setup
instructions. The Research page and the dataset view work without a model.

## 4. Environment variables

| Variable | Purpose |
|---|---|
| `WINEMOOD_MODEL_DIR` | Explicit model directory (exclusive) |
| `WINEMOOD_MODEL_URL` / `WINEMOOD_MODEL_SHA256` | Hosted download of the release archive (both required) |
| `WINEMOOD_RESULTS_DIR` | Alternative location of the `results/` records (tests use this) |
| `WINEMOOD_ROOT` | Override the project root, if the package is installed elsewhere |

## 5. Measured resource use

Measured on 2026-10-13: Windows 10, Intel i5-9400F, Python 3.13, in a fresh
**light** virtual environment without TensorFlow, Keras, PyTorch or
Transformers, using the installed and verified release artifact.

| Measurement | Result |
|---|---|
| Light environment install | 58 s; 557 MB site-packages (full research environment: 2,750 MB) |
| Test suite in the light environment | 190 passed, 2 skipped (neural tests skip themselves) |
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

## 6. Publication procedure (manual; not performed)

The `modernization` branch has **never been pushed**: origin has only `main`,
so GitHub Actions has not run. Suggested steps, for the project owner to run:

```bash
git push -u origin modernization          # triggers CI (light + neural jobs)
# open a pull request modernization -> main; review; wait for green CI
# create a GitHub Release and attach dist/models/E04_tfidf_word.zip (+ its SHA-256)
# Streamlit Community Cloud: New app -> repo, branch, app.py, Python 3.13;
#   Secrets: WINEMOOD_MODEL_URL = "<release asset URL>", WINEMOOD_MODEL_SHA256 = "<sha256>"
```

## 7. Known limitations

- The deployed model is the Phase 3 E04 model. It has learned the A-31
  score-range tokens; the application flags such input (see
  `a31-remediation.md`).
- Estimates are critic-score predictions for English, Wine Enthusiast-style
  tasting notes. They carry no per-prediction uncertainty.
- Licence: non-commercial use only (CC BY-NC-SA 4.0). A public demo must
  attribute the dataset (the About menu and the Explore page do) and must not
  be monetised.
- Deep links (`/research`, `/explore`) trigger two harmless 404 health probes
  in Streamlit's client. This is Streamlit behaviour and does not affect the
  app.
