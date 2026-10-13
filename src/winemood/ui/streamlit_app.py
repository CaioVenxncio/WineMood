"""Streamlit demo for the legacy WineMood classifier.

The UI never fails with a traceback when artifacts are missing: it explains
what is missing and how to obtain it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from winemood.config import Config, load_config
from winemood.data.labels import LabelScheme, get_scheme
from winemood.inference.predictor import ArtifactError, Predictor
from winemood.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

_CLASS_COLOURS = {"negative": "#C0392B", "neutral": "#7F8C8D", "positive": "#27AE60"}

LABEL_DISCLAIMER = (
    "**About these labels.** The model predicts *critic score bands* derived from "
    "the 0-100 WineEnthusiast `points` field (legacy scheme `points_band_v0`), "
    "not sentiment annotated by humans. The *Negative* band (< 80 points) does not "
    "occur in the training data, so the model has never seen an example of it. "
    "See `docs/audit.md`."
)


@st.cache_resource(show_spinner="Loading model...")
def load_predictor(config_path: str | None = None) -> tuple[Predictor | None, str | None]:
    """Return ``(predictor, None)`` or ``(None, user-facing error message)``."""
    config = load_config(config_path)
    scheme = get_scheme(config.labels.scheme)
    try:
        from winemood.inference.legacy_lstm import LegacyLSTMPredictor

        predictor = LegacyLSTMPredictor.from_artifacts(
            config.artifacts, scheme, config.preprocessing
        )
    except ArtifactError as exc:
        logger.warning("model unavailable", extra={"reason": str(exc)})
        return None, str(exc)
    except ImportError as exc:
        logger.exception("model dependencies missing")
        return None, f"A required dependency could not be imported: {exc}"
    return predictor, None


def bar_chart(labels: Sequence[str], values: Sequence[float], colours: Sequence[str], ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(labels, values, color=colours)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(100.0, max(values, default=0) * 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    return fig


def _render_missing_model(message: str, config: Config) -> None:
    st.error("No usable model is available, so predictions are disabled.")
    st.markdown(
        f"""
**Reason:** `{message}`

The original project never committed its trained artifacts. To enable predictions,
place a legacy model in `{config.artifacts.dir}`:

- `{config.artifacts.model_file}` (Keras HDF5 model with
  {get_scheme(config.labels.scheme).n_classes} softmax outputs)
- `{config.artifacts.tokenizer_json}` *or* `{config.artifacts.tokenizer_pickle}`
  (Keras Tokenizer; load pickles only from trusted sources)

See `artifacts/README.md`. A reproducible training pipeline is planned for Phase 3
(`docs/roadmap.md`).
"""
    )


def _render_results(predictor: Predictor, reviews: list[str]) -> None:
    scheme: LabelScheme = predictor.scheme
    colours = [_CLASS_COLOURS.get(c, "#888888") for c in scheme.class_names]
    results = predictor.predict(reviews)

    if any(r.n_known_tokens == 0 for r in results):
        st.warning(
            "At least one review contains no words from the model vocabulary; "
            "its prediction reflects padding only and is not meaningful."
        )

    if len(results) == 1:
        r = results[0]
        st.subheader(f"Predicted band: {r.display_label}")
        st.pyplot(
            bar_chart(
                scheme.display_names,
                [p * 100 for p in r.probabilities],
                colours,
                "Model probability (%)",
            )
        )
        return

    table = pd.DataFrame(
        {
            "Review": [r.text for r in results],
            "Predicted band": [r.display_label for r in results],
            "Confidence": [max(r.probabilities) for r in results],
        }
    )
    st.dataframe(table, width="stretch")
    counts = Counter(r.label_index for r in results)
    shares = [counts.get(i, 0) / len(results) * 100 for i in range(scheme.n_classes)]
    st.pyplot(bar_chart(scheme.display_names, shares, colours, "Share of reviews (%)"))


def _render_about() -> None:
    st.markdown(
        """
### What WineMood is

WineMood studies how the language of professional wine tasting notes relates to
critic scores. The current demo serves the *legacy* model from the original
coursework notebook: an Embedding + LSTM network trained on the Kaggle
*Wine Reviews* dataset (`zynicide/wine-reviews`, ~130k WineEnthusiast reviews).

### What it is not (yet)

- It is **not** a validated sentiment classifier. Labels come from score
  thresholds (>= 90, 80-89, < 80), and the lowest band is empty in the data.
- No evaluation results are reported here. The legacy notebook's evaluation
  has known methodological flaws (see `docs/audit.md`). A rigorous evaluation
  is scheduled for Phase 3.

### Pipeline

1. Text normalisation: lowercase, punctuation removal, NLTK English stopword removal.
2. Tokenisation with the training-time Keras tokenizer, padded to 100 tokens.
3. Embedding -> LSTM -> dense softmax over the score bands.

Project documentation lives in `docs/`.
"""
    )


def main(config_path: str | None = None) -> None:
    configure_logging()
    st.set_page_config(page_title="WineMood", page_icon="🍷", layout="wide")
    config = load_config(config_path)

    st.title("WineMood")
    st.caption("Exploring the language of wine tasting notes and critic score bands")

    tab_predict, tab_about = st.tabs(["Classifier", "About"])

    with tab_predict:
        st.info(LABEL_DISCLAIMER)
        predictor, error = load_predictor(config_path)
        if predictor is None:
            _render_missing_model(error or "unknown error", config)
        user_input = st.text_area(
            "Wine review(s), one per line (English):",
            height=150,
            placeholder="Great complexity with floral aromas and a long finish.",
        )
        if st.button("Classify", type="primary", disabled=predictor is None):
            reviews = [line.strip() for line in user_input.splitlines() if line.strip()]
            if not reviews:
                st.warning("Please enter at least one wine review.")
            elif predictor is not None:
                _render_results(predictor, reviews)

    with tab_about:
        _render_about()
