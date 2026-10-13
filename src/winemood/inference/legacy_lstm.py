"""Loader and predictor for the legacy Keras LSTM artifacts.

The original notebook saved ``model.h5`` (Keras 2 HDF5) and a pickled Keras 2
``Tokenizer``. Recent TensorFlow releases ship Keras 3, which no longer
provides the ``keras.preprocessing.text`` module that such pickles reference,
so both artifacts are loaded through ``tf_keras`` (the Keras 2 compatibility
package) and tokenizer pickles are remapped onto its ``Tokenizer``.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from winemood.config import ArtifactConfig, PreprocessingConfig
from winemood.data.labels import LabelScheme
from winemood.inference.predictor import (
    ArtifactIncompatibleError,
    ArtifactNotFoundError,
    PredictionResult,
)
from winemood.logging_utils import get_logger
from winemood.preprocessing import TextPreprocessor

logger = get_logger(__name__)

# Module paths under which a Keras 2 Tokenizer may have been pickled.
_TOKENIZER_MODULES = {
    "keras.preprocessing.text",
    "keras.src.preprocessing.text",
    "keras.src.legacy.preprocessing.text",
    "keras_preprocessing.text",
    "tensorflow.keras.preprocessing.text",
    "tensorflow.python.keras.preprocessing.text",
    "tf_keras.preprocessing.text",
    "tf_keras.src.preprocessing.text",
}
_ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("builtins", "int"),
    ("builtins", "dict"),
    ("builtins", "list"),
}


class _TokenizerUnpickler(pickle.Unpickler):
    """Unpickler that only materialises a Keras Tokenizer and plain containers.

    This narrows (but does not eliminate) the risk of loading pickles: only
    load tokenizer files from sources you trust.
    """

    def find_class(self, module: str, name: str) -> Any:
        if name == "Tokenizer" and module in _TOKENIZER_MODULES:
            from tf_keras.preprocessing.text import Tokenizer

            return Tokenizer
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Refusing to unpickle global {module}.{name}")


def load_tokenizer(artifacts: ArtifactConfig) -> Any:
    """Prefer the JSON tokenizer; fall back to the restricted pickle loader."""
    json_path, pickle_path = artifacts.tokenizer_json_path, artifacts.tokenizer_pickle_path
    if json_path.is_file():
        from tf_keras.preprocessing.text import tokenizer_from_json

        return tokenizer_from_json(json_path.read_text(encoding="utf-8"))
    if pickle_path.is_file():
        try:
            with pickle_path.open("rb") as fh:
                return _TokenizerUnpickler(fh).load()
        except (pickle.UnpicklingError, AttributeError, EOFError, ImportError) as exc:
            raise ArtifactIncompatibleError(
                f"Could not load tokenizer from {pickle_path}: {exc}"
            ) from exc
    raise ArtifactNotFoundError(f"No tokenizer found. Expected {json_path} or {pickle_path}.")


def load_model(path: Path) -> Any:
    if not path.is_file():
        raise ArtifactNotFoundError(f"Model file not found: {path}")
    import tf_keras

    try:
        return tf_keras.models.load_model(path)
    except Exception as exc:  # Keras raises a wide variety of types here.
        raise ArtifactIncompatibleError(f"Could not load model from {path}: {exc}") from exc


class LegacyLSTMPredictor:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        scheme: LabelScheme,
        preprocessor: TextPreprocessor,
        max_sequence_length: int = 100,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.scheme = scheme
        self.preprocessor = preprocessor
        self.max_sequence_length = max_sequence_length
        n_out = int(model.output_shape[-1])
        if n_out != scheme.n_classes:
            raise ArtifactIncompatibleError(
                f"Model has {n_out} outputs but label scheme '{scheme.name}' "
                f"defines {scheme.n_classes} classes"
            )

    @classmethod
    def from_artifacts(
        cls,
        artifacts: ArtifactConfig,
        scheme: LabelScheme,
        preprocessing: PreprocessingConfig,
    ) -> LegacyLSTMPredictor:
        # Check both files before importing TensorFlow, which is slow.
        if not artifacts.model_path.is_file():
            raise ArtifactNotFoundError(f"Model file not found: {artifacts.model_path}")
        tokenizer = load_tokenizer(artifacts)
        model = load_model(artifacts.model_path)
        logger.info("loaded legacy LSTM", extra={"dir": str(artifacts.dir)})
        return cls(
            model,
            tokenizer,
            scheme,
            TextPreprocessor(preprocessing),
            preprocessing.max_sequence_length,
        )

    def predict(self, texts: Sequence[str]) -> list[PredictionResult]:
        if not texts:
            return []
        from tf_keras.preprocessing.sequence import pad_sequences

        cleaned = self.preprocessor.batch(texts)
        sequences = self.tokenizer.texts_to_sequences(cleaned)
        # Same padding/truncation as the original notebook (post-pad, pre-truncate).
        padded = pad_sequences(sequences, maxlen=self.max_sequence_length, padding="post")
        probs = np.asarray(self.model.predict(padded, verbose=0), dtype=float)
        results = []
        for text, seq, row in zip(texts, sequences, probs, strict=True):
            idx = int(np.argmax(row))
            results.append(
                PredictionResult(
                    text=text,
                    label_index=idx,
                    label=self.scheme.class_names[idx],
                    display_label=self.scheme.display_names[idx],
                    probabilities=tuple(float(p) for p in row),
                    n_known_tokens=len(seq),
                )
            )
        return results
