"""Embedding + (Bi)LSTM score regressor (Keras 3, CPU).

Design (docs/model-development.md):
* Texts are normalised by ``TextPreprocessor`` and tokenised to whitespace-joined
  word tokens in Python; a ``TextVectorization`` layer is adapted on the
  training texts only (vocabulary never sees validation/test).
* The sequence length is the 95th percentile of training token counts.
* The target is standardised with training mean/std; the exported model ends
  in a ``Rescaling`` layer, so it outputs points directly.
* MSE loss, Adam, early stopping on validation MAE with best-weight restore.
* The exported model (vectoriser -> network -> rescaling) is saved in the
  native ``.keras`` format and reloaded with ``safe_mode=True``: no pickle and
  no Lambda layers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DEFAULT_PARAMS: dict[str, Any] = {
    "vocab_size": 20000,
    "embed_dim": 128,
    "units": 64,
    "bidirectional": True,
    "dropout": 0.3,
    "dense_units": 32,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "max_epochs": 12,
    "patience": 2,
    "seq_len_percentile": 95,
}


@dataclass
class LSTMResult:
    model: Any
    history: dict[str, list[float]]
    best_epoch: int
    seq_len: int
    seconds: float
    epoch_seconds: list[float] = field(default_factory=list)


def _keras():
    import keras
    import tensorflow as tf

    return keras, tf


_DETERMINISM_SET = False


def set_seed(seed: int) -> None:
    global _DETERMINISM_SET
    keras, tf = _keras()
    keras.utils.set_random_seed(seed)
    if not _DETERMINISM_SET:
        tf.config.experimental.enable_op_determinism()
        _DETERMINISM_SET = True


def _string_input(texts):
    # Keras 3 rejects NumPy object arrays; a tf.string tensor of shape (n, 1) works.
    _, tf = _keras()
    return tf.reshape(tf.constant(list(texts), dtype=tf.string), (-1, 1))


def build_network(vocab_size: int, params: dict[str, Any]):
    keras, _ = _keras()
    layers = keras.layers
    inputs = keras.Input(shape=(None,), dtype="int64", name="tokens")
    x = layers.Embedding(vocab_size, params["embed_dim"], mask_zero=True, name="embedding")(inputs)
    rnn = layers.LSTM(params["units"], name="lstm")
    x = layers.Bidirectional(rnn, name="bilstm")(x) if params["bidirectional"] else rnn(x)
    x = layers.Dropout(params["dropout"], name="dropout")(x)
    x = layers.Dense(params["dense_units"], activation="relu", name="dense")(x)
    outputs = layers.Dense(1, name="score_z")(x)
    return keras.Model(inputs, outputs, name="lstm_core")


class _EpochTimer:
    def __init__(self):
        keras, _ = _keras()

        class Cb(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.times: list[float] = []

            def on_epoch_begin(self, epoch, logs=None):
                self._t = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                self.times.append(time.perf_counter() - self._t)

        self.callback = Cb()


def train_lstm(
    train_texts: list[str],
    y_train: np.ndarray,
    val_texts: list[str],
    y_val: np.ndarray,
    params: dict[str, Any],
    seed: int,
    *,
    max_epochs: int | None = None,
    verbose: int = 0,
) -> LSTMResult:
    keras, tf = _keras()
    p = {**DEFAULT_PARAMS, **params}
    set_seed(seed)
    started = time.perf_counter()

    lengths = np.array([len(t.split()) for t in train_texts])
    seq_len = max(8, int(np.percentile(lengths, p["seq_len_percentile"])))
    vectorizer = keras.layers.TextVectorization(
        max_tokens=p["vocab_size"],
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=seq_len,
        name="vectorizer",
    )
    vectorizer.adapt(tf.constant(train_texts))  # training texts only
    vocab = len(vectorizer.get_vocabulary())
    x_train = np.asarray(vectorizer(tf.constant(train_texts)))
    x_val = np.asarray(vectorizer(tf.constant(val_texts)))

    mean, std = float(np.mean(y_train)), float(np.std(y_train))
    z_train = (np.asarray(y_train, dtype=float) - mean) / std
    z_val = (np.asarray(y_val, dtype=float) - mean) / std

    network = build_network(vocab, p)
    network.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    timer = _EpochTimer()
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=p["patience"], restore_best_weights=True, mode="min"
    )
    history = network.fit(
        x_train,
        z_train,
        validation_data=(x_val, z_val),
        epochs=max_epochs or p["max_epochs"],
        batch_size=p["batch_size"],
        shuffle=True,
        callbacks=[stopper, timer.callback],
        verbose=verbose,
    )
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    # Report MAE in points, not standardised units.
    for key in ("mae", "val_mae"):
        if key in hist:
            hist[f"{key}_points"] = [v * std for v in hist[key]]
    best_epoch = int(np.argmin(hist["val_mae"])) + 1

    inputs = keras.Input(shape=(1,), dtype="string", name="text")
    tokens = vectorizer(inputs)
    z = network(tokens)
    score = keras.layers.Rescaling(scale=std, offset=mean, name="to_points")(z)
    exported = keras.Model(inputs, score, name="winemood_lstm")
    return LSTMResult(
        model=exported,
        history=hist,
        best_epoch=best_epoch,
        seq_len=seq_len,
        seconds=time.perf_counter() - started,
        epoch_seconds=timer.callback.times,
    )


def predict(model, texts: list[str], batch_size: int = 1024) -> np.ndarray:
    return np.asarray(model.predict(_string_input(texts), batch_size=batch_size, verbose=0)).ravel()


def save(model, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.keras"
    model.save(path)
    return path


def load(directory: Path):
    keras, _ = _keras()
    return keras.models.load_model(directory / "model.keras", safe_mode=True)
