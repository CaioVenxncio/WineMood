# Model artifacts

Trained models are **not** committed. Everything in this directory except this
file is git-ignored.

## Status

The original project never committed a trained model, and its notebook has no
recorded outputs (see `docs/audit.md`, A-05 and A-06). **No trained artifact is
currently available.** Without one, the app runs but prediction is disabled.

## Legacy LSTM layout

If you have artifacts produced by `notebooks/legacy/colab_notebook.ipynb`,
place them here:

```
artifacts/legacy_lstm/
  model.h5           # Keras HDF5 model with 3 softmax outputs
  tokenizer.json     # preferred: Tokenizer.to_json()
  tokenizer.pickle   # accepted: pickled Keras Tokenizer
```

The directory and file names are configured in `configs/default.toml` under
`[artifacts]`.

**Security:** pickle files can execute code when loaded. WineMood loads them
through a restricted unpickler that only accepts the Keras `Tokenizer` and
plain containers. That reduces the risk, but only use files you produced or
fully trust. Prefer `tokenizer.json`.

**Caveats:** a legacy model predicts the `points_band_v0` score bands. Its
"Negative" class was never observed in training (audit A-01), and the last
notebook cells may have overwritten `model.h5` with an inconsistent model
(audit A-04).

Phase 3 will produce reproducible artifacts. Each will include its config, git
commit, data checksum and evaluation metrics.
