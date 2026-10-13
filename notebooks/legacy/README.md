# Legacy notebook (superseded)

`colab_notebook.ipynb` is the original 2025 coursework notebook (in Portuguese).
It is kept **unmodified** as a historical record, because `docs/audit.md` cites
its cells as evidence.

Do not use it to train models or to report results. Known problems include:

- an empty label class (A-01);
- incorrect error-analysis indexing (A-03);
- final cells that retrain against the test set and overwrite `model.h5` (A-04);
- tokenizer leakage (A-07);
- no executed outputs (A-05).

Its role is replaced by the `winemood` package. Training moves there in Phase 3.
