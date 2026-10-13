# results/

Committed, machine-readable records of the Phase 3 experiments. They are
small, they are written only by the pipeline, and they are never edited by
hand.

| Path | Written by | Content |
|---|---|---|
| `validation/<id>.json` | `winemood exp run` | Selected configuration, full validation grid, validation metrics, run metadata (git commit, packages, hardware, dataset and split hashes, timing) |
| `selection.json` | `winemood exp select` | The frozen test scope (configuration hashes) and the model-selection decision |
| `test/<id>.json` | `winemood exp final` | Test metrics of the frozen configurations (single evaluation) |
| `test_access_log.jsonl` | `winemood exp final` | Every test-set access: 19 entries, all after the freeze, no repeats |
| `report.json` | `winemood exp report` | Bootstrap intervals, hypothesis verdicts, error analysis, interpretation, binned test-prediction aggregates |
| `eda/summary.json` | `winemood data eda --publish` | Dataset aggregates (counts, distributions, n-grams) |

**Integrity.** `docs/audit.md` (A-32) records the verification of these
files. The test set must not be evaluated again under this protocol (see
`CLAUDE.md`).

**Licence of quoted text.** `report.json` and the generated docs quote at
most 90 characters from each of 20 reviews, for error analysis, and
`eda/summary.json` lists frequent terms. This text derives from the *Wine
Reviews* dataset (Kaggle, zynicide; WineEnthusiast) and is licensed
**CC BY-NC-SA 4.0**, not under the repository's MIT licence.
