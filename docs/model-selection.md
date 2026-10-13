# WineMood: Model Selection

## 1. Rule (recorded before selection and before any test access)

The rule was recorded in protocol §10 (commit `732ad54`) and is implemented in
`winemood.experiments.runner.Runner.select`.

1. **Within each model family**, the configuration with the best validation
   value of the task's primary metric is selected. That is MAE for T1 and
   macro-F1 for T2/T3, chosen from the pre-declared grid. Ties keep the
   first configuration in grid order.
2. **Across families (T1, `grouped_v1`)**, the candidates in increasing order
   of complexity are B3\* (E04, TF-IDF + Ridge), N2\* (E12, frozen encoder +
   Ridge) and N1\* (E10, BiLSTM, 5 seeds). Selection starts from the simplest.
   A more complex candidate replaces the current choice only if its validation
   MAE is lower **and** the 95 % paired cluster-bootstrap CI of
   MAE(current) − MAE(candidate) on validation excludes 0 (2,000 resamples
   over split groups).
3. **Secondary considerations** (cost, interpretability, size, latency,
   reviewer generalisation) are reported but do not override rule 2. Such
   evidence can only be assessed on the test set, and it is never used for
   selection.
4. `winemood exp select` writes `results/selection.json` with `frozen: true`,
   the rule text, the comparisons and the frozen configuration hashes for
   every test-scope experiment. `winemood exp final` evaluates exactly those
   hashes once and logs every access.

## 2. Outcome

See §3 of [`experimental-results.md`](experimental-results.md) and
`results/selection.json` for the comparisons that were actually computed and
the model that was selected.
