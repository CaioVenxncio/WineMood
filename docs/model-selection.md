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

## 2. Outcome (frozen 2026-10-13, commit `15bc2a6`, before any test access)

| Comparison on validation | MAE(current) − MAE(candidate) [95% CI] | Switch? |
|---|---|---|
| B3\* (1.2906) vs N2\* frozen encoder (1.6843) | −0.394 [−0.416, −0.371] | No (candidate worse) |
| B3\* vs N1\* BiLSTM (1.2862 ± 0.0082 over 5 seeds) | +0.004 [−0.009, +0.018] | No (CI includes 0) |

**Selected: B3\* (E04, word 1–2-gram TF-IDF + Ridge).** It is also the
cheapest model (about 1 s to fit the selected configuration), the most
interpretable, and the smallest.

**After the test evaluation.** On test, the BiLSTM's advantage was 0.016
[0.007, 0.026] MAE, which passes the H2 rule but is practically negligible.
The selection is **not** revised: changing the choice after seeing test
results would be test-set tuning. The model card describes B3\*.
