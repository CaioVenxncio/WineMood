# A-31 Remediation: Critic Score Ranges Embedded in Review Text

**Status:** the correction is implemented as an *isolated, versioned*
preprocessing step (`norm_min_v2`). It is **not applied to the selected
model (E04)**, and no Phase 3 result has been altered. Retraining under the
correction requires a new, separately approved evaluation protocol (§5),
which has not been run.

## 1. Finding

Some reviews of barrel samples begin with the critic's own provisional score
range, for example "86-88 Barrel sample. Densely structured…". The range is
direct information about the target score.

| Quantity (dataset SHA-256 `52af2643…67ac9`) | Value |
|---|---|
| Raw rows beginning with a score range | 680 |
| Unique descriptions beginning with a score range | **669** |
| Stated range bounds | 84–100 |
| True score inside the stated range | 99.85 % |
| Score ranges occurring anywhere other than the start | 0 |

**Count correction (Phase 4).** Phase 3 reported 628 unique descriptions. That
count recognised hyphen and en-dash ranges only. A Phase 4 re-check found 41
more descriptions written with an em dash ("88—90 Barrel sample"), giving 669.
The Phase 3 effect estimate has been recomputed with the full set (§3), and
the conclusion is unchanged.

## 2. The correction (`norm_min_v2`)

- **Function.** `winemood.preprocessing.strip_score_prefix(text)` removes
  only the following, and returns `(text, was_stripped)`:
  - an optional leading whitespace;
  - a score from 80–100;
  - a hyphen, en dash or em dash;
  - a second score from 80–100;
  - optional trailing punctuation.
- **Kept intact:**
  - every other number: vintages, drinking windows ("2018-2025"), percentages
    ("85-90% Cabernet"), ages, case counts;
  - ranges that are not at the start of the text;
  - the phrase "Barrel sample", which is a legitimate description of what was
    tasted. It is correlated with the score format and is documented as such.
- **Version.** `norm_min_v2 = norm_min_v1 + strip_score_prefix`. It is
  registered in `PREPROCESSING_VERSIONS` and enabled by the
  `PreprocessingConfig.strip_score_prefix` flag, which defaults to `False`.
- **Tests.** `tests/test_a31.py` covers affected texts, unaffected texts, the
  dash variants, and the guarantee that `norm_min_v1` output is unchanged.

## 3. Impact on the selected model

The selected model E04 was trained with `norm_min_v1`, so it has seen the
prefixes and learned them. Its coefficients include:
- "92" +5.68 and "94" +5.80;
- the bigram "92 94" +3.11;
- "barrel sample" +4.96.

The aggregate impact is negligible because the prefix is rare. Recomputed on
the **stored** Phase 3 test predictions (no new test access), the 135 affected
test reviews have an E04 MAE of 1.287 against 1.302 for the rest, and the test
MAE is 1.3023 with them and 1.3024 without.

**Compatibility rule.** E04 must be served with `norm_min_v1`, exactly as
trained. Serving it with `norm_min_v2` would silently change the input
distribution. `ScorePredictor` and the application enforce the artifact's
recorded preprocessing version. The application **detects** a leading score
range in user input and shows a notice; it does not alter the input.

## 4. Why the Phase 3 results are not rewritten

The Phase 3 test set was evaluated once, under a frozen protocol, and is now
spent (see `results/test_access_log.jsonl`). Retraining with `norm_min_v2`
and scoring on that test set would be test-set reuse: the outcome would be
known to be relevant before the evaluation. Phase 3 numbers therefore remain
the record, and this document describes their limitation.

## 5. Draft pre-registration for a corrected model (not run; needs approval)

- **Model.** E04's exact configuration (word 1–2-gram TF-IDF, min_df 5,
  Ridge α = 1.0), retrained with `norm_min_v2` on the `grouped_v1` training
  split. There is **no new hyperparameter search**, so the only change is the
  preprocessing.
- **Evaluation data.** The `grouped_v1` test split cannot serve as an
  untouched test set. Options, in order of preference:
  1. a new evaluation set of reviews not in this dataset, such as later
     publications, subject to licence;
  2. a fresh, pre-registered split of a disjoint subset;
  3. if neither is possible, a *descriptive* comparison on validation only,
     labelled as such, with no confirmatory claim.
- **Primary comparison.** MAE of the v2 model against the v1 model on the
  same evaluation set, using a paired cluster bootstrap, with the decision
  rules of protocol §6.
- **Expected.** A negligible aggregate difference. The aim is to remove a
  known leakage path, not to improve scores.

The project owner must approve this plan before it is run.
