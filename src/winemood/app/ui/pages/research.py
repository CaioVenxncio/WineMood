"""Research: the Phase 3 findings, read from the committed result records."""

from __future__ import annotations

from itertools import pairwise

import altair as alt
import pandas as pd
import streamlit as st

from winemood.app.services.results import MODEL_INFO, SPLIT_LABELS, Interval, ResearchResults
from winemood.app.ui import components as c
from winemood.app.ui.pages.shared import aggregate_metrics, limitations, results_unavailable
from winemood.app.ui.state import research_results
from winemood.app.ui.theme import ACCENT, FAMILY_COLORS, MUTED, RAISES

HYPOTHESIS_QUESTIONS = {
    "H1": "Does review text predict the score better than trivial cues (mean, length)?",
    "H2": "Do neural models beat the bag-of-words model?",
    "H3": "Is part of the signal specific to the critics seen in training?",
    "H4": "How much do drinking-window years contribute? (exploratory)",
    "H5": "Does keeping negation words harm the BiLSTM?",
}


def render() -> None:
    c.page_header(
        "Research",
        "How much of a critic's rating is in the words?",
        "Results of the pre-registered Phase 3 evaluation: models trained on Wine Enthusiast "
        "reviews, selected on a validation set, and evaluated once on held-out reviews.",
    )
    results = research_results()
    if not results.available:
        results_unavailable(results)
        return
    _overview(results)
    _comparison(results)
    _hypotheses(results)
    _reviewers(results)
    _ablations(results)
    _errors(results)
    _classification(results)
    c.section("Limitations")
    st.markdown("\n".join(f"- {item}" for item in limitations(results)))
    c.footnote(
        f"Source: results/report.json and related records. The test set was accessed "
        f"{results.test_access_count} times, all logged, for a scope fixed before any access. "
        "Full details: docs/experimental-results.md, docs/model-card.md."
    )


def _overview(results: ResearchResults) -> None:
    sel = results.selected
    c.section("Selected model", rule=False)
    if sel is None:
        st.info("No selected model recorded.")
        return
    st.markdown(f"**{sel.name}** - {sel.metric_note}")
    c.metric_row(aggregate_metrics(results))
    rule = results.selection.get("rule")
    if rule:
        c.note(rule, title="How the model was chosen (frozen before test access)")


def _comparison(results: ResearchResults) -> None:
    c.section(
        "Model comparison",
        "Mean absolute error on the same held-out test reviews (lower is better), with 95% "
        "cluster-bootstrap intervals. Training time is wall-clock CPU time for the full "
        "validation grid.",
    )
    rows = results.main_comparison()
    frame = pd.DataFrame(
        [
            {
                "Model": m.name,
                "Family": m.family,
                "MAE": m.mae.estimate,
                "low": m.mae.low,
                "high": m.mae.high,
                "R²": m.r2.estimate if m.r2 else None,
                "MAE [95% CI]": m.mae.fmt(3),
                "R² [95% CI]": m.r2.fmt(3) if m.r2 else "",
                "Training time (s)": round(m.train_seconds) if m.train_seconds else None,
                "Note": m.metric_note,
            }
            for m in rows
        ]
    )
    order = frame.sort_values("MAE")["Model"].tolist()
    base = alt.Chart(frame).encode(
        y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=340))
    )
    color = alt.Color(
        "Family:N",
        scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())),
        legend=alt.Legend(title=None, columns=3),
    )
    chart = (
        base.mark_rule(strokeWidth=2).encode(
            x=alt.X("low:Q", title="Test MAE (points)", scale=alt.Scale(zero=False)),
            x2="high:Q",
            color=color,
        )
        + base.mark_point(filled=True, size=90).encode(
            x="MAE:Q",
            color=color,
            shape=alt.Shape("Family:N", legend=None),
            tooltip=["Model", "Family", "MAE [95% CI]", "R² [95% CI]"],
        )
    ).properties(height=40 * len(frame) + 30)
    c.chart(
        chart,
        "Dot = test MAE, line = 95% interval. "
        + "; ".join(f"{r.Model}: {r['MAE [95% CI]']}" for _, r in frame.iterrows()),
        data_table=frame[
            ["Model", "Family", "MAE [95% CI]", "R² [95% CI]", "Training time (s)", "Note"]
        ],
    )
    c.footnote(
        "The BiLSTM's MAE is the mean over five independently trained seeds; it is not an "
        "ensemble. The metadata model uses price, critic and origin rather than text and is a "
        "reference point, not a competitor."
    )


def _label(exp_id: str) -> str:
    name, family = MODEL_INFO.get(exp_id, (exp_id, ""))
    if family in ("Unseen tasters", "Size-matched random split"):
        return f"{name} ({family.lower()})"
    return name


def _hypotheses(results: ResearchResults) -> None:
    c.section(
        "Pre-registered hypotheses",
        "Differences are MAE(a) - MAE(b) on the test set (negative = a is better). Verdicts "
        "apply the decision rules registered before any test access.",
    )
    lines = [
        "| | Question | Evidence: MAE(a) - MAE(b) [95% CI] | Verdict |",
        "|---|---|---|---|",
    ]
    for name, h in results.hypotheses.items():
        if name == "exploratory":
            continue
        comps = [x for x in h.get("comparisons", []) if "mae_a_minus_b" in x]
        evidence = "<br>".join(
            f"{_label(x['a'])} - {_label(x['b'])}: "
            f"**{Interval.from_dict(x['mae_a_minus_b']).fmt(3, signed=True)}**"
            for x in comps
        )
        lines.append(
            f"| **{name}** | {HYPOTHESIS_QUESTIONS.get(name, '')} | {evidence} | "
            f"{h.get('verdict', '')} |"
        )
    st.markdown("\n".join(lines), unsafe_allow_html=True)  # trusted, generated from records
    rule = (
        "A difference between neural models counts only if its interval excludes zero AND it "
        "is larger than the variation between training seeds."
    )
    h5 = next(iter(results.hypotheses.get("H5", {}).get("comparisons", [])), None)
    if h5 and "seed_std" in h5:
        d = h5["mae_a_minus_b"]
        rule += (
            f" Keeping negations changed the BiLSTM's MAE by {d['estimate']:+.4f} "
            f"[{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]; the interval excludes zero, but "
            f"the change is smaller than the seed SD ({h5['seed_std']:.4f}), so H5 is recorded "
            "as 'no detectable harm'. An early draft of the report misapplied this rule; the "
            "correction is documented in the protocol's deviation log."
        )
    c.note(rule, title="Decision rule (H2, H5)")


def _reviewers(results: ResearchResults) -> None:
    c.section(
        "Generalisation to unseen critics",
        "Models retrained on a split where test reviews come only from critics absent from "
        "training, compared with a random split of the same rows and sizes. These two test "
        "sets differ from the main split and from each other, so values are not directly "
        "comparable with the model comparison above.",
    )
    ids = ["E40_holdout_tfidf", "E42_random_tfidf", "E41_holdout_lstm", "E43_random_lstm"]
    rows = [results.models[e] for e in ids if e in results.models]
    if not rows:
        st.info("Reviewer-holdout results are not available.")
        return
    frame = pd.DataFrame(
        [
            {
                "Model": m.name,
                "Split": SPLIT_LABELS.get(m.split, m.split),
                "Test set": "Unseen critics" if m.split == "taster_holdout_v1" else "Random split",
                "MAE": m.mae.estimate,
                "low": m.mae.low,
                "high": m.mae.high,
                "MAE [95% CI]": m.mae.fmt(3),
                "n": m.n,
            }
            for m in rows
        ]
    )
    color = alt.Color(
        "Test set:N",
        scale=alt.Scale(domain=["Random split", "Unseen critics"], range=[MUTED, ACCENT]),
        legend=alt.Legend(title=None, labelLimit=300),
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("Model:N", title=None, axis=alt.Axis(labelLimit=240)), yOffset="Test set:N"
    )
    chart = (
        base.mark_rule(strokeWidth=2).encode(
            x=alt.X("low:Q", scale=alt.Scale(zero=False), title="Test MAE (points)"),
            x2="high:Q",
            color=color,
        )
        + base.mark_point(filled=True, size=80).encode(
            x="MAE:Q",
            color=color,
            shape=alt.Shape("Test set:N", legend=None),
            tooltip=["Model", "Split", "MAE [95% CI]", "n"],
        )
    ).properties(height=190)
    c.chart(
        chart,
        "; ".join(f"{r.Model}, {r.Split}: {r['MAE [95% CI]']}" for _, r in frame.iterrows()),
        data_table=frame[["Model", "Split", "MAE [95% CI]", "n"]],
    )
    dist = results.hypotheses.get("H3", {}).get("test_target_distribution", {})
    if dist:
        parts = [
            f"{k.split('_')[1]} test set: mean score {v['mean']:.2f}, SD {v['std']:.2f}"
            for k, v in dist.items()
        ]
        c.note(
            "The test sets differ in composition (" + "; ".join(parts) + "). The gap mixes "
            "unseen-critic effects with differences in critics' scoring habits and score "
            "distribution; it is not a pure measure of reviewer-specific learning.",
            kind="limitation",
            title="Interpretation",
        )


def _ablations(results: ResearchResults) -> None:
    abl = results.ablations
    rows = abl.get("rows", [])
    if not rows:
        return
    c.section(
        "Ablations (validation set)",
        "Controlled variants of the selected TF-IDF model, evaluated on the validation set "
        "only. These are development diagnostics, not test results.",
    )
    frame = pd.DataFrame(
        [
            {
                "Variant": r["description"].split(": ", 1)[-1],
                "Δ MAE": r["vs_B3"]["mae_a_minus_b"]["estimate"],
                "low": r["vs_B3"]["mae_a_minus_b"]["ci95_low"],
                "high": r["vs_B3"]["mae_a_minus_b"]["ci95_high"],
                "Validation MAE": round(r["val_mae"], 4),
            }
            for r in rows
            if r.get("vs_B3")
        ]
    )
    frame["Variant"] = frame["Variant"].str.replace("__", "", regex=False)  # placeholder tokens
    frame["Effect"] = [
        "lower error (CI excludes 0)"
        if hi < 0
        else "higher error (CI excludes 0)"
        if lo > 0
        else "no clear difference"
        for lo, hi in zip(frame["low"], frame["high"], strict=True)
    ]
    order = frame.sort_values("Δ MAE")["Variant"].tolist()  # explicit: layers drop SortField
    base = alt.Chart(frame).encode(
        y=alt.Y("Variant:N", sort=order, title=None, axis=alt.Axis(labelLimit=380))
    )
    chart = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=MUTED, strokeDash=[4, 3])
        .encode(x="x:Q")
        + base.mark_rule(strokeWidth=2, color=RAISES).encode(
            x=alt.X("low:Q", title="Change in validation MAE vs selected model (points)"),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=80, color=RAISES).encode(
            x="Δ MAE:Q",
            shape=alt.Shape("Effect:N", legend=alt.Legend(title=None, labelLimit=300)),
            tooltip=["Variant", alt.Tooltip("Δ MAE:Q", format="+.4f"), "Validation MAE"],
        )
    ).properties(height=40 * len(frame) + 30)
    c.chart(
        chart,
        "Negative values mean the variant had lower validation error than the "
        "selected model. "
        + "; ".join(f"{r.Variant}: {r['Δ MAE']:+.3f}" for _, r in frame.iterrows()),
        data_table=frame[["Variant", "Δ MAE", "low", "high", "Validation MAE"]],
    )


def _errors(results: ResearchResults) -> None:
    agg = results.aggregates.get("models", {})
    ea = results.error_analysis
    if not agg and not ea:
        return
    c.section(
        "Where the model errs",
        "Built from binned summaries of the stored test predictions of the single Phase 3 "
        "evaluation; no review-level test data is shipped with the application.",
    )
    sel_id = results.selection.get("final_model", "E04_tfidf_word")
    sel = agg.get(sel_id)
    col1, col2 = st.columns(2, gap="large")
    if sel:
        q = pd.DataFrame([r for r in sel["pred_quantiles_by_true"] if r["n"] > 0])
        band = (
            alt.Chart(q)
            .mark_area(opacity=0.25, color=ACCENT)
            .encode(
                x=alt.X("true:Q", title="Critic's actual score", scale=alt.Scale(domain=[80, 100])),
                y=alt.Y("p10:Q", title="Model estimate", scale=alt.Scale(domain=[80, 100])),
                y2="p90:Q",
            )
        )
        median = (
            alt.Chart(q)
            .mark_line(color=ACCENT, point=alt.OverlayMarkDef(color=ACCENT, filled=True))
            .encode(
                x="true:Q", y="p50:Q", tooltip=["true", "n", alt.Tooltip("p50:Q", format=".2f")]
            )
        )
        ident = (
            alt.Chart(pd.DataFrame({"a": [80, 100]}))
            .mark_line(color=MUTED, strokeDash=[4, 3])
            .encode(x="a:Q", y="a:Q")
        )
        with col1:
            c.chart(
                (band + median + ident).properties(height=300, title="Estimates vs actual scores"),
                "Median estimate (line) and 10th-90th percentile band for each actual score; "
                "the dashed diagonal is a perfect prediction. Estimates are compressed towards "
                "the middle: low scores are over-estimated and high scores under-estimated.",
                data_table=q,
            )
        edges = results.aggregates["residual_bin_edges"]
        hist = pd.DataFrame(
            [
                {"from": a, "to": b, "reviews": n}
                for (a, b), n in zip(pairwise(edges), sel["residual_counts"], strict=True)
            ]
        )
        hist = hist[hist["reviews"] > 0]
        with col2:
            c.chart(
                alt.Chart(hist)
                .mark_bar(color=RAISES)
                .encode(
                    x=alt.X(
                        "from:Q",
                        title="Estimate minus actual score (points)",
                        axis=alt.Axis(values=list(range(-8, 9, 2)), format="+d"),
                    ),
                    x2="to:Q",
                    y=alt.Y("reviews:Q", title="Test reviews"),
                    tooltip=["from", "to", "reviews"],
                )
                .properties(height=300, title="Error distribution"),
                "Distribution of errors on the test set; values beyond ±10 points are counted "
                "at the edges.",
            )
    models = ea.get("models", {})
    if models:
        rows = [
            {
                "Model": MODEL_INFO.get(m, (m,))[0],
                "Actual score": float(r["bin"]),
                "Mean error": r["mean_residual"],
                "n": r["n"],
            }
            for m, v in models.items()
            for r in v["by_true_score"]
        ]
        frame = pd.DataFrame(rows)
        chart = (
            alt.Chart(frame)
            .mark_line(point=True)
            .encode(
                x=alt.X("Actual score:Q", scale=alt.Scale(domain=[80, 100])),
                y=alt.Y("Mean error:Q", title="Mean error (estimate - actual)"),
                color=alt.Color(
                    "Model:N", legend=alt.Legend(title=None, columns=2, labelLimit=320)
                ),
                strokeDash=alt.StrokeDash("Model:N", legend=None),
                tooltip=["Model", "Actual score", alt.Tooltip("Mean error:Q", format="+.2f"), "n"],
            )
            .properties(height=280)
        )
        st.markdown("##### Error by actual score, all text and baseline models")
        c.chart(
            chart,
            "Every model over-estimates low scores and under-estimates high ones; "
            "text models do so less than the length-only baseline.",
            data_table=frame,
        )
    noise = ea.get("label_noise_floor")
    if noise:
        c.note(
            f"{noise['same_wine_pairs']:,} pairs of distinct reviews of the same wine and vintage "
            f"differ by {noise['same_wine_mean_abs_diff']:.2f} points on average; "
            f"{noise['conflicting_duplicate_groups']} identical texts carry different scores. "
            "The target itself is noisy, which bounds achievable accuracy.",
            title="How noisy are critic scores?",
            kind="limitation",
        )


def _classification(results: ResearchResults) -> None:
    rows = results.classification
    if not rows:
        return
    c.section(
        "Secondary tasks",
        "Separate classification tasks: five Wine Enthusiast categories (T2) and 90+ points "
        "(T3). Their metrics are not comparable with the regression MAE above.",
    )
    names = {
        "E20_t2_majority": "Majority-class baseline",
        "E21_t2_logistic": "Multinomial logistic regression",
        "E22_t2_ordinal": "Ordinal logistic regression",
        "E30_t3_majority": "Majority-class baseline",
        "E31_t3_logistic": "Logistic regression",
    }
    frame = pd.DataFrame(
        [
            {
                "Task": "5 categories" if r["task"] == "T2" else "90+ points",
                "Model": names.get(
                    r["experiment"],
                    "Selected regression model, binned"
                    if r["experiment"].startswith("E16")
                    else r["experiment"],
                ),
                "Macro-F1 [95% CI]": Interval.from_dict(r["macro_f1"]).fmt(3),
                "Balanced accuracy": round(r["balanced_accuracy"]["estimate"], 3),
                "QWK": round(r["qwk"], 3) if r.get("qwk") is not None else None,
            }
            for r in rows
        ]
    ).sort_values(["Task", "Model"])
    st.dataframe(frame, hide_index=True, width="stretch")
