"""Explore: exact explanations, learned associations and dataset composition."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from winemood.app.services import batch
from winemood.app.services.explain import ExplanationError, coefficient_table, explain
from winemood.app.ui import components as c
from winemood.app.ui import content
from winemood.app.ui.pages.shared import model_unavailable, results_unavailable
from winemood.app.ui.state import model_state, research_results
from winemood.app.ui.theme import ACCENT, LOWERS, MUTED, RAISES

EXPLAIN_KEY = "explore_text"
EFFECT_SCALE = alt.Scale(domain=["raises", "lowers"], range=[RAISES, LOWERS])


def render() -> None:
    c.page_header(
        "Explore",
        "Language and ratings",
        "Inspect what the selected model has learned, how it arrives at a single estimate, "
        "and what the underlying dataset looks like.",
    )
    explain_tab, assoc_tab, data_tab = st.tabs(
        ["Explain a review", "Learned associations", "Dataset"]
    )
    state = model_state()
    for tab, view in ((explain_tab, _explain), (assoc_tab, _associations)):
        with tab:
            if state.ok:
                view(state)
            else:
                model_unavailable(state)
    with data_tab:
        _dataset()


def _explain(state) -> None:
    if EXPLAIN_KEY not in st.session_state:
        st.session_state[EXPLAIN_KEY] = st.session_state.get("last_review", content.EXAMPLE_REVIEW)
    st.text_area("Review to explain", key=EXPLAIN_KEY, height=130, max_chars=batch.MAX_CHARS)
    if st.session_state[EXPLAIN_KEY] == content.EXAMPLE_REVIEW:
        c.footnote(content.EXAMPLE_LABEL)
    try:
        text = batch.validate_single(st.session_state[EXPLAIN_KEY])
    except batch.BatchError as exc:
        st.info(str(exc))
        return
    pipe, head = state.predictor.linear_components()
    try:
        exp = explain(pipe, head, text)
    except ExplanationError as exc:
        st.error(f"The explanation could not be verified, so it is not shown: {exc}")
        return
    raises = sum(r.contribution for r in exp.contributions if r.contribution > 0)
    lowers = sum(r.contribution for r in exp.contributions if r.contribution < 0)
    c.metric_row(
        [
            ("Baseline (intercept)", f"{exp.intercept:.2f}", "the model's starting point"),
            ("Features raising", f"{raises:+.2f}", "sum of positive contributions"),
            ("Features lowering", f"{lowers:+.2f}", "sum of negative contributions"),
            (
                "Raw output",
                f"{exp.raw_score:.2f}",
                f"shown as {exp.displayed_score:.1f}" + (" (clipped)" if exp.clipped else ""),
            ),
        ]
    )
    rows = pd.DataFrame(
        [
            {
                "Feature": r.feature,
                "Type": r.kind,
                "TF-IDF weight": round(r.tfidf, 4),
                "Coefficient": round(r.coefficient, 3),
                "Contribution": r.contribution,
                "Effect": "raises" if r.contribution > 0 else "lowers",
            }
            for r in exp.contributions
        ]
    )
    top = rows.head(20)
    # Explicit order: layered Vega-Lite charts drop SortField sorting.
    order = top.sort_values("Contribution", ascending=False)["Feature"].tolist()
    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("Contribution:Q", title="Contribution to the estimate (points)"),
            y=alt.Y("Feature:N", sort=order, title=None),
            color=alt.Color("Effect:N", scale=EFFECT_SCALE, legend=alt.Legend(title=None)),
            tooltip=[
                "Feature",
                "Type",
                "TF-IDF weight",
                "Coefficient",
                alt.Tooltip("Contribution:Q", format="+.3f"),
            ],
        )
        .properties(height=24 * len(top) + 30)
    )

    def _labels(subset: pd.DataFrame, align: str, dx: int) -> alt.Chart:
        return (
            alt.Chart(subset)
            .mark_text(align=align, dx=dx, color=MUTED, fontSize=10)
            .encode(
                x=alt.X("Contribution:Q"),
                y=alt.Y("Feature:N", sort=order),
                text=alt.Text("Contribution:Q", format="+.2f"),
            )
        )

    # Labels sit outside the bar end: right of positive bars, left of negative ones.
    text_labels = _labels(top[top["Contribution"] >= 0], "left", 4) + _labels(
        top[top["Contribution"] < 0], "right", -4
    )
    c.chart(
        chart + text_labels,
        f"The {len(top)} largest of {len(rows)} feature contributions for this review, with "
        "signed values printed next to each bar.",
        data_table=rows,
    )
    c.note(
        f"Each contribution is the feature's TF-IDF weight in this review multiplied by its "
        f"learned coefficient. Intercept + all contributions = {exp.raw_score:.4f} "
        f"(reconstruction error {exp.reconstruction_error:.1e}). Single words and two-word "
        "phrases overlap, and weights depend on the whole review, so a contribution is the "
        "model's arithmetic - not the causal effect of a word on a critic's score.",
        title="How to read this",
    )


def _associations(state) -> None:
    pipe, head = state.predictor.linear_components()
    table = coefficient_table(pipe, head)
    st.markdown(
        f"The selected model weighs **{len(table):,}** single words and two-word phrases. "
        "Coefficients are learned associations between a phrase and higher or lower estimated "
        "scores, holding the other features fixed. They are not causal effects, and correlated "
        "phrases share weight."
    )
    f1, f2, f3 = st.columns([2, 2, 1])
    kinds = f1.multiselect(
        "Feature types", ["unigram", "bigram", "number"], default=["unigram", "bigram"]
    )
    query = f2.text_input("Search phrases", placeholder="e.g. tannins, finish, drink")
    n = f3.number_input("Show top", min_value=5, max_value=40, value=15, step=5)
    view = table[table["kind"].isin(kinds)]
    if query.strip():
        view = view[view["feature"].str.contains(query.strip().lower(), regex=False)]
    if view.empty:
        st.info("No features match these filters.")
    else:
        top = pd.concat(
            [view.nlargest(int(n), "coefficient"), view.nsmallest(int(n), "coefficient")]
        )
        top = top.drop_duplicates("feature")
        left, right = st.columns(2, gap="large")
        for col, direction, title in (
            (left, "raises", "Associated with higher estimates"),
            (right, "lowers", "Associated with lower estimates"),
        ):
            part = top[top["direction"] == direction]
            if part.empty:
                continue
            ch = (
                alt.Chart(part)
                .mark_bar(color=RAISES if direction == "raises" else LOWERS)
                .encode(
                    x=alt.X("coefficient:Q", title="Coefficient (points per unit TF-IDF)"),
                    y=alt.Y(
                        "feature:N",
                        sort=alt.SortField(
                            "coefficient",
                            order="descending" if direction == "raises" else "ascending",
                        ),
                        title=None,
                    ),
                    tooltip=["feature", "kind", alt.Tooltip("coefficient:Q", format="+.2f")],
                )
                .properties(height=22 * len(part) + 30, title=title)
            )
            with col:
                c.chart(
                    ch,
                    f"{title}: " + ", ".join(part["feature"].head(10)) + ".",
                    data_table=part[["feature", "kind", "coefficient"]],
                )
    st.markdown("##### Numerical proxies")
    numbers = table[table["kind"] == "number"].assign(magnitude=lambda d: d["coefficient"].abs())
    numbers = numbers.nlargest(12, "magnitude")
    st.dataframe(numbers[["feature", "coefficient"]], hide_index=True, width="stretch")
    c.note(
        "Some of the strongest number features are drinking-window years (such as 2030), which "
        "signal that a critic expects the wine to age, and score-range tokens (such as 92 and "
        "94) that come from barrel-sample reviews beginning with the critic's own provisional "
        "score (audit A-31). These are proxies for the critic's framing - in the second case "
        "for the target itself - not descriptions of the wine.",
        title="Why numbers appear here",
        kind="limitation",
    )


def _dataset() -> None:
    results = research_results()
    eda = results.eda
    if not results.available or not eda:
        results_unavailable(results)
        return
    d = eda["descriptive"]
    c.metric_row(
        [
            ("Reviews (raw)", f"{d['n_rows_raw']:,}", None),
            ("After deduplication", f"{d['n_rows_dedup']:,}", "exact duplicate texts removed"),
            ("Mean score", f"{d['points']['mean']:.2f}", f"SD {d['points']['std']:.2f}"),
            ("Median length", f"{d['length_words']['p50']:.0f} words", None),
        ]
    )
    left, right = st.columns(2, gap="large")
    counts = pd.DataFrame(
        {
            "score": [int(k) for k in d["points_counts"]],
            "reviews": list(d["points_counts"].values()),
        }
    )
    with left:
        c.chart(
            alt.Chart(counts)
            .mark_bar(color=ACCENT)
            .encode(
                x=alt.X("score:O", title="Critic score"),
                y=alt.Y("reviews:Q", title="Reviews"),
                tooltip=["score", "reviews"],
            )
            .properties(height=260, title="Score distribution"),
            "Scores range from 80 to 100 and cluster around 87-90. Wine Enthusiast does not "
            "publish wines scoring below 80.",
            data_table=counts,
        )
    tasters = (d.get("metadata") or {}).get("tasters", {})
    with right:
        if tasters:
            t = pd.DataFrame(
                {
                    "critic": [
                        f"Critic {i + 1:02d}" for i in range(len(tasters["rows_per_taster_sorted"]))
                    ],
                    "reviews": tasters["rows_per_taster_sorted"],
                }
            )
            c.chart(
                alt.Chart(t)
                .mark_bar(color=RAISES)
                .encode(
                    x=alt.X("critic:N", sort=None, title="Critic (anonymised, ranked)"),
                    y=alt.Y("reviews:Q", title="Reviews"),
                    tooltip=["critic", "reviews"],
                )
                .properties(height=260, title="Reviewer concentration"),
                f"{tasters['n_tasters']} named critics; the most prolific wrote "
                f"{tasters['largest_taster_share']:.1%} of reviews and "
                f"{tasters['rows_missing_taster']:,} reviews have no critic.",
                data_table=t,
            )
    meta = d.get("metadata") or {}
    if meta.get("top_countries"):
        countries = pd.DataFrame(
            {
                "country": list(meta["top_countries"]),
                "reviews": list(meta["top_countries"].values()),
            }
        )
        c.chart(
            alt.Chart(countries)
            .mark_bar(color=MUTED)
            .encode(
                y=alt.Y("country:N", sort="-x", title=None),
                x=alt.X("reviews:Q", title="Reviews"),
                tooltip=["country", "reviews"],
            )
            .properties(height=240, title="Geographic representation (top 10)"),
            f"{meta['n_countries']} countries; coverage is dominated by the United States.",
            data_table=countries,
        )
    c.note(
        "One publication; English only; scraped in 2017; wines are submitted by producers "
        "rather than sampled from the market; scores below 80 are never published; a few "
        "critics wrote most reviews. Statistics shown here are aggregates computed by the "
        "project's data pipeline; no review text is distributed with the application.",
        title="Documented biases",
        kind="limitation",
    )
    c.footnote(
        "Data: Wine Reviews (Kaggle, zynicide; WineEnthusiast reviews scraped in 2017), "
        "CC BY-NC-SA 4.0. Statistics are aggregates from results/eda/summary.json."
    )
