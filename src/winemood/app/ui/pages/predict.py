"""Predict: estimate a critic score for one review or a CSV of reviews."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from winemood.app.services import batch
from winemood.app.services.explain import explain
from winemood.app.ui import components as c
from winemood.app.ui import content
from winemood.app.ui.pages.shared import aggregate_metrics, limitations, model_unavailable
from winemood.app.ui.state import model_state, research_results
from winemood.app.ui.theme import LOWERS, RAISES
from winemood.preprocessing import has_score_prefix

TEXT_KEY = "predict_text"


def _use_example() -> None:
    st.session_state[TEXT_KEY] = content.EXAMPLE_REVIEW
    st.session_state.pop("predict_result", None)


def _clear() -> None:
    st.session_state[TEXT_KEY] = ""
    st.session_state.pop("predict_result", None)


def render() -> None:
    c.page_header(
        "Predict",
        "Predicted critic score",
        "Estimate the score (80-100) a Wine Enthusiast critic would likely have given, "
        "based only on the language of a tasting note.",
    )
    state = model_state()
    results = research_results()
    if not state.ok:
        model_unavailable(state)
        return
    single, many = st.tabs(["Single review", "Batch (CSV)"])
    with single:
        _single(state, results)
    with many:
        _batch(state)


def _single(state, results) -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.text_area(
            "Tasting note",
            key=TEXT_KEY,
            height=190,
            max_chars=batch.MAX_CHARS,
            placeholder="Paste or write a wine review in English...",
            help=f"{batch.MIN_CHARS}-{batch.MAX_CHARS:,} characters. Only the text is used.",
        )
        b1, b2, b3 = st.columns([1.2, 1.4, 1])
        submitted = b1.button("Predict score", type="primary", width="stretch")
        b2.button("Use example review", on_click=_use_example, width="stretch")
        b3.button("Clear", on_click=_clear, width="stretch")
        if st.session_state.get(TEXT_KEY) == content.EXAMPLE_REVIEW:
            c.footnote(content.EXAMPLE_LABEL)
        if submitted:
            try:
                text = batch.validate_single(st.session_state.get(TEXT_KEY, ""))
            except batch.BatchError as exc:
                st.warning(str(exc))
            else:
                with st.spinner("Scoring..."):
                    raw = float(state.predictor.predict_raw([text])[0])
                    pred = state.predictor.predict([text])[0]
                st.session_state["predict_result"] = {
                    "text": text,
                    "raw": raw,
                    "score": pred.score,
                    "category": pred.category,
                }
                st.session_state["last_review"] = text
    with right:
        result = st.session_state.get("predict_result")
        if result:
            _result_panel(state, results, result)
        else:
            c.note(content.WHAT_THIS_IS, title="What this estimate is")


def _result_panel(state, results, result: dict) -> None:
    c.score_display(result["score"], result["category"])
    meaning = content.CATEGORY_MEANING.get(result["category"])
    if meaning:
        st.markdown(meaning)
        c.footnote(
            "Category bands: Wine Enthusiast's published scale (paraphrased). The "
            "category is derived from the rounded estimate."
        )
    if result["raw"] != result["score"]:
        c.note(
            content.CLIP_POLICY.format(raw=result["raw"], shown=result["score"]),
            title="Display policy",
            kind="limitation",
        )
    if has_score_prefix(result["text"]):
        c.note(content.A31_NOTICE, title="Score range detected in the text", kind="warning")
    metrics = aggregate_metrics(results)
    if metrics:
        st.markdown("**How accurate is the model overall?**")
        c.metric_row(metrics)
        c.footnote(content.AGGREGATE_NOT_INDIVIDUAL)
    with st.expander("Why this number? (exact model arithmetic)"):
        pipe, head = state.predictor.linear_components()
        exp = explain(pipe, head, result["text"])
        top = pd.DataFrame(
            [
                {
                    "feature": r.feature,
                    "contribution": r.contribution,
                    "effect": "raises" if r.contribution > 0 else "lowers",
                }
                for r in exp.contributions[:10]
            ]
        )
        if not top.empty:
            bars = (
                alt.Chart(top)
                .mark_bar()
                .encode(
                    x=alt.X("contribution:Q", title="Contribution to the estimate (points)"),
                    y=alt.Y(
                        "feature:N",
                        sort=alt.SortField("contribution", order="descending"),
                        title=None,
                    ),
                    color=alt.Color(
                        "effect:N",
                        scale=alt.Scale(domain=["raises", "lowers"], range=[RAISES, LOWERS]),
                        legend=alt.Legend(title=None),
                    ),
                    tooltip=["feature", alt.Tooltip("contribution:Q", format="+.3f")],
                )
                .properties(height=28 * len(top) + 20)
            )
            c.chart(
                bars,
                "The ten features of this review with the largest contributions. "
                "Positive values raise the estimate and negative values lower it.",
            )
        c.footnote(
            f"Baseline (intercept) {exp.intercept:.2f} + sum of {len(exp.contributions)} feature "
            f"contributions = raw output {exp.raw_score:.2f}. Full breakdown on the Explore page. "
            "Contributions describe this model's arithmetic, not causes of a critic's score."
        )
    with st.expander("Limitations"):
        st.markdown("\n".join(f"- {item}" for item in limitations(results)))


def _batch(state) -> None:
    st.markdown(
        f"Upload a UTF-8 CSV with a column named `review` or `description` "
        f"(up to {batch.MAX_ROWS:,} rows, {batch.MAX_BYTES // 1_000_000} MB)."
    )
    upload = st.file_uploader("CSV file of reviews", type=["csv"], key="batch_upload")
    if upload is None:
        return
    try:
        parsed = batch.parse_csv(upload.getvalue())
    except batch.BatchError as exc:
        st.error(str(exc))
        return
    with st.spinner(f"Scoring {len(parsed.reviews):,} reviews..."):
        frame = batch.predict_batch(state.predictor, parsed)
    st.success(
        f"Estimated scores for {len(frame):,} reviews (column '{parsed.column}'"
        + (f"; {parsed.skipped_empty} empty rows skipped" if parsed.skipped_empty else "")
        + ")."
    )
    flagged = int(frame["starts_with_score_range"].sum())
    if flagged:
        c.note(
            f"{flagged} review(s) begin with a score range; see the A-31 note on single "
            "predictions.",
            kind="warning",
        )
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "predicted_critic_score": st.column_config.NumberColumn(
                "Predicted critic score", format="%.1f"
            )
        },
    )
    st.download_button(
        "Download estimates (CSV)",
        batch.to_csv_bytes(frame),
        file_name="winemood_estimates.csv",
        mime="text/csv",
    )
    c.footnote("Model estimates of critic scores. They are not measurements of wine quality.")
