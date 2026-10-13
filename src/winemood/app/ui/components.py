"""Reusable presentation components.

All dynamic text is HTML-escaped before it reaches ``st.html``; user input is
never rendered as markup.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import altair as alt
import streamlit as st

from winemood.app.ui.theme import CSS


def inject_css() -> None:
    st.html(CSS)


def page_header(kicker: str, title: str, lede: str) -> None:
    st.html(
        f'<p class="wm-kicker">{escape(kicker)}</p>'
        f'<h1 class="wm-title">{escape(title)}</h1>'
        f'<p class="wm-lede">{escape(lede)}</p>'
    )


def section(title: str, caption: str | None = None, *, rule: bool = True) -> None:
    html = '<hr class="wm-rule">' if rule else ""
    html += f'<h2 class="wm-section">{escape(title)}</h2>'
    if caption:
        html += f'<p class="wm-section-caption">{escape(caption)}</p>'
    st.html(html)


def score_display(score: float, category: str, label: str = "Predicted critic score") -> None:
    st.html(
        f'<div class="wm-score-label">{escape(label)}</div>'
        f'<div class="wm-score" role="status" aria-label="{escape(label)}: {score:.1f} out of 100">'
        f'<span class="wm-score-value">{score:.1f}</span>'
        f'<span class="wm-score-unit">/ 100</span></div>'
        f'<span class="wm-badge">{escape(category)}</span>'
    )


def metric_row(items: Sequence[tuple[str, str, str | None]]) -> None:
    """items: (label, value, detail). Values use tabular monospaced numerals."""
    cells = "".join(
        f'<div class="wm-metric"><div class="wm-metric-label">{escape(label)}</div>'
        f'<div class="wm-metric-value">{escape(value)}</div>'
        + (f'<div class="wm-metric-detail">{escape(detail)}</div>' if detail else "")
        + "</div>"
        for label, value, detail in items
    )
    st.html(f'<div class="wm-metrics">{cells}</div>')


def note(body: str, title: str | None = None, kind: str = "method") -> None:
    """kind: 'method' (burgundy), 'limitation' (gold) or 'warning' (rose)."""
    head = f'<span class="wm-note-title">{escape(title)}</span>' if title else ""
    st.html(f'<div class="wm-note {escape(kind)}" role="note">{head}{escape(body)}</div>')


def footnote(text: str) -> None:
    st.html(f'<p class="wm-footnote">{escape(text)}</p>')


def empty_state(title: str, body: str, code: str | None = None) -> None:
    block = f"<pre>{escape(code)}</pre>" if code else ""
    st.html(
        f'<div class="wm-empty" role="status"><strong>{escape(title)}</strong>'
        f"<p>{escape(body)}</p>{block}</div>"
    )


def chart(chart: alt.Chart, description: str, *, data_table=None) -> None:
    """Render a chart with a text description (also attached as the chart's
    accessible description) and, optionally, the underlying numbers."""
    st.altair_chart(chart.properties(description=description), theme=None, width="stretch")
    st.caption(description)
    if data_table is not None:
        with st.expander("Show the numbers behind this chart"):
            st.dataframe(data_table, hide_index=True, width="stretch")


def disclosure(title: str, markdown_body: str) -> None:
    with st.expander(title):
        st.markdown(markdown_body)
