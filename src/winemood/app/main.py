"""Application entry point: page configuration, theme and navigation."""

from __future__ import annotations

import streamlit as st

from winemood.app.ui import components as c
from winemood.app.ui.pages import explore, predict, research
from winemood.app.ui.theme import enable_chart_theme


def run() -> None:
    st.set_page_config(
        page_title="WineMood - language and expert wine ratings",
        page_icon=":material/wine_bar:",
        layout="wide",
        menu_items={
            "About": "WineMood: NLP research on how the language of wine reviews relates to "
            "expert critic scores. Predictions are model estimates of critic ratings, not "
            "measures of wine quality.\n\nData: *Wine Reviews* (Kaggle, zynicide), "
            "WineEnthusiast reviews scraped in 2017, licensed CC BY-NC-SA 4.0. The model is "
            "a derived work under the same licence; non-commercial use only."
        },
    )
    enable_chart_theme()
    c.inject_css()
    pages = [
        st.Page(
            predict.render,
            title="Predict",
            icon=":material/edit_note:",
            url_path="predict",
            default=True,
        ),
        st.Page(
            research.render, title="Research", icon=":material/query_stats:", url_path="research"
        ),
        st.Page(
            explore.render, title="Explore", icon=":material/travel_explore:", url_path="explore"
        ),
    ]
    st.navigation(pages, position="top").run()
