"""Design tokens, scoped CSS and the Altair chart theme.

The Streamlit theme itself (colours, fonts, radius) lives in
``.streamlit/config.toml``. The CSS below styles only WineMood's own ``wm-*``
classes rendered through ``st.html``; it never targets Streamlit's internal DOM.
"""

from __future__ import annotations

import altair as alt

# Tokens (keep in sync with .streamlit/config.toml). Contrast on BG >= 4.5:1
# for every colour used for text.
BG = "#FAF8F5"
PANEL = "#F2EDE7"
TEXT = "#1F1F1F"
MUTED = "#5E5852"
ACCENT = "#7A1F3D"  # burgundy
RAISES = "#2E5E6E"  # slate - features that raise the estimate
LOWERS = "#9C3A4E"  # rose - features that lower the estimate
GOLD = "#8A6A1F"  # non-text marks only
BORDER = "#D9D1C7"
FAMILY_COLORS = {
    "Trivial baseline": "#8C847C",
    "Reference (not text)": GOLD,
    "Classical NLP · selected": ACCENT,
    "Neural (5 seeds)": RAISES,
    "Neural (frozen encoder)": "#6B8E7F",
}
SERIF = "'WineMood Serif', Georgia, serif"  # Source Serif 4 (see config.toml)
SANS = "Inter, 'Segoe UI', Roboto, Arial, sans-serif"
MONO = "'JetBrains Mono', Consolas, monospace"

CSS = f"""
<style>
.wm-kicker {{ font-family: {SANS}; font-size: .78rem; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase; color: {ACCENT}; margin: 0 0 .35rem 0; }}
.wm-title {{ font-family: {SERIF}; font-weight: 600; font-size: 2.3rem; line-height: 1.15;
  color: {TEXT}; margin: 0 0 .6rem 0; }}
.wm-lede {{ font-family: {SANS}; font-size: 1.05rem; line-height: 1.6; color: {MUTED};
  max-width: 46rem; margin: 0 0 1.4rem 0; }}
.wm-rule {{ border: 0; border-top: 1px solid {BORDER}; margin: 1.6rem 0 1.2rem 0; }}
.wm-section {{ font-family: {SERIF}; font-weight: 600; font-size: 1.45rem; color: {TEXT};
  margin: .2rem 0 .25rem 0; }}
.wm-section-caption {{ font-family: {SANS}; color: {MUTED}; font-size: .95rem;
  margin: 0 0 .9rem 0; max-width: 48rem; line-height: 1.55; }}
.wm-score {{ display: flex; align-items: baseline; gap: .45rem; margin: .2rem 0 .1rem 0; }}
.wm-score-value {{ font-family: {MONO}; font-weight: 500; font-size: 3.4rem; color: {ACCENT};
  font-variant-numeric: tabular-nums; line-height: 1; }}
.wm-score-unit {{ font-family: {MONO}; font-size: 1.15rem; color: {MUTED}; }}
.wm-score-label {{ font-family: {SANS}; font-size: .85rem; color: {MUTED}; font-weight: 500;
  letter-spacing: .04em; text-transform: uppercase; }}
.wm-badge {{ display: inline-block; border: 1px solid {ACCENT}; color: {ACCENT};
  border-radius: 3px; padding: .15rem .55rem; font-family: {SANS}; font-weight: 600;
  font-size: .9rem; margin: .5rem 0 .4rem 0; }}
.wm-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: .75rem; margin: .4rem 0 1rem 0; }}
.wm-metric {{ border-top: 2px solid {ACCENT}; padding: .55rem .1rem 0 .1rem; }}
.wm-metric-label {{ font-family: {SANS}; font-size: .78rem; color: {MUTED}; font-weight: 600;
  letter-spacing: .06em; text-transform: uppercase; }}
.wm-metric-value {{ font-family: {MONO}; font-size: 1.45rem; color: {TEXT};
  font-variant-numeric: tabular-nums; margin-top: .15rem; }}
.wm-metric-detail {{ font-family: {SANS}; font-size: .8rem; color: {MUTED}; margin-top: .1rem; }}
.wm-note {{ border-left: 3px solid {ACCENT}; background: {PANEL}; padding: .75rem 1rem;
  margin: .6rem 0 1rem 0; font-family: {SANS}; font-size: .93rem; line-height: 1.55;
  color: {TEXT}; }}
.wm-note.limitation {{ border-left-color: {GOLD}; }}
.wm-note.warning {{ border-left-color: {LOWERS}; }}
.wm-note-title {{ font-weight: 600; display: block; margin-bottom: .2rem; }}
.wm-footnote {{ font-family: {SANS}; font-size: .8rem; color: {MUTED}; line-height: 1.5;
  margin-top: .4rem; }}
.wm-empty {{ border: 1px dashed {BORDER}; padding: 1.2rem 1.3rem; margin: .8rem 0;
  font-family: {SANS}; color: {TEXT}; }}
.wm-empty pre {{ font-family: {MONO}; font-size: .82rem; background: {PANEL};
  padding: .6rem .8rem; white-space: pre-wrap; margin: .6rem 0 0 0; }}
@media (max-width: 640px) {{
  .wm-title {{ font-size: 1.8rem; }}
  .wm-score-value {{ font-size: 2.6rem; }}
}}
</style>
"""


@alt.theme.register("winemood", enable=False)
def altair_theme() -> alt.theme.ThemeConfig:
    return {
        "config": {
            "background": BG,
            "font": SANS,
            "view": {"stroke": None},
            "title": {
                "font": SERIF,
                "fontSize": 15,
                "fontWeight": 600,
                "color": TEXT,
                "anchor": "start",
                "offset": 10,
            },
            "axis": {
                "labelFont": SANS,
                "titleFont": SANS,
                "labelColor": MUTED,
                "titleColor": MUTED,
                "titleFontWeight": 500,
                "gridColor": "#E8E2DA",
                "domainColor": BORDER,
                "tickColor": BORDER,
                "labelFontSize": 11,
                "titleFontSize": 12,
            },
            "legend": {
                "labelFont": SANS,
                "titleFont": SANS,
                "labelColor": TEXT,
                "titleColor": MUTED,
                "orient": "bottom",
            },
            "range": {"category": [ACCENT, RAISES, GOLD, MUTED, LOWERS, "#6B8E7F"]},
        }
    }


def enable_chart_theme() -> None:
    alt.theme.enable("winemood")
