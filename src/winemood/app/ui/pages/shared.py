"""Small helpers shared by pages."""

from __future__ import annotations

from winemood.app.services.model_service import SETUP_INSTRUCTIONS, ModelState
from winemood.app.services.results import ResearchResults
from winemood.app.ui import components as c
from winemood.app.ui import content


def model_unavailable(state: ModelState) -> None:
    c.empty_state(
        "The score model is not available",
        f"{state.error} The Research and Explore pages still work from the committed results.",
        SETUP_INSTRUCTIONS,
    )


def results_unavailable(results: ResearchResults) -> None:
    c.empty_state(
        "Research results are not available",
        "The committed Phase 3 records were not found: " + ", ".join(results.missing),
        "winemood exp report   # regenerates results/report.json from stored predictions",
    )


def limitations(results: ResearchResults) -> list[str]:
    items = list(content.LIMITATIONS)
    noise = (results.error_analysis.get("label_noise_floor") or {}).get("same_wine_mean_abs_diff")
    if noise is not None:
        items.insert(0, content.NOISE_LIMITATION.format(noise=noise))
    return items


def aggregate_metrics(results: ResearchResults) -> list[tuple[str, str, str | None]]:
    m = results.selected
    if m is None:
        return []
    return [
        (
            "Mean absolute error",
            f"{m.mae.estimate:.2f} pts",
            f"95% CI {m.mae.low:.2f}-{m.mae.high:.2f}",
        ),
        ("Within ±2 points", f"{m.within_2:.0%}", f"of {m.n:,} held-out reviews"),
        ("R²", f"{m.r2.estimate:.2f}", "share of score variance explained"),
    ]
