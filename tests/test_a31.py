"""Audit A-31: leading critic score ranges are removed only in norm_min_v2."""

import pytest

from winemood.preprocessing import (
    PREPROCESSING_VERSIONS,
    TextPreprocessor,
    has_score_prefix,
    preprocessing_config,
    strip_score_prefix,
)

EN_DASH, EM_DASH = chr(0x2013), chr(0x2014)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("86-88 Barrel sample. Dense and ripe.", "Barrel sample. Dense and ripe."),
        ("92-94. Barrel sample. Rich.", "Barrel sample. Rich."),
        (f"88{EM_DASH}90 Barrel sample. Spicy.", "Barrel sample. Spicy."),
        (f"  95{EN_DASH}97: powerful", "powerful"),
        ("90-92Barrel sample", "Barrel sample"),
        ("98-100 Barrel sample.", "Barrel sample."),
    ],
)
def test_affected_texts(text, expected):
    assert has_score_prefix(text)
    assert strip_score_prefix(text) == (expected, True)


@pytest.mark.parametrize(
    "text",
    [
        "Drink 2018-2025. Ripe cherry.",
        "A blend of 85-90% Cabernet Sauvignon.",
        "85-90 % Merlot in this blend.",
        "90-day barrel aging gives toast.",
        "12% alcohol, 5-year-old vines.",
        "880-890 cases were made.",
        "Ripe fruit; 86-88 points in other reviews.",  # range not at the start
        "75-79 is below the published scale.",
        "",
    ],
)
def test_unaffected_texts(text):
    assert not has_score_prefix(text)
    assert strip_score_prefix(text) == (text, False)


def test_v1_unchanged_and_v2_strips():
    text = "86-88 Barrel sample. Drink 2020-2025."
    v1 = TextPreprocessor(preprocessing_config("norm_min_v1"))
    v2 = TextPreprocessor(preprocessing_config("norm_min_v2"))
    assert v1(text) == "86-88 barrel sample. drink 2020-2025."  # E04's exact preprocessing
    assert v2(text) == "barrel sample. drink 2020-2025."
    assert "norm_min_v2" in PREPROCESSING_VERSIONS
    assert preprocessing_config("norm_min_v1").strip_score_prefix is False
    assert preprocessing_config("norm_min_v2").strip_score_prefix is True
