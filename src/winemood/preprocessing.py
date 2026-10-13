"""Single text normaliser shared by training and inference.

The default ``PreprocessingConfig`` reproduces the original notebook exactly:
lowercase -> remove non-word characters -> drop NLTK English stopwords.
Phase 3 experiments select a named version with ``preprocessing_config``
(declared in docs/experimental-protocol.md §10). No step is learned from data.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from functools import lru_cache

from winemood.config import PreprocessingConfig

_NON_WORD = re.compile(r"[^\w\s]")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_DIGITS = re.compile(r"\d+")

# Placeholders consist of word characters only, so punctuation stripping and the
# usual tokenisers keep them intact, and they cannot collide with real words.
YEAR_TOKEN = " __year__ "
NUMBER_TOKEN = " __num__ "

PREPROCESSING_VERSIONS = ("norm_min_v1", "norm_legacy_v0", "keep_negations")
MASK_MODES = ("none", "years", "all")


def preprocessing_config(
    version: str, mask_numbers: str = "none", max_sequence_length: int = 100
) -> PreprocessingConfig:
    """Build the PreprocessingConfig for a declared preprocessing version."""
    if mask_numbers not in MASK_MODES:
        raise ValueError(f"mask_numbers must be one of {MASK_MODES}, got '{mask_numbers}'")
    legacy = PreprocessingConfig(max_sequence_length=max_sequence_length)
    if version == "norm_min_v1":
        cfg = replace(legacy, strip_punctuation=False, remove_stopwords=False)
    elif version == "norm_legacy_v0":
        cfg = legacy
    elif version == "keep_negations":
        cfg = replace(legacy, keep_negations=True)
    else:
        raise ValueError(
            f"Unknown preprocessing version '{version}'. Known: {PREPROCESSING_VERSIONS}"
        )
    return replace(cfg, mask_numbers=mask_numbers)


def mask_numbers(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "years":
        return _YEAR.sub(YEAR_TOKEN, text)
    if mode == "all":
        return _DIGITS.sub(NUMBER_TOKEN, text)
    raise ValueError(f"mask_numbers must be one of {MASK_MODES}, got '{mode}'")


# Negation words present in the NLTK English stopword list. The legacy pipeline
# removes them, which erases polarity cues ("not balanced" -> "balanced").
NEGATIONS: frozenset[str] = frozenset({"no", "not", "nor", "never", "none", "nothing", "neither"})


class StopwordsUnavailableError(RuntimeError):
    """Raised when NLTK stopwords are neither installed nor downloadable."""


@lru_cache(maxsize=1)
def nltk_english_stopwords() -> frozenset[str]:
    import nltk
    from nltk.corpus import stopwords

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        if not nltk.download("stopwords", quiet=True):
            raise StopwordsUnavailableError(
                "NLTK stopwords are missing and could not be downloaded. "
                "Run: python -m nltk.downloader stopwords"
            ) from None
    return frozenset(stopwords.words("english"))


class TextPreprocessor:
    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        stopwords: Iterable[str] | None = None,
    ) -> None:
        self.config = config or PreprocessingConfig()
        self._stopwords = frozenset(stopwords) if stopwords is not None else None

    @property
    def stopwords(self) -> frozenset[str]:
        if not self.config.remove_stopwords:
            return frozenset()
        words = self._stopwords if self._stopwords is not None else nltk_english_stopwords()
        if self.config.keep_negations:
            words = words - NEGATIONS
        return words

    def __call__(self, text: str) -> str:
        text = mask_numbers(text, self.config.mask_numbers)
        if self.config.lowercase:
            text = text.lower()
        if self.config.strip_punctuation:
            text = _NON_WORD.sub("", text)
        if self.config.remove_stopwords:
            stop = self.stopwords
            return " ".join(w for w in text.split() if w not in stop)
        return " ".join(text.split())

    def batch(self, texts: Iterable[str]) -> list[str]:
        return [self(t) for t in texts]
