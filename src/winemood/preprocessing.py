"""Single text normaliser shared by training and inference.

The default configuration reproduces the original notebook exactly:
lowercase -> remove non-word characters -> drop NLTK English stopwords.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from winemood.config import PreprocessingConfig

_NON_WORD = re.compile(r"[^\w\s]")

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
