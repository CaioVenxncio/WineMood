from winemood.config import PreprocessingConfig
from winemood.preprocessing import TextPreprocessor


def test_legacy_pipeline(stopwords):
    pre = TextPreprocessor(stopwords=stopwords)
    assert pre("The wine is NOT balanced, with a long finish!") == "wine balanced long finish"


def test_keep_negations_option(stopwords):
    pre = TextPreprocessor(PreprocessingConfig(keep_negations=True), stopwords=stopwords)
    assert pre("The wine is NOT balanced") == "wine not balanced"


def test_disable_stopword_removal(stopwords):
    pre = TextPreprocessor(PreprocessingConfig(remove_stopwords=False), stopwords=stopwords)
    assert pre("The  wine,  is   fine.") == "the wine is fine"


def test_accents_preserved(stopwords):
    pre = TextPreprocessor(stopwords=stopwords)
    assert pre("Rosé from Côtes-du-Rhône") == "rosé from côtesdurhône"


def test_batch(stopwords):
    pre = TextPreprocessor(stopwords=stopwords)
    assert pre.batch(["A red.", "The white"]) == ["red", "white"]
