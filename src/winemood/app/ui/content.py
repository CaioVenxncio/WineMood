"""Fixed explanatory copy (no metrics). Numbers always come from results files."""

from __future__ import annotations

EXAMPLE_REVIEW = (
    "Deep ruby with aromas of black cherry, violet and graphite. The palate is dense yet "
    "focused, with ripe blackberry fruit, firm fine-grained tannins and fresh acidity "
    "carrying a long, spicy finish. Drink 2026-2034."
)
EXAMPLE_LABEL = "Illustrative example written for this demo - not a review from the dataset."

# Wine Enthusiast's published rating categories (paraphrased), merged as in
# the we_category_merged_v1 scheme used by the model's category output.
CATEGORY_MEANING = {
    "Acceptable (80-82)": "The publisher's lowest published band: drinkable, for casual occasions.",
    "Good (83-86)": "Suitable for everyday drinking; often good value.",
    "Very Good (87-89)": "Recommended; frequently good value.",
    "Excellent (90-93)": "Highly recommended.",
    "Superb or Classic (94-100)": "A great achievement, up to the top of the scale.",
}

WHAT_THIS_IS = (
    "WineMood estimates the score a Wine Enthusiast critic would likely have given to a "
    "wine described by this text. It reads language only - it does not taste wine, and the "
    "estimate is not a measure of quality, value or of your enjoyment. It is not sentiment "
    "analysis: the model learned from critics' numeric ratings, not from emotions."
)

AGGREGATE_NOT_INDIVIDUAL = (
    "These are averages over many held-out reviews. They describe how the model performs "
    "overall; they are not an uncertainty range for this particular estimate, which has "
    "not been calibrated."
)

CLIP_POLICY = (
    "The model's raw output was {raw:.2f}, outside the 80-100 range used by Wine "
    "Enthusiast. Following the documented display policy, the score is clipped to {shown:.1f}."
)

A31_NOTICE = (
    "This text begins with a score range (for example '92-94'). Some barrel-sample reviews "
    "in the training data start with the critic's own provisional score, and the model "
    "learned to use those numbers (audit A-31). The estimate is therefore partly reading "
    "the stated range rather than the description."
)

NOISE_LIMITATION = (
    "Scores are one critic's judgement on one occasion. Distinct reviews of the same wine "
    "and vintage in the dataset differ by {noise:.2f} points on average."
)

LIMITATIONS = [
    "Predictions are pulled towards the middle of the scale: very low (80-83) and very high "
    "(94-100) ratings are systematically under- and over-shot.",
    "Accuracy drops for critics the model has never seen.",
    "Longer notes and notes that recommend a drinking window receive higher estimates; part "
    "of the signal is the critic's framing, not sensory description.",
    "Trained on English-language Wine Enthusiast reviews scraped in 2017; other sources, "
    "styles or languages are out of scope.",
    "Data licence CC BY-NC-SA 4.0: non-commercial use only.",
]
