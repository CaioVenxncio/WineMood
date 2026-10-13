import pandas as pd

from winemood.data.schema import validate_reviews


def test_valid_frame_reports_duplicates(reviews_df):
    report = validate_reviews(reviews_df)
    assert report.ok
    assert report.n_rows == 64
    assert report.n_duplicate_text == 3  # exact duplicates only; normalised ones are not counted
    assert report.points_min == 80 and report.points_max == 99
    assert any("duplicate" in w for w in report.warnings)


def test_missing_column_is_error():
    report = validate_reviews(pd.DataFrame({"description": ["x"]}))
    assert not report.ok
    assert "points" in report.errors[0]


def test_non_numeric_target_is_error():
    report = validate_reviews(pd.DataFrame({"description": ["x"], "points": ["ninety"]}))
    assert not report.ok


def test_out_of_range_and_missing_values_warn():
    df = pd.DataFrame({"description": ["a", None, " ", "d"], "points": [79.0, 85.0, None, 101.0]})
    report = validate_reviews(df)
    assert report.ok
    assert report.n_missing_text == 1
    assert report.n_empty_text == 1
    assert report.n_missing_target == 1
    assert any("outside [80, 100]" in w for w in report.warnings)
