import math

import pandas as pd

from dropout.features import add_first_semester_features


def _row(enrolled, approved, evaluations):
    return pd.DataFrame(
        {
            "Curricular units 1st sem (enrolled)": [enrolled],
            "Curricular units 1st sem (approved)": [approved],
            "Curricular units 1st sem (evaluations)": [evaluations],
        }
    )


def test_known_input_gives_expected_features():
    out = add_first_semester_features(_row(enrolled=6, approved=3, evaluations=12))
    assert out["First semester pass rate"].iloc[0] == 0.5
    assert out["Evaluations per enrolled course"].iloc[0] == 2.0
    assert out["No first-semester enrollment"].iloc[0] == 0


def test_zero_enrollment_gives_nan_ratios_and_flag():
    out = add_first_semester_features(_row(enrolled=0, approved=0, evaluations=0))
    assert math.isnan(out["First semester pass rate"].iloc[0])
    assert math.isnan(out["Evaluations per enrolled course"].iloc[0])
    assert out["No first-semester enrollment"].iloc[0] == 1
