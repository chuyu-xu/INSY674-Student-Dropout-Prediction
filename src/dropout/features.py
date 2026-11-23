"""Feature set and preprocessing, reused from the INSY 674 project
("First Semester" scenario)."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Gender",
    "International",
]

NUMERIC = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]

FEATURES = CATEGORICAL + NUMERIC

ENGINEERED = [
    "First semester pass rate",
    "Evaluations per enrolled course",
    "No first-semester enrollment",
]


def add_first_semester_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add ratio features; ratios are NaN when no units were enrolled."""
    data = data.copy()
    enrolled = data["Curricular units 1st sem (enrolled)"]
    denominator = enrolled.replace(0, np.nan)
    data["First semester pass rate"] = (
        data["Curricular units 1st sem (approved)"] / denominator
    )
    data["Evaluations per enrolled course"] = (
        data["Curricular units 1st sem (evaluations)"] / denominator
    )
    data["No first-semester enrollment"] = (enrolled == 0).astype(int)
    return data


def build_preprocessor() -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return Pipeline(
        [
            (
                "feature_engineering",
                FunctionTransformer(add_first_semester_features, validate=False),
            ),
            (
                "preprocessing",
                ColumnTransformer(
                    [
                        ("numeric", numeric, NUMERIC + ENGINEERED),
                        ("categorical", categorical, CATEGORICAL),
                    ]
                ),
            ),
        ]
    )
