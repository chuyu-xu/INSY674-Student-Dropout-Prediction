"""Request/response models and input rules for the API (owner: B)."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from dropout.data import CLASSES
from dropout.features import CATEGORICAL, FEATURES, NUMERIC

# A typical student (mode of each categorical, median of each numeric).
EXAMPLE_STUDENT: dict[str, float] = {
    "Marital status": 1,
    "Application mode": 1,
    "Course": 9500,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Nacionality": 1,
    "Mother's qualification": 1,
    "Father's qualification": 37,
    "Mother's occupation": 9,
    "Father's occupation": 9,
    "Displaced": 1,
    "Educational special needs": 0,
    "Gender": 0,
    "International": 0,
    "Application order": 1,
    "Previous qualification (grade)": 133.1,
    "Admission grade": 126.1,
    "Age at enrollment": 20,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 8,
    "Curricular units 1st sem (approved)": 5,
    "Curricular units 1st sem (grade)": 12.29,
    "Curricular units 1st sem (without evaluations)": 0,
}

# Plausible value ranges; anything outside is rejected with a clear message.
RANGES: dict[str, tuple[float, float]] = {
    "Application order": (0, 9),
    "Previous qualification (grade)": (0, 200),
    "Admission grade": (0, 200),
    "Age at enrollment": (15, 100),
    "Curricular units 1st sem (credited)": (0, 60),
    "Curricular units 1st sem (enrolled)": (0, 60),
    "Curricular units 1st sem (evaluations)": (0, 100),
    "Curricular units 1st sem (approved)": (0, 60),
    "Curricular units 1st sem (grade)": (0, 20),
    "Curricular units 1st sem (without evaluations)": (0, 60),
}

ClassLabel = Literal["Dropout", "Enrolled", "Graduate"]


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="One value per model feature", examples=[EXAMPLE_STUDENT]
    )
    ground_truth: ClassLabel | None = Field(
        None, description="True outcome, if known (used by the simulator)"
    )

    @field_validator("features")
    @classmethod
    def check_features(cls, values: dict[str, float]) -> dict[str, float]:
        problems = []
        missing = [f for f in FEATURES if f not in values]
        unknown = [f for f in values if f not in FEATURES]
        if missing:
            problems.append(f"missing features: {missing}")
        if unknown:
            problems.append(f"unknown features: {unknown}")
        for name, value in values.items():
            if not math.isfinite(value):
                problems.append(f"{name} must be a finite number")
            elif name in CATEGORICAL and (value < 0 or not float(value).is_integer()):
                problems.append(f"{name} must be a non-negative integer code")
            elif name in RANGES:
                low, high = RANGES[name]
                if not low <= value <= high:
                    problems.append(f"{name} must be between {low} and {high}")
        if problems:
            raise ValueError("; ".join(problems))
        return values


class PredictionResponse(BaseModel):
    prediction: ClassLabel
    risk_level: Literal["High", "Medium", "Low"]
    probabilities: dict[str, float]
    model_version: str | None


def risk_level(p_dropout: float) -> str:
    if p_dropout >= 0.5:
        return "High"
    if p_dropout >= 0.25:
        return "Medium"
    return "Low"


SCHEMA_INFO = {
    "features": FEATURES,
    "categorical": CATEGORICAL,
    "numeric": NUMERIC,
    "ranges": RANGES,
    "classes": CLASSES,
    "example": EXAMPLE_STUDENT,
}
