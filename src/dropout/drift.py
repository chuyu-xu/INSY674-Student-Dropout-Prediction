"""Drift injection, drift detection and retraining logic (owner: C).

Evidently is imported inside the functions that need it, so the simulator
can use ``inject_drift`` without installing Evidently.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from dropout import config
from dropout.features import CATEGORICAL, FEATURES, NUMERIC


def inject_drift(df: pd.DataFrame, strength: float = 1.0) -> pd.DataFrame:
    """Return a copy that mimics a weaker incoming cohort.

    Lower grades and pass counts, older students, more missed evaluations.
    ``strength`` scales the shift (0 = no change, 1 = full shift).
    """
    out = df.copy()
    s = float(strength)
    out["Curricular units 1st sem (grade)"] *= 1 - 0.35 * s
    out["Curricular units 1st sem (approved)"] = np.floor(
        out["Curricular units 1st sem (approved)"] * (1 - 0.5 * s)
    )
    out["Curricular units 1st sem (evaluations)"] = np.floor(
        out["Curricular units 1st sem (evaluations)"] * (1 + 0.4 * s)
    )
    out["Curricular units 1st sem (without evaluations)"] += round(2 * s)
    out["Admission grade"] = (out["Admission grade"] - 20 * s).clip(lower=95)
    out["Previous qualification (grade)"] = (
        out["Previous qualification (grade)"] - 20 * s
    ).clip(lower=95)
    out["Age at enrollment"] += round(7 * s)
    return out


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = config.DRIFT_THRESHOLD,
) -> dict:
    """Compare feature distributions with Evidently's dataset drift metric."""
    from evidently import ColumnMapping
    from evidently.metrics import DatasetDriftMetric
    from evidently.report import Report

    ref = reference[FEATURES].copy()
    cur = current[FEATURES].astype(ref.dtypes.to_dict())
    report = Report(metrics=[DatasetDriftMetric()])
    report.run(
        reference_data=ref,
        current_data=cur,
        column_mapping=ColumnMapping(
            numerical_features=NUMERIC, categorical_features=CATEGORICAL
        ),
    )
    result = report.as_dict()["metrics"][0]["result"]
    share = float(result["share_of_drifted_columns"])
    return {
        "share_drifted_features": share,
        "n_drifted_features": int(result["number_of_drifted_columns"]),
        "drift_detected": share >= threshold,
    }


def model_quality(current: pd.DataFrame) -> dict:
    """Accuracy and macro F1 on rows that carry a true label.

    Uses Evidently's classification metric and falls back to scikit-learn
    if the Evidently result format changes.
    """
    labeled = current.dropna(subset=["target"])
    if labeled.empty:
        return {"accuracy": None, "macro_f1": None}
    try:
        from evidently import ColumnMapping
        from evidently.metrics import ClassificationQualityMetric
        from evidently.report import Report

        report = Report(metrics=[ClassificationQualityMetric()])
        report.run(
            reference_data=None,
            current_data=labeled[["target", "prediction"]],
            column_mapping=ColumnMapping(target="target", prediction="prediction"),
        )
        quality = report.as_dict()["metrics"][0]["result"]["current"]
        return {
            "accuracy": float(quality["accuracy"]),
            "macro_f1": float(quality["f1"]),
        }
    except Exception:
        return {
            "accuracy": accuracy_score(labeled["target"], labeled["prediction"]),
            "macro_f1": f1_score(
                labeled["target"], labeled["prediction"], average="macro"
            ),
        }


def should_retrain(
    drift_detected: bool,
    last_retrain_at: datetime | None,
    now: datetime,
    cooldown_minutes: int = config.RETRAIN_COOLDOWN_MINUTES,
    enabled: bool = config.RETRAIN_ON_DRIFT,
) -> bool:
    """Retrain on drift, but not more than once per cooldown window."""
    if not (enabled and drift_detected):
        return False
    if last_retrain_at is None:
        return True
    return now - last_retrain_at >= timedelta(minutes=cooldown_minutes)
