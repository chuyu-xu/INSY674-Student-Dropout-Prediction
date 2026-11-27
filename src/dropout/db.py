"""PostgreSQL helpers for logged predictions and monitoring metrics."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from dropout.data import TARGET
from dropout.features import FEATURES

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    features JSONB NOT NULL,
    prediction TEXT NOT NULL,
    proba_dropout DOUBLE PRECISION,
    proba_enrolled DOUBLE PRECISION,
    proba_graduate DOUBLE PRECISION,
    ground_truth TEXT,
    model_version TEXT,
    latency_ms DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS drift_metrics (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_size INTEGER NOT NULL,
    share_drifted_features DOUBLE PRECISION NOT NULL,
    n_drifted_features INTEGER NOT NULL,
    drift_detected BOOLEAN NOT NULL,
    accuracy DOUBLE PRECISION,
    macro_f1 DOUBLE PRECISION,
    retrain_triggered BOOLEAN NOT NULL DEFAULT FALSE
);
"""


def get_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def create_tables(engine: Engine) -> None:
    with engine.begin() as conn:
        for statement in SCHEMA_SQL.split(";"):
            if statement.strip():
                conn.execute(text(statement))


def log_prediction(
    engine: Engine,
    features: dict,
    prediction: str,
    probabilities: dict[str, float],
    model_version: str | None,
    latency_ms: float,
    ground_truth: str | None = None,
) -> None:
    query = text(
        """
        INSERT INTO predictions (features, prediction, proba_dropout,
            proba_enrolled, proba_graduate, ground_truth, model_version, latency_ms)
        VALUES (CAST(:features AS JSONB), :prediction, :p_dropout, :p_enrolled,
            :p_graduate, :ground_truth, :model_version, :latency_ms)
        """
    )
    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "features": json.dumps(features),
                "prediction": prediction,
                "p_dropout": probabilities.get("Dropout"),
                "p_enrolled": probabilities.get("Enrolled"),
                "p_graduate": probabilities.get("Graduate"),
                "ground_truth": ground_truth,
                "model_version": model_version,
                "latency_ms": latency_ms,
            },
        )


def _expand(rows: list) -> pd.DataFrame:
    """Turn (features_json, prediction, ground_truth) rows into a flat frame."""
    records = []
    for features, prediction, ground_truth in rows:
        record = dict(features if isinstance(features, dict) else json.loads(features))
        record["prediction"] = prediction
        record["target"] = ground_truth
        records.append(record)
    columns = FEATURES + ["prediction", "target"]
    return pd.DataFrame(records, columns=columns)


def load_recent_predictions(engine: Engine, limit: int) -> pd.DataFrame:
    query = text(
        "SELECT features, prediction, ground_truth FROM predictions "
        "ORDER BY created_at DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": limit}).fetchall()
    return _expand(rows)


def load_labeled_predictions(engine: Engine) -> pd.DataFrame:
    """Logged requests that came with a true label, in training format."""
    query = text(
        "SELECT features, prediction, ground_truth FROM predictions "
        "WHERE ground_truth IS NOT NULL"
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    df = _expand(rows).drop(columns=["prediction"])
    return df.rename(columns={"target": TARGET})


def save_drift_metrics(engine: Engine, metrics: dict) -> None:
    query = text(
        """
        INSERT INTO drift_metrics (window_size, share_drifted_features,
            n_drifted_features, drift_detected, accuracy, macro_f1, retrain_triggered)
        VALUES (:window_size, :share_drifted_features, :n_drifted_features,
            :drift_detected, :accuracy, :macro_f1, :retrain_triggered)
        """
    )
    with engine.begin() as conn:
        conn.execute(query, metrics)


def last_retrain_time(engine: Engine) -> datetime | None:
    query = text(
        "SELECT MAX(created_at) FROM drift_metrics WHERE retrain_triggered = TRUE"
    )
    with engine.connect() as conn:
        return conn.execute(query).scalar()
