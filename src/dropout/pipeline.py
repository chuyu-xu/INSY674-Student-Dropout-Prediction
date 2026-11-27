"""Prefect training pipeline (owner: E).

load data -> validate -> (optionally) add labeled production data
-> train and track candidates in MLflow -> register the best model.

Run once:            PYTHONPATH=src python -m dropout.pipeline
Retrain with logs:   PYTHONPATH=src python -m dropout.pipeline --include-feedback
The drift monitor calls ``training_flow(include_feedback=True, reason="drift")``.
"""

from __future__ import annotations

import argparse

import pandas as pd
from prefect import flow, get_run_logger, task

from dropout import config
from dropout.data import TARGET, load_data, validate_data
from dropout.features import FEATURES
from dropout.tracking import run_experiments


def combine_with_feedback(feedback: pd.DataFrame | None) -> pd.DataFrame | None:
    """Keep only complete, labeled feedback rows in training format."""
    if feedback is None or feedback.empty:
        return None
    cleaned = feedback.dropna(subset=FEATURES + [TARGET])
    return cleaned[FEATURES + [TARGET]] if not cleaned.empty else None


@task(retries=2, retry_delay_seconds=5)
def load_dataset() -> pd.DataFrame:
    return load_data()


@task
def validate(df: pd.DataFrame) -> pd.DataFrame:
    validate_data(df, FEATURES)
    return df


@task(retries=2, retry_delay_seconds=10)
def load_feedback() -> pd.DataFrame | None:
    if not config.DATABASE_URL:
        return None
    from dropout.db import get_engine, load_labeled_predictions

    return combine_with_feedback(
        load_labeled_predictions(get_engine(config.DATABASE_URL))
    )


@task
def train_and_register(
    df: pd.DataFrame, feedback: pd.DataFrame | None, reason: str
) -> dict:
    return run_experiments(df=df, extra_train=feedback, run_tag=reason)


@flow(name="dropout-training")
def training_flow(include_feedback: bool = False, reason: str = "manual") -> dict:
    logger = get_run_logger()
    df = validate(load_dataset())
    feedback = load_feedback() if include_feedback else None
    n_extra = 0 if feedback is None else len(feedback)
    logger.info("Training on %d base rows + %d feedback rows", len(df), n_extra)

    result = train_and_register(df, feedback, reason)
    best = result["best"]
    logger.info(
        "Best model %s (cv macro F1 %.4f), registered as version %s",
        best["name"],
        best["cv_macro_f1_mean"],
        result["version"],
    )
    return {
        "best_model": best["name"],
        "version": result["version"],
        "n_extra": n_extra,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-feedback", action="store_true")
    parser.add_argument("--reason", default="manual")
    args = parser.parse_args()
    training_flow(include_feedback=args.include_feedback, reason=args.reason)
