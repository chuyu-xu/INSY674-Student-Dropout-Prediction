"""Drift monitor (owner: C).

Every MONITOR_INTERVAL_SECONDS: read the latest logged predictions from
Postgres, compute data drift and model quality with Evidently, store the
result in ``drift_metrics`` (Grafana reads it), and trigger retraining
when drift is detected.

    PYTHONPATH=src python -m dropout.monitor          # loop forever
    PYTHONPATH=src python -m dropout.monitor --once   # single check
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from dropout import config
from dropout.data import load_data
from dropout.db import (
    create_tables,
    get_engine,
    last_retrain_time,
    load_recent_predictions,
    save_drift_metrics,
)
from dropout.drift import detect_drift, model_quality, should_retrain
from dropout.train import split


def reference_data():
    """The training split is the reference distribution."""
    X_train, _, _, _ = split(load_data())
    return X_train


def trigger_retraining() -> bool:
    try:
        from dropout.pipeline import training_flow

        training_flow(include_feedback=True, reason="drift")
        return True
    except Exception as exc:  # keep monitoring even if retraining fails
        print(f"Retraining failed: {exc}")
        return False


def check_once(engine, reference) -> dict | None:
    current = load_recent_predictions(engine, config.MONITOR_WINDOW)
    if len(current) < config.MONITOR_MIN_ROWS:
        print(f"Only {len(current)} predictions logged, waiting for more")
        return None

    drift = detect_drift(reference, current)
    quality = model_quality(current)
    retrain = should_retrain(
        drift["drift_detected"], last_retrain_time(engine), datetime.now(timezone.utc)
    )
    retrained = trigger_retraining() if retrain else False

    metrics = {
        "window_size": len(current),
        **drift,
        **quality,
        "retrain_triggered": retrained,
    }
    save_drift_metrics(engine, metrics)
    print(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if not config.DATABASE_URL:
        raise SystemExit("DATABASE_URL is not set")
    engine = get_engine(config.DATABASE_URL)
    create_tables(engine)
    reference = reference_data()

    while True:
        try:
            check_once(engine, reference)
        except Exception as exc:
            print(f"Monitoring check failed: {exc}")
        if args.once:
            break
        time.sleep(config.MONITOR_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
