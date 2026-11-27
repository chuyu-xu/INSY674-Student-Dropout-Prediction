"""Central settings, read from environment variables with local defaults.

Every service (trainer, API, monitor, simulator) imports from here, so the
model name, alias and drift threshold are defined in exactly one place.
"""

import os


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "dropout-prediction")
MODEL_NAME = os.getenv("MODEL_NAME", "dropout-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# Optional local model file (used by the CI integration test instead of MLflow)
MODEL_PATH = os.getenv("MODEL_PATH")

# PostgreSQL, e.g. postgresql+psycopg://user:pass@postgres:5432/dropout
DATABASE_URL = os.getenv("DATABASE_URL")
LOG_PREDICTIONS = _bool("LOG_PREDICTIONS", True)

# Drift monitoring
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
MONITOR_WINDOW = int(os.getenv("MONITOR_WINDOW", "300"))
MONITOR_MIN_ROWS = int(os.getenv("MONITOR_MIN_ROWS", "100"))
MONITOR_INTERVAL_SECONDS = int(os.getenv("MONITOR_INTERVAL_SECONDS", "60"))
RETRAIN_ON_DRIFT = _bool("RETRAIN_ON_DRIFT", True)
RETRAIN_COOLDOWN_MINUTES = int(os.getenv("RETRAIN_COOLDOWN_MINUTES", "30"))

# How often the API checks the registry for a new champion version
MODEL_REFRESH_SECONDS = int(os.getenv("MODEL_REFRESH_SECONDS", "60"))
