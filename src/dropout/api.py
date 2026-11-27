"""Model API (owner: B).

Serves predictions from the MLflow registry model ``dropout-model@champion``
(or from MODEL_PATH, a local joblib file used in the CI integration test),
exposes Prometheus metrics at /metrics, and logs every prediction to Postgres.

    PYTHONPATH=src uvicorn dropout.api:app --reload --port 8000
"""

from __future__ import annotations

import threading
import time
from time import perf_counter

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from dropout import config
from dropout.features import CATEGORICAL, FEATURES
from dropout.schemas import (
    SCHEMA_INFO,
    PredictionRequest,
    PredictionResponse,
    risk_level,
)

REQUESTS = Counter(
    "prediction_requests_total", "Prediction requests", ["endpoint", "status"]
)
LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent serving a prediction",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
ERRORS = Counter("prediction_errors_total", "Failed requests by type", ["type"])
PREDICTED = Counter("predictions_by_class_total", "Predicted labels", ["label"])
MODEL_VERSION = Gauge("model_version_info", "Model version being served", ["version"])


class ModelUnavailableError(RuntimeError):
    pass


class ModelStore:
    """Loads the model lazily and reloads it when the champion alias moves."""

    def __init__(self) -> None:
        self._model = None
        self._version: str | None = None
        self._checked_at = 0.0
        self._lock = threading.Lock()

    def _load(self) -> None:
        if config.MODEL_PATH:
            import joblib

            self._model, self._version = joblib.load(config.MODEL_PATH), "local"
            return

        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        client = MlflowClient()
        version = str(
            client.get_model_version_by_alias(
                config.MODEL_NAME, config.MODEL_ALIAS
            ).version
        )
        if version != self._version:
            uri = f"models:/{config.MODEL_NAME}/{version}"
            self._model, self._version = mlflow.sklearn.load_model(uri), version
            MODEL_VERSION.clear()
            MODEL_VERSION.labels(version=version).set(1)

    def get(self):
        with self._lock:
            stale = time.time() - self._checked_at > config.MODEL_REFRESH_SECONDS
            if self._model is None or (stale and not config.MODEL_PATH):
                try:
                    self._load()
                except Exception as exc:
                    if self._model is None:
                        raise ModelUnavailableError(str(exc)) from exc
                self._checked_at = time.time()
            return self._model, self._version


class PredictionLogger:
    def __init__(self) -> None:
        self._engine = None
        if config.LOG_PREDICTIONS and config.DATABASE_URL:
            try:
                from dropout.db import create_tables, get_engine

                engine = get_engine(config.DATABASE_URL)
                create_tables(engine)
                self._engine = engine
            except Exception:
                # Serving must keep working even if the database is down.
                ERRORS.labels(type="logging").inc()

    def log(self, **record) -> None:
        if self._engine is None:
            return
        try:
            from dropout.db import log_prediction

            log_prediction(self._engine, **record)
        except Exception:
            ERRORS.labels(type="logging").inc()


_store = ModelStore()
_logger: PredictionLogger | None = None


def get_model_store() -> ModelStore:
    return _store


def get_prediction_logger() -> PredictionLogger:
    global _logger
    if _logger is None:
        _logger = PredictionLogger()
    return _logger


app = FastAPI(title="Student Dropout Prediction API", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def count_validation_errors(request: Request, exc: RequestValidationError):
    REQUESTS.labels(endpoint=request.url.path, status="422").inc()
    ERRORS.labels(type="validation").inc()
    return await request_validation_exception_handler(request, exc)


@app.get("/health")
def health(store: ModelStore = Depends(get_model_store)) -> dict:
    try:
        _, version = store.get()
        return {"status": "ok", "model_version": version}
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {exc}") from exc


@app.get("/schema")
def schema() -> dict:
    return SCHEMA_INFO


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    store: ModelStore = Depends(get_model_store),
    prediction_logger: PredictionLogger = Depends(get_prediction_logger),
) -> PredictionResponse:
    start = perf_counter()
    try:
        model, version = store.get()
    except ModelUnavailableError as exc:
        REQUESTS.labels(endpoint="/predict", status="503").inc()
        ERRORS.labels(type="model_unavailable").inc()
        raise HTTPException(status_code=503, detail="Model is not available") from exc

    try:
        X = pd.DataFrame([request.features])[FEATURES]
        X[CATEGORICAL] = X[CATEGORICAL].astype(int)
        proba = model.predict_proba(X)[0]
        probabilities = {
            label: round(float(p), 4)
            for label, p in zip(model.classes_, proba, strict=True)
        }
        prediction = max(probabilities, key=probabilities.get)
    except Exception as exc:
        REQUESTS.labels(endpoint="/predict", status="500").inc()
        ERRORS.labels(type="inference").inc()
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    elapsed = perf_counter() - start
    LATENCY.observe(elapsed)
    REQUESTS.labels(endpoint="/predict", status="200").inc()
    PREDICTED.labels(label=prediction).inc()
    prediction_logger.log(
        features=request.features,
        prediction=prediction,
        probabilities=probabilities,
        model_version=version,
        latency_ms=elapsed * 1000,
        ground_truth=request.ground_truth,
    )
    return PredictionResponse(
        prediction=prediction,
        risk_level=risk_level(probabilities.get("Dropout", 0.0)),
        probabilities=probabilities,
        model_version=version,
    )
