"""Runtime paths missed by the fake-store API tests."""

import builtins
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from dropout import api, config, db
from dropout.schemas import EXAMPLE_STUDENT


def test_registry_ensemble_loads_and_serves(registered_ensemble, monkeypatch):
    uri, version, _ = registered_ensemble
    monkeypatch.setattr(config, "MODEL_PATH", None)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "DATABASE_URL", None)
    store = api.ModelStore()
    api.app.dependency_overrides[api.get_model_store] = lambda: store
    api.app.dependency_overrides[api.get_prediction_logger] = api.PredictionLogger
    try:
        response = TestClient(api.app).post(
            "/predict", json={"features": EXAMPLE_STUDENT}
        )
        assert response.status_code == 200, response.text
        assert response.json()["model_version"] == version
        assert abs(sum(response.json()["probabilities"].values()) - 1) < 0.001
        model, _ = store.get()
        assert "xgb" in model.named_steps["model"].named_estimators_
    finally:
        api.app.dependency_overrides.clear()


def test_integer_registry_version_is_normalized(monkeypatch, trained_model):
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    monkeypatch.setattr(config, "MODEL_PATH", None)
    monkeypatch.setattr(
        MlflowClient,
        "get_model_version_by_alias",
        lambda *a: SimpleNamespace(version=1),
    )
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda *a: trained_model)
    _, version = api.ModelStore().get()
    assert version == "1"


def test_missing_logging_import_does_not_break_serving(monkeypatch):
    original_import = builtins.__import__

    def import_without_db(name, *args, **kwargs):
        if name == "dropout.db":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(config, "LOG_PREDICTIONS", True)
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql+psycopg://unused")
    monkeypatch.setattr(builtins, "__import__", import_without_db)
    before = api.ERRORS.labels(type="logging")._value.get()
    logger = api.PredictionLogger()
    assert logger._engine is None
    assert api.ERRORS.labels(type="logging")._value.get() == before + 1


def test_database_connection_failure_is_counted(monkeypatch):
    monkeypatch.setattr(config, "LOG_PREDICTIONS", True)
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql+psycopg://unused")

    def unavailable(*args):
        raise ConnectionError("database unavailable")

    monkeypatch.setattr(db, "get_engine", unavailable)
    before = api.ERRORS.labels(type="logging")._value.get()
    logger = api.PredictionLogger()
    logger.log(features={})
    assert api.ERRORS.labels(type="logging")._value.get() == before + 1


def test_configured_postgres_prediction_is_persisted(monkeypatch, trained_model):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL required; exercised against PostgreSQL in CI")
    monkeypatch.setattr(config, "LOG_PREDICTIONS", True)
    monkeypatch.setattr(config, "DATABASE_URL", url)
    logger = api.PredictionLogger()
    assert logger._engine is not None
    api.app.dependency_overrides[api.get_model_store] = lambda: SimpleNamespace(
        get=lambda: (trained_model, "postgres-test")
    )
    api.app.dependency_overrides[api.get_prediction_logger] = lambda: logger
    try:
        response = TestClient(api.app).post(
            "/predict", json={"features": EXAMPLE_STUDENT, "ground_truth": "Graduate"}
        )
        assert response.status_code == 200, response.text
        with logger._engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT * FROM predictions "
                        "WHERE model_version = 'postgres-test' "
                        "ORDER BY id DESC LIMIT 1"
                    )
                )
                .mappings()
                .one()
            )
        assert row["features"] == EXAMPLE_STUDENT
        assert row["prediction"] == response.json()["prediction"]
        assert row["ground_truth"] == "Graduate"
        assert row["latency_ms"] >= 0
        assert (
            abs(
                sum(
                    row[k]
                    for k in ("proba_dropout", "proba_enrolled", "proba_graduate")
                )
                - 1
            )
            < 0.001
        )
    finally:
        api.app.dependency_overrides.clear()
        logger._engine.dispose()
