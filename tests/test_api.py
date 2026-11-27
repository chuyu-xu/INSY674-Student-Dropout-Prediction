"""Owner: B. API behaviour with a fake model store (no MLflow needed)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")

from fastapi.testclient import TestClient  # noqa: E402

from dropout import api  # noqa: E402
from dropout.schemas import EXAMPLE_STUDENT  # noqa: E402


class FakeStore:
    def __init__(self, model):
        self.model = model

    def get(self):
        return self.model, "test"


class NoLogging:
    def log(self, **record):
        pass


@pytest.fixture()
def client(trained_model):
    api.app.dependency_overrides[api.get_model_store] = lambda: FakeStore(trained_model)
    api.app.dependency_overrides[api.get_prediction_logger] = NoLogging
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_valid_request_returns_readable_prediction(client):
    response = client.post("/predict", json={"features": EXAMPLE_STUDENT})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"Dropout", "Enrolled", "Graduate"}
    assert body["risk_level"] in {"High", "Medium", "Low"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3


def test_missing_feature_is_rejected(client):
    features = dict(EXAMPLE_STUDENT)
    features.pop("Admission grade")
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422
    assert "Admission grade" in response.text


def test_non_numeric_value_is_rejected(client):
    features = {**EXAMPLE_STUDENT, "Age at enrollment": "twenty"}
    assert client.post("/predict", json={"features": features}).status_code == 422


def test_out_of_range_value_is_rejected(client):
    features = {**EXAMPLE_STUDENT, "Curricular units 1st sem (grade)": 45}
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422
    assert "between" in response.text


def test_metrics_endpoint_exposes_request_counter(client):
    client.post("/predict", json={"features": EXAMPLE_STUDENT})
    metrics = client.get("/metrics").text
    assert "prediction_requests_total" in metrics
    assert "prediction_latency_seconds" in metrics


def test_schema_lists_every_feature(client):
    from dropout.features import FEATURES

    assert client.get("/schema").json()["features"] == FEATURES
