"""Owner: F. Integration test against the running API container.

CI starts the container, then runs:  API_URL=http://localhost:8000 pytest -m integration
"""

import os
import time

import pytest
import requests

from dropout.schemas import EXAMPLE_STUDENT

API_URL = os.getenv("API_URL", "http://localhost:8000")
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def wait_for_api():
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            if requests.get(f"{API_URL}/health", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    pytest.fail(f"API at {API_URL} did not become healthy")


def test_container_serves_a_prediction():
    response = requests.post(
        f"{API_URL}/predict", json={"features": EXAMPLE_STUDENT}, timeout=10
    )
    assert response.status_code == 200
    assert response.json()["prediction"] in {"Dropout", "Enrolled", "Graduate"}


def test_container_rejects_bad_input():
    response = requests.post(f"{API_URL}/predict", json={"features": {}}, timeout=10)
    assert response.status_code == 422


def test_container_exposes_metrics():
    assert "prediction_requests_total" in requests.get(f"{API_URL}/metrics").text
