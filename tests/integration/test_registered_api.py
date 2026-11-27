"""Load the real registered ensemble inside the API image, then persist a request."""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import pytest
from sqlalchemy import create_engine, text

from dropout.schemas import EXAMPLE_STUDENT


@pytest.mark.integration
def test_api_image_registry_and_postgres(registered_ensemble):
    uri, version, directory = registered_ensemble
    url = os.environ["TEST_DATABASE_URL"]
    name = "dropout-api-registry-test"
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "--network",
            "host",
            "-v",
            f"{directory}:{directory}",
            "-e",
            f"MLFLOW_TRACKING_URI={uri}",
            "-e",
            f"DATABASE_URL={url}",
            "-e",
            "LOG_PREDICTIONS=true",
            "dropout-api:test",
        ],
        check=True,
    )
    try:
        last_error = "API did not respond"
        for _ in range(60):
            try:
                with urllib.request.urlopen(
                    "http://localhost:8000/health", timeout=2
                ) as r:
                    assert json.load(r)["model_version"] == version
                break
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}: {exc.read().decode()}"
                time.sleep(1)
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = str(exc)
                time.sleep(1)
        else:
            pytest.fail(f"API image could not load registered ensemble: {last_error}")
        request = urllib.request.Request(
            "http://localhost:8000/predict",
            data=json.dumps(
                {"features": EXAMPLE_STUDENT, "ground_truth": "Graduate"}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=15) as r:
            result = json.load(r)
        assert result["model_version"] == version
        assert abs(sum(result["probabilities"].values()) - 1) < 0.001
        engine = create_engine(url)
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT prediction, ground_truth FROM predictions "
                        "ORDER BY id DESC LIMIT 1"
                    )
                ).one()
            assert row.prediction == result["prediction"]
            assert row.ground_truth == "Graduate"
        finally:
            engine.dispose()
    finally:
        subprocess.run(["docker", "logs", name], check=False)
        subprocess.run(["docker", "stop", name], check=False)
