"""Owner: D. Runs are logged and the final ensemble is the champion."""

import json
from pathlib import Path

import pytest

pytest.importorskip("mlflow")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from dropout import config  # noqa: E402
from dropout.tracking import candidate_models, run_experiments  # noqa: E402


def test_voting_ensemble_uses_saved_insy674_settings():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation_results"
        / "model_selection_used.json"
    )
    saved_parameters = json.loads(config_path.read_text())["tuned_parameters"]
    ensemble, _ = candidate_models()["soft_voting_ensemble"]
    expected = {
        "xgb": "XGBoost",
        "gb": "Gradient Boosting",
        "lr": "Logistic Regression",
    }

    assert set(dict(ensemble.estimators)) == set(expected)
    for key, estimator in ensemble.estimators:
        params = estimator.get_params()
        for parameter, value in saved_parameters[expected[key]].items():
            assert params[parameter.removeprefix("model__")] == value


def test_final_ensemble_is_registered_with_champion_alias(tmp_path):
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("test", artifact_location=(tmp_path / "art").as_uri())

    candidates = {
        "soft_voting_ensemble": (
            LogisticRegression(C=0.001, max_iter=500),
            {"C": 0.001},
        ),
        "comparison": (LogisticRegression(C=0.1, max_iter=2000), {"C": 0.1}),
    }
    result = run_experiments(
        candidates=candidates, tracking_uri=uri, experiment_name="test"
    )

    assert len(result["runs"]) == 2
    best = max(result["runs"], key=lambda r: r["cv_macro_f1_mean"])
    assert result["best"]["name"] == best["name"]
    assert result["registered"]["name"] == "soft_voting_ensemble"
    version = MlflowClient().get_model_version_by_alias(
        config.MODEL_NAME, config.MODEL_ALIAS
    )
    assert version.run_id == result["registered"]["run_id"]
