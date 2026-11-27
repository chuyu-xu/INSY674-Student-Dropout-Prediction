import pytest

from dropout.data import load_data
from dropout.train import build_model, split


@pytest.fixture(scope="session")
def data_split():
    return split(load_data())


@pytest.fixture(scope="session")
def trained_model(data_split):
    X_train, _, y_train, _ = data_split
    return build_model().fit(X_train, y_train)


@pytest.fixture()
def registered_ensemble(tmp_path, data_split):
    """Register the actual deployment estimator, including XGBoost."""
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from sklearn.pipeline import Pipeline

    from dropout.features import build_preprocessor
    from dropout.tracking import FINAL_MODEL_NAME, candidate_models

    previous_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    experiment = client.create_experiment(
        "api-registry-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    estimator, _ = candidate_models()[FINAL_MODEL_NAME]
    model = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
    X_train, _, y_train, _ = data_split
    model.fit(X_train, y_train)
    with mlflow.start_run(experiment_id=experiment) as run:
        mlflow.sklearn.log_model(model, "model")
        version = mlflow.register_model(
            f"runs:/{run.info.run_id}/model", "dropout-model"
        )
    client.set_registered_model_alias("dropout-model", "champion", version.version)
    try:
        yield uri, str(version.version), tmp_path
    finally:
        mlflow.set_tracking_uri(previous_uri)
