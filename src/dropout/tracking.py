"""MLflow experiment tracking (owner: D).

Trains several candidate models, logs each one as an MLflow run with its
parameters, metrics and model artifact, then registers the final INSY 674
ensemble as ``dropout-model`` and points the ``champion`` alias at it.

Run locally:  PYTHONPATH=src python -m dropout.tracking
Open the UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from dropout import config
from dropout.data import TARGET, load_data, validate_data
from dropout.features import FEATURES, build_preprocessor
from dropout.train import SEED, split

# Saved training-CV settings from evaluation_results/model_selection_used.json.
TUNED_LOGREG_C = 0.04480970566385288
TUNED_BOOSTING = {
    "learning_rate": 0.12195678219063029,
    "max_depth": 3,
    "min_samples_leaf": 12,
    "n_estimators": 134,
    "subsample": 0.9941308100323758,
}
TUNED_XGBOOST = {
    "colsample_bytree": 0.7810890415965769,
    "learning_rate": 0.1785436060870726,
    "max_depth": 4,
    "min_child_weight": 8,
    "n_estimators": 100,
    "reg_lambda": 0.6154014789262349,
    "subsample": 0.7045980821176709,
}
FINAL_MODEL_NAME = "soft_voting_ensemble"


def _logreg(c: float) -> LogisticRegression:
    return LogisticRegression(
        C=c, class_weight="balanced", max_iter=2000, random_state=SEED
    )


def _forest(n_estimators: int, max_depth: int | None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def _boosting() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=SEED, **TUNED_BOOSTING)


def _xgboost() -> XGBClassifier:
    return XGBClassifier(
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
        eval_metric="mlogloss",
        **TUNED_XGBOOST,
    )


def candidate_models() -> dict[str, tuple[object, dict]]:
    """Name -> (unfitted estimator, params to log)."""
    return {
        "logreg_C0.01": (_logreg(0.01), {"model_type": "logreg", "C": 0.01}),
        "logreg_C0.0448": (
            _logreg(TUNED_LOGREG_C),
            {"model_type": "logreg", "C": TUNED_LOGREG_C},
        ),
        "logreg_C1": (_logreg(1.0), {"model_type": "logreg", "C": 1.0}),
        "rf_200_depth10": (
            _forest(200, 10),
            {"model_type": "random_forest", "n_estimators": 200, "max_depth": 10},
        ),
        "rf_400_full": (
            _forest(400, None),
            {"model_type": "random_forest", "n_estimators": 400, "max_depth": None},
        ),
        "gradient_boosting": (
            _boosting(),
            {"model_type": "gradient_boosting", **TUNED_BOOSTING},
        ),
        "xgboost_tuned": (
            # VotingClassifier maps XGBoost's numeric labels to the original classes.
            VotingClassifier([("xgb", _xgboost())], voting="soft", n_jobs=1),
            {"model_type": "xgboost", **TUNED_XGBOOST},
        ),
        "soft_voting_ensemble": (
            VotingClassifier(
                estimators=[
                    ("xgb", _xgboost()),
                    ("gb", _boosting()),
                    ("lr", _logreg(TUNED_LOGREG_C)),
                ],
                voting="soft",
                n_jobs=1,
            ),
            {
                "model_type": "soft_voting",
                "members": "xgb+gb+lr",
                "lr_C": TUNED_LOGREG_C,
                **{f"xgb_{key}": value for key, value in TUNED_XGBOOST.items()},
                **{f"gb_{key}": value for key, value in TUNED_BOOSTING.items()},
            },
        ),
    }


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = model.predict(X)
    return {
        "test_macro_f1": f1_score(y, pred, average="macro"),
        "test_accuracy": accuracy_score(y, pred),
        "test_balanced_accuracy": balanced_accuracy_score(y, pred),
        "test_dropout_recall": recall_score(
            y, pred, labels=["Dropout"], average="macro"
        ),
    }


def register_best(run_id: str, metric_value: float) -> str:
    """Register the run's model and move the champion alias to it."""
    version = mlflow.register_model(f"runs:/{run_id}/model", config.MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(
        config.MODEL_NAME, config.MODEL_ALIAS, version.version
    )
    client.set_model_version_tag(
        config.MODEL_NAME, version.version, "cv_macro_f1", f"{metric_value:.4f}"
    )
    return version.version


def run_experiments(
    df: pd.DataFrame | None = None,
    extra_train: pd.DataFrame | None = None,
    candidates: dict | None = None,
    register: bool = True,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    run_tag: str = "manual",
) -> dict:
    """Train every candidate and register the final INSY 674 ensemble.

    ``extra_train`` holds labeled production data (from the API logs) that
    is appended to the training split when retraining after drift.
    """
    mlflow.set_tracking_uri(tracking_uri or config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name or config.EXPERIMENT_NAME)

    if df is None:
        df = load_data()
    validate_data(df, FEATURES)
    X_train, X_test, y_train, y_test = split(df)

    if extra_train is not None and not extra_train.empty:
        X_train = pd.concat([X_train, extra_train[FEATURES]], ignore_index=True)
        y_train = pd.concat([y_train, extra_train[TARGET]], ignore_index=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    for name, (estimator, params) in (candidates or candidate_models()).items():
        with mlflow.start_run(run_name=name) as run:
            pipeline = Pipeline(
                [("preprocess", build_preprocessor()), ("model", estimator)]
            )
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="f1_macro"
            )
            pipeline.fit(X_train, y_train)

            metrics = evaluate(pipeline, X_test, y_test)
            metrics["cv_macro_f1_mean"] = float(cv_scores.mean())
            metrics["cv_macro_f1_std"] = float(cv_scores.std())

            mlflow.set_tag("trigger", run_tag)
            mlflow.log_params(
                {
                    **params,
                    "feature_set": "first_semester",
                    "n_train_rows": len(X_train),
                    "n_extra_rows": 0 if extra_train is None else len(extra_train),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=infer_signature(X_test, pipeline.predict(X_test)),
                input_example=X_test.head(3),
            )
            results.append({"name": name, "run_id": run.info.run_id, **metrics})
            print(
                f"{name:<22} cv_f1={metrics['cv_macro_f1_mean']:.4f} "
                f"test_f1={metrics['test_macro_f1']:.4f}"
            )

    # Report the strongest cross-validation result without using the test set
    # for model selection. The course deliverable deploys the archived final
    # INSY 674 ensemble, even when another comparison run has a higher CV score.
    best = max(results, key=lambda r: r["cv_macro_f1_mean"])
    registered = next(
        (result for result in results if result["name"] == FINAL_MODEL_NAME), best
    )
    version = None
    if register:
        version = register_best(registered["run_id"], registered["cv_macro_f1_mean"])
        print(
            f"Registered {registered['name']} as {config.MODEL_NAME} v{version} "
            f"(@{config.MODEL_ALIAS})"
        )
    return {
        "best": best,
        "registered": registered,
        "runs": results,
        "version": version,
    }


if __name__ == "__main__":
    run_experiments()
