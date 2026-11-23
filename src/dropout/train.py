"""Minimal training script. Later steps: wrap in a Prefect flow, log to
MLflow, and swap in the tuned soft-voting ensemble from INSY 674."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from dropout.data import TARGET, load_data, validate_data
from dropout.features import FEATURES, build_preprocessor

SEED = 42


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "model",
                LogisticRegression(C=0.0448, class_weight="balanced", max_iter=2000),
            ),
        ]
    )


def split(df):
    return train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.2,
        stratify=df[TARGET],
        random_state=SEED,
    )


def main() -> None:
    df = load_data()
    validate_data(df, FEATURES)
    X_train, X_test, y_train, y_test = split(df)
    model = build_model().fit(X_train, y_train)
    score = f1_score(y_test, model.predict(X_test), average="macro")
    print(f"Test macro F1: {score:.4f}")


if __name__ == "__main__":
    main()
