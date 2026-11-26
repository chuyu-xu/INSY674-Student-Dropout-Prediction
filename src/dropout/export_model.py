"""Train a model and save it to a local file (owner: F).

Used by the CI integration test, which runs the API container with
MODEL_PATH instead of a live MLflow server.

    PYTHONPATH=src python -m dropout.export_model --out models/model.joblib
"""

import argparse
from pathlib import Path

import joblib

from dropout.data import load_data, validate_data
from dropout.features import FEATURES
from dropout.train import build_model, split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/model.joblib")
    args = parser.parse_args()

    df = load_data()
    validate_data(df, FEATURES)
    X_train, _, y_train, _ = split(df)
    model = build_model().fit(X_train, y_train)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
