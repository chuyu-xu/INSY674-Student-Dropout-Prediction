"""Load and validate the UCI student dropout dataset."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "data.csv"
TARGET = "Target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV (semicolon-separated) and clean column names."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    # Some raw headers contain tabs or trailing spaces.
    df.columns = [c.replace("\t", "").strip() for c in df.columns]
    return df


def validate_data(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise ValueError if the data is not fit for training."""
    missing = [c for c in required_columns + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    unknown = set(df[TARGET].unique()) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown target labels: {sorted(unknown)}")
