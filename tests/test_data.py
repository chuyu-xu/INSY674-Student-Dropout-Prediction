import pytest

from dropout.data import CLASSES, TARGET, load_data, validate_data
from dropout.features import FEATURES


def test_required_columns_present_and_target_complete():
    df = load_data()
    validate_data(df, FEATURES)  # raises if anything is wrong
    assert set(df[TARGET].unique()) == set(CLASSES)


def test_validation_rejects_missing_target():
    df = load_data().head(10).copy()
    df.loc[0, TARGET] = None
    with pytest.raises(ValueError, match="missing"):
        validate_data(df, FEATURES)


def test_validation_rejects_missing_column():
    df = load_data().drop(columns=["Admission grade"])
    with pytest.raises(ValueError, match="Admission grade"):
        validate_data(df, FEATURES)
