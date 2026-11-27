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
