from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from dropout.data import CLASSES, load_data
from dropout.train import build_model, split


def test_model_beats_majority_baseline():
    X_train, X_test, y_train, y_test = split(load_data())

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    model = build_model().fit(X_train, y_train)

    baseline_f1 = f1_score(y_test, baseline.predict(X_test), average="macro")
    model_f1 = f1_score(y_test, model.predict(X_test), average="macro")
    assert model_f1 > baseline_f1 + 0.2


def test_predictions_are_valid_probabilities():
    X_train, X_test, y_train, _ = split(load_data())
    model = build_model().fit(X_train, y_train)

    assert set(model.predict(X_test)) <= set(CLASSES)
    proba = model.predict_proba(X_test)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert abs(proba.sum(axis=1) - 1).max() < 1e-6
