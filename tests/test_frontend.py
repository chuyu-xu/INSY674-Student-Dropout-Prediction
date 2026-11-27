"""Owner: B. The frontend calls the API and shows results or clear errors."""

import frontend_app
import requests

from dropout.schemas import SCHEMA_INFO


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.ok = status_code < 400

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(self.status_code)


def _form():
    return {name: str(value) for name, value in SCHEMA_INFO["example"].items()}


def test_form_renders_with_example_student(monkeypatch):
    monkeypatch.setattr(
        frontend_app.requests, "get", lambda *a, **k: FakeResponse(200, SCHEMA_INFO)
    )
    html = frontend_app.app.test_client().get("/").get_data(as_text=True)
    assert "Admission grade" in html
    assert "Predict" in html


def test_submit_shows_prediction(monkeypatch):
    result = {
        "prediction": "Graduate",
        "risk_level": "Low",
        "model_version": "3",
        "probabilities": {"Dropout": 0.1, "Enrolled": 0.2, "Graduate": 0.7},
    }
    monkeypatch.setattr(
        frontend_app.requests, "get", lambda *a, **k: FakeResponse(200, SCHEMA_INFO)
    )
    monkeypatch.setattr(
        frontend_app.requests, "post", lambda *a, **k: FakeResponse(200, result)
    )
    html = frontend_app.app.test_client().post("/", data=_form()).get_data(as_text=True)
    assert "Predicted outcome: Graduate" in html
    assert "Low dropout risk" in html


def test_bad_input_shows_message_without_calling_api(monkeypatch):
    monkeypatch.setattr(
        frontend_app.requests, "get", lambda *a, **k: FakeResponse(200, SCHEMA_INFO)
    )

    def fail(*args, **kwargs):
        raise AssertionError("API should not be called for invalid input")

    monkeypatch.setattr(frontend_app.requests, "post", fail)
    form = {**_form(), "Admission grade": "abc"}
    html = frontend_app.app.test_client().post("/", data=form).get_data(as_text=True)
    assert "must be a number" in html


def test_api_down_shows_friendly_error(monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError()

    monkeypatch.setattr(frontend_app.requests, "get", down)
    response = frontend_app.app.test_client().get("/")
    assert response.status_code == 503
    assert "Cannot reach the prediction API" in response.get_data(as_text=True)
