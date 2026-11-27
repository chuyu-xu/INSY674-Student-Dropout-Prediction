"""Frontend (owner: B): a form that calls the model API.

Runs in its own container and never loads the model itself; the form fields
come from the API's /schema endpoint.

    API_URL=http://localhost:8000 flask --app frontend_app run --port 8080
"""

from __future__ import annotations

import os

import requests
from flask import Flask, render_template, request

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 10

app = Flask(__name__)


def fetch_schema() -> dict:
    response = requests.get(f"{API_URL}/schema", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def parse_form(form, features: list[str]) -> tuple[dict, list[str]]:
    values, errors = {}, []
    for name in features:
        raw = form.get(name, "").strip()
        if raw == "":
            errors.append(f"{name} is required")
            continue
        try:
            values[name] = float(raw)
        except ValueError:
            errors.append(f"{name} must be a number (got '{raw}')")
    return values, errors


def explain_api_error(response: requests.Response) -> list[str]:
    if response.status_code == 422:
        detail = response.json().get("detail", [])
        if isinstance(detail, list):
            return [str(item.get("msg", item)) for item in detail]
        return [str(detail)]
    if response.status_code == 503:
        return ["The model is not available yet. Please try again in a minute."]
    return [f"The prediction service returned an error ({response.status_code})."]


@app.route("/", methods=["GET", "POST"])
def index():
    try:
        schema = fetch_schema()
    except requests.RequestException:
        return render_template(
            "index.html",
            schema=None,
            values={},
            result=None,
            errors=["Cannot reach the prediction API. Is the api container running?"],
        ), 503

    values = dict(schema["example"])
    result, errors = None, []

    if request.method == "POST":
        values, errors = parse_form(request.form, schema["features"])
        if not errors:
            try:
                response = requests.post(
                    f"{API_URL}/predict", json={"features": values}, timeout=TIMEOUT
                )
                if response.ok:
                    result = response.json()
                else:
                    errors = explain_api_error(response)
            except requests.RequestException:
                errors = ["Cannot reach the prediction API."]

    return render_template(
        "index.html", schema=schema, values=values, result=result, errors=errors
    )


@app.get("/health")
def health():
    return {"status": "ok"}
