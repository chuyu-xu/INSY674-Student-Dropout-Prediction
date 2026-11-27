"""Traffic simulator (owner: C).

Sends unseen test-split students to the API, with their true label so the
monitor can compute model quality. ``--drift`` shifts the inputs to mimic
a weaker cohort.

    PYTHONPATH=src python -m dropout.simulate --api-url http://localhost:8000
    PYTHONPATH=src python -m dropout.simulate --drift --drift-strength 1.0
"""

from __future__ import annotations

import argparse
import os
import time

import requests

from dropout.data import load_data
from dropout.drift import inject_drift
from dropout.features import FEATURES
from dropout.train import split


def wait_for_api(api_url: str, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{api_url}/health", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise SystemExit(f"API at {api_url} did not become ready")


def build_requests(drift: bool, strength: float, n: int, seed: int) -> list[dict]:
    _, X_test, _, y_test = split(load_data())
    sample_idx = X_test.sample(n=n, replace=True, random_state=seed).index
    X = X_test.loc[sample_idx, FEATURES]
    y = y_test.loc[sample_idx]
    if drift:
        X = inject_drift(X, strength)
    return [
        {"features": row, "ground_truth": label}
        for row, label in zip(X.to_dict(orient="records"), y, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url", default=os.getenv("API_URL", "http://localhost:8000")
    )
    parser.add_argument("--rate", type=float, default=float(os.getenv("SIM_RATE", "5")))
    parser.add_argument(
        "--duration", type=int, default=int(os.getenv("SIM_DURATION", "600"))
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        default=os.getenv("DRIFT", "false").lower() == "true",
    )
    parser.add_argument(
        "--drift-strength",
        type=float,
        default=float(os.getenv("DRIFT_STRENGTH", "1.0")),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wait_for_api(args.api_url)
    n = max(1, int(args.rate * args.duration))
    payloads = build_requests(args.drift, args.drift_strength, n, args.seed)
    print(f"Sending {n} requests at {args.rate}/s (drift={args.drift})")

    ok = errors = 0
    for payload in payloads:
        try:
            response = requests.post(
                f"{args.api_url}/predict", json=payload, timeout=10
            )
            ok += response.ok
            errors += not response.ok
        except requests.RequestException:
            errors += 1
        time.sleep(1 / args.rate)
    print(f"Done: {ok} succeeded, {errors} failed")


if __name__ == "__main__":
    main()
