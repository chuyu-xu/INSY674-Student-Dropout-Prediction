-- Tables for logged predictions and monitoring metrics (owner: F).
-- Keep in sync with SCHEMA_SQL in src/dropout/db.py.
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    features JSONB NOT NULL,
    prediction TEXT NOT NULL,
    proba_dropout DOUBLE PRECISION,
    proba_enrolled DOUBLE PRECISION,
    proba_graduate DOUBLE PRECISION,
    ground_truth TEXT,
    model_version TEXT,
    latency_ms DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS drift_metrics (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_size INTEGER NOT NULL,
    share_drifted_features DOUBLE PRECISION NOT NULL,
    n_drifted_features INTEGER NOT NULL,
    drift_detected BOOLEAN NOT NULL,
    accuracy DOUBLE PRECISION,
    macro_f1 DOUBLE PRECISION,
    retrain_triggered BOOLEAN NOT NULL DEFAULT FALSE
);
