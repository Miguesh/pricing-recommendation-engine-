# Operations Runbook

## Local workflow

Create a virtual environment, install, generate deterministic demo data, and
train a portable local model:

    python -m venv .venv
    .venv/Scripts/python -m pip install -e ".[dev]"
    pricing-engine generate-demo-data --output data/demo/observations.parquet
    pricing-engine train --input data/demo/observations.parquet

Set PRICING_MODEL_URI to artifacts/local-model and start FastAPI with:

    uvicorn pricing_engine.interfaces.api.app:create_app --factory --port 8000

Demo data proves the software path only. It is not evidence of real-market
quality.

## Platform

Copy .env.example to .env and start the local platform:

    docker compose --env-file .env up --build

Compose provides PostgreSQL, MinIO, MLflow, and the API. It is for local
integration; use managed databases, object storage, and secret management in
production.

## Model lifecycle

1. Validate a point-in-time extract.
2. Create a candidate:

       pricing-engine retrain --input observations.parquet

3. Review metrics, calibration, slice behavior, and drift.
4. Promote only with named approval:

       pricing-engine promote --version 7 --approved-by "Pricing Owner"

Retraining never promotes automatically. Promotion records the approver.

## Monitoring and rollback

Run a feature-drift comparison:

    pricing-engine drift --reference baseline.parquet --current recent.parquet

PSI at or above 0.20 needs retraining review, not automatic deployment.

GET /health/live tests process liveness. GET /health/ready returns 503 until an
approved model is loaded. GET /metrics exposes Prometheus metrics.

For a bad model or policy, point the MLflow champion alias to the prior verified
version. Do not edit artifacts in place; retain provenance and open an incident.
