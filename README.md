# Pricing Recommendation Engine

A production-oriented dynamic pricing service for nightly rentals. It estimates
occupancy for feasible candidate prices, then recommends the price that maximizes
expected nightly revenue under explicit business constraints.

## Why this is not a price-regression demo

Historical prices are decisions, not ground truth. Regressing directly to them
would reproduce a previous pricing policy. This service instead learns a
point-in-time demand response and optimizes expected nightly revenue as candidate
price multiplied by expected occupancy.

The decision is bounded by price floors, ceilings, increments, occupancy
guardrails, and maximum change rules.

## Architecture

FastAPI routes requests to an application use case and domain pricing policy.
The use case calls the versioned feature factory, a LightGBM quantile demand
bundle, calibrated intervals, and SHAP explanations. MLflow stores registry
metadata and model artifacts.

The source tree uses Clean Architecture:

- domain: pure pricing rules and immutable business models.
- application: use cases and ports; no FastAPI or LightGBM dependency.
- infrastructure: feature engineering, LightGBM, MLflow, data contracts.
- interfaces: FastAPI and CLI delivery adapters.

## Local quick start

Copy .env.example to .env, then run:

    make install
    make demo-data
    make train
    make run

The demo data exists only to prove the end-to-end contract. It is not evidence
of real-world model quality and is never mixed with a production data source.

For the local MLflow registry, artifact store, and API:

    docker compose --env-file .env up --build

## Quality and release policy

- Training uses chronological partitions, never random validation splits.
- The model artifact includes feature contract, calibration state, and version.
- A candidate must meet metric and calibration gates before promotion.
- The API returns model version, calibrated confidence, local SHAP drivers,
  global importance, and every guardrail applied.
- Retraining is a separate workflow, not an API endpoint.

See docs/architecture.md, docs/model-card.md, and docs/runbook.md for
operational details.
