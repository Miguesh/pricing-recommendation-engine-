# Architecture

## Primary decision

This service is a constrained demand-and-revenue optimizer, not a regression
that imitates historic prices. Historic prices are previous policy decisions,
not an optimal label. The target data grain is a pricing snapshot per tenant,
property, stay date, and as-of date.

    expected_revenue = candidate_price × expected_occupancy

The as-of date is mandatory: every feature must have been observable then. This
is the central defense against future-information leakage.

## Component view

~~~mermaid
flowchart LR
    Client["Consumer application"] --> API["FastAPI v1 adapter"]
    API --> UseCase["RecommendPrice use case"]
    UseCase --> Policy["Pricing policy"]
    UseCase --> Features["Versioned feature factory"]
    Features --> Model["LightGBM quantile demand bundle"]
    Model --> Explain["SHAP explanations"]
    Model --> Optimize["Revenue selection"]
    Train["Scheduled retraining"] --> Validate["Data contract and temporal split"]
    Validate --> Model
    Train --> Registry["MLflow registry"]
    Registry --> API
    Monitor["Drift monitor"] --> Train
~~~

## Clean Architecture

| Layer | Responsibility | Must not depend on |
| --- | --- | --- |
| domain | Value objects, guardrails, selection policy | FastAPI, Pandas, MLflow, LightGBM |
| application | Recommendation, training, retraining use cases and ports | HTTP delivery and storage details |
| infrastructure | Features, LightGBM, Pandera, MLflow, drift math | Route handlers |
| interfaces | FastAPI, CLI, health endpoints | Model implementation details |

Training is not exposed through the API. Serving starts only from a local
approved bundle or a configured MLflow model URI.

## Model and decision flow

Three LightGBM models estimate lower, central, and upper occupancy. A held-out
chronological calibration period calculates a split-conformal residual, making
the interval an empirical coverage estimate rather than an uncalibrated tree
spread.

For each request, the engine constructs a candidate grid from commercial bounds
and increments, builds the same point-in-time features used in training, scores
demand, removes candidates below any occupancy floor, and selects the maximum
expected revenue. Exact ties favor the lower price.

Confidence combines interval width with a feature-space support score. It is
not a probability of booking. SHAP explains central-demand prediction for the
selected price; global LightGBM importance supplies portfolio-level context.

## Data contract

Required data includes tenant and property IDs, stay and as-of dates, listed
price, competitor median, historical occupancy, booking pace, bedrooms,
capacity, review score, holiday, event intensity, and realized occupancy.

The contract rejects malformed values, duplicate decision snapshots, and rows
where as-of date is later than stay date. Final realized state must never be
used as an inference feature.

## Model lifecycle

~~~mermaid
stateDiagram-v2
    [*] --> ValidatedData
    ValidatedData --> Candidate: chronological train, calibrate, test
    Candidate --> Rejected: promotion gate fails
    Candidate --> Registered: gates pass
    Registered --> Champion: explicit promote command
    Champion --> RolledBack: restore prior champion alias
    RolledBack --> Champion
~~~

MLflow candidate registration retains parameters, metrics, feature version,
dataset fingerprint, and the complete bundle. It never changes the champion
alias. Promotion records a named human approver.

## Security and operations

- API-key comparison is constant-time and optional only for local development.
- Production authentication must bind tenant identity to the principal instead
  of trusting a caller-provided tenant ID.
- Secrets are environment variables and are never logged or committed.
- Artifact storage is a trusted deployment boundary: load only approved,
  access-controlled MLflow artifacts.
- Liveness is independent of a model; readiness fails until a model is loaded.
- Structured logs and Prometheus metrics are available for external monitoring.
