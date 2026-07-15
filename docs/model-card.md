# Model Card: Pricing Demand and Revenue Optimizer

## Intended use

Recommend a nightly rental price for one property and stay date using
information known at an explicit as-of date. The service is a decision-support
component, not permission to apply fully autonomous price changes without
commercial controls.

## Objective and evaluation

The model estimates expected occupancy conditional on candidate price, then
maximizes expected nightly revenue subject to constraints. Promotion evaluates
occupancy MAE, RMSE, R2, conformal interval coverage and width, expected
revenue MAE, and minimum temporal test size. Random validation splits are
prohibited.

## Inputs

Inputs include historical occupancy, booking pace, seasonal date signals,
holiday/event context, competitor pricing, and property characteristics.
Candidate price is a feature so the optimizer can assess the price-demand
tradeoff. Training and serving share one versioned feature factory.

## Uncertainty and explanation

Prediction intervals use held-out conformal calibration. The confidence score is
a 0–100 operational signal based on interval width and feature-space support;
it is not causal certainty or booking probability.

Local SHAP values describe the selected demand prediction. Global importance
comes from the central LightGBM model. Neither establishes causality.

## Risks and required controls

- Historic prices are observational and are confounded by past pricing policy.
- Competitor, event, and availability data can be stale or incomplete.
- New properties require a cold-start policy and conservative bounds.
- Evaluate performance by market, property type, lead time, and holiday slice.
- Use controlled experiments or causal policy evaluation before asserting
  realized revenue uplift.

Every production decision should retain the request, applied policy, response,
outcome, and model version. Low-confidence recommendations require review.
