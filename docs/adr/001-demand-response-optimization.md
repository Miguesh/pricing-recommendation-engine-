# ADR 001: Use constrained demand-response optimization

## Status

Accepted.

## Context

Historic nightly prices are outcomes of earlier human or automated policies.
Predicting those labels would reproduce historical behavior and cannot prove an
optimal price.

## Decision

Estimate expected occupancy as a function of candidate price and point-in-time
context. Search the feasible candidate grid and select maximum expected
revenue. Use LightGBM quantile models with conformal interval calibration.

## Consequences

The training contract must retain listed price and booking state at the decision
time. The system gains explicit guardrails and uncertainty. It does not
establish causal elasticity from observational data; future iterations need
experimentation or causal evaluation.
