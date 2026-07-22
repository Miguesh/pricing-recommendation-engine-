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
revenue. Use 100-tree LightGBM quantile tails, a 400-tree monotonic central
demand model, and stay-group conformal interval calibration. Apply absolute
bounds, increment alignment, an optional occupancy floor, and a maximum price
change that the public contract hard-caps at ±35%. Offline policy evaluation
exercises the full canonical `0.65, 0.70, ..., 1.35` historical-price multiplier
grid.

Training labels must declare `outcome_available_date`. Chronological train and
calibration windows purge labels unavailable before the next cutoff, plus an
optional embargo, and no `(tenant_id, property_id, stay_date)` group may cross a
partition. Repeated snapshots receive equal total weight per stay; conformal
calibration uses the group's worst residual and evaluation reports simultaneous
stay-level coverage.

Fitting follows the same independence principle: every training snapshot has
weight `1 / snapshots_in_stay`, and all three boosters consume those weights.
Promotion gates cover predictive and economic policy-grid metrics both
aggregate and by required slices, including `holiday/yes` and `holiday/no`.

The feature contract includes a semantic manifest hash, not only a version and
column order. Each bundle and registered-model namespace is bound to exactly one
tenant and one currency. A compatible champion is reevaluated on the
challenger's same immutable current holdout. Retraining may register a gated
candidate but only an explicit, audited `promote` or `rollback` decision may
change the `champion` alias; production serving accepts only
`models:/<name>@champion`.

The registered artifact carries a SHA-256 bundle checksum in matching run and
model-version evidence, plus a PyFunc checksum sidecar that is checked before
joblib deserialization. Promotion requires a `FINISHED` run and rejects stale
benchmark evidence if the champion alias changed after the candidate was
evaluated.

## Consequences

The training contract must retain listed price and booking state at the decision
time, label-availability time, and stay identity. The system gains explicit
guardrails, calibrated uncertainty, feature-contract compatibility checks, and
governed model lineage. It also requires tenant/currency-specific training and
deployment, mature labels, enough independent stays, and serialized
single-writer promotion because alias mutation is not protected by a
distributed lock.

Serving also distinguishes request receipt from model execution: an independent
body-read deadline returns 408, while the inference deadline remains soft and
does not cancel native work already in progress.

Revenue WAPE at historical prices remains a predictive diagnostic. It is not
causal policy uplift or off-policy evaluation. The coherent synthetic demo
validates integration only; real data, multi-year rolling backtests, and
experimentation or defensible causal/OPE evidence remain necessary.

The central price-response curve is constrained to be non-increasing in
candidate price and price-to-competitor ratio. This is a commercial guardrail,
not proof that observational data identifies causal price elasticity.
