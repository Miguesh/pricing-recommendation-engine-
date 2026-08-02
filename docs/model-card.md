# Model Card: Experimental Pricing Demand and Revenue Optimizer

This card applies only to `PERFORMANCE_AWARE_EXPERIMENTAL`. The default
`MARKET_EVIDENCE_STATISTICAL_V1` profile is a deterministic descriptive
algorithm, not a trained model; its contract, evidence policy, and limitations
are documented separately in `docs/contracts/PRICING_CONTRACT_V1.md` and
`docs/limitations/MARKET_EVIDENCE_LIMITATIONS.md`.

## Model details

| Item | Value |
| --- | --- |
| Model family | LightGBM lower/upper quantile tails (100 trees each) plus monotonic central L2 regression (400 trees) |
| Decision objective | Maximize expected nightly revenue under explicit commercial constraints |
| Scope | Exactly one normalized tenant and one currency per bundle and registered-model namespace |
| Feature contract | Versioned ordered semantic manifest and SHA-256 hash serialized with the bundle |
| Uncertainty | Purged chronological, stay-group conformal calibration |
| Explanations | Required local SHAP values and normalized global LightGBM importance; serving fails closed if warm-up fails |
| Lifecycle | MLflow candidate registration, same-current-holdout champion comparison, explicit promotion and rollback |

## Intended use

Recommend a nightly rental price for one property and stay date using only
information known at an explicit as-of date. The service is a decision-support
component for a larger revenue-management workflow. It does not authorize
unbounded or fully autonomous price changes.

Intended users are pricing analysts, revenue managers, and trusted applications
that enforce property- and tenant-specific commercial policy.

The API hard-limits `max_price_change_pct` to 0.35. Absolute floors/ceilings,
increment alignment, a candidate-count budget, and an optional occupancy floor
further constrain the feasible action set. The offline policy gate evaluates
the entire canonical ±35% grid in five-percentage-point steps.

## Out-of-scope use

- long-term leases, sale prices, or non-lodging inventory;
- emergency, discriminatory, retaliatory, or individually targeted pricing;
- autonomous price publication without floors, ceilings, monitoring, and
  rollback;
- claims of causal elasticity or realized revenue uplift from observational
  backtests alone;
- mixing tenants or currencies in one model, or reusing a registered-model
  namespace bound to another tenant/currency;
- inference when critical point-in-time inputs cannot be reconstructed.

## Inputs and target

Inputs include candidate price, lead time, historical occupancy, booking pace,
competitor median, property capacity and quality, holiday/event context,
seasonal date signals, and optional coordinates. Training and serving share one
versioned feature factory.

The supervised target is realized occupancy in `[0, 1]`. Historical listed
price is an input used to learn observational demand response; it is not treated
as an optimal-price label.

## Training and evaluation

Data is partitioned chronologically by unique as-of dates into training,
calibration, and test windows. `outcome_available_date` must fall strictly
before the next window boundary, less any configured embargo, or the row is
purged. A `(tenant_id, property_id, stay_date)` group cannot cross partitions.
Random validation splits are prohibited. The central model applies negative
monotonic constraints to candidate price and the price-to-competitor ratio.
Quantile tails are calibrated on the held-out middle window and evaluated once
on the final test window.

Repeated snapshots do not receive extra fitting or aggregate-metric influence.
Each training snapshot is weighted by the inverse of its stay's snapshot count,
and the lower, central, and upper boosters all fit with those weights. Evaluation
likewise gives each stay group equal total weight. Calibration uses the maximum
nonconformity score per stay, and reported interval coverage is simultaneous
group coverage—a stay passes only when every evaluated snapshot is covered.

Promotion evidence includes:

- occupancy MAE, RMSE, and R²;
- simultaneous stay-group interval coverage and mean interval width;
- expected-revenue MAE and WAPE at historical observed prices;
- minimum sample size;
- operational slice quality;
- price-response, flat-response, policy-boundary, and mean selected-change
  behavior over the full canonical policy grid, both aggregate and by slice;
- required lead-time, `holiday/yes`, `holiday/no`, event, and location slices
  with minimum row/independent-stay evidence and predictive/economic gates; and
- challenger-versus-champion non-regression after both compatible models are
  evaluated on the candidate's same immutable current holdout.

The deterministic synthetic baseline uses 12 properties, 180 decision days,
seed 42, 400 central trees, and 100 trees per quantile tail:

| Evidence | Result |
| --- | ---: |
| Rows | 2,160 |
| Train / calibration / test after purge | 1,416 / 218 / 253 |
| Purged train / calibration | 130 / 143 |
| Occupancy MAE / RMSE / R² | 0.0214351 / 0.0299894 / 0.9308273 |
| Simultaneous group coverage / mean width | 0.9169960 / 0.2014993 |
| Revenue-at-historical-price WAPE | 0.0284972 |
| Mean response / flat rate | 0.1944230 / 0.0039526 |
| Upper / lower / any boundary rate | 0.0553360 / 0.0000000 / 0.0553360 |
| Mean selected price change | -0.0063241 |
| Holiday/no: rows, coverage, MAE, WAPE | 168 / 0.93452 / 0.01880 / 0.02457 |
| Holiday/yes: rows, coverage, MAE, WAPE | 85 / 0.88235 / 0.02664 / 0.03537 |

These are synthetic integration metrics only. The generator is coherent with
the modeled relationships, so the result shows that contracts, training,
calibration, policy gates, registry, and serving execute together. It does not
prove causal identification, real-market generalization, or revenue uplift.
Revenue-at-historical-price WAPE is a predictive demand/revenue diagnostic, not
causal policy uplift or off-policy evaluation (OPE).

## Uncertainty and explanation

Prediction intervals use held-out group-conformal calibration. Independently
fitted tails can cross, so inference repairs interval ordering around the
central estimate. The confidence score is a 0–100 operational signal based on
interval width and feature-space support; it is neither causal certainty nor
booking probability.

Local SHAP values describe the central demand prediction at the selected price.
Global importance comes from the central LightGBM model. Neither establishes
causality, and importance must not be interpreted as business impact.

## Known limitations and risks

- Historic prices are observational and confounded by earlier pricing policy,
  inventory, operator judgment, and unrecorded demand signals.
- Monotonic constraints are economic guardrails, not causal identification.
- Coordinates alone do not capture market identity, neighborhood boundaries,
  regulation, local events, or property type.
- Competitor, event, and availability feeds can be stale, sparse, or incorrect.
- A new property needs a conservative cold-start policy.
- Distribution support is a heuristic and must be calibrated against delayed
  outcomes before it drives automation.
- Aggregate metrics can hide regressions by market, property type, lead time,
  price band, holiday status, or protected-proxy geography.
- The repository has no real-data evaluation and no multi-year rolling
  backtest across seasons, structural breaks, or market regimes. A single final
  holdout and a synthetic generator cannot replace that evidence.
- Optimizing expected nightly revenue does not directly optimize cancellation,
  length of stay, downstream fees, customer lifetime value, or marketplace
  fairness.

## Required production controls

1. Validate data contracts and freshness before every training and inference
   window.
2. Retain request, policy, recommendation, override, outcome, and model version
   for every production decision.
3. Define low-confidence fallback and human-review policies.
4. Monitor drift, calibration, delayed error, recommendation overrides, and
   business guardrails by operational slice.
5. Require named approval and preserve a tested rollback target.
6. Serialize promotion and rollback through one writer per registered-model
   namespace; the alias transition has compensating repair but no distributed
   lock or compare-and-swap.
7. Bind authenticated identity to the deployment's single tenant scope; never
   trust a caller-provided tenant ID by itself.
8. Serve production only from `models:/<name>@champion`; require the model
   artifact SHA-256, run/version integrity evidence, registry binding, feature
   version, and semantic hash to agree. Verify the PyFunc checksum sidecar before
   deserializing joblib.
9. Reject promotion unless its MLflow run is `FINISHED`, its champion benchmark
   is still current, and required aggregate/slice gates pass.
10. Keep the request-body read deadline separate from the soft inference
    deadline; monitor 408, 503, and 504 responses independently.
11. Use controlled experiments or defensible causal/off-policy methods before
   claiming or scaling revenue uplift.

## Approval record

This model card documents the software and evaluation contract, not a specific
production model approval. Each registered model version must carry its own
dataset fingerprint, metrics, feature version, approver, and deployment record
in MLflow or the organization's governed model registry.

An approval of an experimental model version does not validate, calibrate, or
authorize the statistical profile, and a statistical evidence-quality class is
not model confidence or model approval.
