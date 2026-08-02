# ADR: Add an artifact-free market-evidence statistical profile

## Status

Accepted on 2026-08-01.

## Context

The existing engine implements a performance-aware demand and revenue path. It
uses occupancy, booking pace, trained LightGBM bundles, SHAP, and an MLflow
lifecycle. Those capabilities remain useful, but they do not match a consumer
that currently has authorized, normalized, analyst-selected price observations
and no validated operational outcome history.

PLUSBNB owns observation capture, rights and provenance checks, normalization,
comparable selection, similarity scoring, customer/property records, audits,
human review, and report generation. This repository must not duplicate those
responsibilities or fetch evidence on the consumer's behalf.

The integration also needs a stable, versioned contract before the engine as a
whole reaches version 1.0. A stable contract must not imply that the resulting
prices have been commercially validated.

## Decision

Add two explicit engine profiles:

- `MARKET_EVIDENCE_STATISTICAL_V1` is the default profile and is stable for
  contract integration. It is deterministic, uses `Decimal`, performs no I/O,
  loads no artifacts, and has no dependency on occupancy, booking pace,
  revenue outcomes, MLflow, or external providers.
- `PERFORMANCE_AWARE_EXPERIMENTAL` names and preserves the existing
  model-backed behavior. It remains experimental, artifact-dependent, and is
  not intended for the initial PLUSBNB integration.

The stable profile receives comparables already authorized, normalized,
filtered, scored, labeled, and versioned by the consumer. It validates
compatibility, applies documented eligibility and weighted-outlier policies,
calculates weighted empirical P25/P50/P75, reports evidence components, and
either recommends the strategy-specific band point or abstains. A request is one
point-in-time pricing decision, not an analytical batch, and is bounded to 50
comparables with one matching lineage entry per comparable.

Both contracts share `POST /v1/pricing/recommendations` through an explicit
request union. `GET /v1/capabilities` exposes profile metadata so consumers do
not infer requirements from the legacy request. The existing performance-aware
request and response remain accepted, but this does not promise silent full
compatibility across health or profile semantics. The CLI adds `validate`,
`recommend`, `capabilities`, and `export-openapi` operations for the statistical
profile. HTTP and CLI share a 1,048,576-byte statistical request maximum. The
API can be configured lower; capabilities distinguishes that effective limit
from the contractual maximum.

The stable `organization_id` boundary is 128 characters. API-key tenant
binding, serving-tenant routing, and trusted-proxy identity values use that same
explicit maximum, while the retained experimental request keeps its existing
100-character tenant/property identifiers. Stable capabilities enumerates all
required nested contract paths, with a recursive validation-schema test to
detect metadata drift.

Evidence quality compares raw Decimal ESS, average similarity, and dispersion.
Only the response components are quantized. When the initial eligible set is
empty, any stale exclusion takes top-level precedence over low similarity while
the per-comparable map retains all reasons.

The no-query readiness check now means that the stable statistical profile can
execute. Model availability is reported separately in health fields. An
explicit
`/health/ready?profile=PERFORMANCE_AWARE_EXPERIMENTAL` check returns 503 until
that profile's artifact is loaded. Existing monitors that interpreted the
default 503 as a missing-model signal must migrate to this explicit query.

Contract `1.0`, algorithm `weighted-percentile-v1.0.0`, evidence policy
`evidence-quality-v1.0.0`, outlier policy `weighted-tukey-v1.0.0`, and algorithm
configuration `market-evidence-statistical-v1.0.0` are independent version
boundaries. The software remains pre-1.0 (`0.1.0`).

## Parameters fixed by the v1 configuration

| Parameter | Value |
| --- | ---: |
| Minimum comparables after exclusions | 3 |
| Minimum effective sample size | 2.50 |
| Minimum individual similarity | 40 |
| Minimum average similarity | 55 |
| Maximum weighted dispersion `(P75 - P25) / P50` | 1.00 |
| Maximum evidence age | 90 days |
| Weighted Tukey IQR multiplier | 1.50 |
| Maximum comparables / lineage entries per request | 50 / 50 |
| Maximum compact HTTP / CLI request | 1,048,576 bytes |
| Decimal context precision | 28 digits |
| Money rounding / output quantum | `ROUND_HALF_UP` / `0.01` |
| ESS / average similarity / dispersion presentation quantum | `0.0001` / `0.01` / `0.0001` |
| HIGH quality minimum count / ESS / average similarity | 8 / 6 / 80 |
| HIGH quality maximum dispersion | 0.35 |
| MODERATE quality minimum count / ESS / average similarity | 5 / 4 / 65 |
| MODERATE quality maximum dispersion | 0.60 |

These are transparent engineering policy values, not empirically calibrated
market thresholds. Changing behavior requires a new configuration or policy
version rather than an untracked constant change. Quality thresholds consume
raw metrics; the presentation quantums above do not alter policy decisions.

## Alternatives considered

### Extend the trained demand model

Rejected for this use case. Fabricating occupancy or booking-pace inputs would
blur the evidence boundary and create unsupported accuracy claims. It would
also require an artifact before suitable training evidence exists.

### Implement the calculation inside PLUSBNB

Rejected as the integration target. It would duplicate engine behavior in the
consumer and make independent contract evolution and testing harder.

### Create a new service or repository

Rejected for now. The existing application already supplies API, CLI, request
limits, telemetry, packaging, and container controls. A separate deployment is
not justified by current scale or isolation requirements.

### Use an unweighted median or average

Rejected. A versioned weighted empirical distribution preserves the
consumer-provided similarity signal and exposes the band rather than collapsing
all evidence to one opaque number.

## Consequences

Positive consequences:

- PLUSBNB can integrate without trained models or live external services.
- Results and run identifiers are reproducible for canonical equivalent input.
- Evidence weakness is visible through components and typed abstention.
- Existing model-backed capabilities are not removed or silently redefined.

Costs and risks:

- Output quality depends on consumer selection, normalization, and similarity
  scoring, which this engine cannot verify.
- Lineage must exactly cover each comparable tuple of lineage hash, source
  family, source version, and observation hash. Those hash-shaped references
  are still not proof of source rights, authenticity, or correct content.
- Weighted empirical quantiles are discrete and can move when evidence near a
  cumulative-weight boundary changes.
- Current thresholds and evidence classes lack real-market calibration.
- Sharing one endpoint requires consumers to send the profile-specific schema
  explicitly and handle distinct response shapes.

## Compatibility and reversibility

The legacy recommendation contract remains accepted, and the performance-aware
path still requires its governed artifact. Default readiness semantics are not
backward-identical; explicit profile selection provides the migration path. No
automatic fallback crosses profile boundaries. The statistical profile can
later be deployed independently because its domain/application service is pure
and does not depend on the ML infrastructure. Any semantic change to the stable
profile requires a new contract, algorithm, or policy version; silently changing
v1 is not permitted.

## Evidence and non-claims

The repository fixture is synthetic and tests software behavior only. This
decision does not claim price accuracy, forecast quality, revenue uplift,
optimality, causal effects, or permission to publish a price.
