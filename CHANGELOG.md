# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Stable contract `1.0` for the deterministic,
  artifact-free `MARKET_EVIDENCE_STATISTICAL_V1` profile.
- Weighted percentile pricing band, versioned eligibility/outlier policies,
  evidence components/classes, typed abstention, deterministic request/run
  lineage, exact evidence-lineage tuple coverage, and safe decimal
  serialization.
- Discoverable capabilities for the stable statistical profile and the
  retained `PERFORMANCE_AWARE_EXPERIMENTAL` profile.
- Statistical support on `POST /v1/pricing/recommendations` without removing
  the legacy performance-aware request.
- Local `validate`, `recommend`, `capabilities`, and `export-openapi` CLI
  commands.
- Synthetic PLUSBNB consumer-contract fixture and statistical/API/CLI contract
  coverage.
- Architecture decision, contract, algorithm, integration, threat-model, and
  limitation documentation.

### Changed

- Default readiness now represents availability of the stable statistical
  profile. An explicit experimental-profile query preserves missing-model 503
  behavior for migrated monitors, and health output reports the checked profile
  plus trained-model availability.
- Documentation distinguishes evidence quality from calibrated model
  confidence and explicitly rejects accuracy, forecast, uplift, revenue, and
  publication interpretations for the statistical profile.

### Fixed

- Evidence quality now compares raw Decimal ESS, average similarity, and
  dispersion; quantized evidence components are presentation-only and cannot
  promote a boundary value into a higher class.
- Empty recency/similarity eligibility now gives `STALE_EVIDENCE` precedence
  when any observation was excluded by recency, while retaining every
  per-comparable stale and low-similarity reason.
- The single-decision contract now caps both comparables and lineage at 50 and
  shares one 1,048,576-byte maximum across HTTP and CLI. Capabilities expose the
  contractual limit and any lower effective API limit; the reproducible OpenAPI
  schema reflects the same boundaries.
- The lock now resolves GitPython 3.1.57 and the development test runner pytest
  9.1.1, clearing the dependency-audit advisories found during final validation.

### Compatibility

- The existing model-backed contract and MLflow lifecycle remain available
  under `PERFORMANCE_AWARE_EXPERIMENTAL`.
- The software version remains pre-1.0 (`0.1.0`); only the new integration
  contract is versioned as v1.
- These boundary corrections precede the contract's first merge, so contract,
  algorithm, configuration, evidence-policy, and outlier-policy versions remain
  unchanged.

## 0.1.0 - 2026-07-14

### Added

- Initial performance-aware pricing engine with synthetic training/evaluation,
  FastAPI/CLI adapters, governed model lifecycle, tests, and container tooling.
