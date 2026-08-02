# Pricing Contract v1

## Contract identity and scope

This document defines JSON contract `1.0` for
`MARKET_EVIDENCE_STATISTICAL_V1`. The profile converts a bounded set of
consumer-selected comparable prices into weighted evidence percentiles. It does
not select evidence, access providers, convert currencies, forecast occupancy,
optimize revenue, or publish prices. One document represents one point-in-time
pricing decision; contract v1 does not accept an analytical batch.

The contract is stable for integration while the software remains version
`0.1.0`. Stability means that semantic behavior is tied to explicit contract,
algorithm, configuration, evidence-policy, and outlier-policy versions. It does
not mean the product or recommendation quality is commercially validated.

The normative machine-readable schema is the reproducible `docs/openapi.json`.
This document explains semantics that JSON Schema alone cannot express.

## Profile and version constants

| Boundary | Supported value |
| --- | --- |
| `contract_version` | `1.0` |
| `engine_profile` | `MARKET_EVIDENCE_STATISTICAL_V1` |
| `algorithm_version` (output) | `weighted-percentile-v1.0.0` |
| `algorithm_config_version` | `market-evidence-statistical-v1.0.0` |
| `evidence_policy_version` | `evidence-quality-v1.0.0` |
| `outlier_policy_version` | `weighted-tukey-v1.0.0` |
| Engine/software version | `0.1.0` |

Unsupported policy/configuration versions are contract errors. A request that
uses the statistical shape but names another profile returns a typed
`UNSUPPORTED_PROFILE` abstention; consumers must use the legacy request shape
for `PERFORMANCE_AWARE_EXPERIMENTAL`.

## Input envelope

All envelope fields below are required. Profile, policy versions, and both
fixed-false safety flags must be explicit so stored audits remain
self-describing; the parser does not infer them for contract v1.

| Field | Type and constraints | Meaning |
| --- | --- | --- |
| `contract_version` | literal `1.0` | Input schema version |
| `request_id` | string, 1-128 | Consumer request identifier; opaque, not required to be a UUID |
| `correlation_id` | string, 1-128 | Cross-system trace identifier |
| `engine_profile` | profile enum | Must name the intended profile explicitly |
| `organization_id` | string, 1-128 | Opaque authorization/audit scope; no customer name or PII |
| `audit_id` | string, 1-128 | Consumer-owned audit reference |
| `target_property_id` | string, 1-128 | Opaque property reference; no address |
| `market_id` | string, 1-128 | Consumer market configuration key |
| `market_config_version` | string, 1-64 | Version of consumer normalization/selection configuration |
| `currency` | three uppercase letters | Single request currency; no conversion |
| `timezone` | IANA name, 1-64 | Exact timezone required on every comparable |
| `stay_start`, `stay_end` | ISO dates | Target stay interval; end is exclusive for night-count validation |
| `number_of_nights` | integer, 1-366 | Must equal `stay_end - stay_start` |
| `guests` | integer, 1-100 | Must exactly match every comparable in v1 |
| `observed_at` | offset-aware timestamp | Consumer snapshot time; cannot exceed `as_of` |
| `as_of` | offset-aware timestamp | Point-in-time cutoff for all evidence |
| `pricing_strategy` | `conservative`, `balanced`, or `premium` | Selects P25, P50, or P75 after gates pass |
| `evidence_policy_version` | string, 1-64 | Evidence class/gate policy |
| `outlier_policy_version` | string, 1-64 | Outlier inclusion policy |
| `algorithm_config_version` | string, 1-64 | Behavioral parameter set |
| `target_property_features` | object | Bounded descriptive target context |
| `comparables` | array, 1-50 | Already authorized and selected evidence for this decision |
| `input_lineage` | array, 1-50 | One governed lineage entry per comparable |
| `publication_allowed` | required literal `false` | Fixed safety flag |
| `commercial_validation` | required literal `false` | Fixed non-claim flag |

Unknown fields are rejected (`extra="forbid"`). Strings are trimmed. Contract
models are immutable after validation.

`GET /v1/capabilities` publishes a complete recursive inventory of these
required fields. Object children use `parent.child`, array-item children use
`parent[].child`, and the two deliberate safety literals are published as
`publication_allowed=false` and `commercial_validation=false`. Contract tests
derive the required and optional path sets from the validation JSON Schema and
fail if this metadata drifts.

## Target property features

| Field | Type and constraints |
| --- | --- |
| `property_type` | string, 1-64 |
| `bedrooms` | optional decimal, 0-50 |
| `bathrooms` | optional decimal, 0-50 |
| `accommodates` | integer, 1-100 |
| `square_meters` | optional decimal, greater than 0 and at most 10,000 |
| `amenity_codes` | optional array, at most 100 unique normalized lowercase strings, each 1-64 |

These features are retained in the request hash and contract context. Algorithm
v1 does not derive weights or adjust prices from them; PLUSBNB incorporates them
when it selects and scores comparables.

## Comparable object

| Field | Type and constraints | Contract semantics |
| --- | --- | --- |
| `comparable_id` | string, 1-128 | Unique within the request |
| `evidence_type` | string, 1-128 | Consumer taxonomy value |
| `verification_level` | string, 1-128 | Consumer verification taxonomy |
| `collection_method` | string, 1-128 | Consumer collection taxonomy |
| `observed_at` | offset-aware timestamp | When the price evidence was observed; cannot exceed `as_of` |
| `known_at` | offset-aware timestamp | When the consumer knew it; must be >= `observed_at` and <= `as_of` |
| `stay_start`, `stay_end` | ISO dates | Must exactly match the requested stay |
| `number_of_nights` | integer, 1-366 | Must match dates and request |
| `guests` | integer, 1-100 | Must exactly match request in v1 |
| `currency` | three uppercase letters | Must equal request currency |
| `timezone` | IANA name, 1-64 | Must exactly equal request timezone |
| `effective_base_nightly_rate` | optional positive decimal string | Base-only audit component; not used in v1 band |
| `effective_total_nightly_rate` | positive decimal string | Normalized total nightly amount used by the algorithm |
| `similarity_score` | decimal string or integer, 0-100 | Raw statistical weight and eligibility input |
| `similarity_factors` | object, 1-50 entries | Consumer factor scores in `[0, 100]`; not recomputed by the engine |
| `availability_observed` | required boolean or null | Audit context; null means not observed and does not infer occupancy |
| `quality_flags` | required unique array, at most 50 strings | Case-insensitive `NOT_USABLE` makes the evidence contractually unusable |
| `eligible_for_pricing` | optional boolean, default true | False makes the evidence contractually unusable |
| `source_family_id` | string, 1-128 | Opaque consumer source family; never a private URL |
| `source_version` | string, 1-64 | Consumer source/method version |
| `observation_hash` | 64 lowercase hexadecimal characters | Consumer observation digest reference |
| `lineage_hash` | 64 lowercase hexadecimal characters | Participates in the exact lineage tuple match |

Money permits at most 16 total digits and four fractional digits. JSON money
must be encoded as a string; binary floating-point money is rejected. Non-finite
and non-positive values cannot pass the schema. Similarity factor values have
the same `[0, 100]` bound, but v1 does not verify that factors mathematically
reproduce `similarity_score`.

Currency and declared IANA timezone must exactly match the request. Timestamps
must contain a UTC offset, but v1 does not cross-check that supplied offset
against the declared IANA zone at that instant; the consumer must preserve that
consistency.

## Lineage object

Each `input_lineage` entry contains:

| Field | Type and constraints |
| --- | --- |
| `lineage_id` | string, 1-128 |
| `source_family_id` | string, 1-128 |
| `source_version` | string, 1-64 |
| `observation_hash` | SHA-256-shaped lowercase hexadecimal string |
| `lineage_hash` | SHA-256-shaped lowercase hexadecimal string |

`lineage_id` values must be unique. The distinct lineage tuple set must exactly
cover the comparable evidence set, with no unrelated tuple:

```text
(lineage_hash, source_family_id, source_version, observation_hash)
```

Comparable `observation_hash` values must also be unique within the request.
V1 validates format and exact tuple coverage, but it does not recompute hashes,
verify signatures, or prove ownership/licensing. Those remain consumer
controls.

## Explicitly prohibited payload content

Do not place names, emails, phone numbers, exact addresses, unnecessary exact
coordinates, iCal/private URLs, credentials, cookies, messages, payment data,
or complete CSV payloads in any string field. Identifiers are opaque, but the
engine cannot detect whether a consumer has embedded PII in an opaque value.

## Output envelope

| Field | Meaning |
| --- | --- |
| `contract_version` | Response schema `1.0` |
| `engine_version` | Pre-1.0 software version |
| `engine_profile` | Profile processed/requested |
| `algorithm_version` | Weighted-percentile semantic version |
| Policy/configuration versions | Echo the exact behavior boundaries |
| `deterministic_run_id` | UUIDv5 derived from canonical request SHA-256 |
| `request_hash` | SHA-256 of the canonical request |
| `status` | `recommended` or `abstained` |
| `recommendation` | Decimal-string strategy point, or null |
| `pricing_band` | `{low, central, high, currency}`, or null |
| `evidence_quality` | `INSUFFICIENT`, `LOW`, `MODERATE`, or `HIGH` |
| `evidence_components` | Counts, ESS, similarities, dispersion, age, outliers, missing base rates, compatibility |
| Comparable ID arrays | Received, used, and excluded audit sets, sorted by ID |
| `exclusion_reasons` | Known eligibility/outlier reasons keyed by comparable ID |
| `outliers` | Preserved excluded price and versioned reason |
| `reasons` | Human-readable calculation or abstention rationale |
| `warnings` | Required interpretation/non-claim warnings |
| `abstention` | Typed reason/detail, or null for a recommendation |
| `input_lineage` | Canonically sorted lineage echoed for audit |
| `output_lineage` | Request/input lineage hashes and all algorithm/policy versions |
| `duration_ms` | Deterministic contract value `0`; use HTTP telemetry for wall time |
| Safety flags | Both remain literal false |

Recommended money and band amounts serialize as decimal strings. Evidence
decimal components also serialize as strings. A recommendation always obeys
`low <= recommendation <= high`; P25/P50/P75 ordering follows from the weighted
empirical distribution.

Evidence quality is decided from raw Decimal ESS, average similarity, and
dispersion. The response quantizes ESS and dispersion to four decimal places and
average similarity to two solely for presentation. Consumers must not
reclassify a result from those rounded components; for example, raw `0.35004`
may display as `0.3500` without satisfying the HIGH threshold.

## Recommendation, abstention, and error

These are distinct outcomes:

- A recommendation returns HTTP 200 with `status="recommended"`, a band, and
  a strategy-selected amount.
- An evidence abstention returns HTTP 200 with `status="abstained"`, null band
  and recommendation, `INSUFFICIENT` quality, and a typed `abstention` object.
- A contract error returns HTTP 422 and no statistical response envelope.
- An unexpected failure returns a redacted HTTP 500 response.

Typed abstention reasons declared by v1 are:

```text
INSUFFICIENT_COMPARABLES
LOW_EFFECTIVE_SAMPLE_SIZE
EXCESSIVE_DISPERSION
LOW_SIMILARITY
STALE_EVIDENCE
TEMPORAL_INCOMPATIBILITY
CURRENCY_MISMATCH
CONTRACT_INCOMPATIBLE
INVALID_LINEAGE
INSUFFICIENT_PRICE_COVERAGE
UNSUPPORTED_PROFILE
```

Some values are shared taxonomy across errors and abstentions. Current behavior
uses contractual errors for mismatched currency, dates/time, invalid lineage,
unsupported policy versions, duplicate comparable IDs or observation hashes
(`DUPLICATE_COMPARABLE`), exact timezone mismatch (`TIMEZONE_MISMATCH`), and
unusable evidence (`EVIDENCE_NOT_USABLE`). Schema errors cover invalid money,
scores, unknown strategies, malformed timestamps, inconsistent stay lengths,
and unknown fields. `INSUFFICIENT_PRICE_COVERAGE` is reserved: v1 requires every
comparable to contain a total nightly rate, so missing coverage currently fails
schema validation before algorithm execution.

No profile fallback occurs after either abstention or error.

Normative precedence for the empty initial eligible set:

> Cuando no sobrevive evidencia, STALE_EVIDENCE tiene precedencia si al menos
> una observación fue excluida por recencia; de lo contrario, el resultado
> superior es LOW_SIMILARITY. El mapa por comparable contiene el diagnóstico
> completo.

Contract v1 does not add a generic `NO_ELIGIBLE_COMPARABLES` value.

## Deterministic identity

Canonical hashing sorts object keys, comparables by `comparable_id`, and lineage
by `lineage_id`, then uses compact ASCII JSON. Reordering those two arrays does
not change `request_hash`, `deterministic_run_id`, or the calculated output.
Other input changes can change the identity even when they do not change the
band. The engine does not persist requests and does not use these IDs as an
authorization or replay-prevention mechanism.

## HTTP operations

| Operation | Behavior |
| --- | --- |
| `GET /health/live` | Process liveness |
| `GET /health/ready` | Defaults to stable statistical readiness and can succeed without a model |
| `GET /health/ready?profile=PERFORMANCE_AWARE_EXPERIMENTAL` | Checks the experimental profile and returns 503 until its model is loaded |
| `GET /v1/capabilities` | Versioned profile discovery plus contractual/effective request limits; API key applies when configured |
| `POST /v1/pricing/recommendations` | Union of statistical and legacy experimental request contracts |

Successful health responses include `checked_profile`, `model_loaded`, and an
optional `model_version`, so a 200 response is interpreted in the context of the
profile actually checked.

The statistical HTTP and CLI transports share a 1,048,576-byte maximum for the
compact UTF-8 document. The API setting `PRICING_MAX_REQUEST_BODY_BYTES`
defaults to that maximum and may be lowered explicitly, never raised above it;
such an operational reduction can reject a larger contract-valid document and
is reported as `effective_request_body_limit_bytes`. Capabilities also reports
`maximum_request_body_bytes=1048576`, `maximum_comparables=50`,
`maximum_input_lineage_entries=50`, and `batch_supported=false`.

The POST endpoint additionally applies API-key/organization binding when
configured, a body-read deadline, a concurrency bulkhead, and a soft execution
deadline. A configured authorized tenant is compared with statistical
`organization_id`. After surrounding whitespace normalization, the statistical
identity maximum is 128 characters in the contract, `api_key_tenant_id`,
`serving_tenant_id`, and trusted-proxy header value; a longer configured or
supplied identity fails closed. The name of the trusted header retains its
separate HTTP-field-name constraint. These local controls do not replace
production identity, TLS, gateway rate limits, or audit storage.

## Compatibility policy

Within contract `1.0`, consumers must tolerate new response fields but must not
send unknown request fields. Removing fields, widening trust, changing money or
percentile semantics, or changing a gate requires an explicit new version.
Policy versions allow behavior to evolve without pretending that one v1 result
was computed under another configuration.

The existing performance-aware request/response remains accepted, but health
semantics changed deliberately: the no-query readiness check now targets the
stable default profile. Legacy monitors that used a 503 response to detect a
missing model must migrate to the explicit experimental-profile query above.
This is a documented compatibility migration, not an assertion of silent full
compatibility. The experimental request's `tenant_id` and `property_id` remain
bounded to 100 characters; widening statistical authentication configuration
does not change that legacy wire contract.
