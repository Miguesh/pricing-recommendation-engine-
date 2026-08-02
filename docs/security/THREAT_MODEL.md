# Threat Model

## Scope

This threat model covers the HTTP and CLI boundary for
`MARKET_EVIDENCE_STATISTICAL_V1` and its coexistence with the existing
`PERFORMANCE_AWARE_EXPERIMENTAL` profile. It covers repository implementation
controls, not PLUSBNB deployment infrastructure or external evidence providers.

The stable profile is pure application logic: it performs no network or
filesystem I/O, loads no model artifacts, uses no dynamic execution, and does
not call MLflow. The surrounding process still contains the experimental ML
dependencies and must be governed accordingly.

## Assets

- Integrity of comparable prices, weights, lineage references, policy versions,
  pricing bands, abstention, and deterministic identity.
- Confidentiality of opaque organization, property, source, correlation, and
  audit identifiers.
- Availability of the API/CLI and bounded compute capacity.
- Integrity of the OpenAPI and versioned behavioral documentation.
- For the experimental path only, integrity of trained artifacts, registry
  metadata, tenant/currency bindings, and promotion decisions.

## Trust boundaries

```text
PLUSBNB normalization and authorization
        |
        | JSON + authenticated organization context
        v
FastAPI/CLI adapter: size, schema, auth, redaction
        |
        | immutable validated domain objects
        v
Pure statistical service: eligibility, percentiles, abstention
        |
        | response + hashes + lineage references
        v
PLUSBNB audit and human review
```

A separate boundary connects the experimental service to governed model
artifacts and MLflow. That boundary is not entered by the stable statistical
profile.

## Assumed attacker capabilities

An attacker or faulty consumer may submit malformed, oversized, repetitive,
stale, cross-tenant, cross-currency, future-dated, low-quality, or intentionally
skewed evidence. They may place sensitive data in otherwise opaque strings,
manipulate weights, replay a valid request, forge hash-shaped values, or attempt
to exhaust request parsing and calculation capacity. A registry writer can
replace experimental serialized artifacts and is therefore treated as a code
deployment principal.

## Threats and controls

| Threat | Current control | Residual risk / owner |
| --- | --- | --- |
| Cross-organization request | Optional API key plus configured key/tenant or trusted-gateway binding; statistical `organization_id` is checked | Local auth is not enterprise IAM; deployment must supply TLS, key rotation, gateway identity, and authorization |
| PII/private URL exfiltration | Strict bounded fields; documentation prohibits sensitive content; logs omit full payloads | Opaque strings cannot prove content is non-sensitive; PLUSBNB must minimize and scan before sending |
| Currency mixing | Uppercase currency schema and exact per-comparable equality; no FX code path | Three-letter format is not a full ISO registry; consumer owns code validity and prior normalization |
| Temporal leakage | Offset-aware timestamps and `observed_at`/`known_at <= as_of`; exact scenario checks | Consumer controls truthfulness and point-in-time queries; numeric offsets are not cross-checked against the declared IANA zone |
| Forged lineage | Unique comparable observation hashes and exact set coverage of `(lineage_hash, source_family_id, source_version, observation_hash)` | Hashes are not signatures; content is not recomputed and source rights/authenticity are not verified |
| Weight manipulation | Scores/factors bounded `[0,100]`; low individual/average similarity and ESS gates | Engine cannot validate the similarity methodology or factor-to-score consistency; consumer governance required |
| Price poisoning/outliers | Positive decimal money, weighted Tukey audit/exclusion, dispersion abstention | Coordinated plausible prices can remain; outlier rules are not fraud detection |
| Duplicate/replay | Duplicate comparable IDs rejected; deterministic request/run hashes | Requests are not persisted or nonce-checked; consumer owns idempotency and replay controls |
| Schema smuggling | Pydantic strict objects reject unknown fields; bounded strings/collections; no `eval` or dynamic imports | Union request evolution must be reviewed for ambiguous parsing |
| Binary-float money/non-finite values | Money accepted only as decimal strings and stored/calculated with `Decimal` | Downstream systems can reintroduce float error; consumer must preserve string/decimal semantics |
| Payload/compute exhaustion | Shared HTTP/CLI maximum 1,048,576 bytes; API may be configured lower and reports the effective limit; max 50 comparables and 50 lineage entries per single-decision request; read deadline, bulkhead, soft execution timeout | Per-process controls are not distributed rate limits; a lower operational limit can reject a larger contract-valid document; timed-out native work may continue until completion |
| Error data leakage | Validation errors keep only type/location/message; internal errors are generic; full payloads are not logged | Human-readable contract details and identifiers still require log access control and retention |
| Network/provider access | Stable core has no I/O and tests execute without live services | Process-level egress is not denied by the Python type system; deployment should apply network policy |
| Artifact code execution | Stable profile loads no artifacts; experimental lifecycle checks checksums and governance metadata before joblib loading | Serialized model loading can execute code; registry write permission remains code-deployment authority |
| Unauthorized publication | Contract and response fix `publication_allowed=false` and `commercial_validation=false` | Flags do not technically control a downstream channel; PLUSBNB must separate review from publication |
| Dependency compromise | Reproducible `uv.lock`, clean build/CI, dependency inventory and scan workflow | Locking does not eliminate compromised upstream packages; update review and image scanning remain required |

## Availability and transport controls

The body limit applies before FastAPI parses the recommendation endpoint. The
body-read deadline is independent from the soft calculation deadline. A
per-process semaphore limits concurrent recommendation work. The statistical
calculation is bounded further by at most 50 comparables for one pricing
decision and has no external waits. The CLI uses the same 1,048,576-byte
constant. Capabilities exposes the contractual maximum and the API's effective
limit so an operational reduction is observable.

Evidence-class thresholds consume raw Decimal metrics; quantized response
components cannot elevate a class at a boundary.

Liveness reports process health. Default readiness reports the stable profile
as ready even when no experimental model is loaded. Callers that need the model
must use
`/health/ready?profile=PERFORMANCE_AWARE_EXPERIMENTAL`, which returns 503 until
the artifact is loaded, and inspect the reported model fields.

The API Docker image runs as a non-root user. Compose drops Linux capabilities,
uses a read-only filesystem, and provisions bounded writable temporary mounts.
These local controls must be recreated and verified in any other deployment.

## Privacy and logging

HTTP logs record request ID, route/status/timing, failure class, and bounded
pricing completion metadata rather than complete request/response payloads.
Identifiers are still potentially linkable pseudonyms. Logs need access
control, retention, deletion, and incident procedures.

Never send or log names, contact information, addresses, precise coordinates,
listing/calendar/private URLs, credentials, cookies, messages, payment data, or
raw CSV. `source_family_id` is a taxonomy identifier, not a place for a URL.

## Security invariants for the stable profile

- Statistical execution initiates no network call, provider lookup, scraping,
  exchange-rate lookup, or dataset download.
- The stable application service requires and invokes no model, pickle, joblib,
  MLflow, LLM, plugin, or executable artifact. A co-hosted process may still
  initialize the separately configured experimental profile.
- No database/file persistence and no dynamic execution.
- No cross-currency calculation or implicit profile fallback.
- No recommendation when evidence gates require abstention.
- No output can set publication or commercial validation true.

Violating an invariant requires blocking the release or assigning a new profile
and explicit security review; it must not be hidden behind a configuration
default.

## Verification expectations

Release checks should cover locked dependency validation, formatter/lint, strict
type checking, unit/contract/property/API/CLI tests, OpenAPI reproducibility,
build/install smoke, Docker non-root/read-only smoke, dependency audit, secret
scan, forbidden-artifact review, and `git diff --check`. Tests must not require
live external services for the statistical profile.

## Residual risks requiring PLUSBNB controls

PLUSBNB must establish evidence rights, validate normalization, govern scoring,
avoid PII, bind authenticated identity to organization/property access, retain
audits, review recommendations, and control publication. This engine cannot
turn a hash, a high evidence class, or a successful HTTP response into proof of
commercial suitability.
