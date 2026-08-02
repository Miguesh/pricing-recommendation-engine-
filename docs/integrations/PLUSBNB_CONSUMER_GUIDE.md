# PLUSBNB Consumer Guide

## Integration boundary

PLUSBNB consumes `MARKET_EVIDENCE_STATISTICAL_V1` as an analyst decision-support
service. The repositories remain independent; do not copy engine source,
fixtures as production data, model artifacts, or configuration between them.

PLUSBNB is responsible for:

- proving source authorization, terms, privacy, retention, and provenance;
- capturing and normalizing observations into one request currency;
- selecting comparables and calculating versioned similarity scores;
- excluding PII, credentials, private URLs, and raw source payloads;
- owning customer, property, audit, review, override, and report records;
- presenting evidence, warnings, and abstention to a human analyst; and
- deciding whether any reviewed recommendation may be published.

The engine is responsible for validating the pricing contract, applying the
versioned weighted-statistical policy, exposing evidence components, and
returning either a recommendation or typed abstention. It never fetches a
replacement comparable and never publishes a price.

## Supported profile

Use exactly:

```text
contract_version: 1.0
engine_profile: MARKET_EVIDENCE_STATISTICAL_V1
algorithm_config_version: market-evidence-statistical-v1.0.0
evidence_policy_version: evidence-quality-v1.0.0
outlier_policy_version: weighted-tukey-v1.0.0
```

Do not send the legacy occupancy/booking-pace request with this profile. Do not
send a statistical request as `PERFORMANCE_AWARE_EXPERIMENTAL`. The experimental
profile remains a separate, model-backed contract and is not an initial
PLUSBNB dependency.

## Discovery and local contract checks

Install the repository's locked environment, then inspect capabilities without
loading a model or contacting external services:

```powershell
uv sync --locked --extra dev --python 3.12
uv run pricing-engine capabilities
```

Validate the synthetic consumer fixture:

```powershell
uv run pricing-engine validate `
  --input tests/fixtures/statistical/plusbnb-consumer.synthetic.json
```

Run the deterministic calculation locally:

```powershell
uv run pricing-engine recommend `
  --input tests/fixtures/statistical/plusbnb-consumer.synthetic.json
```

Regenerate the OpenAPI document for contract review:

```powershell
uv run pricing-engine export-openapi --output docs/openapi.json
git diff --exit-code -- docs/openapi.json
```

The fixture is wholly synthetic. Its identifiers and prices must never be
treated as a Medellín baseline, market quote, or production default.

## Build a request in PLUSBNB

1. Freeze an explicit `as_of` cutoff and correlation/audit identifiers.
2. Select one market configuration, currency, IANA timezone, target stay, and
   guest count.
3. Retrieve only observations whose `observed_at` and `known_at` do not exceed
   the cutoff.
4. Verify rights before the observation crosses the service boundary. Build one
   exactly matching lineage tuple per evidence record from `lineage_hash`,
   `source_family_id`, `source_version`, and unique `observation_hash` values.
5. Normalize each total nightly amount in PLUSBNB. Do not ask the engine to
   convert currency, allocate fees, or infer taxes.
6. Select and score comparables under the recorded
   `market_config_version`. Every comparable must match currency, timezone,
   dates, nights, and guests exactly in v1.
7. Create opaque organization/property/source identifiers. Never serialize
   display names, addresses, coordinates, listing URLs, calendar URLs, or raw
   source records.
8. Send required `availability_observed` as a boolean or null, and send
   `quality_flags` explicitly (an empty array is valid).
9. Send decimal money as JSON strings and explicit profile/policy versions.
10. Send both required `publication_allowed` and `commercial_validation`
    fields as false.

The statistical engine does not use target features to rescore evidence. A
consumer must not assume the presence of bedrooms, amenities, or area causes an
automatic adjustment.

## HTTP request

Start the API locally without an ML artifact:

```powershell
uv run uvicorn pricing_engine.interfaces.api.app:create_app `
  --factory --port 8000
```

Submit a validated document:

```powershell
$body = Get-Content -Raw `
  tests/fixtures/statistical/plusbnb-consumer.synthetic.json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/v1/pricing/recommendations `
  -ContentType application/json `
  -Headers @{ 'X-Request-ID' = 'plusbnb-contract-test' } `
  -Body $body
```

If `PRICING_API_KEY` is configured, add `X-API-Key`. In a bound deployment,
the authenticated tenant/organization must equal `organization_id`; a mismatch
returns 403. Production also requires TLS, gateway controls, managed secrets,
and an externally auditable identity binding.

`GET /health/ready` confirms that the stable statistical profile can execute
and is expected to return 200 without a model. To check the model-backed path,
call
`GET /health/ready?profile=PERFORMANCE_AWARE_EXPERIMENTAL`; it returns 503 until
the artifact has loaded, then reports `model_loaded=true`. Existing model
monitors must migrate to the explicit query rather than interpreting default
readiness.

## Interpret a successful recommendation

A 200 response with `status="recommended"` provides:

- `pricing_band.low`, `.central`, and `.high` as weighted empirical P25/P50/P75;
- `recommendation` equal to the selected strategy point;
- an ordinal `evidence_quality` and its explicit components;
- complete received/used/excluded ID sets and exclusion reasons;
- preserved outlier records;
- request, input-lineage, algorithm, and policy identity; and
- warnings that forbid forecast, accuracy, uplift, and publication
  interpretations.

Store the complete response beside the PLUSBNB audit and the exact source
request. Display the band, quality components, exclusions, outliers, reasons,
and warnings to the analyst. Do not display `HIGH` as a probability or approval.
The human review record remains a PLUSBNB responsibility.

## Interpret abstention and errors

| Result | Consumer action |
| --- | --- |
| HTTP 200, `status="abstained"` | Preserve the response, show the typed reason, and request evidence/policy review; do not select a fallback price silently |
| HTTP 422, `RequestValidationError` | Correct schema/format errors before retrying |
| HTTP 422, `StatisticalContractError` | Use `code` to route the incompatibility to normalization, lineage, or configuration review |
| HTTP 401/403 | Fix authenticated organization binding; do not rewrite `organization_id` to bypass authorization |
| HTTP 408/413 | Correct request transport or reduce the bounded payload |
| HTTP 503 | Retry with backoff only when service capacity is unavailable; the explicit experimental readiness/recommendation path also returns 503 without a model |
| HTTP 504 | Record the uncertain transport outcome; deterministic request identity permits comparison on a controlled retry |
| HTTP 500 | Record request/correlation IDs, alert the operator, and do not expose internal details to end users |

Abstention is a valid pricing result, not a transport failure. Never substitute
the current price, an average, or an experimental-model result without an
explicit PLUSBNB policy and human audit.

## Idempotency and audit storage

For canonical equivalent requests, the engine returns the same
`request_hash`, `deterministic_run_id`, band, recommendation, and response
content. Reordering comparables or lineage does not change identity. The engine
does not persist or deduplicate requests, so PLUSBNB must enforce its own
idempotency and store:

- request/correlation/audit IDs;
- exact contract and all policy versions;
- normalized input or its governed immutable reference;
- request and lineage hashes;
- complete response and transport status;
- analyst decision, rationale, edits, and timestamp; and
- publication decision in a separate controlled workflow.

Do not use a hash as evidence of source permission or as a digital signature.

## Consumer contract acceptance checklist

- [ ] `GET /v1/capabilities` advertises the stable profile and exact versions.
- [ ] The checked-in synthetic fixture validates and recommends with no model
      URI or external service.
- [ ] PLUSBNB rejects or quarantines PII/private URLs before serialization.
- [ ] Currency and timezone are uniform and no conversion occurs in the engine.
- [ ] Point-in-time queries enforce both `observed_at` and `known_at <= as_of`.
- [ ] Distinct lineage tuples exactly cover all comparables and observation
      hashes are unique within the request.
- [ ] Decimal money is serialized as strings and response money is stored
      without binary-float conversion.
- [ ] Recommended, abstained, contract-error, and internal-error paths have
      separate handling.
- [ ] Evidence components and warnings are visible to the analyst.
- [ ] An abstention cannot trigger silent fallback or automatic publication.
- [ ] Contract tests pin OpenAPI and the synthetic consumer fixture.
- [ ] Monitoring uses the explicit experimental readiness query when model
      availability matters.
- [ ] The integration makes no accuracy, forecast, uplift, revenue, or
      commercial-validation claim.

## Change management

Pin the contract and policy versions in PLUSBNB. Review `CHANGELOG.md`, the
OpenAPI diff, and the consumer fixture before upgrading. Treat a behavior change
without a version change as a defect. Do not couple PLUSBNB to internal Python
classes; consume the HTTP/JSON contract or execute the documented local CLI only
for controlled analyst workflows.
