# Market Evidence Statistical Profile: Limitations

## Correct interpretation

`MARKET_EVIDENCE_STATISTICAL_V1` describes the weighted distribution of a
consumer-provided comparable set. Its low/central/high band is weighted P25,
P50, and P75. The strategy selects one of those three points.

It is not:

- a demand or occupancy forecast;
- a revenue optimizer;
- a causal estimate of price elasticity;
- a booking or availability inference;
- a market-data acquisition or comparable-selection service;
- a probability that a price is correct;
- permission to publish or automate a price; or
- evidence of accuracy, optimality, revenue uplift, or commercial validation.

## Evidence dependency

The result is only as defensible as the consumer's source authorization,
point-in-time capture, fee/tax normalization, comparable selection, similarity
method, and lineage. The engine checks structural compatibility but cannot prove
that an observation is real, licensed, representative, independently sourced,
or correctly normalized.

The engine does not query portals, APIs, databases, calendars, reservation
systems, exchange rates, or replacement evidence. Sparse or biased input stays
sparse or biased; abstention gates reduce but do not eliminate that risk.

## Statistical limitations

- Weighted empirical percentiles do not interpolate. A small change near a
  cumulative-weight boundary can move the selected observed price.
- Similarity is a consumer-provided weight. V1 bounds scores but does not
  recompute them or verify that `similarity_factors` produce the aggregate.
- Effective sample size summarizes weight concentration; it does not prove
  independence, diversity, or market coverage.
- Weighted Tukey fences are a policy rule, not anomaly truth or fraud
  detection. Coordinated or plausible bad data can remain, and legitimate
  extreme properties can be excluded.
- Current thresholds (count, similarity, age, dispersion, ESS, and evidence
  classes) are transparent engineering defaults without real-market
  calibration.
- Evidence quality is ordinal. It is not a calibrated confidence probability
  or expected-error interval.
- V1 reports age, minimum similarity, outlier count, and missing base rates,
  but the final quality class uses only used count, ESS, average similarity, and
  dispersion after eligibility. Those comparisons use raw Decimal values;
  rounded response components are presentation-only and can straddle a class
  boundary.
- When no comparable survives combined age/similarity eligibility,
  `STALE_EVIDENCE` has top-level precedence if at least one observation was
  excluded by recency. Otherwise the reason is `LOW_SIMILARITY`. The
  per-comparable map retains the complete diagnosis, including both reasons.

## Scenario limitations

V1 requires exact currency, IANA timezone, stay dates, night count, and guest
count across the target and every comparable. It does not perform:

- currency conversion or inflation adjustment;
- timezone conversion or fuzzy market matching;
- date-window, weekday, season, lead-time, or event adjustment;
- capacity, bedroom, amenity, property-type, or area adjustment;
- fee/tax allocation or base-to-total price reconstruction;
- length-of-stay or guest-count normalization; or
- availability, occupancy, cancellation, or booking-pace inference.

Target property features remain auditable contract context but do not alter v1
weights or prices. PLUSBNB must incorporate those features before the request
through selection and scoring.

All output money is rounded to `0.01`. The profile does not currently apply
currency-specific minor units or cash-rounding rules. Three uppercase letters
are syntactically accepted; the engine does not maintain a currency registry.

## Temporal and lineage limitations

The engine enforces offset-aware timestamps, request snapshot
`observed_at <= as_of`, comparable `observed_at <= known_at <= as_of`, and a
90-day age gate. It trusts the supplied timestamps and cannot prove that the
consumer's storage/query was genuinely point-in-time.

V1 validates the IANA timezone name and exact request/comparable equality, but
does not cross-check each timestamp's numeric UTC offset against that zone at
the instant. Recency converts both instants to the declared IANA market zone
before comparing calendar dates. PLUSBNB must still emit accurate instants;
an incorrect supplied offset changes the instant before that conversion.

Lineage fields are bounded SHA-256-shaped references. V1 requires unique
comparable observation hashes and exact distinct-tuple coverage across
`lineage_hash`, `source_family_id`, `source_version`, and `observation_hash`.
It still does not recompute a digest, validate a signature, or establish source
ownership. Hashes are not authorization.

## Operational limitations

- The engine does not persist requests, responses, human decisions, overrides,
  or outcomes. PLUSBNB owns audit and idempotency.
- Deterministic request/run IDs do not prevent replay and are not security
  tokens.
- API-key authentication is a local baseline. Production requires TLS,
  managed identity/secrets, tenant/property authorization, rate limits, and
  governed logging.
- The stable and experimental profiles share one API process and endpoint.
  Consumers must select the correct schema and inspect capabilities. Ordinary
  experimental load, validation, warm-up, or service-construction failure is
  isolated from stable startup, but fatal process/runtime failures are not.
- One statistical request represents one pricing decision, not a batch. It is
  limited to 50 comparables, 50 matching lineage entries, and 1,048,576 compact
  UTF-8 bytes in both supported transports. An explicitly lower API setting is
  visible in capabilities and can reject a larger contract-valid request.
- Default statistical readiness does not mean a trained experimental model is
  loaded; model monitoring must use the explicit experimental-profile query.
  Experimental unavailability returns 503 and never invokes the statistical
  calculation as a fallback.
- Statistical identifiers are ASCII-bound after surrounding whitespace is
  trimmed: 1-128 characters matching
  `^[A-Za-z0-9][A-Za-z0-9._:-]*$`. This does not widen or redefine the legacy
  experimental request's 100-character tenant/property wire contract.
- The CLI size check uses a bounded binary read before strict UTF-8 decoding.
  Invalid encoding, JSON, schema, and read failures intentionally share a
  redacted error, which limits diagnostics available to CLI callers.
- The stable core performs no external access, but process-level egress control
  is a deployment responsibility.
- A soft HTTP timeout does not guarantee cancellation of synchronous work.
- The repository fixture and tests are synthetic software evidence only.

## Human and commercial limitations

`conservative`, `balanced`, and `premium` are positions within the observed
weighted price distribution; they are not business-performance promises. The
profile does not consider property-specific floors/ceilings, brand strategy,
regulation, fairness, taxes, operating cost, guest value, cancellation,
inventory controls, or downstream marketplace constraints.

PLUSBNB must present the evidence and warnings to an analyst, record review and
override rationale, and keep publication as a separate controlled action. Both
`publication_allowed` and `commercial_validation` remain false by contract.

## Evidence required before broader use

Before any automation or commercial claim, the owner would need authorized
real-market evidence, documented normalization and selection quality, stable
source coverage, segment-specific error analysis against later outcomes,
threshold calibration, temporal validation across regimes, override/incident
monitoring, and a controlled experiment or defensible causal/off-policy design
for uplift claims. None of that evidence exists in this profile or its
synthetic fixtures.
