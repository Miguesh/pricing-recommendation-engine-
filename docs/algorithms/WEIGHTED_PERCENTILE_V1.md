# Weighted Percentile Algorithm v1

## Scope

`weighted-percentile-v1.0.0` turns consumer-selected comparable nightly rates
into a transparent pricing band. It is a deterministic descriptive statistic,
not a trained model, demand forecast, causal estimator, or revenue optimizer.
All arithmetic runs inside an explicit decimal context with precision 28 and
`ROUND_HALF_UP`, independent of the host process's ambient decimal context.

The full behavior also depends on:

- algorithm configuration `market-evidence-statistical-v1.0.0`;
- evidence policy `evidence-quality-v1.0.0`; and
- outlier policy `weighted-tukey-v1.0.0`.

## Preconditions

The contract layer requires positive decimal-string prices, similarity scores
in `[0, 100]`, one request currency, one exact IANA timezone, matching scenario
dates/nights/guests, point-in-time-compatible timestamps, unique comparable IDs
and observation hashes, and exact lineage coverage by `(lineage_hash,
source_family_id, source_version, observation_hash)`. The consumer is
responsible for source rights, normalization, comparable selection, and
similarity construction. One request represents one pricing decision, not a
batch, and contains at most 50 comparables plus 50 matching lineage entries. A
compact request must fit the shared 1,048,576-byte HTTP/CLI transport boundary.

## Eligibility sequence

Comparables are sorted by `comparable_id` before calculation. The service then:

1. converts `as_of` and comparable timestamps to the declared IANA market
   timezone, then excludes evidence older than 90 calendar days with
   `STALE_EVIDENCE`;
2. excludes an individual similarity below 40 with `LOW_SIMILARITY`;
3. preserves and identifies price outliers when at least four eligible
   comparables remain;
4. excludes identified outliers under `weighted-tukey-v1.0.0`;
5. requires at least three remaining comparables;
6. requires effective sample size of at least 2.50;
7. requires average raw similarity of at least 55; and
8. requires weighted price dispersion no greater than 1.00.

Every exclusion remains visible by comparable ID. An outlier remains in the
response's audit record even though it is not used for the band. The engine does
not search for replacement comparables.

Normative precedence:

> Cuando no sobrevive evidencia, STALE_EVIDENCE tiene precedencia si al menos
> una observación fue excluida por recencia; de lo contrario, el resultado
> superior es LOW_SIMILARITY. El mapa por comparable contiene el diagnóstico
> completo.

Accordingly, one stale exclusion is sufficient for the top-level stale reason;
when none is stale, all initial exclusions are low similarity. There is no
fallback and no `NO_ELIGIBLE_COMPARABLES` reason in contract v1.

## Weight normalization

For used comparable `i`, the raw weight is its consumer-provided
`similarity_score`, denoted by `s_i`. The normalized weight is:

```text
w_i = s_i / sum(s_j)
```

All used weights must be strictly positive after eligibility. Multiplying all
raw weights by the same positive factor leaves normalized weights, percentiles,
and effective sample size unchanged.

## Effective sample size

Weight concentration is measured with Kish effective sample size:

```text
ESS = (sum(w_i))^2 / sum(w_i^2)
```

The implementation can use raw or normalized weights because the expression is
scale invariant. ESS is not a count of independent market observations and does
not establish representativeness; it only summarizes weight concentration.

## Weighted empirical percentile

For percentile `q` in `[0, 1]`:

1. sort `(price, weight)` pairs by ascending price;
2. compute `threshold = q * sum(weight)`;
3. accumulate weights in that order; and
4. return the first price whose cumulative weight is greater than or equal to
   the threshold.

This is a left-continuous weighted empirical quantile. There is no interpolation
between observed prices. Ties are stable because only the cumulative mass at a
price matters.

The pricing band is:

```text
low     = P25
central = P50
high    = P75
```

Each amount is rounded to `0.01` with decimal `ROUND_HALF_UP`. The engine does
not currently apply currency-specific minor-unit rules.

## Weighted Tukey outlier policy

When at least four comparables pass recency and individual-similarity gates,
the algorithm computes weighted Q25 and Q75 using raw similarity weights:

```text
IQR         = Q75 - Q25
lower_fence = Q25 - 1.50 * IQR
upper_fence = Q75 + 1.50 * IQR
```

A total nightly rate strictly outside either fence is recorded as an outlier
and excluded. Values on a fence remain eligible. When IQR is zero, any different
price is outside the zero-width fence. With fewer than four eligible
comparables, this outlier policy is not applied.

This rule does not prove that an observation is wrong. It is only the versioned
inclusion policy for this profile.

## Dispersion and strategy

After outlier exclusion:

```text
dispersion = (P75 - P25) / P50
```

`P50` is positive by contract. A value over 1.00 causes typed abstention. When a
band exists, the strategy selects exactly one observed weighted percentile:

| Strategy | Recommendation |
| --- | --- |
| `conservative` | P25 / band low |
| `balanced` | P50 / band central |
| `premium` | P75 / band high |

These labels express a position within the evidence distribution. They do not
encode demand elasticity, occupancy response, expected revenue, or an optimal
commercial policy.

## Evidence quality

All non-abstained results expose the underlying components. The current class
rules are:

| Class | Conditions |
| --- | --- |
| `HIGH` | used count >= 8, ESS >= 6, average similarity >= 80, dispersion <= 0.35 |
| `MODERATE` | used count >= 5, ESS >= 4, average similarity >= 65, dispersion <= 0.60 |
| `LOW` | recommendation passes all gates but neither higher class applies |
| `INSUFFICIENT` | response abstains |

All inequalities above are inclusive comparisons against the raw Decimal ESS,
average similarity, and dispersion values. Classification never reads the
rounded `evidence_components`. Only response presentation is quantized:
`effective_sample_size` and `dispersion_ratio` to `0.0001`, and
`average_similarity` to `0.01`. Consequently, raw dispersion `0.35004` is not
HIGH even though it serializes as `0.3500`, and raw `0.60004` is not MODERATE
even though it serializes as `0.6000`.

The response also reports minimum similarity, maximum age, outlier count,
missing base-rate count, and scenario compatibility. In v1 those fields aid
review but do not independently change the quality class beyond the eligibility
and abstention gates described above.

Evidence quality is ordinal review metadata. It is not a calibrated probability
that a price is correct.

## Determinism

Money and weights use `Decimal` with fixed precision 28 and `ROUND_HALF_UP`;
there is no randomization. Canonical request hashing sorts JSON object keys,
comparables by `comparable_id`, and lineage by `lineage_id`. The deterministic
run ID is UUIDv5 derived from that SHA-256 request hash. The contractual
`duration_ms` is zero so the pure response remains reproducible; HTTP wall time
belongs to telemetry, not the deterministic result.

Reordering comparables or lineage does not change the result. Changing any
other canonical input, including request/audit identifiers or quality-flag
order, may change the request hash and run ID.

## Worked synthetic example

The safe fixture at
`tests/fixtures/statistical/plusbnb-consumer.synthetic.json` contains eight
equal-weight synthetic rates from `100.00` through `135.00`. Under the
left-continuous definition:

- P25 is `105.00`;
- P50 is `115.00`;
- P75 is `125.00`; and
- `balanced` selects `115.00`.

These invented values demonstrate the calculation only. They are not market
data and do not support accuracy, uplift, or revenue claims.
