"""Contract and mathematical tests for MARKET_EVIDENCE_STATISTICAL_V1."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, getcontext, localcontext
from pathlib import Path

import joblib
import pytest
from pydantic import ValidationError

from pricing_engine.application.statistical_service import (
    MarketEvidenceStatisticalV1,
    StatisticalConfigV1,
    capabilities,
    effective_sample_size,
    weighted_percentile,
)
from pricing_engine.domain.exceptions import StatisticalContractError
from pricing_engine.domain.statistical import (
    MAX_COMPARABLES,
    AbstentionReason,
    EngineProfile,
    EvidenceQuality,
    PricingStrategy,
    RecommendationStatus,
    StatisticalPricingRequest,
)

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "statistical" / "plusbnb-consumer.synthetic.json"
)
SCENARIO_CATALOG_PATH = FIXTURE_PATH.with_name("scenarios.synthetic.json")


def _payload() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _request(payload: dict[str, object] | None = None) -> StatisticalPricingRequest:
    return StatisticalPricingRequest.model_validate(payload or _payload())


def _recommend(
    payload: dict[str, object] | None = None,
):
    return MarketEvidenceStatisticalV1().recommend(_request(payload))


def _comparables(payload: dict[str, object]) -> list[dict[str, object]]:
    comparables = payload["comparables"]
    assert isinstance(comparables, list)
    return comparables


def _synchronize_lineage(payload: dict[str, object]) -> None:
    """Keep scenario variants inside the exact one-entry-per-comparable boundary."""

    observation_hashes = {comparable["observation_hash"] for comparable in _comparables(payload)}
    lineage = payload["input_lineage"]
    assert isinstance(lineage, list)
    payload["input_lineage"] = [
        entry for entry in lineage if entry["observation_hash"] in observation_hashes
    ]


def _append_synthetic_outlier(payload: dict[str, object]) -> None:
    outlier = deepcopy(_comparables(payload)[0])
    outlier.update(
        {
            "comparable_id": "cmp-outlier",
            "effective_base_nightly_rate": "1000.00",
            "effective_total_nightly_rate": "1000.00",
            "observation_hash": "0" * 64,
        }
    )
    _comparables(payload).append(outlier)
    lineage = payload["input_lineage"]
    assert isinstance(lineage, list)
    outlier_lineage = deepcopy(lineage[0])
    outlier_lineage.update(
        {
            "lineage_id": "lineage-synthetic-outlier",
            "observation_hash": "0" * 64,
        }
    )
    lineage.append(outlier_lineage)


def _scenario_catalog() -> dict[str, object]:
    return json.loads(SCENARIO_CATALOG_PATH.read_text(encoding="utf-8"))


def _apply_scenario(payload: dict[str, object], scenario: dict[str, object]) -> None:
    limit = scenario.get("comparable_limit")
    if isinstance(limit, int):
        payload["comparables"] = _comparables(payload)[:limit]
        _synchronize_lineage(payload)
    strategy = scenario.get("pricing_strategy")
    if isinstance(strategy, str):
        payload["pricing_strategy"] = strategy
    rates = scenario.get("total_rates")
    if isinstance(rates, list):
        for comparable, rate in zip(_comparables(payload), rates, strict=True):
            comparable["effective_base_nightly_rate"] = rate
            comparable["effective_total_nightly_rate"] = rate
    scores = scenario.get("similarity_scores")
    if isinstance(scores, list):
        for comparable, score in zip(_comparables(payload), scores, strict=True):
            comparable["similarity_score"] = score
    if scenario.get("append_outlier") is True:
        _append_synthetic_outlier(payload)
    if scenario.get("duplicate_first_comparable") is True:
        _comparables(payload).append(deepcopy(_comparables(payload)[0]))
    currency = scenario.get("first_comparable_currency")
    if isinstance(currency, str):
        _comparables(payload)[0]["currency"] = currency
    timezone = scenario.get("first_comparable_timezone")
    if isinstance(timezone, str):
        _comparables(payload)[0]["timezone"] = timezone
    observed_at = scenario.get("first_observed_at")
    known_at = scenario.get("first_known_at")
    if isinstance(observed_at, str) and isinstance(known_at, str):
        _comparables(payload)[0]["observed_at"] = observed_at
        _comparables(payload)[0]["known_at"] = known_at
    if scenario.get("drop_last_lineage") is True:
        lineage = payload["input_lineage"]
        assert isinstance(lineage, list)
        lineage.pop()
    if scenario.get("base_rates_null") is True:
        for comparable in _comparables(payload):
            comparable["effective_base_nightly_rate"] = None


CATALOG_SCENARIOS = _scenario_catalog()["scenarios"]
assert isinstance(CATALOG_SCENARIOS, list)


def test_weighted_percentile_is_left_continuous_order_independent_and_bounded() -> None:
    values = (Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40"))
    weights = (Decimal("1"), Decimal("1"), Decimal("2"), Decimal("6"))

    assert weighted_percentile(values, weights, Decimal("0.25")) == Decimal("30")
    assert weighted_percentile(values, weights, Decimal("0.50")) == Decimal("40")
    assert weighted_percentile(values, weights, Decimal("0.75")) == Decimal("40")
    assert weighted_percentile(values[::-1], weights[::-1], Decimal("0.25")) == Decimal("30")
    assert weighted_percentile(values, weights, Decimal("0")) == min(values)
    assert weighted_percentile(values, weights, Decimal("1")) == max(values)


@pytest.mark.parametrize(
    ("values", "weights", "percentile"),
    [
        ((), (), Decimal("0.5")),
        ((Decimal("1"),), (Decimal("1"), Decimal("2")), Decimal("0.5")),
        ((Decimal("1"),), (Decimal("0"),), Decimal("0.5")),
        ((Decimal("1"),), (Decimal("NaN"),), Decimal("0.5")),
        ((Decimal("1"),), (Decimal("1"),), Decimal("1.1")),
    ],
)
def test_weighted_percentile_rejects_invalid_inputs(
    values: tuple[Decimal, ...],
    weights: tuple[Decimal, ...],
    percentile: Decimal,
) -> None:
    with pytest.raises(ValueError):
        weighted_percentile(values, weights, percentile)


def test_weight_normalization_and_effective_sample_size_are_scale_invariant() -> None:
    raw = (Decimal("2"), Decimal("3"), Decimal("5"))
    normalized = tuple(weight / sum(raw) for weight in raw)

    assert sum(normalized) == Decimal("1")
    assert effective_sample_size(raw) == effective_sample_size(normalized)
    assert effective_sample_size((Decimal("1"),) * 4) == Decimal("4")
    assert effective_sample_size(()) == Decimal("0")
    assert effective_sample_size((Decimal("1"), Decimal("0"))) == Decimal("0")
    assert effective_sample_size((Decimal("NaN"),)) == Decimal("0")


def test_consumer_fixture_produces_ordered_decimal_band_and_high_quality() -> None:
    response = _recommend()
    assert response.status is RecommendationStatus.RECOMMENDED
    assert response.evidence_quality is EvidenceQuality.HIGH
    assert response.pricing_band is not None
    assert response.pricing_band.low <= response.pricing_band.central <= response.pricing_band.high
    assert response.pricing_band.low <= response.recommendation <= response.pricing_band.high
    assert response.publication_allowed is False
    assert response.commercial_validation is False

    serialized = json.loads(response.model_dump_json())
    assert serialized["recommendation"] == "115.00"
    assert serialized["pricing_band"] == {
        "low": "105.00",
        "central": "115.00",
        "high": "125.00",
        "currency": "COP",
    }
    assert isinstance(serialized["evidence_components"]["effective_sample_size"], str)


@pytest.mark.parametrize(
    ("strategy", "expected_field"),
    [
        (PricingStrategy.CONSERVATIVE, "low"),
        (PricingStrategy.BALANCED, "central"),
        (PricingStrategy.PREMIUM, "high"),
    ],
)
def test_strategy_selects_documented_band_point(
    strategy: PricingStrategy,
    expected_field: str,
) -> None:
    request = _request().model_copy(update={"pricing_strategy": strategy})
    response = MarketEvidenceStatisticalV1().recommend(request)

    assert response.pricing_band is not None
    assert response.recommendation == getattr(response.pricing_band, expected_field)


def test_reordering_comparables_preserves_complete_deterministic_result() -> None:
    request = _request()
    service = MarketEvidenceStatisticalV1()
    first = service.recommend(request)
    reordered = request.model_copy(
        update={
            "comparables": tuple(reversed(request.comparables)),
            "input_lineage": tuple(reversed(request.input_lineage)),
        }
    )
    second = service.recommend(reordered)

    assert first.deterministic_run_id == second.deterministic_run_id
    assert first.request_hash == second.request_hash
    assert first.model_dump_json() == second.model_dump_json()


def test_process_decimal_precision_does_not_change_deterministic_output() -> None:
    request = _request()
    baseline = MarketEvidenceStatisticalV1().recommend(request)

    with localcontext() as caller_context:
        caller_context.prec = 6
        assert getcontext().prec == 6
        constrained = MarketEvidenceStatisticalV1().recommend(request)

    assert constrained.model_dump_json() == baseline.model_dump_json()


@pytest.mark.parametrize("factor", [Decimal("0.5"), Decimal("2"), Decimal("10")])
def test_scaling_all_prices_scales_band_and_recommendation(factor: Decimal) -> None:
    request = _request()
    baseline = MarketEvidenceStatisticalV1().recommend(request)
    scaled_comparables = tuple(
        comparable.model_copy(
            update={
                "effective_base_nightly_rate": (
                    comparable.effective_base_nightly_rate * factor
                    if comparable.effective_base_nightly_rate is not None
                    else None
                ),
                "effective_total_nightly_rate": comparable.effective_total_nightly_rate * factor,
            }
        )
        for comparable in request.comparables
    )
    scaled = MarketEvidenceStatisticalV1().recommend(
        request.model_copy(update={"comparables": scaled_comparables})
    )

    assert baseline.pricing_band is not None
    assert scaled.pricing_band is not None
    assert scaled.pricing_band.low == baseline.pricing_band.low * factor
    assert scaled.pricing_band.central == baseline.pricing_band.central * factor
    assert scaled.pricing_band.high == baseline.pricing_band.high * factor
    assert scaled.recommendation == baseline.recommendation * factor


def test_outlier_is_retained_for_audit_but_excluded_from_calculation() -> None:
    payload = _payload()
    _append_synthetic_outlier(payload)
    response = _recommend(payload)

    assert response.status is RecommendationStatus.RECOMMENDED
    assert [item.comparable_id for item in response.outliers] == ["cmp-outlier"]
    assert "cmp-outlier" in response.comparable_ids_received
    assert "cmp-outlier" not in response.comparable_ids_used
    assert response.exclusion_reasons["cmp-outlier"] == ("OUTLIER_EXCLUDED_BY_WEIGHTED_TUKEY_V1",)


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("insufficient", AbstentionReason.INSUFFICIENT_COMPARABLES),
        ("effective_sample", AbstentionReason.LOW_EFFECTIVE_SAMPLE_SIZE),
        ("dispersion", AbstentionReason.EXCESSIVE_DISPERSION),
        ("similarity", AbstentionReason.LOW_SIMILARITY),
        ("stale", AbstentionReason.STALE_EVIDENCE),
    ],
)
def test_expected_evidence_gates_abstain_without_fallback(
    case: str,
    expected_reason: AbstentionReason,
) -> None:
    payload = _payload()
    comparables = _comparables(payload)
    if case == "insufficient":
        payload["comparables"] = comparables[:2]
    elif case == "effective_sample":
        payload["comparables"] = comparables[:3]
        for comparable, score in zip(_comparables(payload), ("100", "40", "40"), strict=True):
            comparable["similarity_score"] = score
    elif case == "dispersion":
        payload["comparables"] = comparables[:4]
        for comparable, price in zip(
            _comparables(payload), ("10.00", "100.00", "200.00", "300.00"), strict=True
        ):
            comparable["effective_base_nightly_rate"] = price
            comparable["effective_total_nightly_rate"] = price
    elif case == "similarity":
        payload["comparables"] = comparables[:4]
        for comparable in _comparables(payload):
            comparable["similarity_score"] = "50"
    else:
        for comparable in comparables:
            comparable["observed_at"] = "2026-01-01T09:00:00-05:00"
            comparable["known_at"] = "2026-01-01T10:00:00-05:00"
    _synchronize_lineage(payload)

    response = _recommend(payload)

    assert response.status is RecommendationStatus.ABSTAINED
    assert response.recommendation is None
    assert response.pricing_band is None
    assert response.evidence_quality is EvidenceQuality.INSUFFICIENT
    assert response.abstention is not None
    assert response.abstention.reason is expected_reason
    assert "fallback" in response.warnings[0].lower()


def test_total_charge_scenario_does_not_require_a_base_rate() -> None:
    payload = _payload()
    for comparable in _comparables(payload):
        comparable["effective_base_nightly_rate"] = None

    response = _recommend(payload)

    assert response.status is RecommendationStatus.RECOMMENDED
    assert response.evidence_components.missing_base_rate_count == len(_comparables(payload))


def test_quality_classes_are_explicit_and_not_probabilities() -> None:
    payload = _payload()
    payload["comparables"] = _comparables(payload)[:5]
    _synchronize_lineage(payload)
    for comparable in _comparables(payload):
        comparable["similarity_score"] = "70"
    moderate = _recommend(payload)

    payload["comparables"] = _comparables(payload)[:3]
    _synchronize_lineage(payload)
    for comparable in _comparables(payload):
        comparable["similarity_score"] = "60"
    low = _recommend(payload)

    assert moderate.evidence_quality is EvidenceQuality.MODERATE
    assert low.evidence_quality is EvidenceQuality.LOW
    assert any("not a probability" in warning for warning in moderate.warnings)


@pytest.mark.parametrize(
    ("prices", "expected_quality", "serialized_dispersion"),
    [
        (
            (
                "65.0000",
                "65.0000",
                "80.0000",
                "100.0000",
                "100.0000",
                "100.0000",
                "110.0000",
                "115.0000",
            ),
            EvidenceQuality.HIGH,
            "0.3500",
        ),
        (
            (
                "64.9960",
                "64.9960",
                "80.0000",
                "100.0000",
                "100.0000",
                "100.0000",
                "110.0000",
                "115.0000",
            ),
            EvidenceQuality.MODERATE,
            "0.3500",
        ),
        (
            ("40.0000", "40.0000", "100.0000", "100.0000", "110.0000"),
            EvidenceQuality.MODERATE,
            "0.6000",
        ),
        (
            ("39.9960", "39.9960", "100.0000", "100.0000", "110.0000"),
            EvidenceQuality.LOW,
            "0.6000",
        ),
    ],
)
def test_evidence_quality_uses_raw_dispersion_before_response_quantization(
    prices: tuple[str, ...],
    expected_quality: EvidenceQuality,
    serialized_dispersion: str,
) -> None:
    payload = _payload()
    payload["comparables"] = _comparables(payload)[: len(prices)]
    _synchronize_lineage(payload)
    for comparable, price in zip(_comparables(payload), prices, strict=True):
        comparable["effective_base_nightly_rate"] = price
        comparable["effective_total_nightly_rate"] = price

    response = _recommend(payload)
    serialized = json.loads(response.model_dump_json())

    assert response.status is RecommendationStatus.RECOMMENDED
    assert response.evidence_quality is expected_quality
    assert serialized["evidence_components"]["dispersion_ratio"] == serialized_dispersion


def test_evidence_quality_uses_raw_ess_below_high_boundary() -> None:
    raw_ess = Decimal("5.99996")
    assert raw_ess.quantize(Decimal("0.0001")) == Decimal("6.0000")

    quality = MarketEvidenceStatisticalV1()._quality(
        comparable_count_used=8,
        raw_effective_sample_size=raw_ess,
        raw_average_similarity=Decimal("80"),
        raw_dispersion_ratio=Decimal("0.35"),
    )

    assert quality is EvidenceQuality.MODERATE


def test_evidence_quality_uses_raw_average_similarity_below_high_boundary() -> None:
    payload = _payload()
    for comparable in _comparables(payload):
        comparable["similarity_score"] = "79.9999"

    response = _recommend(payload)
    serialized = json.loads(response.model_dump_json())

    assert response.evidence_quality is EvidenceQuality.MODERATE
    assert serialized["evidence_components"]["average_similarity"] == "80.00"


def test_boundary_quality_result_remains_deterministic_when_reordered() -> None:
    payload = _payload()
    prices = (
        "64.9960",
        "64.9960",
        "80.0000",
        "100.0000",
        "100.0000",
        "100.0000",
        "110.0000",
        "115.0000",
    )
    for comparable, price in zip(_comparables(payload), prices, strict=True):
        comparable["effective_base_nightly_rate"] = price
        comparable["effective_total_nightly_rate"] = price
    request = _request(payload)
    reordered = request.model_copy(
        update={
            "comparables": tuple(reversed(request.comparables)),
            "input_lineage": tuple(reversed(request.input_lineage)),
        }
    )

    first = MarketEvidenceStatisticalV1().recommend(request)
    second = MarketEvidenceStatisticalV1().recommend(reordered)

    assert first.evidence_quality is EvidenceQuality.MODERATE
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.parametrize(
    ("case", "expected_reason", "expected_exclusions"),
    [
        (
            "all_stale",
            AbstentionReason.STALE_EVIDENCE,
            {
                "cmp-001": ("STALE_EVIDENCE",),
                "cmp-002": ("STALE_EVIDENCE",),
                "cmp-003": ("STALE_EVIDENCE",),
            },
        ),
        (
            "all_low_similarity",
            AbstentionReason.LOW_SIMILARITY,
            {
                "cmp-001": ("LOW_SIMILARITY",),
                "cmp-002": ("LOW_SIMILARITY",),
                "cmp-003": ("LOW_SIMILARITY",),
            },
        ),
        (
            "mixed",
            AbstentionReason.STALE_EVIDENCE,
            {
                "cmp-001": ("STALE_EVIDENCE",),
                "cmp-002": ("LOW_SIMILARITY",),
                "cmp-003": ("STALE_EVIDENCE", "LOW_SIMILARITY"),
            },
        ),
    ],
)
def test_empty_eligible_set_uses_stale_precedence_and_complete_diagnostics(
    case: str,
    expected_reason: AbstentionReason,
    expected_exclusions: dict[str, tuple[str, ...]],
) -> None:
    payload = _payload()
    payload["comparables"] = _comparables(payload)[:3]
    _synchronize_lineage(payload)
    comparables = _comparables(payload)
    if case == "all_stale":
        for comparable in comparables:
            comparable["observed_at"] = "2026-01-01T09:00:00-05:00"
            comparable["known_at"] = "2026-01-01T10:00:00-05:00"
    elif case == "all_low_similarity":
        for comparable in comparables:
            comparable["similarity_score"] = "39.9999"
    else:
        comparables[0]["observed_at"] = "2026-01-01T09:00:00-05:00"
        comparables[0]["known_at"] = "2026-01-01T10:00:00-05:00"
        comparables[1]["similarity_score"] = "39.9999"
        comparables[2]["observed_at"] = "2026-01-01T09:00:00-05:00"
        comparables[2]["known_at"] = "2026-01-01T10:00:00-05:00"
        comparables[2]["similarity_score"] = "39.9999"

    request = _request(payload)
    response = MarketEvidenceStatisticalV1().recommend(request)
    reordered = request.model_copy(
        update={
            "comparables": tuple(reversed(request.comparables)),
            "input_lineage": tuple(reversed(request.input_lineage)),
        }
    )
    reordered_response = MarketEvidenceStatisticalV1().recommend(reordered)

    assert response.status is RecommendationStatus.ABSTAINED
    assert response.abstention is not None
    assert response.abstention.reason is expected_reason
    assert response.recommendation is None
    assert response.pricing_band is None
    assert response.comparable_ids_used == ()
    assert response.exclusion_reasons == expected_exclusions
    assert set(response.comparable_ids_excluded) == set(expected_exclusions)
    assert "fallback" in response.warnings[0].lower()
    assert response.model_dump_json() == reordered_response.model_dump_json()


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("duplicate", "DUPLICATE_COMPARABLE"),
        ("currency", "CURRENCY_MISMATCH"),
        ("timezone", "TIMEZONE_MISMATCH"),
        ("future", "TEMPORAL_INCOMPATIBILITY"),
        ("guests", "TEMPORAL_INCOMPATIBILITY"),
        ("dates", "TEMPORAL_INCOMPATIBILITY"),
        ("lineage", "INVALID_LINEAGE"),
        ("not_usable", "EVIDENCE_NOT_USABLE"),
        ("policy", "CONTRACT_INCOMPATIBLE"),
    ],
)
def test_contract_incompatibilities_raise_typed_errors(case: str, expected_code: str) -> None:
    payload = _payload()
    comparables = _comparables(payload)
    if case == "duplicate":
        comparables.append(deepcopy(comparables[0]))
    elif case == "currency":
        comparables[0]["currency"] = "USD"
    elif case == "timezone":
        comparables[0]["timezone"] = "UTC"
    elif case == "future":
        comparables[0]["observed_at"] = "2026-07-16T09:00:00-05:00"
        comparables[0]["known_at"] = "2026-07-16T10:00:00-05:00"
    elif case == "guests":
        comparables[0]["guests"] = 3
    elif case == "dates":
        comparables[0]["stay_start"] = "2026-08-11"
        comparables[0]["stay_end"] = "2026-08-13"
    elif case == "lineage":
        comparables[0]["lineage_hash"] = "b" * 64
    elif case == "not_usable":
        comparables[0]["eligible_for_pricing"] = False
    else:
        payload["algorithm_config_version"] = "unsupported-v2"

    with pytest.raises(StatisticalContractError) as captured:
        MarketEvidenceStatisticalV1().recommend(_request(payload))

    assert captured.value.code == expected_code


def test_duplicate_lineage_tuple_with_a_distinct_lineage_id_is_rejected() -> None:
    payload = _payload()
    lineage = payload["input_lineage"]
    assert isinstance(lineage, list)
    duplicate_tuple = deepcopy(lineage[0])
    duplicate_tuple["lineage_id"] = "lineage-synthetic-duplicate-tuple"
    lineage.append(duplicate_tuple)

    with pytest.raises(StatisticalContractError) as captured:
        _recommend(payload)

    assert captured.value.code == AbstentionReason.INVALID_LINEAGE


def test_stable_profile_rejects_unversioned_behavior_override() -> None:
    overridden = StatisticalConfigV1(minimum_comparables=2)

    with pytest.raises(ValueError, match="canonical versioned config"):
        MarketEvidenceStatisticalV1(overridden)


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("contract", "contract_version"),
        ("strategy", "pricing_strategy"),
        ("price", "effective_total_nightly_rate"),
        ("binary_money", "effective_total_nightly_rate"),
        ("score", "similarity_score"),
        ("nights", "number_of_nights"),
        ("pii", "contact_email"),
        ("operational_signal", "historical_occupancy_7d"),
        ("request_count_string", "number_of_nights"),
        ("comparable_count_string", "guests"),
        ("request_datetime_date", "stay_start"),
        ("comparable_datetime_date", "stay_start"),
        ("similarity_factor_float", "similarity_factors"),
        ("long_identifier", "request_id"),
    ],
)
def test_strict_schema_rejects_invalid_or_extraneous_values(
    case: str,
    expected_message: str,
) -> None:
    payload = _payload()
    comparable = _comparables(payload)[0]
    if case == "contract":
        payload["contract_version"] = "2.0"
    elif case == "strategy":
        payload["pricing_strategy"] = "invented"
    elif case == "price":
        comparable["effective_total_nightly_rate"] = "0"
    elif case == "binary_money":
        comparable["effective_total_nightly_rate"] = 100.0
    elif case == "score":
        comparable["similarity_score"] = "101"
    elif case == "nights":
        comparable["number_of_nights"] = 3
    elif case == "pii":
        payload["contact_email"] = "synthetic@example.invalid"
    elif case == "operational_signal":
        payload["historical_occupancy_7d"] = "0.80"
    elif case == "request_count_string":
        payload["number_of_nights"] = "2"
    elif case == "comparable_count_string":
        comparable["guests"] = "2"
    elif case == "request_datetime_date":
        payload["stay_start"] = "2026-08-10T00:00:00-05:00"
    elif case == "comparable_datetime_date":
        comparable["stay_start"] = "2026-08-10T00:00:00-05:00"
    elif case == "similarity_factor_float":
        factors = comparable["similarity_factors"]
        assert isinstance(factors, dict)
        factors["capacity"] = 90.0
    else:
        payload["request_id"] = "r" * 129

    with pytest.raises(ValidationError, match=expected_message):
        _request(payload)


def test_point_in_time_contract_requires_aware_ordered_timestamps() -> None:
    payload = _payload()
    comparable = _comparables(payload)[0]
    comparable["known_at"] = "2026-07-09T10:00:00-05:00"
    with pytest.raises(ValidationError, match="known_at cannot precede observed_at"):
        _request(payload)

    payload = _payload()
    payload["as_of"] = "2026-07-09T10:00:00-05:00"
    with pytest.raises(ValidationError, match="observed_at cannot be after as_of"):
        _request(payload)

    payload = _payload()
    payload["as_of"] = datetime(2026, 7, 15, 12, 0).isoformat()
    with pytest.raises(ValidationError, match="explicit UTC offset"):
        _request(payload)


def test_each_request_supports_one_currency_without_conversion() -> None:
    baseline = _recommend()
    payload = _payload()
    payload["currency"] = "EUR"
    for comparable in _comparables(payload):
        comparable["currency"] = "EUR"
    euro = _recommend(payload)

    assert baseline.pricing_band is not None
    assert euro.pricing_band is not None
    assert euro.pricing_band.currency == "EUR"
    assert euro.pricing_band.low == baseline.pricing_band.low
    assert euro.pricing_band.high == baseline.pricing_band.high


@pytest.mark.parametrize("field", ["comparables", "input_lineage"])
def test_comparable_and_lineage_collections_reject_51_items(field: str) -> None:
    payload = _payload()
    comparable_template = deepcopy(_comparables(payload)[0])
    lineage = payload["input_lineage"]
    assert isinstance(lineage, list)
    lineage_template = deepcopy(lineage[0])
    bounded_items: list[dict[str, object]] = []
    for index in range(MAX_COMPARABLES + 1):
        observation_hash = f"{index:064x}"
        if field == "comparables":
            comparable = deepcopy(comparable_template)
            comparable["comparable_id"] = f"cmp-bounded-{index:03d}"
            comparable["observation_hash"] = observation_hash
            bounded_items.append(comparable)
        else:
            entry = deepcopy(lineage_template)
            entry["lineage_id"] = f"lineage-bounded-{index:03d}"
            entry["observation_hash"] = observation_hash
            bounded_items.append(entry)
    payload[field] = bounded_items

    with pytest.raises(ValidationError, match=field):
        _request(payload)


@pytest.mark.parametrize(
    "scenario",
    CATALOG_SCENARIOS,
    ids=[str(item["id"]) for item in CATALOG_SCENARIOS],
)
def test_versioned_synthetic_scenario_catalog_is_materializable(
    scenario: dict[str, object],
) -> None:
    catalog = _scenario_catalog()
    assert catalog["catalog_version"] == "1.0"
    assert catalog["synthetic_only"] is True
    assert "not observations from any real" in str(catalog["disclaimer"])
    payload = _payload()
    _apply_scenario(payload, scenario)

    if scenario["expectation"] == "contract_error":
        with pytest.raises(StatisticalContractError) as captured:
            _recommend(payload)
        assert captured.value.code == scenario["expected_code"]
        return

    response = _recommend(payload)
    assert response.status.value == scenario["expectation"]
    if response.status is RecommendationStatus.ABSTAINED:
        assert response.abstention is not None
        assert response.abstention.reason == scenario["expected_reason"]
    if scenario.get("append_outlier") is True:
        assert [item.comparable_id for item in response.outliers] == ["cmp-outlier"]
    if scenario.get("base_rates_null") is True:
        assert response.evidence_components.missing_base_rate_count == len(_comparables(payload))
    strategy = scenario.get("pricing_strategy")
    if isinstance(strategy, str):
        assert response.pricing_band is not None
        selected_field = {
            "balanced": "central",
            "conservative": "low",
            "premium": "high",
        }[strategy]
        assert response.recommendation == getattr(response.pricing_band, selected_field)


def test_experimental_profile_remains_separate_and_is_not_silently_executed() -> None:
    request = _request().model_copy(
        update={"engine_profile": EngineProfile.PERFORMANCE_AWARE_EXPERIMENTAL}
    )
    response = MarketEvidenceStatisticalV1().recommend(request)
    discovered = {item.profile: item for item in capabilities().profiles}

    assert response.status is RecommendationStatus.ABSTAINED
    assert response.abstention is not None
    assert response.abstention.reason is AbstentionReason.UNSUPPORTED_PROFILE
    assert discovered[EngineProfile.MARKET_EVIDENCE_STATISTICAL_V1].uses_artifacts is False
    assert discovered[EngineProfile.MARKET_EVIDENCE_STATISTICAL_V1].deterministic is True
    assert discovered[EngineProfile.PERFORMANCE_AWARE_EXPERIMENTAL].uses_artifacts is True
    assert discovered[EngineProfile.PERFORMANCE_AWARE_EXPERIMENTAL].deterministic is False


def test_statistical_execution_performs_no_file_network_or_artifact_io(monkeypatch) -> None:
    request = _request()
    from pricing_engine.infrastructure.model_registry import (
        LocalModelBundleStore,
        MLflowModelRegistry,
    )

    def fail_io(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("The statistical profile attempted external or artifact I/O.")

    monkeypatch.setattr("builtins.open", fail_io)
    monkeypatch.setattr(Path, "open", fail_io)
    monkeypatch.setattr("socket.create_connection", fail_io)
    monkeypatch.setattr(joblib, "load", fail_io)
    monkeypatch.setattr(LocalModelBundleStore, "load", fail_io)
    monkeypatch.setattr(MLflowModelRegistry, "load", fail_io)

    response = MarketEvidenceStatisticalV1().recommend(request)

    assert response.status is RecommendationStatus.RECOMMENDED


def test_clean_process_does_not_import_ml_or_artifact_runtimes(tmp_path: Path) -> None:
    script = "\n".join(
        (
            "import json, sys",
            "from pathlib import Path",
            (
                "from pricing_engine.application.statistical_service "
                "import MarketEvidenceStatisticalV1"
            ),
            "from pricing_engine.domain.statistical import StatisticalPricingRequest",
            ("payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))"),
            "request = StatisticalPricingRequest.model_validate(payload)",
            "response = MarketEvidenceStatisticalV1().recommend(request)",
            "prefixes = ('mlflow', 'lightgbm', 'joblib')",
            (
                "loaded = sorted(name for name in sys.modules "
                "if any(name == prefix or name.startswith(prefix + '.') "
                "for prefix in prefixes))"
            ),
            ("print(json.dumps({'status': response.status.value, 'forbidden_modules': loaded}))"),
        )
    )
    clean_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COV_CORE_") and key != "COVERAGE_PROCESS_START"
    }
    completed = subprocess.run(  # noqa: S603 - sys.executable is the trusted test runtime.
        [sys.executable, "-I", "-c", script, str(FIXTURE_PATH)],
        cwd=tmp_path,
        env=clean_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "status": "recommended",
        "forbidden_modules": [],
    }
