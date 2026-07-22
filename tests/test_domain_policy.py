"""Unit tests for deterministic commercial pricing rules."""

from datetime import date
from decimal import Decimal

import pytest

from pricing_engine.domain.exceptions import InvalidPricingRequestError
from pricing_engine.domain.models import DemandEstimate, PriceConstraints, PricingContext
from pricing_engine.domain.policies import CandidatePrice, PricingPolicy


def _estimate(occupancy: float) -> DemandEstimate:
    return DemandEstimate(
        expected_occupancy=occupancy,
        lower_occupancy=max(0, occupancy - 0.1),
        upper_occupancy=min(1, occupancy + 0.1),
        in_distribution_score=0.9,
    )


def test_feasible_prices_enforce_change_guardrail() -> None:
    prices = PricingPolicy().feasible_prices(
        current_price=Decimal("100.00"),
        constraints=PriceConstraints(
            min_price=Decimal("50.00"),
            max_price=Decimal("200.00"),
            price_increment=Decimal("5.00"),
            max_price_change_pct=Decimal("0.10"),
        ),
        maximum_candidates=20,
    )

    assert prices == (
        Decimal("90.00"),
        Decimal("95.00"),
        Decimal("100.00"),
        Decimal("105.00"),
        Decimal("110.00"),
    )


def test_minimum_occupancy_is_an_enforced_business_constraint() -> None:
    selected = PricingPolicy().select(
        [
            CandidatePrice(Decimal("100.00"), _estimate(0.45)),
            CandidatePrice(Decimal("110.00"), _estimate(0.61)),
        ],
        min_expected_occupancy=0.50,
    )

    assert selected.price == Decimal("110.00")


def test_selection_rejects_when_no_price_meets_occupancy_constraint() -> None:
    with pytest.raises(InvalidPricingRequestError, match="occupancy constraints"):
        PricingPolicy().select(
            [CandidatePrice(Decimal("100.00"), _estimate(0.40))],
            min_expected_occupancy=0.50,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("historical_occupancy_7d", 1.1),
        ("booking_pace_7d", 1.1),
        ("event_intensity", 5.1),
    ],
)
def test_pricing_context_enforces_shared_signal_ranges(field: str, value: float) -> None:
    values = {
        "tenant_id": "tenant-a",
        "property_id": "property-a",
        "stay_date": date(2026, 8, 1),
        "as_of_date": date(2026, 7, 18),
        "currency": "USD",
        "current_price": Decimal("100"),
        "historical_occupancy_7d": 0.5,
        "booking_pace_7d": 0.4,
        "competitor_price_median": Decimal("105"),
        "bedrooms": 2,
        "accommodates": 4,
        "review_score": 4.5,
        "event_intensity": 0.0,
    }
    values[field] = value

    with pytest.raises(InvalidPricingRequestError, match=r"\[0, [15]\]"):
        PricingContext(**values)


def test_pricing_context_requires_paired_valid_coordinates() -> None:
    with pytest.raises(InvalidPricingRequestError, match="both be provided"):
        PricingContext(
            tenant_id="tenant-a",
            property_id="property-a",
            stay_date=date(2026, 8, 1),
            as_of_date=date(2026, 7, 18),
            currency="USD",
            current_price=Decimal("100"),
            historical_occupancy_7d=0.5,
            booking_pace_7d=0.4,
            competitor_price_median=Decimal("105"),
            bedrooms=2,
            accommodates=4,
            review_score=4.5,
            latitude=25.7,
        )


def test_domain_constraints_reject_excessive_price_change() -> None:
    with pytest.raises(InvalidPricingRequestError, match=r"between 0 and 0\.35"):
        PriceConstraints(
            min_price=Decimal("50"),
            max_price=Decimal("200"),
            max_price_change_pct=Decimal("0.36"),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_price": Decimal("NaN")},
        {"min_price": Decimal("0")},
        {"min_price": Decimal("201")},
        {"price_increment": Decimal("0")},
        {"max_price_change_pct": Decimal("NaN")},
        {"max_price_change_pct": Decimal("-0.01")},
        {"max_price_change_pct": None},
        {"min_expected_occupancy": 1.01},
    ],
)
def test_price_constraints_reject_invalid_commercial_bounds(overrides) -> None:
    values = {
        "min_price": Decimal("50"),
        "max_price": Decimal("200"),
        "price_increment": Decimal("1"),
        "max_price_change_pct": Decimal("0.35"),
        "min_expected_occupancy": 0.5,
    }
    values.update(overrides)

    with pytest.raises(InvalidPricingRequestError):
        PriceConstraints(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": " tenant-a"},
        {"as_of_date": date(2026, 8, 2)},
        {"currency": "usd"},
        {"current_price": Decimal("NaN")},
        {"competitor_price_median": Decimal("0")},
        {"historical_occupancy_7d": -0.01},
        {"review_score": 5.01},
        {"bedrooms": True},
        {"is_holiday": 1},
        {"latitude": 91.0, "longitude": 0.0},
        {"latitude": 0.0, "longitude": float("inf")},
    ],
)
def test_pricing_context_rejects_invalid_domain_state(overrides) -> None:
    values = {
        "tenant_id": "tenant-a",
        "property_id": "property-a",
        "stay_date": date(2026, 8, 1),
        "as_of_date": date(2026, 7, 18),
        "currency": "USD",
        "current_price": Decimal("100"),
        "historical_occupancy_7d": 0.5,
        "booking_pace_7d": 0.4,
        "competitor_price_median": Decimal("105"),
        "bedrooms": 2,
        "accommodates": 4,
        "review_score": 4.5,
        "is_holiday": False,
        "latitude": None,
        "longitude": None,
    }
    values.update(overrides)

    with pytest.raises(InvalidPricingRequestError):
        PricingContext(**values)


@pytest.mark.parametrize(
    "estimate",
    [
        (0.5, 0.6, 0.8, 0.9),
        (0.5, 0.4, 0.8, 1.1),
    ],
)
def test_demand_estimate_rejects_invalid_probability_state(estimate) -> None:
    with pytest.raises(InvalidPricingRequestError):
        DemandEstimate(*estimate)
