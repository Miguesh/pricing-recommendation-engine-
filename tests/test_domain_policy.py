"""Unit tests for deterministic commercial pricing rules."""

from decimal import Decimal

import pytest

from pricing_engine.domain.exceptions import InvalidPricingRequestError
from pricing_engine.domain.models import DemandEstimate, PriceConstraints
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
