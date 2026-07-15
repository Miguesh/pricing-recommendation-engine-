"""Tests for data quality and training-serving feature parity."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pricing_engine.domain.exceptions import DataContractError
from pricing_engine.domain.models import PriceConstraints, PricingContext
from pricing_engine.infrastructure.data_contracts import validate_training_frame
from pricing_engine.infrastructure.features import PricingFeatureFactory


def test_training_contract_rejects_future_information(demo_observations) -> None:
    invalid = demo_observations.head(2).copy()
    invalid.loc[invalid.index[0], "as_of_date"] = pd.Timestamp("2026-01-01")

    with pytest.raises(DataContractError, match="as_of_date"):
        validate_training_frame(invalid)


def test_candidate_features_match_the_versioned_contract() -> None:
    context = PricingContext(
        tenant_id="tenant-a",
        property_id="property-a",
        stay_date=date(2026, 7, 24),
        as_of_date=date(2026, 7, 14),
        currency="USD",
        current_price=Decimal("125.00"),
        historical_occupancy_7d=0.72,
        booking_pace_7d=0.35,
        competitor_price_median=Decimal("130.00"),
        bedrooms=2,
        accommodates=4,
        review_score=4.7,
        constraints=PriceConstraints(
            min_price=Decimal("90.00"),
            max_price=Decimal("170.00"),
        ),
    )

    features = PricingFeatureFactory().for_candidate_prices(context, [100.0, 130.0])

    assert tuple(features.columns) == PricingFeatureFactory.FEATURE_COLUMNS
    assert features.loc[1, "price_to_competitor_ratio"] == pytest.approx(1.0)
    assert features.loc[0, "lead_time_days"] == 10
