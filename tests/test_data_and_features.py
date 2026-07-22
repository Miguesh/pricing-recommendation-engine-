"""Tests for data quality and training-serving feature parity."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pricing_engine.application.training_service import ChronologicalSplitter, TrainDemandModel
from pricing_engine.domain.exceptions import DataContractError
from pricing_engine.domain.models import PriceConstraints, PricingContext
from pricing_engine.infrastructure.data_contracts import validate_training_frame
from pricing_engine.infrastructure.demo_data import generate_demo_observations
from pricing_engine.infrastructure.features import PricingFeatureFactory
from pricing_engine.infrastructure.models.lightgbm_demand import DemandTrainingConfig
from pricing_engine.infrastructure.training import (
    LightGBMDemandModelTrainer,
    PanderaTrainingDataValidator,
)


def test_training_contract_rejects_future_information(demo_observations) -> None:
    invalid = demo_observations.head(2).copy()
    invalid.loc[invalid.index[0], "as_of_date"] = pd.Timestamp("2026-01-01")

    with pytest.raises(DataContractError, match="as_of_date"):
        validate_training_frame(invalid)


def test_training_contract_requires_auditable_label_availability(demo_observations) -> None:
    missing_availability = demo_observations.head(2).drop(columns=["outcome_available_date"])
    with pytest.raises(DataContractError, match="outcome_available_date"):
        validate_training_frame(missing_availability)

    invalid_availability = demo_observations.head(2).copy()
    invalid_availability.loc[invalid_availability.index[0], "outcome_available_date"] = (
        invalid_availability.loc[invalid_availability.index[0], "stay_date"]
    )
    with pytest.raises(DataContractError, match="must be after stay_date"):
        validate_training_frame(invalid_availability)


def test_training_contract_rejects_conflicting_labels_for_one_stay(demo_observations) -> None:
    original = demo_observations.head(1).copy()
    conflicting_snapshot = original.copy()
    conflicting_snapshot["as_of_date"] = conflicting_snapshot["as_of_date"] - pd.Timedelta(days=1)
    conflicting_snapshot["observed_occupancy"] = 1.0 - conflicting_snapshot["observed_occupancy"]

    with pytest.raises(DataContractError, match="share one final outcome"):
        validate_training_frame(pd.concat([original, conflicting_snapshot], ignore_index=True))


def test_training_contract_rejects_invalid_or_mixed_model_currency(demo_observations) -> None:
    normalized = demo_observations.head(2).copy()
    normalized["currency"] = "usd"
    assert validate_training_frame(normalized)["currency"].tolist() == ["USD", "USD"]

    invalid = demo_observations.head(2).copy()
    invalid.loc[invalid.index[0], "currency"] = "US1"
    with pytest.raises(DataContractError, match="schema validation"):
        validate_training_frame(invalid)

    mixed = demo_observations.copy()
    eur_property = str(mixed.loc[mixed.index[-1], "property_id"])
    mixed.loc[mixed["property_id"] == eur_property, "currency"] = "EUR"
    with pytest.raises(DataContractError, match="exactly one normalized currency"):
        TrainDemandModel(
            validator=PanderaTrainingDataValidator(),
            feature_factory=PricingFeatureFactory(),
            model_trainer=LightGBMDemandModelTrainer(),
        ).execute(mixed)


def test_training_is_single_tenant_and_legacy_champion_is_non_comparable(
    demo_observations,
) -> None:
    service = TrainDemandModel(
        validator=PanderaTrainingDataValidator(),
        feature_factory=PricingFeatureFactory(),
        model_trainer=LightGBMDemandModelTrainer(DemandTrainingConfig(n_estimators=10)),
    )
    mixed_tenants = demo_observations.copy()
    mixed_tenants.loc[mixed_tenants.index[-1], "tenant_id"] = "another-tenant"
    with pytest.raises(DataContractError, match="single-tenant"):
        service.execute(mixed_tenants)

    class LegacyChampion:
        version = "legacy-v1"

    outcome = service.execute(demo_observations, benchmark_model=LegacyChampion())

    assert outcome.benchmark_status == "incompatible_contract"
    assert outcome.benchmark_model_version == "legacy-v1"
    assert outcome.benchmark_metrics is None


def test_training_contract_parses_booleans_without_truthiness_coercion(
    demo_observations,
) -> None:
    valid = demo_observations.head(2).copy()
    valid["is_holiday"] = ["False", "true"]

    assert validate_training_frame(valid)["is_holiday"].tolist() == [False, True]

    invalid = demo_observations.head(2).copy()
    invalid["is_holiday"] = invalid["is_holiday"].astype(object)
    invalid.loc[invalid.index[0], "is_holiday"] = "not-a-boolean"
    with pytest.raises(DataContractError, match="accepts only booleans"):
        validate_training_frame(invalid)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("bedrooms", 2.7, "whole numbers"),
        ("accommodates", 4.9, "whole numbers"),
        ("listed_price", float("inf"), "finite"),
        ("competitor_price_median", float("inf"), "finite"),
        ("stay_date", 20250101, "numeric timestamp"),
    ],
)
def test_training_contract_rejects_lossy_or_ambiguous_coercion(
    demo_observations,
    column: str,
    value: object,
    message: str,
) -> None:
    invalid = demo_observations.head(2).copy()
    if column in {"bedrooms", "accommodates"}:
        invalid[column] = invalid[column].astype(float)
    invalid.loc[invalid.index[0], column] = value

    with pytest.raises(DataContractError, match=message):
        validate_training_frame(invalid)


def test_training_contract_canonicalizes_ids_and_discards_external_index(
    demo_observations,
) -> None:
    valid = demo_observations.head(4).copy()
    valid["tenant_id"] = "  demo-tenant  "
    valid.index = [4, 4, 9, 9]

    validated = validate_training_frame(valid)

    assert validated["tenant_id"].unique().tolist() == ["demo-tenant"]
    assert validated.index.tolist() == [0, 1, 2, 3]

    invalid = demo_observations.head(2).copy()
    invalid["property_id"] = "   "
    with pytest.raises(DataContractError, match="non-blank printable"):
        validate_training_frame(invalid)


def test_synthetic_outcomes_preserve_one_coherent_decision_per_stay(
    demo_observations,
) -> None:
    stay_keys = ["tenant_id", "property_id", "stay_date"]
    assert not demo_observations.duplicated(stay_keys).any()
    price_ratio = demo_observations["listed_price"] / demo_observations["competitor_price_median"]
    assert price_ratio.corr(demo_observations["observed_occupancy"]) < -0.30


def test_synthetic_temporal_holdout_has_independent_holiday_evidence(
    demo_observations,
) -> None:
    stay_keys = ["tenant_id", "property_id", "stay_date"]
    baseline = generate_demo_observations(properties=12, decision_days=180, seed=42)
    for observations in (demo_observations, baseline):
        assert not observations.duplicated(stay_keys).any()
        validated = validate_training_frame(observations)
        features = PricingFeatureFactory().build_training_features(validated)
        test_context = ChronologicalSplitter().split(validated, features).test_context
        unique_test_stays = test_context.drop_duplicates(stay_keys)
        holiday_counts = unique_test_stays["is_holiday"].value_counts()

        assert holiday_counts[True] >= 20
        assert holiday_counts[False] >= 20


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
    assert features["location_known"].tolist() == [0, 0]
    assert PricingFeatureFactory.VERSION == "2.0.0"
    assert (
        PricingFeatureFactory.SCHEMA_HASH
        == "7ba3eba5b7667915eed94eeb6ac45812ef361e38162641d1a10c3c74103cef4c"
    )


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("historical_occupancy_7d", 1.01),
        ("booking_pace_7d", 1.01),
        ("event_intensity", 5.01),
    ],
)
def test_training_contract_enforces_shared_signal_ranges(
    demo_observations,
    column: str,
    invalid_value: float,
) -> None:
    invalid = demo_observations.head(2).copy()
    invalid.loc[invalid.index[0], column] = invalid_value

    with pytest.raises(DataContractError, match="schema validation"):
        validate_training_frame(invalid)


def test_location_features_are_optional_but_coordinates_must_be_paired(
    demo_observations,
) -> None:
    without_location = demo_observations.drop(columns=["latitude", "longitude"])
    validated = validate_training_frame(without_location)
    features = PricingFeatureFactory().build_training_features(validated.head(2))

    assert features["location_known"].tolist() == [0, 0]
    assert features["latitude"].tolist() == [0.0, 0.0]

    with pytest.raises(DataContractError, match="both be present"):
        validate_training_frame(demo_observations.drop(columns=["longitude"]))


def test_known_location_is_encoded_without_losing_coordinates() -> None:
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
        latitude=25.7617,
        longitude=-80.1918,
    )

    features = PricingFeatureFactory().for_candidate_prices(context, [125.0])

    assert features.loc[0, "location_known"] == 1
    assert features.loc[0, "latitude"] == pytest.approx(25.7617)
    assert features.loc[0, "longitude"] == pytest.approx(-80.1918)
