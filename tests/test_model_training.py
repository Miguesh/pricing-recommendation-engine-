"""Tests for calibrated LightGBM training and portable bundles."""

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pricing_engine.application.evaluation import (
    EvaluationMetrics,
    PromotionCriteria,
    SliceEvaluationMetrics,
    evaluate_demand_predictions,
)
from pricing_engine.application.training_service import (
    ChronologicalSplitter,
    TrainDemandModel,
    _equal_stay_training_weights,
)
from pricing_engine.domain.exceptions import PromotionRejectedError
from pricing_engine.domain.models import DemandEstimate, PricingContext
from pricing_engine.infrastructure.data_contracts import validate_training_frame
from pricing_engine.infrastructure.features import PricingFeatureFactory
from pricing_engine.infrastructure.model_registry import LocalModelBundleStore
from pricing_engine.infrastructure.models.lightgbm_demand import (
    DemandTrainingConfig,
    QuantileLightGBMDemandModel,
)


def test_trained_model_emits_bounded_calibrated_estimates(
    training_outcome, demo_observations
) -> None:
    features = PricingFeatureFactory().build_training_features(demo_observations.head(3))
    estimates = training_outcome.model.predict(features)

    assert len(estimates) == 3
    assert all(
        0
        <= estimate.lower_occupancy
        <= estimate.expected_occupancy
        <= estimate.upper_occupancy
        <= 1
        for estimate in estimates
    )
    assert training_outcome.metrics.sample_size > 50
    assert 0 <= training_outcome.metrics.interval_coverage <= 1
    assert training_outcome.metrics.expected_revenue_wape is not None
    assert training_outcome.metrics.unique_stays is not None
    assert training_outcome.metrics.unique_stays >= 50
    assert training_outcome.metrics.slices
    assert any(key.startswith("slice.") for key in training_outcome.metrics.as_dict())


def test_default_demo_candidate_satisfies_every_promotion_gate(training_outcome) -> None:
    PromotionCriteria().evaluate(training_outcome.metrics)

    slices = {item.name: item for item in training_outcome.metrics.slices}
    assert slices["holiday/yes"].unique_stays >= 20
    assert slices["holiday/no"].unique_stays >= 20


def test_evaluation_evidence_round_trips_through_registry_metric_mapping() -> None:
    original = _policy_metrics()
    original = replace(
        original,
        slices=(
            SliceEvaluationMetrics(
                name="lead_time/0_7_days",
                occupancy_mae=0.08,
                interval_coverage=0.90,
                expected_revenue_wape=0.09,
                sample_size=30,
                unique_stays=30,
                mean_price_response=0.08,
                flat_price_response_rate=0.10,
                upper_price_boundary_rate=0.05,
                lower_price_boundary_rate=0.10,
                any_price_boundary_rate=0.15,
                mean_selected_price_change_pct=-0.03,
            ),
        ),
    )

    restored = EvaluationMetrics.from_mapping(original.as_dict())

    assert restored == original


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_estimators": 0},
        {"quantile_n_estimators": 0},
        {"interval_coverage": 1.0},
        {"subsample": 0.0},
        {"subsample_freq": -1},
        {"inference_num_threads": 0},
        {"training_num_threads": 0},
    ],
)
def test_training_configuration_rejects_invalid_capacity(overrides) -> None:
    with pytest.raises(ValueError):
        DemandTrainingConfig(**overrides)


def test_repeated_snapshots_cannot_overwhelm_independent_stay_errors() -> None:
    estimates = [DemandEstimate(0.0, 0.0, 0.0, 1.0) for _ in range(1_050)]
    metrics = evaluate_demand_predictions(
        actual_occupancy=([0.0] * 1_000) + ([1.0] * 50),
        estimates=estimates,
        listed_prices=[100.0] * 1_050,
        group_keys=([("tenant", "perfect-stay")] * 1_000)
        + [("tenant", f"failed-stay-{index}") for index in range(50)],
        price_response_deltas=[0.1] * 1_050,
        selected_price_change_pcts=[0.0] * 1_050,
    )

    assert metrics.unique_stays == 51
    assert metrics.occupancy_mae == pytest.approx(50 / 51)
    with pytest.raises(PromotionRejectedError, match="occupancy MAE"):
        PromotionCriteria(required_slices=()).evaluate(metrics)


def test_training_weights_give_every_stay_equal_total_influence() -> None:
    context = pd.DataFrame(
        {
            "tenant_id": ["tenant"] * 4,
            "property_id": ["property-a"] * 3 + ["property-b"],
            "stay_date": [date(2026, 8, 1)] * 3 + [date(2026, 8, 2)],
        }
    )

    weights = _equal_stay_training_weights(context)
    weighted = context.assign(sample_weight=weights)
    total_by_stay = weighted.groupby(
        ["tenant_id", "property_id", "stay_date"],
        sort=False,
    )["sample_weight"].sum()

    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0])
    assert total_by_stay.tolist() == pytest.approx([1.0, 1.0])


def test_slice_behavior_uses_the_same_equal_stay_weighting_as_aggregate() -> None:
    metrics = evaluate_demand_predictions(
        actual_occupancy=[0.5, 0.5, 0.5],
        estimates=[DemandEstimate(0.5, 0.4, 0.6, 1.0)] * 3,
        listed_prices=[100.0] * 3,
        group_keys=["stay-a", "stay-a", "stay-b"],
        slice_memberships={"holiday/yes": [True, True, True]},
        price_response_deltas=[0.0, 0.0, 1.0],
        selected_price_change_pcts=[-0.35, -0.35, 0.35],
    )

    holiday = metrics.slices[0]
    assert metrics.mean_price_response == pytest.approx(0.5)
    assert holiday.mean_price_response == pytest.approx(0.5)
    assert holiday.upper_price_boundary_rate == pytest.approx(0.5)
    assert holiday.lower_price_boundary_rate == pytest.approx(0.5)


@pytest.mark.parametrize(
    "sample_weight",
    ([1.0], [1.0, 0.0], [1.0, float("nan")]),
)
def test_model_fit_rejects_invalid_training_sample_weights(sample_weight) -> None:
    features = pd.DataFrame({"candidate_price": [100.0, 110.0]})
    target = pd.Series([0.7, 0.6])

    with pytest.raises(ValueError, match="sample weights"):
        QuantileLightGBMDemandModel.fit(
            train_features=features,
            train_target=target,
            train_sample_weight=sample_weight,
            calibration_features=features,
            calibration_target=target,
            calibration_group_keys=("stay-a", "stay-b"),
            feature_contract_version="test-contract",
            feature_schema_hash="test-hash",
            currency="USD",
            tenant_ids=("tenant-a",),
            config=DemandTrainingConfig(n_estimators=1, quantile_n_estimators=1),
        )


def test_interval_coverage_requires_every_snapshot_in_a_stay_to_be_covered() -> None:
    metrics = evaluate_demand_predictions(
        actual_occupancy=[0.5, 0.5, 0.5],
        estimates=[
            DemandEstimate(0.5, 0.0, 1.0, 1.0),
            DemandEstimate(0.2, 0.0, 0.4, 1.0),
            DemandEstimate(0.5, 0.0, 1.0, 1.0),
        ],
        listed_prices=[100.0, 100.0, 100.0],
        group_keys=["stay-a", "stay-a", "stay-b"],
    )

    assert metrics.interval_coverage == 0.5


def test_purged_split_prevents_label_and_stay_group_leakage(demo_observations) -> None:
    validated = validate_training_frame(demo_observations)
    features = PricingFeatureFactory().build_training_features(validated)
    partitions = ChronologicalSplitter().split(validated, features)

    assert partitions.train_context["outcome_available_date"].max() < partitions.calibration_start
    assert partitions.calibration_context["outcome_available_date"].max() < partitions.test_start
    assert partitions.purged_train_rows > 0
    assert partitions.purged_calibration_rows > 0

    group_columns = ["tenant_id", "property_id", "stay_date"]

    def groups(frame: pd.DataFrame) -> set[tuple[object, ...]]:
        return set(frame[group_columns].itertuples(index=False, name=None))

    train_groups = groups(partitions.train_context)
    calibration_groups = groups(partitions.calibration_context)
    test_groups = groups(partitions.test_context)
    assert train_groups.isdisjoint(calibration_groups)
    assert train_groups.isdisjoint(test_groups)
    assert calibration_groups.isdisjoint(test_groups)


def test_valid_crossing_snapshots_are_purged_before_stay_isolation(
    demo_observations,
) -> None:
    dates = pd.Index(
        pd.to_datetime(demo_observations["as_of_date"]).dt.normalize().unique()
    ).sort_values()
    train_end = max(1, int(len(dates) * 0.70))
    calibration_end = max(train_end + 1, int(len(dates) * 0.85))
    calibration_as_of = pd.Timestamp(dates[calibration_end - 1])
    test_as_of = pd.Timestamp(dates[calibration_end])
    stay_date = test_as_of + pd.Timedelta(days=1)
    outcome_available_date = stay_date + pd.Timedelta(days=1)
    crossing = pd.concat([demo_observations.head(1)] * 2, ignore_index=True)
    crossing["property_id"] = "crossing-property"
    crossing["as_of_date"] = [calibration_as_of, test_as_of]
    crossing["stay_date"] = stay_date
    crossing["outcome_available_date"] = outcome_available_date
    augmented = pd.concat([demo_observations, crossing], ignore_index=True)
    validated = validate_training_frame(augmented)
    features = PricingFeatureFactory().build_training_features(validated)

    partitions = ChronologicalSplitter().split(validated, features)

    group_columns = ["tenant_id", "property_id", "stay_date"]
    crossing_key = tuple(
        validated.loc[validated["property_id"] == "crossing-property", group_columns].iloc[0]
    )

    def groups(frame: pd.DataFrame) -> set[tuple[object, ...]]:
        return set(frame[group_columns].itertuples(index=False, name=None))

    assert crossing_key not in groups(partitions.train_context)
    assert crossing_key not in groups(partitions.calibration_context)
    assert crossing_key in groups(partitions.test_context)


def test_temporal_split_rejects_invalid_embargo_and_insufficient_evidence(
    demo_observations,
) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        ChronologicalSplitter(embargo_days=0.5)  # type: ignore[arg-type]

    validated = validate_training_frame(demo_observations.head(60))
    features = PricingFeatureFactory().build_training_features(validated)
    with pytest.raises(ValueError, match=r"Insufficient evidence|empty partition"):
        ChronologicalSplitter().split(validated, features)


def test_local_bundle_round_trip(training_outcome, tmp_path: Path) -> None:
    store = LocalModelBundleStore()
    feature_row = PricingFeatureFactory().build_training_features(
        pd.DataFrame(
            {
                "stay_date": [date(2026, 8, 1)],
                "as_of_date": [date(2026, 7, 18)],
                "listed_price": [125.0],
                "historical_occupancy_7d": [0.7],
                "booking_pace_7d": [0.4],
                "competitor_price_median": [130.0],
                "bedrooms": [2],
                "accommodates": [4],
                "review_score": [4.7],
                "is_holiday": [False],
                "event_intensity": [0.0],
            }
        )
    )
    training_outcome.model.explain(feature_row, top_k=3)
    path = store.save_local(training_outcome.model, tmp_path)
    loaded = store.load(str(path))

    assert loaded.version == training_outcome.model.version
    assert path.exists()
    assert loaded.explain(feature_row, top_k=3)


def test_promotion_policy_rejects_insufficient_evidence() -> None:
    insufficient = EvaluationMetrics(
        occupancy_mae=0.35,
        occupancy_rmse=0.40,
        occupancy_r2=0.1,
        interval_coverage=0.50,
        mean_interval_width=0.9,
        expected_revenue_mae=20,
        sample_size=10,
    )

    try:
        PromotionCriteria().evaluate(insufficient)
    except Exception as error:
        assert "occupancy MAE" in str(error)
    else:
        raise AssertionError("An underperforming candidate must not pass promotion gates.")


def test_expected_demand_is_non_increasing_across_candidate_price_grid(
    training_outcome,
) -> None:
    context = PricingContext(
        tenant_id="tenant-a",
        property_id="property-a",
        stay_date=date(2026, 8, 15),
        as_of_date=date(2026, 7, 18),
        currency="USD",
        current_price=Decimal("135"),
        historical_occupancy_7d=0.70,
        booking_pace_7d=0.45,
        competitor_price_median=Decimal("140"),
        bedrooms=2,
        accommodates=4,
        review_score=4.6,
        latitude=25.7617,
        longitude=-80.1918,
    )
    candidate_prices = np.linspace(60.0, 240.0, 181)
    features = PricingFeatureFactory().for_candidate_prices(context, candidate_prices)

    expected = np.asarray(
        [item.expected_occupancy for item in training_outcome.model.predict(features)]
    )

    assert np.all(np.diff(expected) <= 1e-12)
    constraints = training_outcome.model.median_model.get_params()["monotone_constraints"]
    assert constraints[PricingFeatureFactory.FEATURE_COLUMNS.index("candidate_price")] == -1
    assert (
        constraints[PricingFeatureFactory.FEATURE_COLUMNS.index("price_to_competitor_ratio")] == -1
    )


def test_central_and_quantile_model_capacities_are_independently_versioned(
    training_outcome,
) -> None:
    config = training_outcome.model.config

    assert training_outcome.model.median_model.n_estimators == config.n_estimators
    assert training_outcome.model.lower_model.n_estimators == config.quantile_n_estimators
    assert training_outcome.model.upper_model.n_estimators == config.quantile_n_estimators


def test_feature_support_score_penalizes_extreme_extrapolation(
    training_outcome,
    demo_observations,
) -> None:
    factory = PricingFeatureFactory()
    in_distribution = factory.build_training_features(demo_observations.head(1))
    extreme = in_distribution.copy()
    extreme.loc[0, "candidate_price"] = 100_000.0
    extreme.loc[0, "price_to_competitor_ratio"] = 1_000.0

    inside_score = training_outcome.model.predict(in_distribution)[0].in_distribution_score
    extreme_score = training_outcome.model.predict(extreme)[0].in_distribution_score

    assert 0 <= extreme_score < inside_score <= 1


def test_promotion_rejects_excessive_revenue_error() -> None:
    candidate = EvaluationMetrics(
        occupancy_mae=0.12,
        occupancy_rmse=0.14,
        occupancy_r2=0.75,
        interval_coverage=0.90,
        mean_interval_width=0.25,
        expected_revenue_mae=15,
        sample_size=100,
        expected_revenue_wape=0.40,
    )

    with pytest.raises(PromotionRejectedError, match="revenue WAPE"):
        PromotionCriteria().evaluate(candidate)


def test_promotion_rejects_material_champion_regression() -> None:
    champion = EvaluationMetrics(
        occupancy_mae=0.10,
        occupancy_rmse=0.12,
        occupancy_r2=0.8,
        interval_coverage=0.90,
        mean_interval_width=0.25,
        expected_revenue_mae=10,
        sample_size=100,
        expected_revenue_wape=0.10,
    )
    candidate = EvaluationMetrics(
        occupancy_mae=0.11,
        occupancy_rmse=0.13,
        occupancy_r2=0.78,
        interval_coverage=0.90,
        mean_interval_width=0.25,
        expected_revenue_mae=11,
        sample_size=100,
        expected_revenue_wape=0.11,
    )

    with pytest.raises(PromotionRejectedError, match="versus champion"):
        PromotionCriteria().evaluate(candidate, champion_metrics=champion)


def _policy_metrics(
    *,
    lower_boundary_rate: float = 0.0,
    upper_boundary_rate: float = 0.0,
    mean_selected_price_change_pct: float = 0.05,
) -> EvaluationMetrics:
    return EvaluationMetrics(
        occupancy_mae=0.10,
        occupancy_rmse=0.12,
        occupancy_r2=0.80,
        interval_coverage=0.90,
        mean_interval_width=0.25,
        expected_revenue_mae=10.0,
        sample_size=100,
        unique_stays=100,
        expected_revenue_wape=0.10,
        mean_price_response=0.10,
        flat_price_response_rate=0.0,
        upper_price_boundary_rate=upper_boundary_rate,
        lower_price_boundary_rate=lower_boundary_rate,
        any_price_boundary_rate=lower_boundary_rate + upper_boundary_rate,
        mean_selected_price_change_pct=mean_selected_price_change_pct,
    )


def _healthy_slice(name: str) -> SliceEvaluationMetrics:
    return SliceEvaluationMetrics(
        name=name,
        occupancy_mae=0.10,
        interval_coverage=0.90,
        expected_revenue_wape=0.10,
        sample_size=30,
        unique_stays=30,
        mean_price_response=0.10,
        flat_price_response_rate=0.0,
        upper_price_boundary_rate=0.0,
        lower_price_boundary_rate=0.0,
        any_price_boundary_rate=0.0,
        mean_selected_price_change_pct=0.05,
    )


@pytest.mark.parametrize("location_slice", ["location/known", "location/unknown"])
def test_location_promotion_slice_is_conditional_on_observed_cohort(
    location_slice: str,
) -> None:
    criteria = PromotionCriteria()
    assert "location/known" not in criteria.required_slices
    assert "location/unknown" not in criteria.required_slices
    metrics = replace(
        _policy_metrics(),
        slices=tuple(_healthy_slice(name) for name in (*criteria.required_slices, location_slice)),
    )

    criteria.evaluate(metrics)


@pytest.mark.parametrize(
    ("latitude", "longitude", "expected_slice"),
    [(25.7617, -80.1918, "location/known"), (None, None, "location/unknown")],
)
def test_location_slices_emit_only_non_empty_observed_cohorts(
    latitude: float | None,
    longitude: float | None,
    expected_slice: str,
) -> None:
    frame = pd.DataFrame(
        {
            "stay_date": [date(2026, 8, 15)],
            "as_of_date": [date(2026, 7, 18)],
            "is_holiday": [False],
            "event_intensity": [0.0],
            "latitude": [latitude],
            "longitude": [longitude],
        }
    )

    memberships = TrainDemandModel._build_slice_memberships(frame)

    assert expected_slice in memberships
    absent_slice = "location/unknown" if expected_slice == "location/known" else "location/known"
    assert absent_slice not in memberships


def test_promotion_rejects_policy_saturated_at_lower_boundary() -> None:
    with pytest.raises(PromotionRejectedError, match="lower price-boundary rate"):
        PromotionCriteria(required_slices=()).evaluate(
            _policy_metrics(
                lower_boundary_rate=1.0,
                mean_selected_price_change_pct=-0.25,
            )
        )


def test_promotion_rejects_material_policy_shift_from_champion() -> None:
    champion = _policy_metrics(mean_selected_price_change_pct=-0.20)
    candidate = _policy_metrics(mean_selected_price_change_pct=0.20)

    with pytest.raises(
        PromotionRejectedError,
        match=r"mean_selected_price_change_pct.*versus champion",
    ):
        PromotionCriteria(required_slices=()).evaluate(
            candidate,
            champion_metrics=champion,
        )


def test_promotion_rejects_weak_high_volume_slice() -> None:
    metrics = EvaluationMetrics(
        occupancy_mae=0.10,
        occupancy_rmse=0.12,
        occupancy_r2=0.8,
        interval_coverage=0.90,
        mean_interval_width=0.25,
        expected_revenue_mae=10,
        sample_size=100,
        expected_revenue_wape=0.10,
        slices=(
            SliceEvaluationMetrics(
                name="lead_time/0_7_days",
                occupancy_mae=0.40,
                interval_coverage=0.70,
                expected_revenue_wape=0.20,
                sample_size=30,
            ),
        ),
    )

    with pytest.raises(PromotionRejectedError, match="slice lead_time/0_7_days"):
        PromotionCriteria().evaluate(metrics)


def test_promotion_rejects_economically_degenerate_required_slice() -> None:
    metrics = replace(
        _policy_metrics(),
        slices=(
            SliceEvaluationMetrics(
                name="holiday/yes",
                occupancy_mae=0.10,
                interval_coverage=0.90,
                expected_revenue_wape=0.10,
                sample_size=30,
                unique_stays=30,
                mean_price_response=0.10,
                flat_price_response_rate=0.0,
                upper_price_boundary_rate=0.0,
                lower_price_boundary_rate=1.0,
                any_price_boundary_rate=1.0,
                mean_selected_price_change_pct=-0.35,
            ),
        ),
    )

    with pytest.raises(
        PromotionRejectedError,
        match=r"slice holiday/yes lower price-boundary rate",
    ):
        PromotionCriteria(required_slices=("holiday/yes",)).evaluate(metrics)


class _ConstantPredictor:
    def __init__(self, value: float) -> None:
        self.value = value
        self.feature_importances_ = np.asarray([1.0])

    def predict(self, features: pd.DataFrame, **kwargs) -> np.ndarray:
        return np.full(len(features), self.value)


def test_crossing_quantiles_are_repaired_around_the_central_estimate() -> None:
    model = QuantileLightGBMDemandModel(
        config=DemandTrainingConfig(n_estimators=1),
        feature_names=("candidate_price",),
        lower_model=_ConstantPredictor(0.9),
        median_model=_ConstantPredictor(0.5),
        upper_model=_ConstantPredictor(0.1),
        calibration_radius=0.0,
        feature_means={"candidate_price": 100.0},
        feature_stds={"candidate_price": 10.0},
        model_version="crossing-test",
        feature_contract_version="test-contract",
        feature_schema_hash="test-hash",
        currency="USD",
        tenant_ids=("tenant-a",),
    )

    estimate = model.predict(pd.DataFrame({"candidate_price": [100.0]}))[0]

    assert estimate.lower_occupancy == estimate.expected_occupancy == 0.5
    assert estimate.upper_occupancy == estimate.expected_occupancy
