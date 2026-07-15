"""Tests for calibrated LightGBM training and portable bundles."""

from pathlib import Path

from pricing_engine.application.evaluation import EvaluationMetrics, PromotionCriteria
from pricing_engine.infrastructure.features import PricingFeatureFactory
from pricing_engine.infrastructure.model_registry import LocalModelBundleStore


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


def test_local_bundle_round_trip(training_outcome, tmp_path: Path) -> None:
    store = LocalModelBundleStore()
    path = store.save_local(training_outcome.model, tmp_path)
    loaded = store.load(str(path))

    assert loaded.version == training_outcome.model.version
    assert path.exists()


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
