"""Shared deterministic fixtures for the pricing engine test suite."""

from __future__ import annotations

import pytest

from pricing_engine.application.training_service import TrainDemandModel, TrainingOutcome
from pricing_engine.infrastructure.demo_data import generate_demo_observations
from pricing_engine.infrastructure.features import PricingFeatureFactory
from pricing_engine.infrastructure.models.lightgbm_demand import DemandTrainingConfig
from pricing_engine.infrastructure.training import (
    LightGBMDemandModelTrainer,
    PanderaTrainingDataValidator,
)


@pytest.fixture(scope="session")
def demo_observations():
    """A sufficiently long synthetic history for temporal model tests."""

    return generate_demo_observations(properties=6, decision_days=180, seed=7)


@pytest.fixture(scope="session")
def training_outcome(demo_observations) -> TrainingOutcome:
    """One trained bundle shared across API and model-level tests."""

    return TrainDemandModel(
        validator=PanderaTrainingDataValidator(),
        feature_factory=PricingFeatureFactory(),
        model_trainer=LightGBMDemandModelTrainer(
            DemandTrainingConfig(
                n_estimators=50,
                min_child_samples=10,
                random_state=7,
            )
        ),
    ).execute(demo_observations)
