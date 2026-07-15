"""Chronological demand-model training use case."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pricing_engine.application.evaluation import EvaluationMetrics, evaluate_demand_predictions
from pricing_engine.infrastructure.data_contracts import validate_training_frame
from pricing_engine.infrastructure.features import PricingFeatureFactory
from pricing_engine.infrastructure.models.lightgbm_demand import (
    DemandTrainingConfig,
    QuantileLightGBMDemandModel,
)


@dataclass(frozen=True, slots=True)
class TemporalPartitions:
    """Strictly ordered feature and target partitions."""

    train_features: pd.DataFrame
    train_target: pd.Series
    calibration_features: pd.DataFrame
    calibration_target: pd.Series
    test_features: pd.DataFrame
    test_target: pd.Series
    test_listed_prices: pd.Series


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Result of a complete, reproducible model-training attempt."""

    model: QuantileLightGBMDemandModel
    metrics: EvaluationMetrics
    train_rows: int
    calibration_rows: int
    test_rows: int
    feature_version: str


class ChronologicalSplitter:
    """Splits by unique decision dates, preventing future rows in training."""

    def __init__(self, train_fraction: float = 0.70, calibration_fraction: float = 0.15) -> None:
        if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
            raise ValueError("Split fractions must be between zero and one.")
        if train_fraction + calibration_fraction >= 1:
            raise ValueError("Train and calibration fractions must leave a test partition.")
        self._train_fraction = train_fraction
        self._calibration_fraction = calibration_fraction

    def split(self, frame: pd.DataFrame, features: pd.DataFrame) -> TemporalPartitions:
        ordered_source = frame.sort_values(["as_of_date", "stay_date", "property_id"])
        ordered_features = features.loc[ordered_source.index].reset_index(drop=True)
        ordered = ordered_source.reset_index(drop=True)
        dates = pd.Index(ordered["as_of_date"].dt.normalize().unique()).sort_values()
        if len(dates) < 10:
            raise ValueError(
                "At least 10 unique as_of_date values are required for temporal splitting."
            )
        train_end = max(1, int(len(dates) * self._train_fraction))
        calibration_end = max(
            train_end + 1, int(len(dates) * (self._train_fraction + self._calibration_fraction))
        )
        if calibration_end >= len(dates):
            calibration_end = len(dates) - 1
        train_dates = dates[:train_end]
        calibration_dates = dates[train_end:calibration_end]
        test_dates = dates[calibration_end:]
        if not len(calibration_dates) or not len(test_dates):
            raise ValueError("Temporal split did not create calibration and test windows.")

        train_mask = ordered["as_of_date"].dt.normalize().isin(train_dates)
        calibration_mask = ordered["as_of_date"].dt.normalize().isin(calibration_dates)
        test_mask = ordered["as_of_date"].dt.normalize().isin(test_dates)
        target = ordered["observed_occupancy"].astype(float)
        return TemporalPartitions(
            train_features=ordered_features.loc[train_mask].reset_index(drop=True),
            train_target=target.loc[train_mask].reset_index(drop=True),
            calibration_features=ordered_features.loc[calibration_mask].reset_index(drop=True),
            calibration_target=target.loc[calibration_mask].reset_index(drop=True),
            test_features=ordered_features.loc[test_mask].reset_index(drop=True),
            test_target=target.loc[test_mask].reset_index(drop=True),
            test_listed_prices=ordered.loc[test_mask, "listed_price"]
            .astype(float)
            .reset_index(drop=True),
        )


class TrainDemandModel:
    """Trains and evaluates a versioned demand model without registry side effects."""

    def __init__(
        self,
        *,
        feature_factory: PricingFeatureFactory | None = None,
        splitter: ChronologicalSplitter | None = None,
        training_config: DemandTrainingConfig | None = None,
    ) -> None:
        self._feature_factory = feature_factory or PricingFeatureFactory()
        self._splitter = splitter or ChronologicalSplitter()
        self._training_config = training_config or DemandTrainingConfig()

    def execute(self, raw_frame: pd.DataFrame) -> TrainingOutcome:
        validated = validate_training_frame(raw_frame)
        features = self._feature_factory.build_training_features(validated)
        partitions = self._splitter.split(validated, features)
        model = QuantileLightGBMDemandModel.fit(
            train_features=partitions.train_features,
            train_target=partitions.train_target,
            calibration_features=partitions.calibration_features,
            calibration_target=partitions.calibration_target,
            config=self._training_config,
        )
        estimates = model.predict(partitions.test_features)
        metrics = evaluate_demand_predictions(
            actual_occupancy=partitions.test_target.tolist(),
            estimates=estimates,
            listed_prices=partitions.test_listed_prices.tolist(),
        )
        return TrainingOutcome(
            model=model,
            metrics=metrics,
            train_rows=len(partitions.train_features),
            calibration_rows=len(partitions.calibration_features),
            test_rows=len(partitions.test_features),
            feature_version=self._feature_factory.VERSION,
        )
