"""Concrete offline-training adapters for application-layer ports."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import pandas as pd

from pricing_engine.application.ports import DemandPredictor
from pricing_engine.infrastructure.data_contracts import validate_training_frame
from pricing_engine.infrastructure.models.lightgbm_demand import (
    DemandTrainingConfig,
    QuantileLightGBMDemandModel,
)


class PanderaTrainingDataValidator:
    """Pandera-backed implementation of the offline data-validation port."""

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        return validate_training_frame(frame)


class LightGBMDemandModelTrainer:
    """LightGBM implementation of the calibrated demand-model trainer port."""

    def __init__(self, config: DemandTrainingConfig | None = None) -> None:
        self._config = config or DemandTrainingConfig()

    def fit(
        self,
        *,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        train_sample_weight: Sequence[float],
        calibration_features: pd.DataFrame,
        calibration_target: pd.Series,
        calibration_group_keys: Sequence[Hashable],
        feature_contract_version: str,
        feature_schema_hash: str,
        currency: str,
        tenant_ids: Sequence[str],
    ) -> DemandPredictor:
        return QuantileLightGBMDemandModel.fit(
            train_features=train_features,
            train_target=train_target,
            train_sample_weight=train_sample_weight,
            calibration_features=calibration_features,
            calibration_target=calibration_target,
            calibration_group_keys=calibration_group_keys,
            feature_contract_version=feature_contract_version,
            feature_schema_hash=feature_schema_hash,
            currency=currency,
            tenant_ids=tenant_ids,
            config=self._config,
        )
