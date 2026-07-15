"""Retraining orchestration with quality gates and candidate-only registration."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from pricing_engine.application.evaluation import PromotionCriteria
from pricing_engine.application.training_service import TrainDemandModel, TrainingOutcome
from pricing_engine.infrastructure.model_registry import MLflowModelRegistry, RegisteredModel


@dataclass(frozen=True, slots=True)
class RetrainingResult:
    """Auditable result of a retraining attempt."""

    outcome: TrainingOutcome
    registered_model: RegisteredModel
    dataset_fingerprint: str


class RetrainingPipeline:
    """Runs validation, training, promotion gates, and candidate registration."""

    def __init__(
        self,
        *,
        trainer: TrainDemandModel,
        registry: MLflowModelRegistry,
        promotion_criteria: PromotionCriteria | None = None,
    ) -> None:
        self._trainer = trainer
        self._registry = registry
        self._promotion_criteria = promotion_criteria or PromotionCriteria()

    def execute(self, observations: pd.DataFrame) -> RetrainingResult:
        outcome = self._trainer.execute(observations)
        self._promotion_criteria.evaluate(outcome.metrics)
        fingerprint = self._fingerprint(observations)
        registered = self._registry.log_candidate(
            model=outcome.model,
            metrics=outcome.metrics,
            feature_version=outcome.feature_version,
            dataset_fingerprint=fingerprint,
        )
        return RetrainingResult(
            outcome=outcome,
            registered_model=registered,
            dataset_fingerprint=fingerprint,
        )

    @staticmethod
    def _fingerprint(observations: pd.DataFrame) -> str:
        """Create a stable provenance fingerprint without retaining raw records."""

        hashed_rows = np.asarray(pd.util.hash_pandas_object(observations, index=True).values)
        return sha256(hashed_rows.tobytes()).hexdigest()
