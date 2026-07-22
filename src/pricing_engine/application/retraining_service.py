"""Retraining orchestration with quality gates and candidate-only registration."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

import numpy as np
import pandas as pd

from pricing_engine.application.evaluation import PromotionCriteria
from pricing_engine.application.ports import (
    CandidateModelRegistry,
    DemandPredictor,
    RegisteredModel,
)
from pricing_engine.application.training_service import TrainingOutcome


class TrainingRunner(Protocol):
    """Training boundary required by the retraining orchestrator."""

    def execute(
        self,
        observations: pd.DataFrame,
        *,
        benchmark_model: DemandPredictor | None = None,
    ) -> TrainingOutcome: ...


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
        trainer: TrainingRunner,
        registry: CandidateModelRegistry,
        promotion_criteria: PromotionCriteria | None = None,
    ) -> None:
        self._trainer = trainer
        self._registry = registry
        self._promotion_criteria = promotion_criteria or PromotionCriteria()

    def execute(self, observations: pd.DataFrame) -> RetrainingResult:
        champion = self._registry.load_champion()
        outcome = self._trainer.execute(observations, benchmark_model=champion)
        self._promotion_criteria.evaluate(
            outcome.metrics,
            champion_metrics=outcome.benchmark_metrics,
        )
        fingerprint = self._fingerprint(observations)
        registered = self._registry.log_candidate(
            model=outcome.model,
            metrics=outcome.metrics,
            feature_version=outcome.feature_version,
            dataset_fingerprint=fingerprint,
            benchmark_metrics=outcome.benchmark_metrics,
            benchmark_model_version=outcome.benchmark_model_version,
            benchmark_status=outcome.benchmark_status,
        )
        return RetrainingResult(
            outcome=outcome,
            registered_model=registered,
            dataset_fingerprint=fingerprint,
        )

    @staticmethod
    def _fingerprint(observations: pd.DataFrame) -> str:
        """Hash logical records independent of row order, index, and CSV/Parquet dtypes."""

        canonical = observations.copy()
        for column in ("stay_date", "as_of_date", "outcome_available_date"):
            if column in canonical:
                canonical[column] = pd.to_datetime(canonical[column]).dt.normalize()
        for column in ("tenant_id", "property_id", "currency"):
            if column in canonical:
                canonical[column] = canonical[column].astype(str).str.strip()
        protected = {
            "tenant_id",
            "property_id",
            "currency",
            "stay_date",
            "as_of_date",
            "outcome_available_date",
        }
        for column in set(canonical.columns).difference(protected):
            if pd.api.types.is_numeric_dtype(canonical[column]):
                canonical[column] = canonical[column].astype(float)
        canonical = canonical.reindex(sorted(canonical.columns), axis=1)
        row_hashes = np.sort(np.asarray(pd.util.hash_pandas_object(canonical, index=False).values))
        digest = sha256()
        digest.update("\0".join(canonical.columns).encode())
        digest.update(row_hashes.tobytes())
        return digest.hexdigest()
