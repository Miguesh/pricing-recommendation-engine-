"""Dependency inversion ports used by application services."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import pandas as pd

from pricing_engine.application.evaluation import EvaluationMetrics
from pricing_engine.domain.models import (
    DemandEstimate,
    FeatureContribution,
    PricingContext,
)

BenchmarkStatus: TypeAlias = Literal[
    "not_available",
    "evaluated",
    "incompatible_contract",
]


class FeatureFactory(Protocol):
    """Transforms a domain request into model-ready rows."""

    def for_candidate_prices(
        self, context: PricingContext, candidate_prices: Sequence[float]
    ) -> pd.DataFrame: ...


class DemandPredictor(Protocol):
    """Predicts calibrated demand and model metadata."""

    @property
    def version(self) -> str: ...

    @property
    def currency(self) -> str: ...

    @property
    def tenant_ids(self) -> Sequence[str]: ...

    @property
    def feature_contract_version(self) -> str: ...

    @property
    def feature_schema_hash(self) -> str: ...

    @property
    def feature_names(self) -> Sequence[str]: ...

    @property
    def feature_means(self) -> Mapping[str, float]: ...

    @property
    def interval_coverage(self) -> float: ...

    @property
    def training_parameters(self) -> Mapping[str, str | int | float | bool]: ...

    def predict(self, features: pd.DataFrame) -> Sequence[DemandEstimate]: ...

    def explain(self, features: pd.DataFrame, *, top_k: int) -> Sequence[FeatureContribution]: ...

    def global_feature_importance(self, *, top_k: int) -> dict[str, float]: ...


class ModelBundleStore(Protocol):
    """Stores and retrieves a complete, immutable model bundle."""

    def load(self, uri: str) -> DemandPredictor: ...

    def save_local(self, model: DemandPredictor, destination: Path) -> Path: ...


class TrainingDataValidator(Protocol):
    """Validates and normalizes offline observations without leaking framework details."""

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame: ...


class TrainingFeatureFactory(Protocol):
    """Builds the versioned offline feature matrix."""

    VERSION: str
    SCHEMA_HASH: str

    def build_training_features(self, frame: pd.DataFrame) -> pd.DataFrame: ...


class DemandModelTrainer(Protocol):
    """Fits a demand predictor from already partitioned matrices."""

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
    ) -> DemandPredictor: ...


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """Registry identity emitted for an immutable candidate."""

    run_id: str
    model_name: str
    model_version: str
    model_uri: str


class CandidateModelRegistry(Protocol):
    """Candidate-only model registry boundary used by retraining orchestration."""

    def log_candidate(
        self,
        *,
        model: DemandPredictor,
        metrics: EvaluationMetrics,
        feature_version: str,
        dataset_fingerprint: str,
        benchmark_metrics: EvaluationMetrics | None,
        benchmark_model_version: str | None,
        benchmark_status: BenchmarkStatus,
    ) -> RegisteredModel: ...

    def load_champion(self) -> DemandPredictor | None: ...
