"""Dependency inversion ports used by application services."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import pandas as pd

from pricing_engine.domain.models import (
    DemandEstimate,
    FeatureContribution,
    PricingContext,
)


class FeatureFactory(Protocol):
    """Transforms a domain request into model-ready rows."""

    def for_candidate_prices(
        self, context: PricingContext, candidate_prices: Sequence[float]
    ) -> pd.DataFrame: ...


class DemandPredictor(Protocol):
    """Predicts calibrated demand and model metadata."""

    @property
    def version(self) -> str: ...

    def predict(self, features: pd.DataFrame) -> Sequence[DemandEstimate]: ...

    def explain(self, features: pd.DataFrame, *, top_k: int) -> Sequence[FeatureContribution]: ...

    def global_feature_importance(self, *, top_k: int) -> dict[str, float]: ...


class ModelBundleStore(Protocol):
    """Stores and retrieves a complete, immutable model bundle."""

    def load(self, uri: str) -> DemandPredictor: ...

    def save_local(self, model: DemandPredictor, destination: Path) -> Path: ...
