"""Composition root for API dependencies."""

from __future__ import annotations

from pathlib import Path

from pricing_engine.application.recommendation_service import RecommendPrice
from pricing_engine.config import Settings
from pricing_engine.domain.exceptions import ModelUnavailableError
from pricing_engine.domain.policies import PricingPolicy
from pricing_engine.infrastructure.features import PricingFeatureFactory
from pricing_engine.infrastructure.model_registry import LocalModelBundleStore, MLflowModelRegistry
from pricing_engine.infrastructure.models.lightgbm_demand import QuantileLightGBMDemandModel


class ApplicationContainer:
    """Owns mutable runtime model state while keeping routes framework-thin."""

    def __init__(
        self,
        settings: Settings,
        *,
        predictor: QuantileLightGBMDemandModel | None = None,
    ) -> None:
        self.settings = settings
        self._predictor = predictor
        self._service: RecommendPrice | None = (
            self._build_service(predictor) if predictor is not None else None
        )

    @property
    def model_loaded(self) -> bool:
        return self._service is not None

    @property
    def model_version(self) -> str | None:
        return self._predictor.version if self._predictor else None

    @property
    def recommendation_service(self) -> RecommendPrice:
        if self._service is None:
            raise ModelUnavailableError(
                "No champion model is loaded. Configure PRICING_MODEL_URI to an approved "
                "MLflow URI or local model bundle."
            )
        return self._service

    def load_configured_model(self) -> None:
        """Load only a configured artifact; service never trains at startup."""

        if self._predictor is not None:
            return
        if not self.settings.model_uri:
            return
        uri = self.settings.model_uri
        if Path(uri).exists():
            predictor = LocalModelBundleStore().load(uri)
        else:
            predictor = MLflowModelRegistry(tracking_uri=self.settings.mlflow_tracking_uri).load(
                uri
            )
        self._predictor = predictor
        self._service = self._build_service(predictor)

    def _build_service(self, predictor: QuantileLightGBMDemandModel) -> RecommendPrice:
        return RecommendPrice(
            predictor=predictor,
            feature_factory=PricingFeatureFactory(),
            pricing_policy=PricingPolicy(),
            settings=self.settings,
        )
