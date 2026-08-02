"""Composition root for API dependencies."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

import structlog

from pricing_engine.application.statistical_service import MarketEvidenceStatisticalV1
from pricing_engine.config import Settings
from pricing_engine.domain.exceptions import ModelUnavailableError

if TYPE_CHECKING:
    from pricing_engine.application.recommendation_service import RecommendPrice
    from pricing_engine.infrastructure.models.lightgbm_demand import QuantileLightGBMDemandModel

logger = structlog.get_logger(__name__)


class ExperimentalProfileState(StrEnum):
    """Internal availability state for the optional artifact-backed profile."""

    NOT_CONFIGURED = "NOT_CONFIGURED"
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"


class ExperimentalProfileFailureCode(StrEnum):
    """Safe internal failure taxonomy; provider details are deliberately excluded."""

    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"


class ApplicationContainer:
    """Owns mutable runtime model state while keeping routes framework-thin."""

    def __init__(
        self,
        settings: Settings,
        *,
        predictor: QuantileLightGBMDemandModel | None = None,
    ) -> None:
        self.settings = settings
        self._injected_predictor_candidate = predictor
        self._predictor: QuantileLightGBMDemandModel | None = None
        self._service: RecommendPrice | None = None
        self._experimental_profile_state = ExperimentalProfileState.NOT_CONFIGURED
        self._experimental_failure_code: ExperimentalProfileFailureCode | None = None
        self._statistical_service = MarketEvidenceStatisticalV1()

    @property
    def model_loaded(self) -> bool:
        return self._service is not None

    @property
    def model_version(self) -> str | None:
        return self._predictor.version if self._predictor else None

    @property
    def experimental_profile_state(self) -> ExperimentalProfileState:
        """Return internal optional-profile state without provider details."""

        return self._experimental_profile_state

    @property
    def experimental_failure_code(self) -> ExperimentalProfileFailureCode | None:
        """Return the bounded failure code for operational assertions."""

        return self._experimental_failure_code

    @property
    def recommendation_service(self) -> RecommendPrice:
        if self._service is None:
            raise ModelUnavailableError(
                "No champion model is loaded. Configure PRICING_MODEL_URI to an approved "
                "MLflow URI or local model bundle."
            )
        return self._service

    @property
    def statistical_service(self) -> MarketEvidenceStatisticalV1:
        """Return the artifact-free stable profile, which is always available."""

        return self._statistical_service

    def load_configured_model(self) -> None:
        """Strictly load a configured artifact without publishing partial state."""

        if self._service is not None:
            self._experimental_profile_state = ExperimentalProfileState.READY
            self._experimental_failure_code = None
            return
        candidate = self._injected_predictor_candidate
        configured_uri = self.settings.model_uri
        if candidate is None and not configured_uri:
            self._clear_experimental_profile(ExperimentalProfileState.NOT_CONFIGURED)
            return
        try:
            if candidate is not None:
                predictor = candidate
            else:
                predictor = self._load_configured_predictor(cast(str, configured_uri))
            self._validate_predictor_contract(predictor)
            self._warm_predictor(predictor)
            service = self._build_service(predictor)
        except Exception:
            self._clear_experimental_profile(
                ExperimentalProfileState.UNAVAILABLE,
                failure_code=ExperimentalProfileFailureCode.MODEL_LOAD_FAILED,
            )
            raise
        self._publish_experimental_profile(predictor, service)

    def initialize_optional_experimental_profile(self) -> ExperimentalProfileState:
        """Initialize the optional model profile without blocking the stable profile."""

        if self._service is not None:
            self._experimental_profile_state = ExperimentalProfileState.READY
            self._experimental_failure_code = None
            return self._experimental_profile_state
        if self._injected_predictor_candidate is None and not self.settings.model_uri:
            self._clear_experimental_profile(ExperimentalProfileState.NOT_CONFIGURED)
            return self._experimental_profile_state
        try:
            self.load_configured_model()
        except Exception as error:
            self._clear_experimental_profile(
                ExperimentalProfileState.UNAVAILABLE,
                failure_code=ExperimentalProfileFailureCode.MODEL_LOAD_FAILED,
            )
            logger.warning(
                "experimental_profile_initialization_failed",
                state=ExperimentalProfileState.UNAVAILABLE.value,
                code=ExperimentalProfileFailureCode.MODEL_LOAD_FAILED.value,
                failure_type=type(error).__name__,
            )
        return self._experimental_profile_state

    def _load_configured_predictor(self, uri: str) -> QuantileLightGBMDemandModel:
        """Load from the configured source without mutating container state."""

        if Path(uri).exists():
            from pricing_engine.infrastructure.model_registry import LocalModelBundleStore

            return LocalModelBundleStore().load(uri)
        from pricing_engine.infrastructure.model_registry import MLflowModelRegistry

        return MLflowModelRegistry(
            tracking_uri=self.settings.mlflow_tracking_uri,
            dependency_project_path=self.settings.dependency_project_path,
        ).load(uri)

    def _publish_experimental_profile(
        self,
        predictor: QuantileLightGBMDemandModel,
        service: RecommendPrice,
    ) -> None:
        """Atomically publish a fully validated and warmed experimental profile."""

        self._predictor = predictor
        self._service = service
        self._injected_predictor_candidate = None
        self._experimental_profile_state = ExperimentalProfileState.READY
        self._experimental_failure_code = None

    def _clear_experimental_profile(
        self,
        state: ExperimentalProfileState,
        *,
        failure_code: ExperimentalProfileFailureCode | None = None,
    ) -> None:
        """Remove all artifact-backed state after an optional-profile failure."""

        self._predictor = None
        self._service = None
        self._injected_predictor_candidate = None
        self._experimental_profile_state = state
        self._experimental_failure_code = failure_code

    def _validate_predictor_contract(
        self,
        predictor: QuantileLightGBMDemandModel,
    ) -> None:
        """Reject stale artifacts before the API can report itself ready."""

        from pricing_engine.infrastructure.features import PricingFeatureFactory

        expected = tuple(PricingFeatureFactory.FEATURE_COLUMNS)
        actual = tuple(predictor.feature_names)
        actual_version = getattr(predictor, "feature_contract_version", None)
        actual_hash = getattr(predictor, "feature_schema_hash", None)
        if (
            actual != expected
            or actual_version != PricingFeatureFactory.VERSION
            or actual_hash != PricingFeatureFactory.SCHEMA_HASH
        ):
            raise ModelUnavailableError(
                "Configured model uses an incompatible feature contract. "
                f"Expected version={PricingFeatureFactory.VERSION}, "
                f"schema_hash={PricingFeatureFactory.SCHEMA_HASH}, columns={expected}; "
                f"got version={actual_version}, schema_hash={actual_hash}, columns={actual}. "
                "Retrain and promote a compatible model."
            )
        if not getattr(predictor, "currency", None):
            raise ModelUnavailableError("Configured model does not declare its training currency.")
        tenant_ids = getattr(predictor, "tenant_ids", ())
        if not tenant_ids:
            raise ModelUnavailableError(
                "Configured model does not declare its governed tenant scope."
            )
        configured_tenant = self.settings.serving_tenant_id or self.settings.api_key_tenant_id
        if configured_tenant is not None and configured_tenant not in tenant_ids:
            raise ModelUnavailableError(
                "Configured API tenant binding is incompatible with the model tenant scope."
            )

    @staticmethod
    def _warm_predictor(predictor: QuantileLightGBMDemandModel) -> None:
        """Fail readiness closed and pre-initialize model/SHAP native state."""

        try:
            import pandas as pd

            feature_names = tuple(predictor.feature_names)
            row = pd.DataFrame(
                [{name: predictor.feature_means[name] for name in feature_names}],
                columns=feature_names,
            )
            estimates = predictor.predict(row)
            if len(estimates) != 1:
                raise ValueError("Warm-up inference did not return exactly one estimate.")
            predictor.global_feature_importance(top_k=1)
            predictor.explain(row, top_k=1)
        except Exception as error:
            raise ModelUnavailableError(
                "Configured model failed inference and explainability warm-up."
            ) from error

    def _build_service(self, predictor: QuantileLightGBMDemandModel) -> RecommendPrice:
        from pricing_engine.application.recommendation_service import RecommendPrice
        from pricing_engine.domain.policies import PricingPolicy
        from pricing_engine.infrastructure.features import PricingFeatureFactory

        return RecommendPrice(
            predictor=predictor,
            feature_factory=PricingFeatureFactory(),
            pricing_policy=PricingPolicy(),
            maximum_candidates=self.settings.recommendation_max_candidates,
        )
