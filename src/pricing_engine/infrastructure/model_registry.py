"""MLflow-backed versioning plus portable local model-bundle persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

from pricing_engine.application.evaluation import EvaluationMetrics
from pricing_engine.domain.exceptions import ModelUnavailableError
from pricing_engine.infrastructure.models.lightgbm_demand import QuantileLightGBMDemandModel


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """Registry identity emitted after a candidate model is logged."""

    run_id: str
    model_name: str
    model_version: str
    model_uri: str


class DemandModelPyfunc(mlflow.pyfunc.PythonModel):  # type: ignore[name-defined, misc]
    """MLflow wrapper for interoperable batch demand inference."""

    def load_context(self, context: Any) -> None:
        self._bundle: QuantileLightGBMDemandModel = joblib.load(context.artifacts["bundle"])

    def predict(
        self,
        context: Any,
        model_input: list[dict[str, float]],
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, float]]:
        raw_input: Any = model_input
        frame = raw_input.copy() if isinstance(raw_input, pd.DataFrame) else pd.DataFrame(raw_input)
        estimates = self._bundle.predict(frame)
        return [
            {
                "expected_occupancy": estimate.expected_occupancy,
                "lower_occupancy": estimate.lower_occupancy,
                "upper_occupancy": estimate.upper_occupancy,
                "in_distribution_score": estimate.in_distribution_score,
            }
            for estimate in estimates
        ]


class LocalModelBundleStore:
    """Simple filesystem store used by local development and integration tests."""

    BUNDLE_NAME = "pricing_demand_bundle.joblib"

    def load(self, uri: str) -> QuantileLightGBMDemandModel:
        path = Path(uri)
        if path.is_dir():
            path = path / self.BUNDLE_NAME
        if not path.exists():
            raise ModelUnavailableError(f"Model bundle does not exist at {path}.")
        model = joblib.load(path)
        if not isinstance(model, QuantileLightGBMDemandModel):
            raise ModelUnavailableError(f"Unexpected model bundle type at {path}.")
        return model

    def save_local(self, model: QuantileLightGBMDemandModel, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        bundle_path = destination / self.BUNDLE_NAME
        joblib.dump(model, bundle_path)
        return bundle_path


class MLflowModelRegistry:
    """Logs immutable candidate bundles and promotes only explicit versions."""

    ARTIFACT_PATH = "pricing_demand_model"

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str = "pricing-demand",
        registered_model_name: str = "pricing-demand",
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._registered_model_name = registered_model_name

    def log_candidate(
        self,
        *,
        model: QuantileLightGBMDemandModel,
        metrics: EvaluationMetrics,
        feature_version: str,
        dataset_fingerprint: str,
    ) -> RegisteredModel:
        """Log a model bundle and register it as a candidate, never as champion."""

        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        with TemporaryDirectory(prefix="pricing-model-") as temporary:
            bundle_path = Path(temporary) / LocalModelBundleStore.BUNDLE_NAME
            joblib.dump(model, bundle_path)
            with mlflow.start_run() as run:
                mlflow.log_params(
                    {
                        "model_family": "lightgbm_quantile_ensemble",
                        "model_version": model.version,
                        "feature_version": feature_version,
                        "interval_coverage": model.config.interval_coverage,
                        "dataset_fingerprint": dataset_fingerprint,
                        **{f"training.{key}": value for key, value in asdict(model.config).items()},
                    }
                )
                mlflow.log_metrics(metrics.as_dict())
                model_info = mlflow.pyfunc.log_model(
                    name=self.ARTIFACT_PATH,
                    python_model=DemandModelPyfunc(),
                    artifacts={"bundle": str(bundle_path)},
                )
                model_uri = str(model_info.model_uri)
                registered = mlflow.register_model(
                    model_uri=model_uri,
                    name=self._registered_model_name,
                    tags={
                        "lifecycle": "candidate",
                        "model_version": model.version,
                        "feature_version": feature_version,
                    },
                )
        return RegisteredModel(
            run_id=run.info.run_id,
            model_name=self._registered_model_name,
            model_version=str(registered.version),
            model_uri=f"models:/{self._registered_model_name}/{registered.version}",
        )

    def promote(self, *, version: str, approved_by: str) -> None:
        """Assign the champion alias after an explicit human approval event."""

        client = MlflowClient(tracking_uri=self._tracking_uri)
        client.set_registered_model_alias(
            name=self._registered_model_name,
            alias="champion",
            version=version,
        )
        client.set_model_version_tag(
            name=self._registered_model_name,
            version=version,
            key="approved_by",
            value=approved_by,
        )
        client.set_model_version_tag(
            name=self._registered_model_name,
            version=version,
            key="lifecycle",
            value="champion",
        )

    def load(self, uri: str) -> QuantileLightGBMDemandModel:
        """Download the bundle embedded in an MLflow model URI."""

        try:
            local_model_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
            possible_paths = (
                local_model_dir / "artifacts" / LocalModelBundleStore.BUNDLE_NAME,
                local_model_dir / "artifacts" / "bundle" / LocalModelBundleStore.BUNDLE_NAME,
                local_model_dir / "bundle" / LocalModelBundleStore.BUNDLE_NAME,
            )
            bundle_path = next((path for path in possible_paths if path.exists()), None)
            if bundle_path is None:
                raise ModelUnavailableError(
                    f"MLflow artifact at {uri} does not contain "
                    f"{LocalModelBundleStore.BUNDLE_NAME}."
                )
            return LocalModelBundleStore().load(str(bundle_path))
        except Exception as error:
            raise ModelUnavailableError(f"Could not load model URI {uri}: {error}") from error
