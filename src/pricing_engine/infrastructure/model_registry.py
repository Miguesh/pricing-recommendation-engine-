"""MLflow-backed versioning plus portable local model-bundle persistence."""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from secrets import compare_digest
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from pricing_engine.application.evaluation import EvaluationMetrics, PromotionCriteria
from pricing_engine.application.ports import BenchmarkStatus, DemandPredictor, RegisteredModel
from pricing_engine.domain.exceptions import ModelUnavailableError, PromotionRejectedError
from pricing_engine.infrastructure.features import PricingFeatureFactory
from pricing_engine.infrastructure.models.lightgbm_demand import QuantileLightGBMDemandModel

BUNDLE_CHECKSUM_NAME = "pricing_demand_bundle.sha256"


def _sha256_file(path: Path) -> str:
    """Hash an artifact without loading the full model bundle into memory."""

    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class DemandModelPyfunc(mlflow.pyfunc.PythonModel):  # type: ignore[name-defined, misc]
    """MLflow wrapper for interoperable batch demand inference."""

    def load_context(self, context: Any) -> None:
        bundle_path = Path(context.artifacts["bundle"])
        checksum_path = Path(context.artifacts["bundle_checksum"])
        expected_content_hash = checksum_path.read_text(encoding="ascii").strip()
        actual_content_hash = _sha256_file(bundle_path)
        if len(expected_content_hash) != 64 or not compare_digest(
            actual_content_hash, expected_content_hash
        ):
            raise ModelUnavailableError(
                "MLflow PyFunc bundle failed SHA-256 integrity verification."
            )
        self._bundle: QuantileLightGBMDemandModel = joblib.load(bundle_path)

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        estimates = self._bundle.predict(model_input)
        return pd.DataFrame(
            [
                {
                    "expected_occupancy": estimate.expected_occupancy,
                    "lower_occupancy": estimate.lower_occupancy,
                    "upper_occupancy": estimate.upper_occupancy,
                    "in_distribution_score": estimate.in_distribution_score,
                }
                for estimate in estimates
            ]
        )


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

    def save_local(self, model: DemandPredictor, destination: Path) -> Path:
        if not isinstance(model, QuantileLightGBMDemandModel):
            raise ModelUnavailableError("Local bundle store only supports LightGBM demand models.")
        destination.mkdir(parents=True, exist_ok=True)
        bundle_path = destination / self.BUNDLE_NAME
        joblib.dump(model, bundle_path)
        return bundle_path


class MLflowModelRegistry:
    """Logs immutable candidate bundles and promotes only explicit versions."""

    ARTIFACT_PATH = "pricing_demand_model"

    @staticmethod
    def _is_missing_alias(error: mlflow.exceptions.MlflowException) -> bool:
        """Normalize MLflow backend differences for an alias that does not exist."""

        error_code = getattr(error, "error_code", None)
        message = str(error).lower()
        return error_code == "RESOURCE_DOES_NOT_EXIST" or (
            error_code == "INVALID_PARAMETER_VALUE"
            and "alias" in message
            and "not found" in message
        )

    @staticmethod
    def _validate_uv_project_root(candidate: Path) -> Path | None:
        """Return a dependency contract only when it belongs to this package."""

        pyproject = candidate / "pyproject.toml"
        lock = candidate / "uv.lock"
        if not pyproject.is_file() or not lock.is_file():
            return None
        try:
            project_name = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["name"]
        except (KeyError, tomllib.TOMLDecodeError) as error:
            raise ModelUnavailableError(f"Invalid dependency contract at {candidate}.") from error
        if project_name != "pricing-recommendation-engine":
            raise ModelUnavailableError(
                f"Dependency contract at {candidate} belongs to project {project_name!r}."
            )
        return candidate.resolve()

    def _find_uv_project_root(self) -> Path:
        """Locate the immutable dependency contract required for portable model logging."""

        if self._dependency_project_path is not None:
            validated = self._validate_uv_project_root(self._dependency_project_path)
            if validated is None:
                raise ModelUnavailableError(
                    "Configured dependency_project_path must contain pyproject.toml and uv.lock."
                )
            return validated
        candidates = (
            Path(__file__).resolve().parents[3],
            Path("/app/model-contract"),
        )
        for candidate in candidates:
            validated = self._validate_uv_project_root(candidate)
            if validated is not None:
                return validated
        raise ModelUnavailableError(
            "Candidate logging requires an explicit pricing-engine dependency contract. "
            "Configure PRICING_DEPENDENCY_PROJECT_PATH."
        )

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str = "pricing-demand",
        registered_model_name: str = "pricing-demand",
        dependency_project_path: Path | None = None,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._registered_model_name = registered_model_name
        self._dependency_project_path = dependency_project_path

    def _ensure_registered_model_tenant_binding(
        self,
        *,
        tenant_id: str,
        tenant_scope_hash: str,
        currency: str,
    ) -> None:
        """Bind a registered-model namespace immutably to one governed tenant."""

        client = MlflowClient(tracking_uri=self._tracking_uri)
        binding_tags = {
            "tenant_id": tenant_id,
            "tenant_scope_hash": tenant_scope_hash,
            "tenant_scope": "single-tenant",
            "currency": currency,
        }
        try:
            client.create_registered_model(
                name=self._registered_model_name,
                tags=binding_tags,
            )
            return
        except mlflow.exceptions.MlflowException as error:
            if getattr(error, "error_code", None) != "RESOURCE_ALREADY_EXISTS":
                raise
        registered_model = client.get_registered_model(self._registered_model_name)
        existing_hash = registered_model.tags.get("tenant_scope_hash")
        existing_tenant = registered_model.tags.get("tenant_id")
        existing_currency = registered_model.tags.get("currency")
        if (
            existing_hash != tenant_scope_hash
            or existing_tenant != tenant_id
            or existing_currency != currency
        ):
            raise ModelUnavailableError(
                "Registered model namespace is unbound or belongs to a different tenant/currency. "
                "Use an independently governed registered_model_name per tenant and currency."
            )

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
    ) -> RegisteredModel:
        """Log a model bundle and register it as a candidate, never as champion."""

        if not isinstance(model, QuantileLightGBMDemandModel):
            raise ModelUnavailableError("MLflow registry only supports LightGBM demand models.")
        if feature_version != model.feature_contract_version:
            raise ModelUnavailableError(
                "Registry feature version does not match the serialized model contract."
            )
        if len(dataset_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in dataset_fingerprint
        ):
            raise ModelUnavailableError("dataset_fingerprint must be a lowercase SHA-256 digest.")
        if len(model.tenant_ids) != 1:
            raise ModelUnavailableError("Registry accepts only single-tenant model bundles.")
        self._validate_benchmark_evidence(
            benchmark_status=benchmark_status,
            benchmark_metrics=benchmark_metrics,
            benchmark_model_version=benchmark_model_version,
        )
        PromotionCriteria().evaluate(
            metrics,
            champion_metrics=(benchmark_metrics if benchmark_status == "evaluated" else None),
        )

        mlflow.set_tracking_uri(self._tracking_uri)
        experiment = mlflow.set_experiment(self._experiment_name)
        if self._tracking_uri.startswith(
            ("http://", "https://")
        ) and experiment.artifact_location.startswith(("s3://", "gs://", "wasbs://", "abfss://")):
            raise ModelUnavailableError(
                "Experiment artifact root bypasses the MLflow artifact proxy. "
                "Use a new proxied experiment or migrate the existing experiment before retraining."
            )
        input_example = pd.DataFrame(
            [{name: model.feature_means[name] for name in model.feature_names}],
            columns=model.feature_names,
        )
        output_example = pd.DataFrame(
            [
                {
                    "expected_occupancy": estimate.expected_occupancy,
                    "lower_occupancy": estimate.lower_occupancy,
                    "upper_occupancy": estimate.upper_occupancy,
                    "in_distribution_score": estimate.in_distribution_score,
                }
                for estimate in model.predict(input_example)
            ]
        )
        signature = infer_signature(input_example, output_example)
        uv_project_root = self._find_uv_project_root()
        package_source = Path(__file__).resolve().parents[1]
        tenant_scope_hash = sha256("\0".join(model.tenant_ids).encode()).hexdigest()
        tenant_id = model.tenant_ids[0]
        self._ensure_registered_model_tenant_binding(
            tenant_id=tenant_id,
            tenant_scope_hash=tenant_scope_hash,
            currency=model.currency,
        )
        benchmark_registry_version = self._resolve_benchmark_registry_version(
            benchmark_model_version=benchmark_model_version,
        )
        with TemporaryDirectory(prefix="pricing-model-") as temporary:
            bundle_path = Path(temporary) / LocalModelBundleStore.BUNDLE_NAME
            joblib.dump(model, bundle_path)
            model_content_hash = _sha256_file(bundle_path)
            checksum_path = Path(temporary) / BUNDLE_CHECKSUM_NAME
            checksum_path.write_text(model_content_hash, encoding="ascii")
            with mlflow.start_run() as run:
                parameters: dict[str, str | int | float | bool] = {
                    "model_family": "lightgbm_quantile_ensemble",
                    "model_version": model.version,
                    "feature_version": feature_version,
                    "feature_schema_hash": model.feature_schema_hash,
                    "currency": model.currency,
                    "tenant_count": len(model.tenant_ids),
                    "tenant_id": tenant_id,
                    "tenant_scope_hash": tenant_scope_hash,
                    "interval_coverage": model.interval_coverage,
                    "dataset_fingerprint": dataset_fingerprint,
                    "model_content_hash": model_content_hash,
                    "benchmark_status": benchmark_status,
                    "promotion_gate": "passed",
                    **{
                        f"training.{key}": value for key, value in model.training_parameters.items()
                    },
                }
                if benchmark_model_version is not None:
                    parameters["benchmark_model_version"] = benchmark_model_version
                if benchmark_registry_version is not None:
                    parameters["benchmark_registry_version"] = benchmark_registry_version
                mlflow.log_params(parameters)
                candidate_metrics = metrics.as_dict()
                logged_metrics = dict(candidate_metrics)
                logged_metrics.update(
                    {
                        f"candidate.current_holdout.{name}": value
                        for name, value in candidate_metrics.items()
                    }
                )
                if benchmark_metrics is not None:
                    benchmark_values = benchmark_metrics.as_dict()
                    logged_metrics.update(
                        {
                            f"benchmark.current_holdout.{name}": value
                            for name, value in benchmark_values.items()
                        }
                    )
                    logged_metrics["delta.current_holdout.occupancy_mae"] = (
                        metrics.occupancy_mae - benchmark_metrics.occupancy_mae
                    )
                    if (
                        metrics.expected_revenue_wape is not None
                        and benchmark_metrics.expected_revenue_wape is not None
                    ):
                        logged_metrics["delta.current_holdout.expected_revenue_wape"] = (
                            metrics.expected_revenue_wape - benchmark_metrics.expected_revenue_wape
                        )
                mlflow.log_metrics(logged_metrics)
                model_info = mlflow.pyfunc.log_model(
                    name=self.ARTIFACT_PATH,
                    python_model=DemandModelPyfunc(),
                    artifacts={
                        "bundle": str(bundle_path),
                        "bundle_checksum": str(checksum_path),
                    },
                    code_paths=[str(package_source)],
                    input_example=input_example,
                    signature=signature,
                    uv_project_path=uv_project_root,
                    metadata={
                        "dependency_contract": "uv.lock",
                        "feature_contract_version": feature_version,
                    },
                )
                model_uri = str(model_info.model_uri)
                version_tags = {
                    "lifecycle": "candidate",
                    "model_version": model.version,
                    "feature_version": feature_version,
                    "feature_schema_hash": model.feature_schema_hash,
                    "currency": model.currency,
                    "tenant_count": str(len(model.tenant_ids)),
                    "tenant_id": tenant_id,
                    "tenant_scope_hash": tenant_scope_hash,
                    "benchmark_status": benchmark_status,
                    "promotion_gate": "passed",
                    "model_content_hash": model_content_hash,
                }
                if benchmark_registry_version is not None:
                    version_tags["benchmark_registry_version"] = benchmark_registry_version
            registered = mlflow.register_model(
                model_uri=model_uri,
                name=self._registered_model_name,
                tags=version_tags,
            )
        return RegisteredModel(
            run_id=run.info.run_id,
            model_name=self._registered_model_name,
            model_version=str(registered.version),
            model_uri=f"models:/{self._registered_model_name}/{registered.version}",
        )

    @staticmethod
    def _validate_benchmark_evidence(
        *,
        benchmark_status: BenchmarkStatus,
        benchmark_metrics: EvaluationMetrics | None,
        benchmark_model_version: str | None,
    ) -> None:
        """Reject contradictory comparison evidence at the registry boundary."""

        evidence_is_valid = {
            "evaluated": benchmark_metrics is not None and benchmark_model_version is not None,
            "not_available": benchmark_metrics is None and benchmark_model_version is None,
            "incompatible_contract": (
                benchmark_metrics is None and benchmark_model_version is not None
            ),
        }[benchmark_status]
        if not evidence_is_valid:
            raise ModelUnavailableError(
                f"Benchmark evidence is inconsistent with status={benchmark_status}."
            )

    def promote(self, *, version: str, approved_by: str) -> None:
        """Validate, audit, and switch the alias with compensating rollback."""

        self._set_champion(
            version=version,
            approved_by=approved_by,
            required_lifecycle="candidate",
            decision="promotion",
            reason=None,
        )

    def _resolve_benchmark_registry_version(
        self,
        *,
        benchmark_model_version: str | None,
    ) -> str | None:
        """Bind evaluation evidence to the exact champion alias observed at registration."""

        client = MlflowClient(tracking_uri=self._tracking_uri)
        try:
            champion = client.get_model_version_by_alias(
                name=self._registered_model_name,
                alias="champion",
            )
        except mlflow.exceptions.MlflowException as error:
            if self._is_missing_alias(error):
                if benchmark_model_version is not None:
                    raise ModelUnavailableError(
                        "Benchmark evidence references a champion that no longer exists. "
                        "Restart retraining against the current registry state."
                    ) from error
                return None
            raise
        self._validate_champion_governance(
            client=client,
            model_name=self._registered_model_name,
            model_version=champion,
        )
        if champion.tags.get("model_version") != benchmark_model_version:
            raise ModelUnavailableError(
                "Champion changed while the candidate was being evaluated. "
                "Restart retraining against the current champion."
            )
        return str(champion.version)

    def rollback(self, *, version: str, approved_by: str, reason: str) -> None:
        """Restore a superseded champion through an explicit audited decision."""

        normalized_reason = reason.strip()
        if (
            len(normalized_reason) < 10
            or len(normalized_reason) > 500
            or not all(character.isprintable() for character in normalized_reason)
        ):
            raise PromotionRejectedError(
                "Rollback reason must be a printable explanation of 10 to 500 characters."
            )
        self._set_champion(
            version=version,
            approved_by=approved_by,
            required_lifecycle="superseded",
            decision="rollback",
            reason=normalized_reason,
        )

    def _set_champion(
        self,
        *,
        version: str,
        approved_by: str,
        required_lifecycle: str,
        decision: str,
        reason: str | None,
    ) -> None:
        """Shared governed alias transition used by promotion and rollback."""

        approver = approved_by.strip()
        if (
            not approver
            or len(approver) > 200
            or not all(character.isprintable() for character in approver)
        ):
            raise PromotionRejectedError("approved_by must identify one printable human approver.")

        client = MlflowClient(tracking_uri=self._tracking_uri)
        candidate = client.get_model_version(
            name=self._registered_model_name,
            version=version,
        )
        required_tags = {
            "lifecycle": required_lifecycle,
            "promotion_gate": "passed",
        }
        invalid_tags = {
            key: candidate.tags.get(key)
            for key, expected in required_tags.items()
            if candidate.tags.get(key) != expected
        }
        if invalid_tags:
            raise PromotionRejectedError(
                f"Version {version} is not an eligible gated candidate: {invalid_tags}."
            )
        if not candidate.run_id:
            raise PromotionRejectedError("Candidate does not reference an auditable MLflow run.")
        run = client.get_run(candidate.run_id)
        if run.info.status != "FINISHED":
            raise PromotionRejectedError(
                f"Candidate run is not complete: status={run.info.status}."
            )
        if run.data.params.get("promotion_gate") != "passed":
            raise PromotionRejectedError("Candidate run is missing passed promotion evidence.")
        benchmark_status = run.data.params.get("benchmark_status")
        if (
            benchmark_status not in {"evaluated", "not_available", "incompatible_contract"}
            or candidate.tags.get("benchmark_status") != benchmark_status
        ):
            raise PromotionRejectedError(
                "Candidate run and model version have inconsistent benchmark governance."
            )
        if len(run.data.params.get("dataset_fingerprint", "")) != 64:
            raise PromotionRejectedError("Candidate run is missing a valid dataset fingerprint.")
        model_content_hash = run.data.params.get("model_content_hash", "")
        if (
            len(model_content_hash) != 64
            or candidate.tags.get("model_content_hash") != model_content_hash
        ):
            raise PromotionRejectedError(
                "Candidate run and model version lack matching artifact integrity evidence."
            )
        candidate_metrics = EvaluationMetrics.from_mapping(run.data.metrics)
        PromotionCriteria().evaluate(candidate_metrics)

        registered_model = client.get_registered_model(self._registered_model_name)
        tenant_id = registered_model.tags.get("tenant_id")
        currency = registered_model.tags.get("currency")
        tenant_scope_hash = registered_model.tags.get("tenant_scope_hash")
        if not tenant_id or not currency or not tenant_scope_hash:
            raise PromotionRejectedError("Registered model lacks tenant/currency governance tags.")

        try:
            bundle = self.load(f"models:/{self._registered_model_name}/{version}")
        except ModelUnavailableError as error:
            raise PromotionRejectedError(
                "Candidate artifact integrity validation failed."
            ) from error
        expected_bundle_metadata = {
            "model_version": candidate.tags.get("model_version"),
            "feature_version": PricingFeatureFactory.VERSION,
            "feature_schema_hash": PricingFeatureFactory.SCHEMA_HASH,
            "currency": currency,
            "tenant_id": tenant_id,
            "tenant_scope_hash": tenant_scope_hash,
        }
        actual_bundle_metadata = {
            "model_version": getattr(bundle, "version", None),
            "feature_version": getattr(bundle, "feature_contract_version", None),
            "feature_schema_hash": getattr(bundle, "feature_schema_hash", None),
            "currency": getattr(bundle, "currency", None),
            "tenant_id": bundle.tenant_ids[0]
            if len(getattr(bundle, "tenant_ids", ())) == 1
            else None,
            "tenant_scope_hash": sha256(
                "\0".join(getattr(bundle, "tenant_ids", ())).encode()
            ).hexdigest(),
        }
        version_metadata = {key: candidate.tags.get(key) for key in expected_bundle_metadata}
        if (
            actual_bundle_metadata != expected_bundle_metadata
            or version_metadata != expected_bundle_metadata
        ):
            raise PromotionRejectedError(
                "Candidate bundle, version tags, and registered-model governance are incompatible."
            )

        previous = None
        try:
            previous = client.get_model_version_by_alias(
                name=self._registered_model_name,
                alias="champion",
            )
        except mlflow.exceptions.MlflowException as error:
            if not self._is_missing_alias(error):
                raise

        if decision == "promotion":
            benchmark_registry_version = run.data.params.get("benchmark_registry_version")
            current_champion_version = str(previous.version) if previous is not None else None
            if benchmark_registry_version != current_champion_version:
                raise PromotionRejectedError(
                    "Candidate benchmark evidence is stale because the champion alias changed. "
                    "Retrain and reevaluate the candidate against the current champion."
                )

        tracked_keys = (
            "lifecycle",
            "approved_by",
            "approved_at",
            "approval_status",
            "champion_decision",
            "champion_decision_reason",
        )
        affected_versions = {version}
        if previous is not None and str(previous.version) != str(version):
            affected_versions.add(str(previous.version))
        originals = {
            affected_version: {
                key: client.get_model_version(
                    self._registered_model_name,
                    affected_version,
                ).tags.get(key)
                for key in tracked_keys
            }
            for affected_version in affected_versions
        }

        def restore_tags() -> None:
            for affected_version, tags in originals.items():
                current_tags = client.get_model_version(
                    self._registered_model_name,
                    affected_version,
                ).tags
                for key, value in tags.items():
                    if value is None:
                        if key in current_tags:
                            client.delete_model_version_tag(
                                self._registered_model_name,
                                affected_version,
                                key,
                            )
                    else:
                        client.set_model_version_tag(
                            self._registered_model_name,
                            affected_version,
                            key,
                            value,
                        )

        try:
            client.set_model_version_tag(
                self._registered_model_name,
                version,
                "approved_by",
                approver,
            )
            client.set_model_version_tag(
                self._registered_model_name,
                version,
                "approved_at",
                datetime.now(UTC).isoformat(),
            )
            client.set_model_version_tag(
                self._registered_model_name,
                version,
                "approval_status",
                "approved",
            )
            client.set_model_version_tag(
                self._registered_model_name,
                version,
                "champion_decision",
                decision,
            )
            if reason is not None:
                client.set_model_version_tag(
                    self._registered_model_name,
                    version,
                    "champion_decision_reason",
                    reason,
                )
            client.set_model_version_tag(
                self._registered_model_name,
                version,
                "lifecycle",
                "champion",
            )
            if previous is not None and str(previous.version) != str(version):
                client.set_model_version_tag(
                    self._registered_model_name,
                    str(previous.version),
                    "lifecycle",
                    "superseded",
                )
            # Alias mutation is deliberately last: no required audit write can
            # fail after the serving source of truth has changed.
            client.set_registered_model_alias(
                name=self._registered_model_name,
                alias="champion",
                version=version,
            )
        except Exception as error:
            try:
                current = None
                try:
                    current = client.get_model_version_by_alias(
                        name=self._registered_model_name,
                        alias="champion",
                    )
                except mlflow.exceptions.MlflowException as alias_error:
                    if not self._is_missing_alias(alias_error):
                        raise
                if current is not None and str(current.version) == str(version):
                    if previous is None:
                        client.delete_registered_model_alias(
                            self._registered_model_name,
                            "champion",
                        )
                    else:
                        client.set_registered_model_alias(
                            self._registered_model_name,
                            "champion",
                            str(previous.version),
                        )
                restore_tags()
            except Exception as rollback_error:
                raise PromotionRejectedError(
                    "Promotion failed and automatic rollback requires operator reconciliation."
                ) from rollback_error
            raise PromotionRejectedError(
                "Promotion failed; prior alias and tags were restored."
            ) from error

    def get_champion_metrics(self) -> EvaluationMetrics | None:
        """Return aggregate holdout evidence for the current champion, if one exists."""

        client = MlflowClient(tracking_uri=self._tracking_uri)
        try:
            champion = client.get_model_version_by_alias(
                name=self._registered_model_name,
                alias="champion",
            )
        except mlflow.exceptions.MlflowException as error:
            if self._is_missing_alias(error):
                return None
            raise
        self._validate_champion_governance(
            client=client,
            model_name=self._registered_model_name,
            model_version=champion,
        )
        if not champion.run_id:
            return None
        run = client.get_run(champion.run_id)
        return EvaluationMetrics.from_mapping(run.data.metrics)

    def load_champion(self) -> QuantileLightGBMDemandModel | None:
        """Load the champion bundle for evaluation on the candidate's current holdout."""

        client = MlflowClient(tracking_uri=self._tracking_uri)
        try:
            champion = client.get_model_version_by_alias(
                name=self._registered_model_name,
                alias="champion",
            )
        except mlflow.exceptions.MlflowException as error:
            if self._is_missing_alias(error):
                return None
            raise
        self._validate_champion_governance(
            client=client,
            model_name=self._registered_model_name,
            model_version=champion,
        )
        return self.load(f"models:/{self._registered_model_name}/{champion.version}")

    def load(self, uri: str) -> QuantileLightGBMDemandModel:
        """Download the bundle embedded in an MLflow model URI."""

        try:
            mlflow.set_tracking_uri(self._tracking_uri)
            expected_content_hash, immutable_uri = self._resolve_model_uri(uri)
            local_model_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=immutable_uri))
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
            actual_content_hash = _sha256_file(bundle_path)
            if not compare_digest(actual_content_hash, expected_content_hash):
                raise ModelUnavailableError(
                    "MLflow model bundle failed SHA-256 integrity verification."
                )
            return LocalModelBundleStore().load(str(bundle_path))
        except Exception as error:
            raise ModelUnavailableError(f"Could not load model URI {uri}: {error}") from error

    @staticmethod
    def _is_lowercase_sha256(value: str) -> bool:
        """Return whether a value is a canonical lowercase SHA-256 digest."""

        return len(value) == 64 and all(character in "0123456789abcdef" for character in value)

    def _validate_champion_governance(
        self,
        *,
        client: MlflowClient,
        model_name: str,
        model_version: Any,
    ) -> None:
        """Reject a manually aliased version that bypassed promotion or rollback."""

        required_version_tags = {
            "lifecycle": "champion",
            "promotion_gate": "passed",
            "approval_status": "approved",
            "feature_version": PricingFeatureFactory.VERSION,
            "feature_schema_hash": PricingFeatureFactory.SCHEMA_HASH,
        }
        invalid_tags = {
            key: model_version.tags.get(key)
            for key, expected in required_version_tags.items()
            if model_version.tags.get(key) != expected
        }
        if invalid_tags:
            raise ModelUnavailableError(
                "Champion alias lacks approved lifecycle and feature-contract evidence."
            )

        approver = model_version.tags.get("approved_by", "").strip()
        if (
            not approver
            or len(approver) > 200
            or not all(character.isprintable() for character in approver)
        ):
            raise ModelUnavailableError("Champion alias lacks a valid named approver.")
        try:
            approved_at = datetime.fromisoformat(model_version.tags.get("approved_at", ""))
        except ValueError as error:
            raise ModelUnavailableError(
                "Champion alias lacks a valid approval timestamp."
            ) from error
        if approved_at.tzinfo is None:
            raise ModelUnavailableError("Champion approval timestamp must include a timezone.")

        decision = model_version.tags.get("champion_decision")
        if decision not in {"promotion", "rollback"}:
            raise ModelUnavailableError("Champion alias lacks an auditable deployment decision.")
        if decision == "rollback":
            reason = model_version.tags.get("champion_decision_reason", "").strip()
            if (
                len(reason) < 10
                or len(reason) > 500
                or not all(character.isprintable() for character in reason)
            ):
                raise ModelUnavailableError("Rollback champion lacks an auditable reason.")

        if not model_version.run_id:
            raise ModelUnavailableError("Champion alias does not reference an auditable run.")
        run = client.get_run(model_version.run_id)
        if run.info.status != "FINISHED":
            raise ModelUnavailableError("Champion alias references a run that is not FINISHED.")

        registered_model = client.get_registered_model(model_name)
        registered_binding = {
            "tenant_id": registered_model.tags.get("tenant_id"),
            "tenant_scope_hash": registered_model.tags.get("tenant_scope_hash"),
            "currency": registered_model.tags.get("currency"),
        }
        if (
            registered_model.tags.get("tenant_scope") != "single-tenant"
            or not registered_binding["tenant_id"]
            or not self._is_lowercase_sha256(registered_binding["tenant_scope_hash"] or "")
            or not registered_binding["currency"]
        ):
            raise ModelUnavailableError(
                "Champion registered-model namespace lacks a valid tenant/currency binding."
            )

        version_binding = {key: model_version.tags.get(key) for key in registered_binding}
        run_binding = {key: run.data.params.get(key) for key in registered_binding}
        if version_binding != registered_binding or run_binding != registered_binding:
            raise ModelUnavailableError(
                "Champion run, version, and registered-model tenant/currency bindings differ."
            )
        if run.data.params.get("promotion_gate") != "passed":
            raise ModelUnavailableError("Champion run lacks passed promotion evidence.")
        if run.data.params.get("model_version") != model_version.tags.get("model_version"):
            raise ModelUnavailableError("Champion run and version identify different models.")
        if (
            run.data.params.get("feature_version") != PricingFeatureFactory.VERSION
            or run.data.params.get("feature_schema_hash") != PricingFeatureFactory.SCHEMA_HASH
        ):
            raise ModelUnavailableError(
                "Champion run lacks the current semantic feature-contract evidence."
            )
        if not self._is_lowercase_sha256(run.data.params.get("dataset_fingerprint", "")):
            raise ModelUnavailableError("Champion run lacks a valid dataset fingerprint.")

        content_hash = model_version.tags.get("model_content_hash", "")
        if (
            not self._is_lowercase_sha256(content_hash)
            or run.data.params.get("model_content_hash") != content_hash
        ):
            raise ModelUnavailableError(
                "Champion run and version lack matching artifact integrity evidence."
            )

    def _resolve_model_uri(self, uri: str) -> tuple[str, str]:
        """Resolve integrity evidence and freeze aliases to an immutable version URI."""

        if not uri.startswith("models:/"):
            raise ModelUnavailableError("MLflow registry loading requires a governed models:/ URI.")
        reference = uri.removeprefix("models:/")
        client = MlflowClient(tracking_uri=self._tracking_uri)
        if "@" in reference:
            model_name, alias = reference.rsplit("@", 1)
            if not model_name or not alias:
                raise ModelUnavailableError("Malformed MLflow model alias URI.")
            model_version = client.get_model_version_by_alias(model_name, alias)
            if alias == "champion":
                self._validate_champion_governance(
                    client=client,
                    model_name=model_name,
                    model_version=model_version,
                )
        else:
            model_name, separator, version = reference.rpartition("/")
            if not separator or not model_name or not version:
                raise ModelUnavailableError("Malformed MLflow model version URI.")
            model_version = client.get_model_version(model_name, version)
        content_hash = model_version.tags.get("model_content_hash", "")
        if not self._is_lowercase_sha256(content_hash):
            raise ModelUnavailableError(
                "MLflow model version is missing a valid bundle integrity checksum."
            )
        immutable_uri = f"models:/{model_name}/{model_version.version}"
        return content_hash, immutable_uri
