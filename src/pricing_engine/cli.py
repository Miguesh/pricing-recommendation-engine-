"""Operational command-line interface for local workflows and scheduled jobs."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from pydantic import ValidationError

from pricing_engine.application.statistical_service import (
    MarketEvidenceStatisticalV1,
    capabilities,
    request_hash,
)
from pricing_engine.config import get_settings
from pricing_engine.domain.exceptions import StatisticalContractError
from pricing_engine.domain.statistical import (
    MAX_STATISTICAL_REQUEST_BYTES,
    StatisticalPricingRequest,
)

if TYPE_CHECKING:
    import pandas as pd

    from pricing_engine.application.training_service import TrainDemandModel

app = typer.Typer(
    name="pricing-engine",
    help="Dynamic pricing training, registry, and monitoring operations.",
    no_args_is_help=True,
)


@app.callback()
def configure_console_encoding() -> None:
    """Make third-party CLI output portable across Windows code pages."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def _build_training_service() -> TrainDemandModel:
    """Compose application training ports at the CLI boundary."""

    from pricing_engine.application.training_service import TrainDemandModel
    from pricing_engine.infrastructure.features import PricingFeatureFactory
    from pricing_engine.infrastructure.training import (
        LightGBMDemandModelTrainer,
        PanderaTrainingDataValidator,
    )

    return TrainDemandModel(
        validator=PanderaTrainingDataValidator(),
        feature_factory=PricingFeatureFactory(),
        model_trainer=LightGBMDemandModelTrainer(),
    )


def _read_frame(path: Path) -> pd.DataFrame:
    import pandas as pd

    if not path.exists():
        raise typer.BadParameter(f"Input file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise typer.BadParameter("Input must be a .csv or .parquet file.")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise typer.BadParameter("Output must be a .csv or .parquet file.")


def _read_statistical_request(path: Path) -> StatisticalPricingRequest:
    if not path.is_file():
        raise typer.BadParameter("Input file does not exist.")
    if path.stat().st_size > MAX_STATISTICAL_REQUEST_BYTES:
        raise typer.BadParameter(f"Input exceeds the {MAX_STATISTICAL_REQUEST_BYTES:,}-byte limit.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return StatisticalPricingRequest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise typer.BadParameter(
            "Input is not a valid statistical contract v1 document."
        ) from error


@app.command("capabilities")
def show_capabilities() -> None:
    """Print discoverable engine profiles without loading models or external services."""

    typer.echo(capabilities().model_dump_json(indent=2))


@app.command("validate")
def validate_statistical_contract(
    input: Path = typer.Option(..., help="Statistical contract v1 JSON document."),
) -> None:
    """Validate a stable statistical request without producing a recommendation."""

    request = _read_statistical_request(input)
    try:
        MarketEvidenceStatisticalV1().validate(request)
    except StatisticalContractError as error:
        typer.echo(json.dumps({"valid": False, "code": error.code}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"valid": True, "request_hash": request_hash(request)}, indent=2))


@app.command("recommend")
def recommend_statistical_price(
    input: Path = typer.Option(..., help="Statistical contract v1 JSON document."),
) -> None:
    """Run the pure MARKET_EVIDENCE_STATISTICAL_V1 profile."""

    request = _read_statistical_request(input)
    try:
        response = MarketEvidenceStatisticalV1().recommend(request)
    except StatisticalContractError as error:
        typer.echo(json.dumps({"error": "StatisticalContractError", "code": error.code}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(response.model_dump_json(indent=2))


@app.command("export-openapi")
def export_openapi(
    output: Path = typer.Option(Path("docs/openapi.json"), help="OpenAPI JSON destination."),
) -> None:
    """Export the deterministic API schema without starting the server."""

    from pricing_engine.config import Settings
    from pricing_engine.interfaces.api.app import create_app

    schema_settings = Settings.model_construct(environment="test")
    schema = create_app(schema_settings).openapi()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(json.dumps({"output": str(output), "openapi": schema["openapi"]}, indent=2))


@app.command("generate-demo-data")
def generate_demo_data(
    output: Path = typer.Option(..., help="Destination CSV or Parquet file."),
    properties: int = typer.Option(12, min=1, max=500),
    decision_days: int = typer.Option(180, min=10, max=5_000),
    seed: int = typer.Option(42),
) -> None:
    """Create deterministic non-production data for an end-to-end demonstration."""

    from pricing_engine.infrastructure.demo_data import generate_demo_observations

    frame = generate_demo_observations(
        properties=properties,
        decision_days=decision_days,
        seed=seed,
    )
    _write_frame(frame, output)
    typer.echo(
        json.dumps(
            {"output": str(output), "rows": len(frame), "warning": "synthetic demo data"},
            indent=2,
        )
    )


@app.command()
def train(
    input: Path = typer.Option(..., help="Validated historical CSV or Parquet observations."),
    output: Path = typer.Option(
        Path("artifacts/local-model"),
        help="Local bundle directory for API development.",
    ),
) -> None:
    """Train locally, evaluate chronologically, and persist a portable bundle."""

    from pricing_engine.infrastructure.model_registry import LocalModelBundleStore

    outcome = _build_training_service().execute(_read_frame(input))
    bundle = LocalModelBundleStore().save_local(outcome.model, output)
    typer.echo(
        json.dumps(
            {
                "bundle": str(bundle),
                "model_version": outcome.model.version,
                "metrics": outcome.metrics.as_dict(),
                "partitions": {
                    "train": outcome.train_rows,
                    "calibration": outcome.calibration_rows,
                    "test": outcome.test_rows,
                    "purged_train": outcome.purged_train_rows,
                    "purged_calibration": outcome.purged_calibration_rows,
                },
            },
            indent=2,
        )
    )


@app.command()
def retrain(
    input: Path = typer.Option(..., help="Validated historical CSV or Parquet observations."),
    experiment_name: str = typer.Option("pricing-demand"),
    registered_model_name: str = typer.Option("pricing-demand"),
) -> None:
    """Train and register a gated candidate. This never changes the champion alias."""

    from pricing_engine.application.retraining_service import RetrainingPipeline
    from pricing_engine.infrastructure.model_registry import MLflowModelRegistry

    settings = get_settings()
    registry = MLflowModelRegistry(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=experiment_name,
        registered_model_name=registered_model_name,
        dependency_project_path=settings.dependency_project_path,
    )
    with redirect_stdout(sys.stderr):
        result = RetrainingPipeline(
            trainer=_build_training_service(),
            registry=registry,
        ).execute(_read_frame(input))
    typer.echo(
        json.dumps(
            {
                "candidate": {
                    "model_name": result.registered_model.model_name,
                    "model_version": result.registered_model.model_version,
                    "model_uri": result.registered_model.model_uri,
                    "run_id": result.registered_model.run_id,
                },
                "metrics": result.outcome.metrics.as_dict(),
                "partitions": {
                    "train": result.outcome.train_rows,
                    "calibration": result.outcome.calibration_rows,
                    "test": result.outcome.test_rows,
                    "purged_train": result.outcome.purged_train_rows,
                    "purged_calibration": result.outcome.purged_calibration_rows,
                },
                "dataset_fingerprint": result.dataset_fingerprint,
                "compared_with_champion": result.outcome.benchmark_metrics is not None,
                "benchmark": {
                    "status": result.outcome.benchmark_status,
                    "model_version": result.outcome.benchmark_model_version,
                    "current_holdout_metrics": (
                        result.outcome.benchmark_metrics.as_dict()
                        if result.outcome.benchmark_metrics is not None
                        else None
                    ),
                },
                "next_step": "Review the candidate, then run promote explicitly.",
            },
            indent=2,
        )
    )


@app.command()
def promote(
    version: str = typer.Option(..., help="MLflow registered model version to promote."),
    approved_by: str = typer.Option(..., help="Named human approver recorded in MLflow."),
    registered_model_name: str = typer.Option("pricing-demand"),
) -> None:
    """Set the MLflow champion alias after an explicit human approval."""

    from pricing_engine.infrastructure.model_registry import MLflowModelRegistry

    settings = get_settings()
    with redirect_stdout(sys.stderr):
        MLflowModelRegistry(
            tracking_uri=settings.mlflow_tracking_uri,
            registered_model_name=registered_model_name,
        ).promote(version=version, approved_by=approved_by)
    typer.echo(json.dumps({"promoted_version": version, "alias": "champion"}, indent=2))


@app.command()
def rollback(
    version: str = typer.Option(..., help="Superseded MLflow version to restore."),
    approved_by: str = typer.Option(..., help="Named human approver recorded in MLflow."),
    reason: str = typer.Option(..., help="Auditable operational reason for the rollback."),
    registered_model_name: str = typer.Option("pricing-demand"),
) -> None:
    """Restore a superseded model version without bypassing governance checks."""

    from pricing_engine.infrastructure.model_registry import MLflowModelRegistry

    settings = get_settings()
    with redirect_stdout(sys.stderr):
        MLflowModelRegistry(
            tracking_uri=settings.mlflow_tracking_uri,
            registered_model_name=registered_model_name,
            dependency_project_path=settings.dependency_project_path,
        ).rollback(version=version, approved_by=approved_by, reason=reason)
    typer.echo(
        json.dumps(
            {"rolled_back_to_version": version, "alias": "champion", "reason": reason},
            indent=2,
        )
    )


@app.command()
def drift(
    reference: Path = typer.Option(..., help="Baseline historical observations."),
    current: Path = typer.Option(..., help="Recent historical observations."),
    threshold: float = typer.Option(0.20, min=0.01, max=1.0),
) -> None:
    """Report population stability drift over the immutable feature contract."""

    from pricing_engine.infrastructure.data_contracts import validate_training_frame
    from pricing_engine.infrastructure.features import PricingFeatureFactory
    from pricing_engine.infrastructure.monitoring import build_drift_report

    factory = PricingFeatureFactory()
    reference_features = factory.build_training_features(
        validate_training_frame(_read_frame(reference))
    )
    current_features = factory.build_training_features(
        validate_training_frame(_read_frame(current))
    )
    report = build_drift_report(reference_features, current_features, threshold=threshold)
    typer.echo(
        json.dumps(
            {
                "threshold": threshold,
                "requires_retraining_review": report.requires_retraining_review,
                "features": [
                    {
                        "feature": item.feature,
                        "status": item.status.value,
                        "psi": (
                            round(item.population_stability_index, 5)
                            if isfinite(item.population_stability_index)
                            else None
                        ),
                        "reference_missing_rate": item.reference_missing_rate,
                        "current_missing_rate": item.current_missing_rate,
                    }
                    for item in report.features
                ],
            },
            indent=2,
        )
    )
