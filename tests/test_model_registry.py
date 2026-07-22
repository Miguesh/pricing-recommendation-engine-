"""Integration tests for local MLflow candidate registration and artifact loading."""

from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import pytest
from mlflow import MlflowClient

from pricing_engine.domain.exceptions import ModelUnavailableError, PromotionRejectedError
from pricing_engine.infrastructure.model_registry import MLflowModelRegistry


@pytest.mark.parametrize(
    ("benchmark_status", "include_metrics", "benchmark_model_version"),
    [
        ("evaluated", False, "champion-v1"),
        ("evaluated", True, None),
        ("not_available", True, None),
        ("not_available", False, "champion-v1"),
        ("incompatible_contract", True, "champion-v1"),
        ("incompatible_contract", False, None),
    ],
)
def test_registry_rejects_inconsistent_benchmark_evidence(
    training_outcome,
    tmp_path,
    benchmark_status,
    include_metrics,
    benchmark_model_version,
) -> None:
    registry = MLflowModelRegistry(
        tracking_uri=f"sqlite:///{(tmp_path / 'invalid-benchmark.db').as_posix()}",
        experiment_name="invalid-benchmark",
        registered_model_name="invalid-benchmark",
    )

    with pytest.raises(ModelUnavailableError, match="Benchmark evidence is inconsistent"):
        registry.log_candidate(
            model=training_outcome.model,
            metrics=training_outcome.metrics,
            feature_version=training_outcome.feature_version,
            dataset_fingerprint="a" * 64,
            benchmark_metrics=(training_outcome.metrics if include_metrics else None),
            benchmark_model_version=benchmark_model_version,
            benchmark_status=benchmark_status,
        )


def test_mlflow_candidate_registration_promotion_and_load(
    training_outcome,
    tmp_path,
    monkeypatch,
) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    registry = MLflowModelRegistry(
        tracking_uri=tracking_uri,
        experiment_name="pricing-test",
        registered_model_name="pricing-test-demand",
    )
    assert registry.get_champion_metrics() is None

    registered = registry.log_candidate(
        model=training_outcome.model,
        metrics=training_outcome.metrics,
        feature_version=training_outcome.feature_version,
        dataset_fingerprint="a" * 64,
        benchmark_metrics=None,
        benchmark_model_version=None,
        benchmark_status="not_available",
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    registered_version = client.get_model_version(
        registered.model_name,
        registered.model_version,
    )
    client.set_registered_model_alias(
        registered.model_name,
        "champion",
        registered.model_version,
    )
    manual_alias_uri = f"models:/{registered.model_name}@champion"
    with pytest.raises(ModelUnavailableError, match="approved lifecycle"):
        registry.load(manual_alias_uri)
    with pytest.raises(ModelUnavailableError, match="approved lifecycle"):
        registry.load_champion()
    with pytest.raises(ModelUnavailableError, match="approved lifecycle"):
        registry.get_champion_metrics()
    client.delete_registered_model_alias(registered.model_name, "champion")

    source_run = client.get_run(registered.run_id)
    incomplete_run = client.create_run(source_run.info.experiment_id)
    incomplete_version = client.create_model_version(
        name=registered.model_name,
        source=registered_version.source,
        run_id=incomplete_run.info.run_id,
        tags=registered_version.tags,
    )
    with pytest.raises(PromotionRejectedError, match="run is not complete"):
        registry.promote(
            version=str(incomplete_version.version),
            approved_by="test-reviewer",
        )
    client.set_terminated(incomplete_run.info.run_id, status="KILLED")
    original_set_tag = MlflowClient.set_model_version_tag
    calls = 0

    def fail_during_approval(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated audit write failure")
        return original_set_tag(self, *args, **kwargs)

    monkeypatch.setattr(MlflowClient, "set_model_version_tag", fail_during_approval)
    with pytest.raises(PromotionRejectedError, match="restored"):
        registry.promote(version=registered.model_version, approved_by="test-reviewer")
    rolled_back = client.get_model_version(registered.model_name, registered.model_version)
    assert rolled_back.tags["lifecycle"] == "candidate"
    assert "approved_by" not in rolled_back.tags

    monkeypatch.setattr(MlflowClient, "set_model_version_tag", original_set_tag)
    registry.promote(version=registered.model_version, approved_by="test-reviewer")
    second_registered = registry.log_candidate(
        model=training_outcome.model,
        metrics=training_outcome.metrics,
        feature_version=training_outcome.feature_version,
        dataset_fingerprint="b" * 64,
        benchmark_metrics=training_outcome.metrics,
        benchmark_model_version=training_outcome.model.version,
        benchmark_status="evaluated",
    )
    stale_registered = registry.log_candidate(
        model=training_outcome.model,
        metrics=training_outcome.metrics,
        feature_version=training_outcome.feature_version,
        dataset_fingerprint="c" * 64,
        benchmark_metrics=training_outcome.metrics,
        benchmark_model_version=training_outcome.model.version,
        benchmark_status="evaluated",
    )
    registry.promote(version=second_registered.model_version, approved_by="test-reviewer")
    with pytest.raises(PromotionRejectedError, match="benchmark evidence is stale"):
        registry.promote(version=stale_registered.model_version, approved_by="test-reviewer")
    tampered_registered = registry.log_candidate(
        model=training_outcome.model,
        metrics=training_outcome.metrics,
        feature_version=training_outcome.feature_version,
        dataset_fingerprint="d" * 64,
        benchmark_metrics=training_outcome.metrics,
        benchmark_model_version=training_outcome.model.version,
        benchmark_status="evaluated",
    )
    tampered_model_dir = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=tampered_registered.model_uri)
    )
    tampered_bundle = next(
        path
        for path in (
            tampered_model_dir / "artifacts" / "pricing_demand_bundle.joblib",
            tampered_model_dir / "artifacts" / "bundle" / "pricing_demand_bundle.joblib",
            tampered_model_dir / "bundle" / "pricing_demand_bundle.joblib",
        )
        if path.exists()
    )
    with tampered_bundle.open("ab") as stream:
        stream.write(b"tampered-after-registration")

    with pytest.raises(ModelUnavailableError, match="PyFunc bundle failed SHA-256"):
        mlflow.pyfunc.load_model(tampered_registered.model_uri)
    with pytest.raises(ModelUnavailableError, match="integrity verification"):
        registry.load(tampered_registered.model_uri)
    with pytest.raises(PromotionRejectedError, match="artifact integrity"):
        registry.promote(
            version=tampered_registered.model_version,
            approved_by="test-reviewer",
        )
    registry.rollback(
        version=registered.model_version,
        approved_by="incident-commander",
        reason="Restore the previously validated model after a serving regression.",
    )
    champion_metrics = registry.get_champion_metrics()
    loaded = registry.load(registered.model_uri)
    model_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=registered.model_uri))
    pyfunc_model = mlflow.pyfunc.load_model(registered.model_uri)
    input_frame = pd.DataFrame(
        [
            {
                name: training_outcome.model.feature_means[name]
                for name in training_outcome.model.feature_names
            }
        ],
        columns=training_outcome.model.feature_names,
    )
    pyfunc_prediction = pyfunc_model.predict(input_frame)
    requirements = (model_dir / "requirements.txt").read_text(encoding="utf-8")
    logged_model = mlflow.models.Model.load(model_dir)

    assert registered.model_version
    assert champion_metrics is not None
    assert champion_metrics.occupancy_mae == training_outcome.metrics.occupancy_mae
    champion = registry.load_champion()
    assert champion is not None
    assert champion.version == training_outcome.model.version
    champion_version = client.get_model_version_by_alias(registered.model_name, "champion")
    assert str(champion_version.version) == registered.model_version
    assert champion_version.tags["champion_decision"] == "rollback"
    assert loaded.version == training_outcome.model.version
    assert logged_model.flavors["python_function"]["code"] == "code"
    assert logged_model.signature is not None
    assert logged_model.signature.inputs.input_names() == list(training_outcome.model.feature_names)
    assert logged_model.signature.outputs.input_names() == [
        "expected_occupancy",
        "lower_occupancy",
        "upper_occupancy",
        "in_distribution_score",
    ]
    assert (model_dir / "code" / "pricing_engine").is_dir()
    assert (model_dir / "uv.lock").is_file()
    assert (model_dir / "pyproject.toml").is_file()
    for dependency in ("joblib", "lightgbm", "numba", "pandas", "scikit-learn", "shap"):
        assert f"{dependency}==" in requirements
    assert list(pyfunc_prediction.columns) == [
        "expected_occupancy",
        "lower_occupancy",
        "upper_occupancy",
        "in_distribution_score",
    ]
    assert len(pyfunc_prediction) == 1
