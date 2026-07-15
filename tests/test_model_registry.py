"""Integration tests for local MLflow candidate registration and artifact loading."""

from pricing_engine.infrastructure.model_registry import MLflowModelRegistry


def test_mlflow_candidate_registration_promotion_and_load(training_outcome, tmp_path) -> None:
    registry = MLflowModelRegistry(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        experiment_name="pricing-test",
        registered_model_name="pricing-test-demand",
    )

    registered = registry.log_candidate(
        model=training_outcome.model,
        metrics=training_outcome.metrics,
        feature_version=training_outcome.feature_version,
        dataset_fingerprint="test-fingerprint",
    )
    registry.promote(version=registered.model_version, approved_by="test-reviewer")
    loaded = registry.load(registered.model_uri)

    assert registered.model_version
    assert loaded.version == training_outcome.model.version
