"""Tests for retraining orchestration without external registry side effects."""

from dataclasses import dataclass

from pricing_engine.application.retraining_service import RetrainingPipeline
from pricing_engine.infrastructure.model_registry import RegisteredModel


@dataclass
class StubTrainer:
    outcome: object

    def execute(self, observations):
        return self.outcome


class RecordingRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def log_candidate(self, **kwargs):
        self.calls.append(kwargs)
        return RegisteredModel(
            run_id="run-1",
            model_name="pricing-demand",
            model_version="7",
            model_uri="models:/pricing-demand/7",
        )


def test_retraining_registers_a_gated_candidate_with_dataset_provenance(
    training_outcome,
    demo_observations,
) -> None:
    registry = RecordingRegistry()
    pipeline = RetrainingPipeline(
        trainer=StubTrainer(training_outcome),
        registry=registry,
    )

    result = pipeline.execute(demo_observations)

    assert result.registered_model.model_version == "7"
    assert len(result.dataset_fingerprint) == 64
    assert registry.calls[0]["model"] is training_outcome.model
