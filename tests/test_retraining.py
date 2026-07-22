"""Tests for retraining orchestration without external registry side effects."""

from dataclasses import dataclass, replace

from pricing_engine.application.ports import RegisteredModel
from pricing_engine.application.retraining_service import RetrainingPipeline


@dataclass
class StubTrainer:
    outcome: object

    def execute(self, observations, *, benchmark_model=None):
        if benchmark_model is None:
            return self.outcome
        return replace(
            self.outcome,
            benchmark_metrics=self.outcome.metrics,
            benchmark_model_version=benchmark_model.version,
            benchmark_status="evaluated",
        )


class RecordingRegistry:
    def __init__(self, champion=None) -> None:
        self.calls: list[dict[str, object]] = []
        self.champion = champion

    def load_champion(self):
        return self.champion

    def log_candidate(self, **kwargs):
        self.calls.append(kwargs)
        return RegisteredModel(
            run_id="run-1",
            model_name="pricing-demand",
            model_version="7",
            model_uri="models:/pricing-demand/7",
        )


class RecordingPromotionCriteria:
    def __init__(self) -> None:
        self.champion_metrics = None

    def evaluate(self, metrics, *, champion_metrics=None) -> None:
        self.champion_metrics = champion_metrics


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


def test_retraining_compares_candidate_with_champion_on_the_current_holdout(
    training_outcome,
    demo_observations,
) -> None:
    criteria = RecordingPromotionCriteria()
    registry = RecordingRegistry(champion=training_outcome.model)
    pipeline = RetrainingPipeline(
        trainer=StubTrainer(training_outcome),
        registry=registry,
        promotion_criteria=criteria,
    )

    pipeline.execute(demo_observations)

    assert criteria.champion_metrics is training_outcome.metrics
    assert registry.calls[0]["benchmark_status"] == "evaluated"


def test_dataset_fingerprint_is_stable_across_row_order_and_index(demo_observations) -> None:
    reordered = demo_observations.sample(frac=1.0, random_state=17)
    reordered.index = range(10_000, 10_000 + len(reordered))

    assert RetrainingPipeline._fingerprint(demo_observations) == RetrainingPipeline._fingerprint(
        reordered
    )
