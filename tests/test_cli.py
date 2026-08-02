"""Tests for CLI command wiring and machine-readable operational output."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import pricing_engine.cli as cli
from pricing_engine.application.ports import RegisteredModel
from pricing_engine.config import Settings

runner = CliRunner()


def test_generate_and_drift_commands(demo_observations, tmp_path: Path) -> None:
    generated = tmp_path / "demo.parquet"
    result = runner.invoke(
        cli.app,
        [
            "generate-demo-data",
            "--output",
            str(generated),
            "--properties",
            "2",
            "--decision-days",
            "12",
        ],
    )
    assert result.exit_code == 0, result.output
    assert generated.exists()

    reference = tmp_path / "reference.parquet"
    current = tmp_path / "current.parquet"
    demo_observations.to_parquet(reference, index=False)
    shifted = demo_observations.copy()
    shifted["listed_price"] = shifted["listed_price"] * 1.5
    shifted.to_parquet(current, index=False)
    drift_result = runner.invoke(
        cli.app,
        ["drift", "--reference", str(reference), "--current", str(current)],
    )
    assert drift_result.exit_code == 0, drift_result.output
    assert "requires_retraining_review" in drift_result.output


def test_train_command_persists_bundle(
    monkeypatch, training_outcome, demo_observations, tmp_path: Path
) -> None:
    input_path = tmp_path / "observations.parquet"
    output_path = tmp_path / "model"
    demo_observations.to_parquet(input_path, index=False)

    class StubTrainer:
        def execute(self, frame):
            assert len(frame) == len(demo_observations)
            return training_outcome

    monkeypatch.setattr(cli, "_build_training_service", lambda: StubTrainer())
    result = runner.invoke(
        cli.app,
        ["train", "--input", str(input_path), "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert (output_path / "pricing_demand_bundle.joblib").exists()


def test_registry_commands_are_explicit_and_do_not_require_network(
    monkeypatch,
    training_outcome,
    demo_observations,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "observations.parquet"
    demo_observations.to_parquet(input_path, index=False)
    events: dict[str, object] = {}

    class FakeRegistry:
        def __init__(self, **kwargs):
            events["registry_args"] = kwargs

        def promote(self, *, version: str, approved_by: str) -> None:
            events["promotion"] = (version, approved_by)

        def rollback(self, *, version: str, approved_by: str, reason: str) -> None:
            events["rollback"] = (version, approved_by, reason)

    class FakePipeline:
        def __init__(self, **kwargs):
            events["pipeline_args"] = kwargs

        def execute(self, frame):
            events["retrain_rows"] = len(frame)

            class Result:
                outcome = training_outcome
                registered_model = RegisteredModel(
                    run_id="run-1",
                    model_name="pricing-demand",
                    model_version="5",
                    model_uri="models:/pricing-demand/5",
                )
                dataset_fingerprint = "fingerprint"

            return Result()

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(environment="test"))
    from pricing_engine.application import retraining_service
    from pricing_engine.infrastructure import model_registry

    monkeypatch.setattr(model_registry, "MLflowModelRegistry", FakeRegistry)
    monkeypatch.setattr(retraining_service, "RetrainingPipeline", FakePipeline)

    retrain_result = runner.invoke(cli.app, ["retrain", "--input", str(input_path)])
    promote_result = runner.invoke(
        cli.app,
        ["promote", "--version", "5", "--approved-by", "Miguel"],
    )
    rollback_result = runner.invoke(
        cli.app,
        [
            "rollback",
            "--version",
            "4",
            "--approved-by",
            "Miguel",
            "--reason",
            "Restore stable model after detected regression.",
        ],
    )

    assert retrain_result.exit_code == 0, retrain_result.output
    assert promote_result.exit_code == 0, promote_result.output
    assert rollback_result.exit_code == 0, rollback_result.output
    assert json.loads(retrain_result.stdout)["candidate"]["model_version"] == "5"
    assert json.loads(promote_result.stdout) == {"promoted_version": "5", "alias": "champion"}
    assert json.loads(rollback_result.stdout)["rolled_back_to_version"] == "4"
    assert events["retrain_rows"] == len(demo_observations)
    assert events["promotion"] == ("5", "Miguel")
    assert events["rollback"] == (
        "4",
        "Miguel",
        "Restore stable model after detected regression.",
    )
