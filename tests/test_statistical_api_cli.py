"""API and CLI integration tests for the statistical contract v1."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr
from typer.testing import CliRunner

import pricing_engine.cli as cli
from pricing_engine.config import Settings
from pricing_engine.interfaces.api.app import create_app

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "statistical" / "plusbnb-consumer.synthetic.json"
)
runner = CliRunner()


def _payload() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _comparables(payload: dict[str, object]) -> list[dict[str, object]]:
    comparables = payload["comparables"]
    assert isinstance(comparables, list)
    return comparables


def test_capabilities_endpoint_discovers_stable_and_experimental_profiles() -> None:
    with TestClient(create_app(Settings(environment="test"))) as client:
        response = client.get("/v1/capabilities")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["default_profile"] == "MARKET_EVIDENCE_STATISTICAL_V1"
    profiles = {item["profile"]: item for item in body["profiles"]}
    assert profiles["MARKET_EVIDENCE_STATISTICAL_V1"]["status"] == "stable_contract"
    assert profiles["MARKET_EVIDENCE_STATISTICAL_V1"]["uses_artifacts"] is False
    assert profiles["MARKET_EVIDENCE_STATISTICAL_V1"]["deterministic"] is True
    assert profiles["PERFORMANCE_AWARE_EXPERIMENTAL"]["status"] == "experimental"
    assert profiles["PERFORMANCE_AWARE_EXPERIMENTAL"]["uses_artifacts"] is True


def test_capabilities_honors_api_key_without_exposing_it() -> None:
    settings = Settings(environment="test", api_key=SecretStr("synthetic-test-key"))
    with TestClient(create_app(settings)) as client:
        denied = client.get("/v1/capabilities")
        allowed = client.get(
            "/v1/capabilities",
            headers={"X-API-Key": "synthetic-test-key"},
        )

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert "synthetic-test-key" not in denied.text + allowed.text


def test_statistical_api_recommends_without_loading_experimental_model() -> None:
    with TestClient(create_app(Settings(environment="test"))) as client:
        response = client.post("/v1/pricing/recommendations", json=_payload())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["engine_profile"] == "MARKET_EVIDENCE_STATISTICAL_V1"
    assert body["status"] == "recommended"
    assert body["recommendation"] == "115.00"
    assert body["deterministic_run_id"]
    assert body["publication_allowed"] is False
    assert body["commercial_validation"] is False


def test_statistical_api_enforces_organization_binding() -> None:
    settings = Settings(
        environment="test",
        api_key=SecretStr("synthetic-test-key"),
        api_key_tenant_id="bound-organization",
    )
    payload = _payload()
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/pricing/recommendations",
            json=payload,
            headers={"X-API-Key": "synthetic-test-key"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Caller is not authorized for the requested tenant."


def test_statistical_api_returns_typed_redacted_contract_error() -> None:
    payload = _payload()
    _comparables(payload)[0]["currency"] = "USD"
    with TestClient(create_app(Settings(environment="test"))) as client:
        response = client.post("/v1/pricing/recommendations", json=payload)

    assert response.status_code == 422
    assert response.json() == {
        "error": "StatisticalContractError",
        "detail": ("Every comparable must use the request currency; conversion is not supported."),
        "code": "CURRENCY_MISMATCH",
    }
    assert "comparables" not in response.text


def test_statistical_payload_uses_existing_body_limit() -> None:
    settings = Settings(environment="test", max_request_body_bytes=1_024)
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/pricing/recommendations", json=_payload())

    assert response.status_code == 413
    assert response.json()["error"] == "RequestTooLarge"


def test_openapi_exposes_capabilities_and_both_pricing_contracts() -> None:
    schema = create_app(Settings(environment="test")).openapi()
    recommendation = schema["paths"]["/v1/pricing/recommendations"]["post"]
    request_schema = recommendation["requestBody"]["content"]["application/json"]["schema"]

    assert "/v1/capabilities" in schema["paths"]
    assert "StatisticalPricingRequest" in schema["components"]["schemas"]
    assert "StatisticalPricingResponse" in schema["components"]["schemas"]
    assert "PricingRecommendationRequest" in schema["components"]["schemas"]
    assert len(request_schema["anyOf"]) == 2


def test_cli_validate_recommend_capabilities_and_export_openapi(tmp_path: Path) -> None:
    validate = runner.invoke(cli.app, ["validate", "--input", str(FIXTURE_PATH)])
    recommend = runner.invoke(cli.app, ["recommend", "--input", str(FIXTURE_PATH)])
    discovered = runner.invoke(cli.app, ["capabilities"])
    openapi_path = tmp_path / "openapi.json"
    second_openapi_path = tmp_path / "openapi-second.json"
    exported = runner.invoke(
        cli.app,
        ["export-openapi", "--output", str(openapi_path)],
    )
    exported_again = runner.invoke(
        cli.app,
        ["export-openapi", "--output", str(second_openapi_path)],
    )

    assert validate.exit_code == 0, validate.output
    assert json.loads(validate.stdout)["valid"] is True
    assert len(json.loads(validate.stdout)["request_hash"]) == 64
    assert recommend.exit_code == 0, recommend.output
    assert json.loads(recommend.stdout)["status"] == "recommended"
    assert discovered.exit_code == 0, discovered.output
    assert json.loads(discovered.stdout)["default_profile"] == ("MARKET_EVIDENCE_STATISTICAL_V1")
    assert exported.exit_code == 0, exported.output
    assert exported_again.exit_code == 0, exported_again.output
    assert openapi_path.is_file()
    assert json.loads(openapi_path.read_text(encoding="utf-8"))["openapi"].startswith("3.")
    assert openapi_path.read_bytes() == second_openapi_path.read_bytes()


def test_cli_contract_failure_is_typed_and_does_not_echo_payload(tmp_path: Path) -> None:
    payload = _payload()
    comparables = _comparables(payload)
    comparables.append(deepcopy(comparables[0]))
    invalid_path = tmp_path / "invalid.synthetic.json"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(cli.app, ["recommend", "--input", str(invalid_path)])

    assert result.exit_code == 2
    assert json.loads(result.stderr) == {
        "error": "StatisticalContractError",
        "code": "DUPLICATE_COMPARABLE",
    }
    assert "property-synthetic-001" not in result.output


def test_openapi_export_ignores_pricing_environment_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline-openapi.json"
    overridden_path = tmp_path / "overridden-openapi.json"
    monkeypatch.delenv("PRICING_API_TITLE", raising=False)
    baseline = runner.invoke(
        cli.app,
        ["export-openapi", "--output", str(baseline_path)],
    )
    monkeypatch.setenv("PRICING_API_TITLE", "Environment Injected Title")
    overridden = runner.invoke(
        cli.app,
        ["export-openapi", "--output", str(overridden_path)],
    )

    assert baseline.exit_code == 0, baseline.output
    assert overridden.exit_code == 0, overridden.output
    assert baseline_path.read_bytes() == overridden_path.read_bytes()
    schema = json.loads(overridden_path.read_text(encoding="utf-8"))
    assert schema["info"]["title"] == "Pricing Recommendation Engine"
