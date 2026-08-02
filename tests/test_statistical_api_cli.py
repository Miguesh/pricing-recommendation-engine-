"""API and CLI integration tests for the statistical contract v1."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from typer.testing import CliRunner

import pricing_engine.cli as cli
import pricing_engine.interfaces.api.app as api_app_module
from pricing_engine.application.statistical_service import capabilities
from pricing_engine.config import Settings
from pricing_engine.domain.statistical import (
    MAX_COMPARABLES,
    MAX_STATISTICAL_IDENTITY_LENGTH,
    MAX_STATISTICAL_REQUEST_BYTES,
    EngineProfile,
    StatisticalPricingRequest,
)
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


def _maximum_identifier(prefix: str, index: int | None = None, length: int = 128) -> str:
    suffix = f"-{index:03d}" if index is not None else ""
    seed = prefix + suffix
    assert len(seed) <= length
    return seed + ("x" * (length - len(seed)))


def _maximum_statistical_payload() -> dict[str, object]:
    payload = _payload()
    payload.update(
        {
            "request_id": _maximum_identifier("request"),
            "correlation_id": _maximum_identifier("correlation"),
            "organization_id": _maximum_identifier("organization"),
            "audit_id": _maximum_identifier("audit"),
            "target_property_id": _maximum_identifier("property"),
            "market_id": _maximum_identifier("market"),
            "market_config_version": _maximum_identifier("version", length=64),
        }
    )
    target_features = payload["target_property_features"]
    assert isinstance(target_features, dict)
    target_features["property_type"] = "p" * 64
    target_features["amenity_codes"] = [
        _maximum_identifier("amenity", index, 64) for index in range(100)
    ]

    comparable_template = deepcopy(_comparables(payload)[0])
    lineage = payload["input_lineage"]
    assert isinstance(lineage, list)
    lineage_template = deepcopy(lineage[0])
    comparables: list[dict[str, object]] = []
    lineage_entries: list[dict[str, object]] = []
    for index in range(MAX_COMPARABLES):
        observation_hash = f"{index + 1:064x}"
        comparable = deepcopy(comparable_template)
        comparable.update(
            {
                "comparable_id": _maximum_identifier("comparable", index),
                "evidence_type": _maximum_identifier("evidence", index),
                "verification_level": _maximum_identifier("verification", index),
                "collection_method": _maximum_identifier("collection", index),
                "effective_base_nightly_rate": "999999999999.9999",
                "effective_total_nightly_rate": "999999999999.9999",
                "similarity_score": "100.0000",
                "similarity_factors": {
                    _maximum_identifier("factor", factor): "100.0000" for factor in range(50)
                },
                "quality_flags": [_maximum_identifier("flag", flag) for flag in range(50)],
                "source_family_id": _maximum_identifier("source", index),
                "source_version": _maximum_identifier("source-version", index, 64),
                "observation_hash": observation_hash,
                "lineage_hash": "a" * 64,
            }
        )
        entry = deepcopy(lineage_template)
        entry.update(
            {
                "lineage_id": _maximum_identifier("lineage", index),
                "source_family_id": comparable["source_family_id"],
                "source_version": comparable["source_version"],
                "observation_hash": observation_hash,
                "lineage_hash": comparable["lineage_hash"],
            }
        )
        comparables.append(comparable)
        lineage_entries.append(entry)
    payload["comparables"] = comparables
    payload["input_lineage"] = lineage_entries
    return payload


def _compact_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _resolve_schema_ref(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> dict[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str):
        return schema
    assert reference.startswith("#/")
    resolved: Any = root_schema
    for token in reference.removeprefix("#/").split("/"):
        assert isinstance(resolved, dict)
        resolved = resolved[token.replace("~1", "/").replace("~0", "~")]
    assert isinstance(resolved, dict)
    return {
        **resolved,
        **{key: value for key, value in schema.items() if key != "$ref"},
    }


def _schema_capability_inventory(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    prefix: str = "",
) -> tuple[set[str], set[str]]:
    resolved = _resolve_schema_ref(schema, root_schema)
    properties = resolved.get("properties", {})
    required_names = set(resolved.get("required", ()))
    assert isinstance(properties, dict)

    required_paths: set[str] = set()
    optional_paths: set[str] = set()
    for name, raw_property_schema in properties.items():
        assert isinstance(name, str)
        assert isinstance(raw_property_schema, dict)
        path = f"{prefix}.{name}" if prefix else name
        property_schema = _resolve_schema_ref(raw_property_schema, root_schema)
        if name not in required_names:
            optional_paths.add(path)
            continue

        required_paths.add(f"{path}=false" if property_schema.get("const") is False else path)
        nested_schema: dict[str, Any] | None = None
        nested_prefix = path
        if property_schema.get("type") == "array":
            raw_items = property_schema.get("items")
            assert isinstance(raw_items, dict)
            nested_schema = _resolve_schema_ref(raw_items, root_schema)
            nested_prefix = f"{path}[]"
        elif "properties" in property_schema:
            nested_schema = property_schema

        if nested_schema is not None and "properties" in nested_schema:
            nested_required, nested_optional = _schema_capability_inventory(
                nested_schema,
                root_schema,
                prefix=nested_prefix,
            )
            required_paths.update(nested_required)
            optional_paths.update(nested_optional)

    return required_paths, optional_paths


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
    assert profiles["MARKET_EVIDENCE_STATISTICAL_V1"]["maximum_comparables"] == 50
    assert profiles["MARKET_EVIDENCE_STATISTICAL_V1"]["maximum_input_lineage_entries"] == 50
    assert profiles["MARKET_EVIDENCE_STATISTICAL_V1"]["maximum_request_body_bytes"] == (
        MAX_STATISTICAL_REQUEST_BYTES
    )
    assert (
        profiles["MARKET_EVIDENCE_STATISTICAL_V1"]["effective_request_body_limit_bytes"]
        == MAX_STATISTICAL_REQUEST_BYTES
    )
    assert profiles["MARKET_EVIDENCE_STATISTICAL_V1"]["batch_supported"] is False
    assert profiles["PERFORMANCE_AWARE_EXPERIMENTAL"]["status"] == "experimental"
    assert profiles["PERFORMANCE_AWARE_EXPERIMENTAL"]["uses_artifacts"] is True


def test_statistical_capabilities_match_required_and_optional_json_schema_paths() -> None:
    first = capabilities()
    second = capabilities()
    stable = next(
        profile
        for profile in first.profiles
        if profile.profile is EngineProfile.MARKET_EVIDENCE_STATISTICAL_V1
    )
    schema = StatisticalPricingRequest.model_json_schema(mode="validation", by_alias=True)
    required_paths, optional_paths = _schema_capability_inventory(schema, schema)

    assert MAX_STATISTICAL_IDENTITY_LENGTH == 128
    assert first.model_dump_json() == second.model_dump_json()
    assert len(stable.required_fields) == len(set(stable.required_fields))
    assert len(stable.optional_fields) == len(set(stable.optional_fields))
    assert set(stable.required_fields) == required_paths
    assert set(stable.optional_fields) == optional_paths


@pytest.mark.parametrize("identity_length", [101, MAX_STATISTICAL_IDENTITY_LENGTH])
@pytest.mark.parametrize("identity_mode", ["api_key", "trusted_proxy"])
def test_statistical_identity_up_to_128_is_contract_valid_and_authorized_fail_closed(
    identity_length: int,
    identity_mode: str,
) -> None:
    identity = "o" * identity_length
    padded_identity = f" {identity} "
    api_key = "k" * 32
    payload = _payload()
    payload["organization_id"] = identity
    StatisticalPricingRequest.model_validate(payload)

    identity_settings: dict[str, object]
    allowed_headers = {"X-API-Key": api_key}
    if identity_mode == "api_key":
        identity_settings = {
            "api_key_tenant_id": padded_identity,
            "serving_tenant_id": padded_identity,
        }
        missing_identity_headers: dict[str, str] = {}
        mismatched_payload = deepcopy(payload)
        mismatched_payload["organization_id"] = "other-organization"
        mismatched_headers = allowed_headers
    else:
        identity_settings = {
            "serving_tenant_id": padded_identity,
            "trust_proxy_identity": True,
            "trusted_tenant_header": "X-Authenticated-Tenant",
        }
        allowed_headers["X-Authenticated-Tenant"] = padded_identity
        missing_identity_headers = {"X-API-Key": api_key}
        mismatched_payload = payload
        mismatched_headers = {
            "X-API-Key": api_key,
            "X-Authenticated-Tenant": "other-organization",
        }

    settings = Settings(
        environment="production",
        api_key=SecretStr(api_key),
        trusted_hosts=["testserver"],
        **identity_settings,
    )
    with TestClient(create_app(settings)) as client:
        missing_identity = client.post(
            "/v1/pricing/recommendations",
            json=payload,
            headers=missing_identity_headers,
        )
        mismatched_identity = client.post(
            "/v1/pricing/recommendations",
            json=mismatched_payload,
            headers=mismatched_headers,
        )
        allowed = client.post(
            "/v1/pricing/recommendations",
            json=payload,
            headers=allowed_headers,
        )

    assert missing_identity.status_code == 401
    assert mismatched_identity.status_code == 403
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["engine_profile"] == "MARKET_EVIDENCE_STATISTICAL_V1"


@pytest.mark.parametrize("field", ["api_key_tenant_id", "serving_tenant_id"])
def test_configured_identity_rejects_129_characters(field: str) -> None:
    values: dict[str, object] = {
        "environment": "test",
        "api_key": SecretStr("k" * 32),
        field: "o" * (MAX_STATISTICAL_IDENTITY_LENGTH + 1),
    }

    with pytest.raises(ValidationError, match=field):
        Settings(**values)


def test_trusted_tenant_header_name_limit_remains_100_characters() -> None:
    header_name = "X" * 100

    settings = Settings(trusted_tenant_header=f" {header_name} ")

    assert settings.trusted_tenant_header == header_name
    with pytest.raises(ValidationError, match="trusted_tenant_header"):
        Settings(trusted_tenant_header="X" * 101)


def test_trusted_proxy_rejects_129_character_identity_without_disclosure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized_identity = "o" * (MAX_STATISTICAL_IDENTITY_LENGTH + 1)
    api_key = "k" * 32
    log_events: list[tuple[str, dict[str, object]]] = []

    class RecordingLogger:
        def info(self, event: str, **context: object) -> None:
            log_events.append((event, context))

    monkeypatch.setattr(api_app_module, "logger", RecordingLogger())
    settings = Settings(
        environment="test",
        api_key=SecretStr(api_key),
        serving_tenant_id="organization-synthetic",
        trust_proxy_identity=True,
        trusted_tenant_header="X-Authenticated-Tenant",
    )
    with TestClient(create_app(settings)) as client:
        trusted_header_response = client.post(
            "/v1/pricing/recommendations",
            json=_payload(),
            headers={
                "X-API-Key": api_key,
                "X-Authenticated-Tenant": oversized_identity,
            },
        )
        oversized_payload = _payload()
        oversized_payload["organization_id"] = oversized_identity
        contract_response = client.post(
            "/v1/pricing/recommendations",
            json=oversized_payload,
            headers={
                "X-API-Key": api_key,
                "X-Authenticated-Tenant": "organization-synthetic",
            },
        )

    assert trusted_header_response.status_code == 401
    assert trusted_header_response.json() == {
        "error": "HTTPException",
        "detail": "Trusted tenant identity is invalid.",
    }
    assert trusted_header_response.headers["WWW-Authenticate"] == "ApiKey"
    assert contract_response.status_code == 422
    assert oversized_identity not in trusted_header_response.text + contract_response.text
    assert oversized_identity not in repr(log_events)


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


def test_statistical_api_exposes_and_enforces_lower_operational_body_limit() -> None:
    settings = Settings(environment="test", max_request_body_bytes=1_024)
    with TestClient(create_app(settings)) as client:
        capabilities_response = client.get("/v1/capabilities")
        response = client.post("/v1/pricing/recommendations", json=_payload())

    stable_profile = next(
        profile
        for profile in capabilities_response.json()["profiles"]
        if profile["profile"] == "MARKET_EVIDENCE_STATISTICAL_V1"
    )
    assert stable_profile["maximum_request_body_bytes"] == MAX_STATISTICAL_REQUEST_BYTES
    assert stable_profile["effective_request_body_limit_bytes"] == 1_024
    assert response.status_code == 413
    assert response.json()["error"] == "RequestTooLarge"


def test_default_and_maximum_transport_settings_share_the_contract_limit() -> None:
    assert Settings(environment="test").max_request_body_bytes == MAX_STATISTICAL_REQUEST_BYTES
    assert (
        Settings(
            environment="test",
            max_request_body_bytes=MAX_STATISTICAL_REQUEST_BYTES,
        ).max_request_body_bytes
        == MAX_STATISTICAL_REQUEST_BYTES
    )
    with pytest.raises(ValidationError, match="max_request_body_bytes"):
        Settings(
            environment="test",
            max_request_body_bytes=MAX_STATISTICAL_REQUEST_BYTES + 1,
        )


def test_statistical_api_rejects_body_above_contract_transport_limit() -> None:
    oversized_body = b"x" * (MAX_STATISTICAL_REQUEST_BYTES + 1)
    with TestClient(create_app(Settings(environment="test"))) as client:
        response = client.post(
            "/v1/pricing/recommendations",
            content=oversized_body,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json() == {
        "error": "RequestTooLarge",
        "detail": "Request body exceeds the configured limit.",
    }


def test_openapi_exposes_capabilities_and_both_pricing_contracts() -> None:
    schema = create_app(Settings(environment="test")).openapi()
    recommendation = schema["paths"]["/v1/pricing/recommendations"]["post"]
    request_schema = recommendation["requestBody"]["content"]["application/json"]["schema"]

    assert "/v1/capabilities" in schema["paths"]
    assert "StatisticalPricingRequest" in schema["components"]["schemas"]
    assert "StatisticalPricingResponse" in schema["components"]["schemas"]
    assert "PricingRecommendationRequest" in schema["components"]["schemas"]
    assert len(request_schema["anyOf"]) == 2
    statistical_request = schema["components"]["schemas"]["StatisticalPricingRequest"]
    legacy_request = schema["components"]["schemas"]["PricingRecommendationRequest"]
    assert statistical_request["properties"]["organization_id"]["maxLength"] == 128
    assert statistical_request["properties"]["comparables"]["maxItems"] == MAX_COMPARABLES
    assert statistical_request["properties"]["input_lineage"]["maxItems"] == MAX_COMPARABLES
    assert legacy_request["properties"]["tenant_id"]["maxLength"] == 100
    assert legacy_request["properties"]["property_id"]["maxLength"] == 100
    assert recommendation["responses"]["413"]["description"] == "Content Too Large"
    assert recommendation["responses"]["422"]["description"] == "Unprocessable Content"


def test_maximum_contract_payload_fits_and_is_accepted_by_api_and_cli(
    tmp_path: Path,
) -> None:
    payload = _maximum_statistical_payload()
    compact_body = _compact_json(payload)
    request = StatisticalPricingRequest.model_validate(payload)
    maximum_path = tmp_path / "maximum-statistical-request.synthetic.json"
    maximum_path.write_bytes(compact_body)

    assert len(request.comparables) == MAX_COMPARABLES
    assert len(request.input_lineage) == MAX_COMPARABLES
    assert len(compact_body) <= MAX_STATISTICAL_REQUEST_BYTES

    with TestClient(create_app(Settings(environment="test"))) as client:
        api_response = client.post(
            "/v1/pricing/recommendations",
            content=compact_body,
            headers={"Content-Type": "application/json"},
        )
    cli_response = runner.invoke(cli.app, ["recommend", "--input", str(maximum_path)])

    assert api_response.status_code == 200, api_response.text
    assert api_response.json()["status"] in {"recommended", "abstained"}
    assert api_response.json()["evidence_components"]["dispersion_ratio"] == "0.0000"
    assert cli_response.exit_code == 0, cli_response.output
    assert json.loads(cli_response.stdout)["status"] in {"recommended", "abstained"}


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


def test_cli_rejects_oversized_input_without_echoing_content(tmp_path: Path) -> None:
    private_marker = b"private-marker-must-not-be-echoed"
    oversized_path = tmp_path / "oversized.synthetic.json"
    oversized_path.write_bytes(
        private_marker + (b"x" * (MAX_STATISTICAL_REQUEST_BYTES + 1 - len(private_marker)))
    )

    result = runner.invoke(cli.app, ["validate", "--input", str(oversized_path)])

    assert result.exit_code == 2
    assert f"{MAX_STATISTICAL_REQUEST_BYTES:,}-byte limit" in result.output
    assert private_marker.decode("ascii") not in result.output


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
