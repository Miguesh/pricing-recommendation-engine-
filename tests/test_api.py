"""Contract tests for the public pricing API."""

import asyncio
import json
from copy import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

import pricing_engine.interfaces.api.container as container_module
from pricing_engine.config import Settings
from pricing_engine.domain.exceptions import ModelUnavailableError
from pricing_engine.infrastructure.model_registry import MLflowModelRegistry
from pricing_engine.interfaces.api.app import (
    RECOMMENDATION_PATH,
    RequestBodyLimitMiddleware,
    create_app,
)
from pricing_engine.interfaces.api.container import (
    ApplicationContainer,
    ExperimentalProfileFailureCode,
    ExperimentalProfileState,
)
from pricing_engine.interfaces.api.schemas import PricingRecommendationRequest

PRODUCTION_API_KEY = "production-key-with-at-least-32-characters"
STATISTICAL_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "statistical" / "plusbnb-consumer.synthetic.json"
)


def _payload() -> dict[str, object]:
    return {
        "tenant_id": "demo-tenant",
        "property_id": "property-a",
        "stay_date": "2026-07-24",
        "as_of_date": "2026-07-14",
        "currency": "USD",
        "current_price": "125.00",
        "historical_occupancy_7d": 0.72,
        "booking_pace_7d": 0.35,
        "competitor_price_median": "130.00",
        "bedrooms": 2,
        "accommodates": 4,
        "review_score": 4.7,
        "constraints": {
            "min_price": "90.00",
            "max_price": "170.00",
            "price_increment": "5.00",
            "max_price_change_pct": "0.20",
            "min_expected_occupancy": 0.20,
        },
    }


def _statistical_payload() -> dict[str, object]:
    return json.loads(STATISTICAL_FIXTURE_PATH.read_text(encoding="utf-8"))


def _assert_experimental_failure_is_isolated(
    client: TestClient,
    *,
    api_key: str | None = None,
    organization_id: str = "org-synthetic",
) -> None:
    headers = {"X-API-Key": api_key} if api_key is not None else {}
    statistical_payload = _statistical_payload()
    statistical_payload["organization_id"] = organization_id
    legacy_payload = _payload()
    legacy_payload["tenant_id"] = organization_id

    live_response = client.get("/health/live")
    stable_readiness = client.get("/health/ready")
    capabilities_response = client.get("/v1/capabilities", headers=headers)
    statistical_response = client.post(
        "/v1/pricing/recommendations",
        json=statistical_payload,
        headers=headers,
    )
    experimental_readiness = client.get(
        "/health/ready",
        params={"profile": "PERFORMANCE_AWARE_EXPERIMENTAL"},
    )
    legacy_response = client.post(
        "/v1/pricing/recommendations",
        json=legacy_payload,
        headers=headers,
    )

    assert live_response.status_code == 200
    assert stable_readiness.status_code == 200
    assert capabilities_response.status_code == 200
    assert statistical_response.status_code == 200, statistical_response.text
    assert statistical_response.json()["engine_profile"] == ("MARKET_EVIDENCE_STATISTICAL_V1")
    assert experimental_readiness.status_code == 503
    assert legacy_response.status_code == 503
    assert legacy_response.json() == {
        "error": "ModelUnavailableError",
        "detail": "Pricing model is temporarily unavailable.",
    }


def test_recommendation_contract_requires_auth_and_returns_explanation(training_outcome) -> None:
    settings = Settings(
        environment="test",
        api_key=SecretStr("test-key"),
        recommendation_max_candidates=100,
    )
    container = ApplicationContainer(settings, predictor=training_outcome.model)
    app = create_app(settings, container=container)
    with TestClient(app) as client:
        unauthorized = client.post("/v1/pricing/recommendations", json=_payload())
        response = client.post(
            "/v1/pricing/recommendations",
            json=_payload(),
            headers={"X-API-Key": "test-key", "X-Request-ID": "request-123"},
        )

    assert container.experimental_profile_state is ExperimentalProfileState.READY
    assert unauthorized.status_code == 401
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["currency"] == "USD"
    assert 90 <= float(body["recommended_price"]) <= 150
    assert body["model_version"] == training_outcome.model.version
    assert body["local_feature_contributions"]
    assert body["global_feature_importance"]
    assert response.headers["X-Request-ID"] == "request-123"


def test_readiness_reports_stable_profile_without_an_approved_model() -> None:
    settings = Settings(environment="test")
    container = ApplicationContainer(settings)
    with TestClient(create_app(settings, container=container)) as client:
        live_response = client.get("/health/live")
        response = client.get("/health/ready")
        experimental_response = client.get(
            "/health/ready",
            params={"profile": "PERFORMANCE_AWARE_EXPERIMENTAL"},
        )
        recommendation_response = client.post("/v1/pricing/recommendations", json=_payload())
        invalid_payload = _payload()
        invalid_payload["constraints"] = {
            "min_price": "120.00",
            "max_price": "90.00",
            "price_increment": "5.00",
        }
        validation_response = client.post("/v1/pricing/recommendations", json=invalid_payload)

    assert live_response.status_code == 200
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_loaded": False,
        "model_version": None,
        "stable_statistical_profile_ready": True,
        "checked_profile": "MARKET_EVIDENCE_STATISTICAL_V1",
    }
    assert experimental_response.status_code == 503
    assert recommendation_response.status_code == 503
    assert validation_response.status_code == 422
    assert container.experimental_profile_state is ExperimentalProfileState.NOT_CONFIGURED
    assert container.experimental_failure_code is None
    assert container.model_loaded is False
    assert container.model_version is None


def test_production_settings_fail_closed_and_blank_local_key_is_unconfigured() -> None:
    assert Settings(environment="local", api_key="").api_key is None

    with pytest.raises(ValidationError, match="api_key is required"):
        Settings(environment="production")
    with pytest.raises(ValidationError, match="trusted_hosts"):
        Settings(
            environment="production",
            api_key=SecretStr(PRODUCTION_API_KEY),
            api_key_tenant_id="demo-tenant",
        )
    with pytest.raises(ValidationError, match="at least 32"):
        Settings(
            environment="production",
            api_key=SecretStr("weak-key"),
            api_key_tenant_id="demo-tenant",
            trusted_hosts=["api.example.com"],
        )
    statistical_only = Settings(
        environment="production",
        api_key=SecretStr(PRODUCTION_API_KEY),
        api_key_tenant_id="demo-tenant",
        serving_tenant_id="demo-tenant",
        trusted_hosts=["api.example.com"],
    )
    assert statistical_only.model_uri is None


def test_production_security_controls_and_tenant_binding(training_outcome) -> None:
    settings = Settings(
        environment="production",
        api_key=SecretStr(PRODUCTION_API_KEY),
        api_key_tenant_id="demo-tenant",
        serving_tenant_id="demo-tenant",
        trusted_hosts=["testserver"],
        model_uri="models:/pricing-demand@champion",
        max_request_body_bytes=1_024,
    )
    container = ApplicationContainer(settings, predictor=training_outcome.model)
    with TestClient(create_app(settings, container=container)) as client:
        docs_response = client.get("/docs")
        openapi_response = client.get("/openapi.json")
        unauthorized_tenant = _payload()
        unauthorized_tenant["tenant_id"] = "tenant-b"
        forbidden_response = client.post(
            "/v1/pricing/recommendations",
            json=unauthorized_tenant,
            headers={"X-API-Key": PRODUCTION_API_KEY},
        )
        response = client.post(
            "/v1/pricing/recommendations",
            json=_payload(),
            headers={"X-API-Key": PRODUCTION_API_KEY},
        )
        oversized_response = client.post(
            "/v1/pricing/recommendations",
            content=b"{" + (b" " * 1_024) + b"}",
            headers={"Content-Type": "application/json", "X-API-Key": PRODUCTION_API_KEY},
        )
        untrusted_host_response = client.get(
            "/health/live",
            headers={"Host": "attacker.example"},
        )

    assert docs_response.status_code == 404
    assert openapi_response.status_code == 404
    assert forbidden_response.status_code == 403
    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert oversized_response.status_code == 413
    assert untrusted_host_response.status_code == 400


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stay_date", 1_782_345_600),
        ("stay_date", "2026-07-24T00:00:00"),
        ("bedrooms", True),
        ("accommodates", "4"),
        ("historical_occupancy_7d", "0.72"),
    ],
)
def test_public_contract_rejects_semantic_coercion(
    training_outcome,
    field: str,
    value: object,
) -> None:
    settings = Settings(environment="test")
    container = ApplicationContainer(settings, predictor=training_outcome.model)
    payload = _payload()
    payload[field] = value

    with TestClient(create_app(settings, container=container)) as client:
        response = client.post("/v1/pricing/recommendations", json=payload)

    assert response.status_code == 422


def test_legacy_tenant_identifier_limit_remains_100_characters() -> None:
    payload = _payload()
    payload["tenant_id"] = "t" * 100

    assert len(PricingRecommendationRequest.model_validate(payload).tenant_id) == 100

    payload["tenant_id"] = "t" * 101
    with pytest.raises(ValidationError, match="tenant_id"):
        PricingRecommendationRequest.model_validate(payload)


def test_api_rejects_price_change_outside_validated_policy_space(training_outcome) -> None:
    settings = Settings(environment="test")
    container = ApplicationContainer(settings, predictor=training_outcome.model)
    payload = _payload()
    constraints = payload["constraints"]
    assert isinstance(constraints, dict)
    constraints["max_price_change_pct"] = "0.36"

    with TestClient(create_app(settings, container=container)) as client:
        response = client.post("/v1/pricing/recommendations", json=payload)

    assert response.status_code == 422


def test_api_rejects_null_price_change_guardrail(training_outcome) -> None:
    settings = Settings(environment="test")
    container = ApplicationContainer(settings, predictor=training_outcome.model)
    payload = _payload()
    constraints = payload["constraints"]
    assert isinstance(constraints, dict)
    constraints["max_price_change_pct"] = None

    with TestClient(create_app(settings, container=container)) as client:
        response = client.post("/v1/pricing/recommendations", json=payload)

    assert response.status_code == 422


def test_api_key_is_checked_before_body_parsing_and_handles_unicode(training_outcome) -> None:
    settings = Settings(environment="test", api_key=SecretStr("test-key"))
    container = ApplicationContainer(settings, predictor=training_outcome.model)
    app = create_app(settings, container=container)
    with TestClient(app) as client:
        malformed = client.post(
            "/v1/pricing/recommendations",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        unicode_key = client.post(
            "/v1/pricing/recommendations",
            json=_payload(),
            headers=[(b"X-API-Key", "tést-key".encode())],
        )

    assert malformed.status_code == 401
    assert unicode_key.status_code == 401
    assert "test-key" not in repr(app.user_middleware)


def test_body_reader_times_out_before_request_validation() -> None:
    async def never_called_app(scope, receive, send) -> None:
        raise AssertionError("A timed-out body must not reach FastAPI.")

    middleware = RequestBodyLimitMiddleware(
        never_called_app,
        maximum_bytes=1_024,
        read_timeout_seconds=0.001,
        path=RECOMMENDATION_PATH,
        api_key_digest=None,
    )
    sent_messages = []

    async def delayed_receive():
        await asyncio.sleep(0.05)
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def capture_send(message) -> None:
        sent_messages.append(message)

    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": RECOMMENDATION_PATH,
                "headers": [],
            },
            delayed_receive,
            capture_send,
        )
    )

    assert sent_messages[0]["status"] == 408
    assert b"RequestTimeout" in sent_messages[1]["body"]


def test_proxy_tenant_must_match_single_tenant_deployment(training_outcome) -> None:
    settings = Settings(
        environment="production",
        api_key=SecretStr(PRODUCTION_API_KEY),
        serving_tenant_id="demo-tenant",
        trust_proxy_identity=True,
        trusted_tenant_header="X-Authenticated-Tenant",
        trusted_hosts=["testserver"],
        model_uri="models:/pricing-demand@champion",
    )
    container = ApplicationContainer(settings, predictor=training_outcome.model)
    headers = {
        "X-API-Key": PRODUCTION_API_KEY,
        "X-Authenticated-Tenant": "other-tenant",
    }
    with TestClient(create_app(settings, container=container)) as client:
        wrong_route = client.post(
            "/v1/pricing/recommendations",
            json=_payload(),
            headers=headers,
        )
        headers["X-Authenticated-Tenant"] = "demo-tenant"
        correct_route = client.post(
            "/v1/pricing/recommendations",
            json=_payload(),
            headers=headers,
        )

    assert wrong_route.status_code == 403
    assert correct_route.status_code == 200


def test_unexpected_errors_use_safe_stable_envelope(training_outcome) -> None:
    settings = Settings(environment="test")
    container = ApplicationContainer(settings, predictor=training_outcome.model)

    class BrokenService:
        def execute(self, context):
            raise RuntimeError("secret internal details")

    with TestClient(
        create_app(settings, container=container),
        raise_server_exceptions=False,
    ) as client:
        container.__dict__["_service"] = BrokenService()
        response = client.post(
            "/v1/pricing/recommendations",
            json=_payload(),
            headers={"X-Request-ID": "failure-request"},
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": "InternalServerError",
        "detail": "An unexpected internal error occurred.",
    }
    assert response.headers["X-Request-ID"] == "failure-request"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_coordinates_are_optional_but_must_be_provided_together(training_outcome) -> None:
    settings = Settings(environment="test")
    container = ApplicationContainer(settings, predictor=training_outcome.model)
    invalid_payload = _payload()
    invalid_payload["latitude"] = 25.7617

    with TestClient(create_app(settings, container=container)) as client:
        invalid_response = client.post("/v1/pricing/recommendations", json=invalid_payload)
        valid_payload = _payload()
        valid_payload.update({"latitude": 25.7617, "longitude": -80.1918})
        valid_response = client.post("/v1/pricing/recommendations", json=valid_payload)

    assert invalid_response.status_code == 422
    assert valid_response.status_code == 200


def test_recommendation_rejects_currency_mismatch(training_outcome) -> None:
    settings = Settings(environment="test")
    container = ApplicationContainer(settings, predictor=training_outcome.model)
    payload = _payload()
    payload["currency"] = "EUR"

    with TestClient(create_app(settings, container=container)) as client:
        response = client.post("/v1/pricing/recommendations", json=payload)

    assert response.status_code == 422
    assert "Model currency is USD" in response.json()["detail"]


def test_container_rejects_stale_feature_contract(training_outcome) -> None:
    stale_predictor = copy(training_outcome.model)
    stale_predictor.feature_names = ("stale_feature",)
    container = ApplicationContainer(
        Settings(environment="test"),
        predictor=stale_predictor,
    )

    with pytest.raises(ModelUnavailableError, match="incompatible feature contract"):
        container.load_configured_model()


def test_container_fails_readiness_when_required_shap_warmup_fails(training_outcome) -> None:
    broken_predictor = copy(training_outcome.model)

    class BrokenExplainer:
        def shap_values(self, features):
            raise RuntimeError("simulated SHAP native failure")

    broken_predictor.__dict__["_shap_explainer"] = BrokenExplainer()
    container = ApplicationContainer(
        Settings(environment="test"),
        predictor=broken_predictor,
    )

    with pytest.raises(ModelUnavailableError, match="warm-up"):
        container.load_configured_model()


def test_production_loader_failure_preserves_stable_profile_and_logs_safely(
    monkeypatch,
) -> None:
    model_uri = "models:/private-artifact-path@champion"
    sensitive_failure = "provider said credential-marker for a private artifact"
    tenant_id = "org-synthetic"
    settings = Settings(
        environment="production",
        api_key=SecretStr(PRODUCTION_API_KEY),
        api_key_tenant_id=tenant_id,
        serving_tenant_id=tenant_id,
        trusted_hosts=["testserver"],
        model_uri=model_uri,
        mlflow_tracking_uri="https://private.example.invalid/artifacts",
    )
    container = ApplicationContainer(settings)

    def fail_to_load(_: MLflowModelRegistry, configured_uri: str) -> None:
        assert configured_uri == model_uri
        raise RuntimeError(sensitive_failure)

    class RecordingLogger:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def warning(self, event: str, **context: object) -> None:
            self.events.append((event, context))

    recording_logger = RecordingLogger()
    monkeypatch.setattr(MLflowModelRegistry, "load", fail_to_load)
    monkeypatch.setattr(container_module, "logger", recording_logger)

    with TestClient(create_app(settings, container=container)) as client:
        _assert_experimental_failure_is_isolated(
            client,
            api_key=PRODUCTION_API_KEY,
            organization_id=tenant_id,
        )

    assert container.experimental_profile_state is ExperimentalProfileState.UNAVAILABLE
    assert container.experimental_failure_code is ExperimentalProfileFailureCode.MODEL_LOAD_FAILED
    assert container.model_loaded is False
    assert container.model_version is None
    assert recording_logger.events == [
        (
            "experimental_profile_initialization_failed",
            {
                "state": ExperimentalProfileState.UNAVAILABLE.value,
                "code": ExperimentalProfileFailureCode.MODEL_LOAD_FAILED.value,
                "failure_type": "RuntimeError",
            },
        )
    ]
    rendered_events = repr(recording_logger.events)
    for sensitive_value in (
        model_uri,
        sensitive_failure,
        tenant_id,
        PRODUCTION_API_KEY,
        "private-artifact-path",
    ):
        assert sensitive_value not in rendered_events


@pytest.mark.parametrize("failure_stage", ["contract", "warmup"])
def test_optional_artifact_validation_failures_leave_only_stable_profile_available(
    training_outcome,
    monkeypatch,
    failure_stage: str,
) -> None:
    predictor = copy(training_outcome.model)
    if failure_stage == "contract":
        predictor.feature_names = ("stale_feature",)
    else:

        class BrokenExplainer:
            def shap_values(self, features):
                raise RuntimeError("simulated optional SHAP failure")

        predictor.__dict__["_shap_explainer"] = BrokenExplainer()

    settings = Settings(
        environment="test",
        model_uri="models:/synthetic-optional-model@champion",
    )
    container = ApplicationContainer(settings)
    monkeypatch.setattr(container, "_load_configured_predictor", lambda _: predictor)

    with TestClient(create_app(settings, container=container)) as client:
        _assert_experimental_failure_is_isolated(client)

    assert container.experimental_profile_state is ExperimentalProfileState.UNAVAILABLE
    assert container.experimental_failure_code is ExperimentalProfileFailureCode.MODEL_LOAD_FAILED
    assert container.model_loaded is False
    assert container.model_version is None
    with pytest.raises(ModelUnavailableError):
        _ = container.recommendation_service


def test_optional_service_construction_failure_publishes_no_partial_model(
    training_outcome,
    monkeypatch,
) -> None:
    settings = Settings(
        environment="test",
        model_uri="models:/synthetic-optional-model@champion",
    )
    container = ApplicationContainer(settings)
    monkeypatch.setattr(
        container,
        "_load_configured_predictor",
        lambda _: training_outcome.model,
    )

    def fail_service_construction(predictor) -> None:
        assert predictor is training_outcome.model
        raise RuntimeError("simulated service construction failure")

    monkeypatch.setattr(container, "_build_service", fail_service_construction)

    with TestClient(create_app(settings, container=container)) as client:
        _assert_experimental_failure_is_isolated(client)

    assert container.experimental_profile_state is ExperimentalProfileState.UNAVAILABLE
    assert container.experimental_failure_code is ExperimentalProfileFailureCode.MODEL_LOAD_FAILED
    assert container.model_loaded is False
    assert container.model_version is None
    assert container.__dict__["_predictor"] is None
    assert container.__dict__["_service"] is None


def test_valid_configured_model_is_published_ready_after_optional_initialization(
    training_outcome,
    monkeypatch,
) -> None:
    settings = Settings(
        environment="test",
        model_uri="models:/synthetic-optional-model@champion",
    )
    container = ApplicationContainer(settings)
    monkeypatch.setattr(
        container,
        "_load_configured_predictor",
        lambda _: training_outcome.model,
    )

    with TestClient(create_app(settings, container=container)) as client:
        stable_readiness = client.get("/health/ready")
        experimental_readiness = client.get(
            "/health/ready",
            params={"profile": "PERFORMANCE_AWARE_EXPERIMENTAL"},
        )
        recommendation_response = client.post(
            "/v1/pricing/recommendations",
            json=_payload(),
        )
        statistical_response = client.post(
            "/v1/pricing/recommendations",
            json=_statistical_payload(),
        )

    assert stable_readiness.status_code == 200
    assert experimental_readiness.status_code == 200
    assert recommendation_response.status_code == 200, recommendation_response.text
    assert statistical_response.status_code == 200, statistical_response.text
    assert statistical_response.json()["engine_profile"] == "MARKET_EVIDENCE_STATISTICAL_V1"
    assert container.experimental_profile_state is ExperimentalProfileState.READY
    assert container.experimental_failure_code is None
    assert container.model_loaded is True
    assert container.model_version == training_outcome.model.version


def test_strict_configured_model_loader_rethrows_and_clears_partial_state(
    monkeypatch,
) -> None:
    settings = Settings(
        environment="test",
        model_uri="models:/synthetic-optional-model@champion",
    )
    container = ApplicationContainer(settings)

    def fail_to_load(_: str) -> None:
        raise RuntimeError("strict loader failure")

    monkeypatch.setattr(container, "_load_configured_predictor", fail_to_load)

    with pytest.raises(RuntimeError, match="strict loader failure"):
        container.load_configured_model()

    assert container.experimental_profile_state is ExperimentalProfileState.UNAVAILABLE
    assert container.experimental_failure_code is ExperimentalProfileFailureCode.MODEL_LOAD_FAILED
    assert container.model_loaded is False
    assert container.model_version is None
