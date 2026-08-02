"""Contract tests for the public pricing API."""

import asyncio
from copy import copy

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from pricing_engine.config import Settings
from pricing_engine.domain.exceptions import ModelUnavailableError
from pricing_engine.interfaces.api.app import (
    RECOMMENDATION_PATH,
    RequestBodyLimitMiddleware,
    create_app,
)
from pricing_engine.interfaces.api.container import ApplicationContainer
from pricing_engine.interfaces.api.schemas import PricingRecommendationRequest

PRODUCTION_API_KEY = "production-key-with-at-least-32-characters"


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
    with TestClient(create_app(settings)) as client:
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

    container.__dict__["_service"] = BrokenService()
    with TestClient(
        create_app(settings, container=container),
        raise_server_exceptions=False,
    ) as client:
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

    with pytest.raises(ModelUnavailableError, match="incompatible feature contract"):
        ApplicationContainer(Settings(environment="test"), predictor=stale_predictor)


def test_container_fails_readiness_when_required_shap_warmup_fails(training_outcome) -> None:
    broken_predictor = copy(training_outcome.model)

    class BrokenExplainer:
        def shap_values(self, features):
            raise RuntimeError("simulated SHAP native failure")

    broken_predictor.__dict__["_shap_explainer"] = BrokenExplainer()

    with pytest.raises(ModelUnavailableError, match="warm-up"):
        ApplicationContainer(Settings(environment="test"), predictor=broken_predictor)
