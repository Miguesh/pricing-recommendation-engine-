"""Contract tests for the public pricing API."""

from fastapi.testclient import TestClient
from pydantic import SecretStr

from pricing_engine.config import Settings
from pricing_engine.interfaces.api.app import create_app
from pricing_engine.interfaces.api.container import ApplicationContainer


def _payload() -> dict[str, object]:
    return {
        "tenant_id": "tenant-a",
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


def test_readiness_is_unavailable_without_an_approved_model() -> None:
    settings = Settings(environment="test")
    with TestClient(create_app(settings)) as client:
        live_response = client.get("/health/live")
        response = client.get("/health/ready")
        recommendation_response = client.post("/v1/pricing/recommendations", json=_payload())
        invalid_payload = _payload()
        invalid_payload["constraints"] = {
            "min_price": "120.00",
            "max_price": "90.00",
            "price_increment": "5.00",
        }
        validation_response = client.post("/v1/pricing/recommendations", json=invalid_payload)

    assert live_response.status_code == 200
    assert response.status_code == 503
    assert recommendation_response.status_code == 503
    assert validation_response.status_code == 422
