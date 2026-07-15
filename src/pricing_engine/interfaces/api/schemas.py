"""Pydantic contracts for the versioned public pricing API."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
)

from pricing_engine.domain.models import PriceConstraints, PricingContext, PricingRecommendation

CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
PositiveMoney = Annotated[Decimal, Field(gt=0, max_digits=12, decimal_places=2)]


class PriceConstraintsRequest(BaseModel):
    """Client-specified commercial bounds validated before model inference."""

    model_config = ConfigDict(extra="forbid")

    min_price: PositiveMoney
    max_price: PositiveMoney
    price_increment: PositiveMoney = Decimal("1.00")
    max_price_change_pct: Decimal | None = Field(default=Decimal("0.35"), ge=0, le=1)
    min_expected_occupancy: float | None = Field(default=None, ge=0, le=1)

    @field_validator("max_price")
    @classmethod
    def max_price_must_exceed_minimum(cls, max_price: Decimal, info: object) -> Decimal:
        values = getattr(info, "data", {})
        min_price = values.get("min_price")
        if min_price is not None and max_price < min_price:
            raise ValueError("max_price must be greater than or equal to min_price.")
        return max_price

    def to_domain(self) -> PriceConstraints:
        return PriceConstraints(
            min_price=self.min_price,
            max_price=self.max_price,
            price_increment=self.price_increment,
            max_price_change_pct=self.max_price_change_pct,
            min_expected_occupancy=self.min_expected_occupancy,
        )


class PricingRecommendationRequest(BaseModel):
    """A point-in-time request for one property and one stay date."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: Annotated[str, Field(min_length=1, max_length=100)]
    property_id: Annotated[str, Field(min_length=1, max_length=100)]
    stay_date: date
    as_of_date: date
    currency: CurrencyCode = "USD"
    current_price: PositiveMoney
    historical_occupancy_7d: float = Field(ge=0, le=1)
    booking_pace_7d: float = Field(ge=0, le=1)
    competitor_price_median: PositiveMoney
    bedrooms: int = Field(ge=0, le=20)
    accommodates: int = Field(ge=1, le=50)
    review_score: float = Field(ge=0, le=5)
    is_holiday: bool = False
    event_intensity: float = Field(default=0, ge=0, le=5)
    constraints: PriceConstraintsRequest

    def to_domain(self) -> PricingContext:
        return PricingContext(
            tenant_id=self.tenant_id,
            property_id=self.property_id,
            stay_date=self.stay_date,
            as_of_date=self.as_of_date,
            currency=self.currency,
            current_price=self.current_price,
            historical_occupancy_7d=self.historical_occupancy_7d,
            booking_pace_7d=self.booking_pace_7d,
            competitor_price_median=self.competitor_price_median,
            bedrooms=self.bedrooms,
            accommodates=self.accommodates,
            review_score=self.review_score,
            is_holiday=self.is_holiday,
            event_intensity=self.event_intensity,
            constraints=self.constraints.to_domain(),
        )


class FeatureContributionResponse(BaseModel):
    feature: str
    contribution: float
    direction: str


class FeatureImportanceResponse(BaseModel):
    feature: str
    importance_pct: float


class PricingRecommendationResponse(BaseModel):
    """Stable response contract; fields are additive-only within API v1."""

    recommended_price: Decimal
    currency: CurrencyCode
    expected_occupancy: float
    expected_occupancy_interval: tuple[float, float]
    confidence_score: float = Field(ge=0, le=100)
    confidence_method: str
    expected_revenue: Decimal
    model_version: str
    explanation: str
    local_feature_contributions: list[FeatureContributionResponse]
    global_feature_importance: list[FeatureImportanceResponse]
    constraints_applied: list[str]

    @field_serializer("recommended_price", "expected_revenue")
    def serialize_money(self, value: Decimal) -> str:
        return f"{value:.2f}"

    @classmethod
    def from_domain(
        cls,
        recommendation: PricingRecommendation,
        *,
        candidate_count: int,
    ) -> PricingRecommendationResponse:
        drivers = ", ".join(
            f"{item.feature} ({item.direction.replace('_', ' ')})"
            for item in recommendation.explanation[:3]
        )
        return cls(
            recommended_price=recommendation.recommended_price,
            currency=recommendation.currency,
            expected_occupancy=round(recommendation.expected_occupancy, 4),
            expected_occupancy_interval=tuple(
                round(value, 4) for value in recommendation.occupancy_interval
            ),
            confidence_score=recommendation.confidence_score,
            confidence_method=(
                "Calibrated occupancy-interval width combined with feature-space "
                "in-distribution support; it is not a booking probability."
            ),
            expected_revenue=recommendation.expected_revenue,
            model_version=recommendation.model_version,
            explanation=(
                f"Selected the revenue-maximizing price from {candidate_count} feasible "
                f"candidates. Main demand drivers: {drivers or 'no dominant drivers'}."
            ),
            local_feature_contributions=[
                FeatureContributionResponse(
                    feature=item.feature,
                    contribution=item.contribution,
                    direction=item.direction,
                )
                for item in recommendation.explanation
            ],
            global_feature_importance=[
                FeatureImportanceResponse(feature=name, importance_pct=value)
                for name, value in recommendation.feature_importance.items()
            ],
            constraints_applied=list(recommendation.constraints_applied),
        )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
