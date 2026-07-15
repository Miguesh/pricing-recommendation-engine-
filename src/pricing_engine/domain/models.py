"""Immutable domain models for a single pricing decision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from pricing_engine.domain.exceptions import InvalidPricingRequestError


@dataclass(frozen=True, slots=True)
class PriceConstraints:
    """Commercial and safety bounds applied before optimization."""

    min_price: Decimal
    max_price: Decimal
    price_increment: Decimal = Decimal("1.00")
    max_price_change_pct: Decimal | None = Decimal("0.35")
    min_expected_occupancy: float | None = None

    def __post_init__(self) -> None:
        if self.min_price <= 0 or self.max_price <= 0:
            raise InvalidPricingRequestError("Price bounds must be strictly positive.")
        if self.min_price > self.max_price:
            raise InvalidPricingRequestError("min_price cannot exceed max_price.")
        if self.price_increment <= 0:
            raise InvalidPricingRequestError("price_increment must be strictly positive.")
        if self.max_price_change_pct is not None and self.max_price_change_pct < 0:
            raise InvalidPricingRequestError("max_price_change_pct cannot be negative.")
        if self.min_expected_occupancy is not None and not 0 <= self.min_expected_occupancy <= 1:
            raise InvalidPricingRequestError("min_expected_occupancy must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class PricingContext:
    """Point-in-time context used to price one property-night.

    The as-of date is deliberately explicit: all signals must have been known
    then. This is the primary guard against future-information leakage.
    """

    tenant_id: str
    property_id: str
    stay_date: date
    as_of_date: date
    currency: str
    current_price: Decimal
    historical_occupancy_7d: float
    booking_pace_7d: float
    competitor_price_median: Decimal
    bedrooms: int
    accommodates: int
    review_score: float
    is_holiday: bool = False
    event_intensity: float = 0.0
    constraints: PriceConstraints = field(
        default_factory=lambda: PriceConstraints(
            min_price=Decimal("25.00"),
            max_price=Decimal("2500.00"),
        )
    )

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.property_id:
            raise InvalidPricingRequestError("tenant_id and property_id are required.")
        if self.as_of_date > self.stay_date:
            raise InvalidPricingRequestError("as_of_date cannot be after stay_date.")
        if self.current_price <= 0 or self.competitor_price_median <= 0:
            raise InvalidPricingRequestError("Current and competitor prices must be positive.")
        for name, value in (
            ("historical_occupancy_7d", self.historical_occupancy_7d),
            ("booking_pace_7d", self.booking_pace_7d),
            ("event_intensity", self.event_intensity),
        ):
            if value < 0:
                raise InvalidPricingRequestError(f"{name} cannot be negative.")
        if not 0 <= self.review_score <= 5:
            raise InvalidPricingRequestError("review_score must be in [0, 5].")
        if self.bedrooms < 0 or self.accommodates < 1:
            raise InvalidPricingRequestError("Invalid property capacity.")


@dataclass(frozen=True, slots=True)
class DemandEstimate:
    """Calibrated occupancy estimate for one candidate price."""

    expected_occupancy: float
    lower_occupancy: float
    upper_occupancy: float
    in_distribution_score: float

    def __post_init__(self) -> None:
        if not (0 <= self.lower_occupancy <= self.expected_occupancy <= self.upper_occupancy <= 1):
            raise InvalidPricingRequestError(
                "Demand interval must be ordered and bounded in [0, 1]."
            )
        if not 0 <= self.in_distribution_score <= 1:
            raise InvalidPricingRequestError("in_distribution_score must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One human-readable local explanation factor."""

    feature: str
    contribution: float
    direction: str


@dataclass(frozen=True, slots=True)
class PricingRecommendation:
    """Decision returned by the application layer, independent of HTTP."""

    recommended_price: Decimal
    currency: str
    expected_occupancy: float
    occupancy_interval: tuple[float, float]
    confidence_score: float
    expected_revenue: Decimal
    model_version: str
    explanation: tuple[FeatureContribution, ...]
    feature_importance: Mapping[str, float]
    constraints_applied: tuple[str, ...]
