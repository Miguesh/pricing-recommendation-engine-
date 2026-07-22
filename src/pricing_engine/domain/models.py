"""Immutable domain models for a single pricing decision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from math import isfinite

from pricing_engine.domain.exceptions import InvalidPricingRequestError


@dataclass(frozen=True, slots=True)
class PriceConstraints:
    """Commercial and safety bounds applied before optimization."""

    min_price: Decimal
    max_price: Decimal
    price_increment: Decimal = Decimal("1.00")
    max_price_change_pct: Decimal = Decimal("0.35")
    min_expected_occupancy: float | None = None

    def __post_init__(self) -> None:
        monetary_values = (self.min_price, self.max_price, self.price_increment)
        if not all(value.is_finite() for value in monetary_values):
            raise InvalidPricingRequestError("Price constraints must be finite.")
        if self.min_price <= 0 or self.max_price <= 0:
            raise InvalidPricingRequestError("Price bounds must be strictly positive.")
        if self.min_price > self.max_price:
            raise InvalidPricingRequestError("min_price cannot exceed max_price.")
        if self.price_increment <= 0:
            raise InvalidPricingRequestError("price_increment must be strictly positive.")
        if not isinstance(self.max_price_change_pct, Decimal) or (
            not self.max_price_change_pct.is_finite()
            or not 0 <= self.max_price_change_pct <= Decimal("0.35")
        ):
            raise InvalidPricingRequestError("max_price_change_pct must be between 0 and 0.35.")
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
    latitude: float | None = None
    longitude: float | None = None

    def __post_init__(self) -> None:
        for identifier_name, identifier_value in (
            ("tenant_id", self.tenant_id),
            ("property_id", self.property_id),
        ):
            if (
                not identifier_value
                or identifier_value != identifier_value.strip()
                or len(identifier_value) > 100
                or not all(
                    character.isprintable() and character not in "\r\n\t"
                    for character in identifier_value
                )
            ):
                raise InvalidPricingRequestError(
                    f"{identifier_name} must be a normalized printable identifier."
                )
        if self.as_of_date > self.stay_date:
            raise InvalidPricingRequestError("as_of_date cannot be after stay_date.")
        if len(self.currency) != 3 or not self.currency.isalpha() or not self.currency.isupper():
            raise InvalidPricingRequestError("currency must be an uppercase three-letter code.")
        if not self.current_price.is_finite() or not self.competitor_price_median.is_finite():
            raise InvalidPricingRequestError("Current and competitor prices must be finite.")
        if self.current_price <= 0 or self.competitor_price_median <= 0:
            raise InvalidPricingRequestError("Current and competitor prices must be positive.")
        for signal_name, signal_value in (
            ("historical_occupancy_7d", self.historical_occupancy_7d),
            ("booking_pace_7d", self.booking_pace_7d),
        ):
            if not 0 <= signal_value <= 1:
                raise InvalidPricingRequestError(f"{signal_name} must be in [0, 1].")
        if not 0 <= self.event_intensity <= 5:
            raise InvalidPricingRequestError("event_intensity must be in [0, 5].")
        if not 0 <= self.review_score <= 5:
            raise InvalidPricingRequestError("review_score must be in [0, 5].")
        if (
            isinstance(self.bedrooms, bool)
            or isinstance(self.accommodates, bool)
            or not 0 <= self.bedrooms <= 20
            or not 1 <= self.accommodates <= 50
        ):
            raise InvalidPricingRequestError("Invalid property capacity.")
        if not isinstance(self.is_holiday, bool):
            raise InvalidPricingRequestError("is_holiday must be a boolean.")
        if (self.latitude is None) != (self.longitude is None):
            raise InvalidPricingRequestError(
                "latitude and longitude must either both be provided or both be omitted."
            )
        if self.latitude is not None and self.longitude is not None:
            if not isfinite(self.latitude) or not -90 <= self.latitude <= 90:
                raise InvalidPricingRequestError("latitude must be finite and in [-90, 90].")
            if not isfinite(self.longitude) or not -180 <= self.longitude <= 180:
                raise InvalidPricingRequestError("longitude must be finite and in [-180, 180].")


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
