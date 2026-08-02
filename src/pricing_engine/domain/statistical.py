"""Stable, artifact-free contracts for market-evidence statistical pricing."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

CONTRACT_VERSION = "1.0"
ALGORITHM_VERSION = "weighted-percentile-v1.0.0"
DEFAULT_ALGORITHM_CONFIG_VERSION = "market-evidence-statistical-v1.0.0"
DEFAULT_EVIDENCE_POLICY_VERSION = "evidence-quality-v1.0.0"
DEFAULT_OUTLIER_POLICY_VERSION = "weighted-tukey-v1.0.0"
ENGINE_VERSION = "0.1.0"
MAX_COMPARABLES = 200

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
VersionIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]*$",
    ),
]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PositiveMoney = Annotated[Decimal, Field(gt=0, max_digits=16, decimal_places=4)]
Score = Annotated[Decimal, Field(ge=0, le=100, max_digits=7, decimal_places=4)]


def _require_json_integer(value: object, field_group: str) -> object:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_group} must be JSON integers.")
    return value


def _require_iso_calendar_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str) or (
        len(value) != 10
        or value[4] != "-"
        or value[7] != "-"
        or not (value[:4] + value[5:7] + value[8:]).isdigit()
    ):
        raise ValueError("Stay dates must use ISO YYYY-MM-DD calendar dates.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Stay dates must use ISO YYYY-MM-DD calendar dates.") from error


def _require_aware_iso_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and "T" in value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("Timestamps must use ISO 8601 with an explicit offset.") from error
    else:
        raise ValueError("Timestamps must use ISO 8601 with an explicit offset.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamps must include an explicit UTC offset.")
    return parsed


class StrictContractModel(BaseModel):
    """Base class for immutable additive-only v1 contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class EngineProfile(StrEnum):
    MARKET_EVIDENCE_STATISTICAL_V1 = "MARKET_EVIDENCE_STATISTICAL_V1"
    PERFORMANCE_AWARE_EXPERIMENTAL = "PERFORMANCE_AWARE_EXPERIMENTAL"


class PricingStrategy(StrEnum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    PREMIUM = "premium"


class RecommendationStatus(StrEnum):
    RECOMMENDED = "recommended"
    ABSTAINED = "abstained"


class EvidenceQuality(StrEnum):
    INSUFFICIENT = "INSUFFICIENT"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class AbstentionReason(StrEnum):
    INSUFFICIENT_COMPARABLES = "INSUFFICIENT_COMPARABLES"
    LOW_EFFECTIVE_SAMPLE_SIZE = "LOW_EFFECTIVE_SAMPLE_SIZE"
    EXCESSIVE_DISPERSION = "EXCESSIVE_DISPERSION"
    LOW_SIMILARITY = "LOW_SIMILARITY"
    STALE_EVIDENCE = "STALE_EVIDENCE"
    TEMPORAL_INCOMPATIBILITY = "TEMPORAL_INCOMPATIBILITY"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    CONTRACT_INCOMPATIBLE = "CONTRACT_INCOMPATIBLE"
    INVALID_LINEAGE = "INVALID_LINEAGE"
    INSUFFICIENT_PRICE_COVERAGE = "INSUFFICIENT_PRICE_COVERAGE"
    UNSUPPORTED_PROFILE = "UNSUPPORTED_PROFILE"


class LineageEntry(StrictContractModel):
    lineage_id: Identifier
    source_family_id: Identifier
    source_version: VersionIdentifier
    observation_hash: Sha256Digest
    lineage_hash: Sha256Digest


class TargetPropertyFeatures(StrictContractModel):
    property_type: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    bedrooms: Decimal | None = Field(default=None, ge=0, le=50)
    bathrooms: Decimal | None = Field(default=None, ge=0, le=50)
    accommodates: int = Field(ge=1, le=100)
    square_meters: Decimal | None = Field(default=None, gt=0, le=10_000)
    amenity_codes: tuple[Annotated[str, StringConstraints(min_length=1, max_length=64)], ...] = (
        Field(default=(), max_length=100)
    )

    @field_validator("accommodates", mode="before")
    @classmethod
    def require_json_integer(cls, value: object) -> object:
        return _require_json_integer(value, "Capacity fields")

    @field_validator("bedrooms", "bathrooms", "square_meters", mode="before")
    @classmethod
    def reject_binary_decimal_features(cls, value: object) -> object:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (str, Decimal, int))
        ):
            raise ValueError("Decimal property features must be strings or integers.")
        return value

    @field_validator("amenity_codes")
    @classmethod
    def normalize_amenities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted({value.strip().lower() for value in values}))
        if len(normalized) != len(values):
            raise ValueError("amenity_codes must be normalized and unique.")
        return normalized


class StatisticalComparable(StrictContractModel):
    comparable_id: Identifier
    evidence_type: Identifier
    verification_level: Identifier
    collection_method: Identifier
    observed_at: datetime
    known_at: datetime
    stay_start: date
    stay_end: date
    number_of_nights: int = Field(ge=1, le=366)
    guests: int = Field(ge=1, le=100)
    currency: CurrencyCode
    timezone: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    effective_base_nightly_rate: PositiveMoney | None = None
    effective_total_nightly_rate: PositiveMoney
    similarity_score: Score
    similarity_factors: dict[Identifier, Score] = Field(min_length=1, max_length=50)
    availability_observed: StrictBool | None
    quality_flags: tuple[Identifier, ...] = Field(max_length=50)
    eligible_for_pricing: StrictBool = True
    source_family_id: Identifier
    source_version: VersionIdentifier
    observation_hash: Sha256Digest
    lineage_hash: Sha256Digest

    @field_validator("effective_base_nightly_rate", "effective_total_nightly_rate", mode="before")
    @classmethod
    def reject_binary_money(cls, value: object) -> object:
        if value is not None and not isinstance(value, (str, Decimal)):
            raise ValueError("Money must be encoded as a decimal string.")
        return value

    @field_validator("similarity_score", mode="before")
    @classmethod
    def reject_binary_score(cls, value: object) -> object:
        if not isinstance(value, (str, Decimal, int)) or isinstance(value, bool):
            raise ValueError("similarity_score must be a decimal string or integer.")
        return value

    @field_validator("similarity_factors", mode="before")
    @classmethod
    def reject_binary_similarity_factors(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("similarity_factors must be a JSON object.")
        if any(
            isinstance(item, bool) or not isinstance(item, (str, Decimal, int))
            for item in value.values()
        ):
            raise ValueError("Similarity factors must be decimal strings or integers.")
        return value

    @field_validator("number_of_nights", "guests", mode="before")
    @classmethod
    def require_json_integer(cls, value: object) -> object:
        return _require_json_integer(value, "Night and guest counts")

    @field_validator("stay_start", "stay_end", mode="before")
    @classmethod
    def require_iso_calendar_date(cls, value: object) -> date:
        return _require_iso_calendar_date(value)

    @field_validator("observed_at", "known_at", mode="before")
    @classmethod
    def require_aware_timestamp(cls, value: object) -> datetime:
        return _require_aware_iso_timestamp(value)

    @field_validator("timezone")
    @classmethod
    def require_iana_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("timezone must be a supported IANA timezone.") from error
        return value

    @field_validator("quality_flags")
    @classmethod
    def require_unique_quality_flags(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("quality_flags must be unique.")
        return values

    @model_validator(mode="after")
    def validate_stay(self) -> StatisticalComparable:
        if (self.stay_end - self.stay_start).days != self.number_of_nights:
            raise ValueError("Comparable stay dates and number_of_nights are inconsistent.")
        if self.known_at < self.observed_at:
            raise ValueError("known_at cannot precede observed_at.")
        return self


class StatisticalPricingRequest(StrictContractModel):
    contract_version: Literal["1.0"]
    request_id: Identifier
    correlation_id: Identifier
    engine_profile: EngineProfile
    organization_id: Identifier
    audit_id: Identifier
    target_property_id: Identifier
    market_id: Identifier
    market_config_version: VersionIdentifier
    currency: CurrencyCode
    timezone: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    stay_start: date
    stay_end: date
    number_of_nights: int = Field(ge=1, le=366)
    guests: int = Field(ge=1, le=100)
    observed_at: datetime
    as_of: datetime
    pricing_strategy: PricingStrategy
    evidence_policy_version: VersionIdentifier
    outlier_policy_version: VersionIdentifier
    algorithm_config_version: VersionIdentifier
    target_property_features: TargetPropertyFeatures
    comparables: tuple[StatisticalComparable, ...] = Field(min_length=1, max_length=MAX_COMPARABLES)
    input_lineage: tuple[LineageEntry, ...] = Field(min_length=1, max_length=MAX_COMPARABLES)
    publication_allowed: Literal[False]
    commercial_validation: Literal[False]

    @field_validator("publication_allowed", "commercial_validation", mode="before")
    @classmethod
    def require_explicit_false(cls, value: object) -> object:
        if value is not False:
            raise ValueError("Safety flags must be explicit JSON false values.")
        return value

    @field_validator("observed_at", "as_of", mode="before")
    @classmethod
    def require_aware_timestamp(cls, value: object) -> datetime:
        return _require_aware_iso_timestamp(value)

    @field_validator("number_of_nights", "guests", mode="before")
    @classmethod
    def require_json_integer(cls, value: object) -> object:
        return _require_json_integer(value, "Night and guest counts")

    @field_validator("stay_start", "stay_end", mode="before")
    @classmethod
    def require_iso_calendar_date(cls, value: object) -> date:
        return _require_iso_calendar_date(value)

    @field_validator("timezone")
    @classmethod
    def require_iana_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("timezone must be a supported IANA timezone.") from error
        return value

    @model_validator(mode="after")
    def validate_scenario(self) -> StatisticalPricingRequest:
        if (self.stay_end - self.stay_start).days != self.number_of_nights:
            raise ValueError("stay dates and number_of_nights are inconsistent.")
        if self.observed_at > self.as_of:
            raise ValueError("observed_at cannot be after as_of.")
        return self


class PricingBandV1(StrictContractModel):
    low: Decimal
    central: Decimal
    high: Decimal
    currency: CurrencyCode

    @field_serializer("low", "central", "high")
    def serialize_money(self, value: Decimal) -> str:
        return format(value, "f")


class EvidenceComponents(StrictContractModel):
    comparable_count_received: int
    comparable_count_used: int
    effective_sample_size: Decimal
    average_similarity: Decimal
    minimum_similarity: Decimal
    dispersion_ratio: Decimal | None
    maximum_age_days: int
    outlier_count: int
    missing_base_rate_count: int
    scenario_compatible: StrictBool

    @field_serializer(
        "effective_sample_size",
        "average_similarity",
        "minimum_similarity",
        "dispersion_ratio",
    )
    def serialize_decimal(self, value: Decimal | None) -> str | None:
        return format(value, "f") if value is not None else None


class OutlierRecord(StrictContractModel):
    comparable_id: Identifier
    effective_total_nightly_rate: Decimal
    reason: str

    @field_serializer("effective_total_nightly_rate")
    def serialize_money(self, value: Decimal) -> str:
        return format(value, "f")


class Abstention(StrictContractModel):
    reason: AbstentionReason
    detail: Annotated[str, StringConstraints(min_length=1, max_length=500)]


class OutputLineage(StrictContractModel):
    request_hash: Sha256Digest
    input_lineage_hash: Sha256Digest
    algorithm_version: VersionIdentifier
    algorithm_config_version: VersionIdentifier
    evidence_policy_version: VersionIdentifier
    outlier_policy_version: VersionIdentifier


class StatisticalPricingResponse(StrictContractModel):
    contract_version: Literal["1.0"] = "1.0"
    engine_version: VersionIdentifier = ENGINE_VERSION
    engine_profile: EngineProfile
    algorithm_version: VersionIdentifier = ALGORITHM_VERSION
    algorithm_config_version: VersionIdentifier
    evidence_policy_version: VersionIdentifier
    outlier_policy_version: VersionIdentifier
    deterministic_run_id: Identifier
    request_hash: Sha256Digest
    status: RecommendationStatus
    recommendation: Decimal | None
    pricing_band: PricingBandV1 | None
    evidence_quality: EvidenceQuality
    evidence_components: EvidenceComponents
    comparable_ids_received: tuple[Identifier, ...]
    comparable_ids_used: tuple[Identifier, ...]
    comparable_ids_excluded: tuple[Identifier, ...]
    exclusion_reasons: dict[Identifier, tuple[str, ...]]
    outliers: tuple[OutlierRecord, ...]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    abstention: Abstention | None
    input_lineage: tuple[LineageEntry, ...]
    output_lineage: OutputLineage
    duration_ms: int = Field(ge=0)
    publication_allowed: Literal[False] = False
    commercial_validation: Literal[False] = False

    @field_serializer("recommendation")
    def serialize_money(self, value: Decimal | None) -> str | None:
        return format(value, "f") if value is not None else None


class ProfileCapability(StrictContractModel):
    profile: EngineProfile
    status: Literal["stable_contract", "experimental"]
    version: VersionIdentifier
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    limitations: tuple[str, ...]
    dependencies: tuple[str, ...]
    currency_support: str
    timezone_support: str
    uses_artifacts: StrictBool
    deterministic: StrictBool


class CapabilitiesResponse(StrictContractModel):
    engine_version: VersionIdentifier = ENGINE_VERSION
    default_profile: EngineProfile
    profiles: tuple[ProfileCapability, ...]
