"""Deterministic weighted-percentile pricing from consumer-selected evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from decimal import ROUND_HALF_UP, Context, Decimal, localcontext
from hashlib import sha256
from typing import Any
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

from pricing_engine.domain.exceptions import StatisticalContractError
from pricing_engine.domain.statistical import (
    ALGORITHM_VERSION,
    DEFAULT_ALGORITHM_CONFIG_VERSION,
    DEFAULT_EVIDENCE_POLICY_VERSION,
    DEFAULT_OUTLIER_POLICY_VERSION,
    Abstention,
    AbstentionReason,
    CapabilitiesResponse,
    EngineProfile,
    EvidenceComponents,
    EvidenceQuality,
    OutlierRecord,
    OutputLineage,
    PricingBandV1,
    PricingStrategy,
    ProfileCapability,
    RecommendationStatus,
    StatisticalComparable,
    StatisticalPricingRequest,
    StatisticalPricingResponse,
)

MONEY_QUANTUM = Decimal("0.01")
DECIMAL_PRECISION = 28


@dataclass(frozen=True, slots=True)
class StatisticalConfigV1:
    """All behavioral parameters for the stable statistical profile."""

    version: str = DEFAULT_ALGORITHM_CONFIG_VERSION
    minimum_comparables: int = 3
    minimum_effective_sample_size: Decimal = Decimal("2.50")
    minimum_individual_similarity: Decimal = Decimal("40")
    minimum_average_similarity: Decimal = Decimal("55")
    maximum_dispersion_ratio: Decimal = Decimal("1.00")
    maximum_evidence_age_days: int = 90
    outlier_iqr_multiplier: Decimal = Decimal("1.50")
    high_quality_minimum_comparables: int = 8
    high_quality_minimum_effective_sample_size: Decimal = Decimal("6")
    high_quality_minimum_average_similarity: Decimal = Decimal("80")
    high_quality_maximum_dispersion_ratio: Decimal = Decimal("0.35")
    moderate_quality_minimum_comparables: int = 5
    moderate_quality_minimum_effective_sample_size: Decimal = Decimal("4")
    moderate_quality_minimum_average_similarity: Decimal = Decimal("65")
    moderate_quality_maximum_dispersion_ratio: Decimal = Decimal("0.60")


def _canonical_payload(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _canonical_request(request: StatisticalPricingRequest) -> dict[str, Any]:
    payload = request.model_dump(mode="json")
    payload["comparables"] = sorted(payload["comparables"], key=lambda item: item["comparable_id"])
    payload["input_lineage"] = sorted(payload["input_lineage"], key=lambda item: item["lineage_id"])
    return payload


def request_hash(request: StatisticalPricingRequest) -> str:
    return sha256(_canonical_payload(_canonical_request(request))).hexdigest()


def weighted_percentile(
    values: tuple[Decimal, ...],
    weights: tuple[Decimal, ...],
    percentile: Decimal,
) -> Decimal:
    """Return the left-continuous weighted empirical quantile."""

    if not values or len(values) != len(weights):
        raise ValueError("values and weights must be non-empty and aligned")
    if not percentile.is_finite() or not Decimal("0") <= percentile <= Decimal("1"):
        raise ValueError("percentile must be in [0, 1]")
    if any(not value.is_finite() for value in values):
        raise ValueError("values must be finite")
    if any(not weight.is_finite() or weight <= 0 for weight in weights):
        raise ValueError("weights must be finite and strictly positive")
    with localcontext(Context(prec=DECIMAL_PRECISION, rounding=ROUND_HALF_UP)):
        ordered = sorted(zip(values, weights, strict=True), key=lambda item: item[0])
        total = sum((weight for _, weight in ordered), Decimal("0"))
        threshold = total * percentile
        cumulative = Decimal("0")
        for value, weight in ordered:
            cumulative += weight
            if cumulative >= threshold:
                return value
        return ordered[-1][0]


def effective_sample_size(weights: tuple[Decimal, ...]) -> Decimal:
    if not weights or any(not weight.is_finite() or weight <= 0 for weight in weights):
        return Decimal("0")
    with localcontext(Context(prec=DECIMAL_PRECISION, rounding=ROUND_HALF_UP)):
        total = sum(weights, Decimal("0"))
        return (total * total) / sum((weight * weight for weight in weights), Decimal("0"))


def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(
        default_profile=EngineProfile.MARKET_EVIDENCE_STATISTICAL_V1,
        profiles=(
            ProfileCapability(
                profile=EngineProfile.MARKET_EVIDENCE_STATISTICAL_V1,
                status="stable_contract",
                version="1.0",
                required_fields=(
                    "contract_version",
                    "request_id",
                    "correlation_id",
                    "engine_profile",
                    "organization_id",
                    "audit_id",
                    "target_property_id",
                    "market_id",
                    "market_config_version",
                    "currency",
                    "timezone",
                    "stay_start",
                    "stay_end",
                    "number_of_nights",
                    "guests",
                    "observed_at",
                    "as_of",
                    "pricing_strategy",
                    "evidence_policy_version",
                    "outlier_policy_version",
                    "algorithm_config_version",
                    "target_property_features",
                    "comparables",
                    "input_lineage",
                    "publication_allowed=false",
                    "commercial_validation=false",
                    "comparables[].comparable_id",
                    "comparables[].evidence_type",
                    "comparables[].verification_level",
                    "comparables[].collection_method",
                    "comparables[].observed_at",
                    "comparables[].known_at",
                    "comparables[].stay_start",
                    "comparables[].stay_end",
                    "comparables[].number_of_nights",
                    "comparables[].guests",
                    "comparables[].currency",
                    "comparables[].timezone",
                    "comparables[].effective_total_nightly_rate",
                    "comparables[].similarity_score",
                    "comparables[].similarity_factors",
                    "comparables[].availability_observed",
                    "comparables[].quality_flags",
                    "comparables[].source_family_id",
                    "comparables[].source_version",
                    "comparables[].observation_hash",
                    "comparables[].lineage_hash",
                ),
                optional_fields=(
                    "comparables[].effective_base_nightly_rate",
                    "comparables[].eligible_for_pricing",
                    "target_property_features.bedrooms",
                    "target_property_features.bathrooms",
                    "target_property_features.square_meters",
                    "target_property_features.amenity_codes",
                ),
                limitations=(
                    "Consumer selects and authorizes comparables.",
                    "One currency and one IANA timezone per request; no conversion.",
                    "Evidence quality is not a probability or commercial accuracy claim.",
                    "No forecast, occupancy inference, uplift, or automatic publication.",
                ),
                dependencies=("Python Decimal", "Pydantic contract validation"),
                currency_support="Any ISO-style uppercase code, one currency per run.",
                timezone_support="IANA timezone required and exact-match across evidence.",
                uses_artifacts=False,
                deterministic=True,
            ),
            ProfileCapability(
                profile=EngineProfile.PERFORMANCE_AWARE_EXPERIMENTAL,
                status="experimental",
                version="0.1.0",
                required_fields=(
                    "tenant_id",
                    "property_id",
                    "stay_date",
                    "as_of_date",
                    "currency",
                    "current_price",
                    "historical_occupancy_7d",
                    "booking_pace_7d",
                    "competitor_price_median",
                    "bedrooms",
                    "accommodates",
                    "review_score",
                    "constraints",
                ),
                optional_fields=(
                    "is_holiday",
                    "event_intensity",
                    "latitude",
                    "longitude",
                    "constraints.min_expected_occupancy",
                ),
                limitations=(
                    "Requires a governed trained model and operational signals.",
                    "Not the default profile and not yet intended for PLUSBNB.",
                ),
                dependencies=("LightGBM", "SHAP", "MLflow", "trained artifact"),
                currency_support="One currency bound to each trained artifact.",
                timezone_support="Stay-date contract; no market-timezone conversion.",
                uses_artifacts=True,
                deterministic=False,
            ),
        ),
    )


class MarketEvidenceStatisticalV1:
    """Pure statistical pricing profile; it performs no I/O and loads no artifacts."""

    def __init__(self, config: StatisticalConfigV1 | None = None) -> None:
        resolved_config = config or StatisticalConfigV1()
        if resolved_config != StatisticalConfigV1():
            raise ValueError(
                "MARKET_EVIDENCE_STATISTICAL_V1 accepts only its canonical versioned config."
            )
        self.config = resolved_config

    def validate(self, request: StatisticalPricingRequest) -> None:
        if request.engine_profile is not EngineProfile.MARKET_EVIDENCE_STATISTICAL_V1:
            return
        expected_versions = {
            "algorithm_config_version": self.config.version,
            "evidence_policy_version": DEFAULT_EVIDENCE_POLICY_VERSION,
            "outlier_policy_version": DEFAULT_OUTLIER_POLICY_VERSION,
        }
        actual_versions = {
            "algorithm_config_version": request.algorithm_config_version,
            "evidence_policy_version": request.evidence_policy_version,
            "outlier_policy_version": request.outlier_policy_version,
        }
        if actual_versions != expected_versions:
            raise StatisticalContractError(
                AbstentionReason.CONTRACT_INCOMPATIBLE,
                "Unsupported algorithm, evidence, or outlier policy version.",
            )
        comparable_ids = [item.comparable_id for item in request.comparables]
        if len(set(comparable_ids)) != len(comparable_ids):
            raise StatisticalContractError(
                "DUPLICATE_COMPARABLE",
                "comparable_id values must be unique within one request.",
            )
        observation_hashes = [item.observation_hash for item in request.comparables]
        if len(set(observation_hashes)) != len(observation_hashes):
            raise StatisticalContractError(
                "DUPLICATE_COMPARABLE",
                "observation_hash values must be unique within one request.",
            )
        lineage_ids = [entry.lineage_id for entry in request.input_lineage]
        if len(set(lineage_ids)) != len(lineage_ids):
            raise StatisticalContractError(
                AbstentionReason.INVALID_LINEAGE,
                "lineage_id values must be unique within one request.",
            )
        lineage_keys = {
            (
                entry.lineage_hash,
                entry.source_family_id,
                entry.source_version,
                entry.observation_hash,
            )
            for entry in request.input_lineage
        }
        if len(lineage_keys) != len(request.input_lineage):
            raise StatisticalContractError(
                AbstentionReason.INVALID_LINEAGE,
                "Each input_lineage entry must identify one unique evidence tuple.",
            )
        expected_lineage_keys = {
            (
                comparable.lineage_hash,
                comparable.source_family_id,
                comparable.source_version,
                comparable.observation_hash,
            )
            for comparable in request.comparables
        }
        if lineage_keys != expected_lineage_keys:
            raise StatisticalContractError(
                AbstentionReason.INVALID_LINEAGE,
                "input_lineage must exactly cover the comparable evidence in this request.",
            )
        for comparable in request.comparables:
            self._validate_comparable(request, comparable, lineage_keys)

    @staticmethod
    def _validate_comparable(
        request: StatisticalPricingRequest,
        comparable: StatisticalComparable,
        lineage_keys: set[tuple[str, str, str, str]],
    ) -> None:
        if comparable.currency != request.currency:
            raise StatisticalContractError(
                AbstentionReason.CURRENCY_MISMATCH,
                "Every comparable must use the request currency; conversion is not supported.",
            )
        if comparable.timezone != request.timezone:
            raise StatisticalContractError(
                "TIMEZONE_MISMATCH",
                "Every comparable must use the request IANA timezone.",
            )
        if (
            comparable.stay_start != request.stay_start
            or comparable.stay_end != request.stay_end
            or comparable.number_of_nights != request.number_of_nights
            or comparable.guests != request.guests
        ):
            raise StatisticalContractError(
                AbstentionReason.TEMPORAL_INCOMPATIBILITY,
                "Comparable dates, nights, and guests must match the request scenario.",
            )
        comparable_observed = comparable.observed_at.astimezone(UTC)
        comparable_known = comparable.known_at.astimezone(UTC)
        cutoff = request.as_of.astimezone(UTC)
        if comparable_observed > cutoff or comparable_known > cutoff:
            raise StatisticalContractError(
                AbstentionReason.TEMPORAL_INCOMPATIBILITY,
                "Comparable observed_at and known_at must not exceed as_of.",
            )
        marked_not_usable = any(flag.upper() == "NOT_USABLE" for flag in comparable.quality_flags)
        if not comparable.eligible_for_pricing or marked_not_usable:
            raise StatisticalContractError(
                "EVIDENCE_NOT_USABLE",
                "Evidence marked as not usable cannot enter the pricing profile.",
            )
        comparable_lineage = (
            comparable.lineage_hash,
            comparable.source_family_id,
            comparable.source_version,
            comparable.observation_hash,
        )
        if comparable_lineage not in lineage_keys:
            raise StatisticalContractError(
                AbstentionReason.INVALID_LINEAGE,
                "Every comparable must have matching lineage, source, version, and "
                "observation hashes in input_lineage.",
            )

    def recommend(self, request: StatisticalPricingRequest) -> StatisticalPricingResponse:
        """Execute with a fixed Decimal context independent of the host process."""

        with localcontext(Context(prec=DECIMAL_PRECISION, rounding=ROUND_HALF_UP)):
            return self._recommend(request)

    def _recommend(self, request: StatisticalPricingRequest) -> StatisticalPricingResponse:
        digest = request_hash(request)
        if request.engine_profile is not EngineProfile.MARKET_EVIDENCE_STATISTICAL_V1:
            return self._abstain(
                request,
                digest,
                reason=AbstentionReason.UNSUPPORTED_PROFILE,
                detail="Use the legacy performance-aware request contract for this profile.",
                comparable_ids_used=(),
                exclusions={
                    item.comparable_id: (AbstentionReason.UNSUPPORTED_PROFILE.value,)
                    for item in request.comparables
                },
                outliers=(),
            )
        self.validate(request)
        ordered = tuple(sorted(request.comparables, key=lambda item: item.comparable_id))
        exclusions: dict[str, tuple[str, ...]] = {}
        recent: list[StatisticalComparable] = []
        for item in ordered:
            age_days = self._market_age_days(request, item)
            reasons: list[str] = []
            if age_days > self.config.maximum_evidence_age_days:
                reasons.append("STALE_EVIDENCE")
            if item.similarity_score < self.config.minimum_individual_similarity:
                reasons.append("LOW_SIMILARITY")
            if reasons:
                exclusions[item.comparable_id] = tuple(reasons)
            else:
                recent.append(item)
        if not recent and ordered:
            all_stale = all("STALE_EVIDENCE" in exclusions[item.comparable_id] for item in ordered)
            empty_reason = (
                AbstentionReason.STALE_EVIDENCE if all_stale else AbstentionReason.LOW_SIMILARITY
            )
            return self._abstain(
                request,
                digest,
                reason=empty_reason,
                detail="No comparable remains within the versioned recency and similarity gates.",
                comparable_ids_used=(),
                exclusions=exclusions,
                outliers=(),
            )

        outliers = self._outliers(tuple(recent))
        outlier_ids = {item.comparable_id for item in outliers}
        for comparable_id in sorted(outlier_ids):
            exclusions[comparable_id] = ("OUTLIER_EXCLUDED_BY_WEIGHTED_TUKEY_V1",)
        used = tuple(item for item in recent if item.comparable_id not in outlier_ids)
        if len(used) < self.config.minimum_comparables:
            return self._abstain(
                request,
                digest,
                reason=AbstentionReason.INSUFFICIENT_COMPARABLES,
                detail="Too few compatible non-outlier comparables remain.",
                comparable_ids_used=tuple(item.comparable_id for item in used),
                exclusions=exclusions,
                outliers=outliers,
            )

        raw_weights = tuple(item.similarity_score for item in used)
        weight_total = sum(raw_weights, Decimal("0"))
        if weight_total <= 0:
            return self._abstain(
                request,
                digest,
                reason=AbstentionReason.LOW_SIMILARITY,
                detail="Similarity weights do not provide usable evidence.",
                comparable_ids_used=tuple(item.comparable_id for item in used),
                exclusions=exclusions,
                outliers=outliers,
            )
        weights = tuple(weight / weight_total for weight in raw_weights)
        sample_size = effective_sample_size(weights)
        if sample_size < self.config.minimum_effective_sample_size:
            return self._abstain(
                request,
                digest,
                reason=AbstentionReason.LOW_EFFECTIVE_SAMPLE_SIZE,
                detail="Similarity concentration leaves too little effective evidence.",
                comparable_ids_used=tuple(item.comparable_id for item in used),
                exclusions=exclusions,
                outliers=outliers,
            )
        average_similarity = sum(raw_weights, Decimal("0")) / Decimal(len(raw_weights))
        if average_similarity < self.config.minimum_average_similarity:
            return self._abstain(
                request,
                digest,
                reason=AbstentionReason.LOW_SIMILARITY,
                detail="Average comparable similarity is below the documented gate.",
                comparable_ids_used=tuple(item.comparable_id for item in used),
                exclusions=exclusions,
                outliers=outliers,
            )

        prices = tuple(item.effective_total_nightly_rate for item in used)
        p25 = weighted_percentile(prices, weights, Decimal("0.25"))
        p50 = weighted_percentile(prices, weights, Decimal("0.50"))
        p75 = weighted_percentile(prices, weights, Decimal("0.75"))
        dispersion = (p75 - p25) / p50 if p50 > 0 else None
        if dispersion is None or dispersion > self.config.maximum_dispersion_ratio:
            return self._abstain(
                request,
                digest,
                reason=AbstentionReason.EXCESSIVE_DISPERSION,
                detail="Weighted price dispersion exceeds the versioned evidence gate.",
                comparable_ids_used=tuple(item.comparable_id for item in used),
                exclusions=exclusions,
                outliers=outliers,
                dispersion=dispersion,
            )
        band = PricingBandV1(
            low=p25.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP),
            central=p50.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP),
            high=p75.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP),
            currency=request.currency,
        )
        selected = {
            PricingStrategy.CONSERVATIVE: band.low,
            PricingStrategy.BALANCED: band.central,
            PricingStrategy.PREMIUM: band.high,
        }[request.pricing_strategy]
        components = self._components(request, used, outliers, dispersion)
        quality = self._quality(components)
        return self._response(
            request=request,
            digest=digest,
            status=RecommendationStatus.RECOMMENDED,
            recommendation=selected,
            band=band,
            quality=quality,
            components=components,
            used_ids=tuple(item.comparable_id for item in used),
            exclusions=exclusions,
            outliers=outliers,
            reasons=(
                "P25, P50, and P75 use normalized consumer-provided similarity weights.",
                f"Strategy {request.pricing_strategy.value} selects its documented band point.",
            ),
            warnings=(
                "Evidence quality is not a probability that the price is correct.",
                "The result is not a forecast, uplift estimate, or authorization to publish.",
            ),
            abstention=None,
        )

    @staticmethod
    def _market_age_days(
        request: StatisticalPricingRequest,
        comparable: StatisticalComparable,
    ) -> int:
        market_timezone = ZoneInfo(request.timezone)
        return (
            request.as_of.astimezone(market_timezone).date()
            - comparable.observed_at.astimezone(market_timezone).date()
        ).days

    def _outliers(
        self, comparables: tuple[StatisticalComparable, ...]
    ) -> tuple[OutlierRecord, ...]:
        if len(comparables) < 4:
            return ()
        values = tuple(item.effective_total_nightly_rate for item in comparables)
        weights = tuple(item.similarity_score for item in comparables)
        q25 = weighted_percentile(values, weights, Decimal("0.25"))
        q75 = weighted_percentile(values, weights, Decimal("0.75"))
        spread = q75 - q25
        lower = q25 - self.config.outlier_iqr_multiplier * spread
        upper = q75 + self.config.outlier_iqr_multiplier * spread
        return tuple(
            OutlierRecord(
                comparable_id=item.comparable_id,
                effective_total_nightly_rate=item.effective_total_nightly_rate,
                reason="Outside weighted Tukey fences under weighted-tukey-v1.0.0.",
            )
            for item in comparables
            if item.effective_total_nightly_rate < lower
            or item.effective_total_nightly_rate > upper
        )

    def _components(
        self,
        request: StatisticalPricingRequest,
        used: tuple[StatisticalComparable, ...],
        outliers: tuple[OutlierRecord, ...],
        dispersion: Decimal | None,
        scenario_compatible: bool = True,
    ) -> EvidenceComponents:
        weights = tuple(item.similarity_score for item in used)
        count = len(used)
        return EvidenceComponents(
            comparable_count_received=len(request.comparables),
            comparable_count_used=count,
            effective_sample_size=effective_sample_size(weights).quantize(Decimal("0.0001")),
            average_similarity=(
                (sum(weights, Decimal("0")) / Decimal(count)).quantize(Decimal("0.01"))
                if count
                else Decimal("0")
            ),
            minimum_similarity=min(weights, default=Decimal("0")),
            dispersion_ratio=(dispersion.quantize(Decimal("0.0001")) if dispersion else dispersion),
            maximum_age_days=max(self._market_age_days(request, item) for item in used)
            if used
            else 0,
            outlier_count=len(outliers),
            missing_base_rate_count=sum(item.effective_base_nightly_rate is None for item in used),
            scenario_compatible=scenario_compatible,
        )

    def _quality(self, components: EvidenceComponents) -> EvidenceQuality:
        dispersion = (
            components.dispersion_ratio
            if components.dispersion_ratio is not None
            else Decimal("99")
        )
        if components.comparable_count_used < 3:
            return EvidenceQuality.INSUFFICIENT
        if (
            components.comparable_count_used >= self.config.high_quality_minimum_comparables
            and components.effective_sample_size
            >= self.config.high_quality_minimum_effective_sample_size
            and components.average_similarity >= self.config.high_quality_minimum_average_similarity
            and dispersion <= self.config.high_quality_maximum_dispersion_ratio
        ):
            return EvidenceQuality.HIGH
        if (
            components.comparable_count_used >= self.config.moderate_quality_minimum_comparables
            and components.effective_sample_size
            >= self.config.moderate_quality_minimum_effective_sample_size
            and components.average_similarity
            >= self.config.moderate_quality_minimum_average_similarity
            and dispersion <= self.config.moderate_quality_maximum_dispersion_ratio
        ):
            return EvidenceQuality.MODERATE
        return EvidenceQuality.LOW

    def _abstain(
        self,
        request: StatisticalPricingRequest,
        digest: str,
        *,
        reason: AbstentionReason,
        detail: str,
        comparable_ids_used: tuple[str, ...],
        exclusions: dict[str, tuple[str, ...]],
        outliers: tuple[OutlierRecord, ...],
        dispersion: Decimal | None = None,
    ) -> StatisticalPricingResponse:
        used_by_id = {item.comparable_id: item for item in request.comparables}
        used = tuple(used_by_id[item] for item in comparable_ids_used)
        return self._response(
            request=request,
            digest=digest,
            status=RecommendationStatus.ABSTAINED,
            recommendation=None,
            band=None,
            quality=EvidenceQuality.INSUFFICIENT,
            components=self._components(
                request,
                used,
                outliers,
                dispersion,
                scenario_compatible=reason is not AbstentionReason.UNSUPPORTED_PROFILE,
            ),
            used_ids=comparable_ids_used,
            exclusions=exclusions,
            outliers=outliers,
            reasons=(detail,),
            warnings=("Abstention is an expected safe result; no fallback was selected.",),
            abstention=Abstention(reason=reason, detail=detail),
        )

    def _response(
        self,
        *,
        request: StatisticalPricingRequest,
        digest: str,
        status: RecommendationStatus,
        recommendation: Decimal | None,
        band: PricingBandV1 | None,
        quality: EvidenceQuality,
        components: EvidenceComponents,
        used_ids: tuple[str, ...],
        exclusions: dict[str, tuple[str, ...]],
        outliers: tuple[OutlierRecord, ...],
        reasons: tuple[str, ...],
        warnings: tuple[str, ...],
        abstention: Abstention | None,
    ) -> StatisticalPricingResponse:
        input_lineage_hash = sha256(
            _canonical_payload(
                sorted(
                    (item.model_dump(mode="json") for item in request.input_lineage),
                    key=lambda item: item["lineage_id"],
                )
            )
        ).hexdigest()
        received = tuple(sorted(item.comparable_id for item in request.comparables))
        used = tuple(sorted(used_ids))
        excluded = tuple(sorted(set(received) - set(used)))
        return StatisticalPricingResponse(
            engine_profile=request.engine_profile,
            algorithm_config_version=request.algorithm_config_version,
            evidence_policy_version=request.evidence_policy_version,
            outlier_policy_version=request.outlier_policy_version,
            deterministic_run_id=str(uuid5(NAMESPACE_URL, digest)),
            request_hash=digest,
            status=status,
            recommendation=recommendation,
            pricing_band=band,
            evidence_quality=quality,
            evidence_components=components,
            comparable_ids_received=received,
            comparable_ids_used=used,
            comparable_ids_excluded=excluded,
            exclusion_reasons={key: exclusions[key] for key in sorted(exclusions)},
            outliers=tuple(sorted(outliers, key=lambda item: item.comparable_id)),
            reasons=reasons,
            warnings=warnings,
            abstention=abstention,
            input_lineage=tuple(sorted(request.input_lineage, key=lambda item: item.lineage_id)),
            output_lineage=OutputLineage(
                request_hash=digest,
                input_lineage_hash=input_lineage_hash,
                algorithm_version=ALGORITHM_VERSION,
                algorithm_config_version=request.algorithm_config_version,
                evidence_policy_version=request.evidence_policy_version,
                outlier_policy_version=request.outlier_policy_version,
            ),
            # HTTP telemetry measures wall time. Keeping this contractual value
            # at zero makes the pure response byte-for-byte reproducible.
            duration_ms=0,
        )
