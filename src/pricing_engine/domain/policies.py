"""Pure business policies for feasible-price generation and selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from pricing_engine.domain.exceptions import InvalidPricingRequestError
from pricing_engine.domain.models import DemandEstimate, PriceConstraints


@dataclass(frozen=True, slots=True)
class CandidatePrice:
    price: Decimal
    demand: DemandEstimate

    @property
    def expected_revenue(self) -> Decimal:
        return self.price * Decimal(str(self.demand.expected_occupancy))


class PricingPolicy:
    """Optimizes expected nightly revenue within explicit business guardrails."""

    def feasible_prices(
        self,
        *,
        current_price: Decimal,
        constraints: PriceConstraints,
        maximum_candidates: int,
    ) -> tuple[Decimal, ...]:
        minimum = constraints.min_price
        maximum = constraints.max_price
        if constraints.max_price_change_pct is not None:
            change = constraints.max_price_change_pct
            minimum = max(minimum, current_price * (Decimal("1") - change))
            maximum = min(maximum, current_price * (Decimal("1") + change))

        increment = constraints.price_increment
        first = (minimum / increment).to_integral_value(rounding=ROUND_CEILING) * increment
        last = (maximum / increment).to_integral_value(rounding=ROUND_FLOOR) * increment
        if first > last:
            raise InvalidPricingRequestError(
                "No feasible price remains after applying constraints."
            )

        count = int((last - first) / increment) + 1
        if count > maximum_candidates:
            raise InvalidPricingRequestError(
                f"Candidate grid has {count} prices; maximum is {maximum_candidates}. "
                "Use a larger increment or narrower bounds."
            )
        return tuple(first + i * increment for i in range(count))

    def select(
        self,
        candidates: Iterable[CandidatePrice],
        *,
        min_expected_occupancy: float | None = None,
    ) -> CandidatePrice:
        feasible = [
            candidate
            for candidate in candidates
            if (
                candidate.demand.expected_occupancy >= 0
                and (
                    min_expected_occupancy is None
                    or candidate.demand.expected_occupancy >= min_expected_occupancy
                )
            )
        ]
        if not feasible:
            raise InvalidPricingRequestError(
                "No candidate price satisfies the demand and occupancy constraints."
            )
        # Deterministic tie-breaker: lower price minimizes unnecessary guest cost
        # when two prices produce the same expected revenue.
        return max(feasible, key=lambda candidate: (candidate.expected_revenue, -candidate.price))
