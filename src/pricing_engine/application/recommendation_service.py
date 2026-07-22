"""The use case for producing a pricing recommendation."""

from __future__ import annotations

from decimal import Decimal

from pricing_engine.application.ports import DemandPredictor, FeatureFactory
from pricing_engine.domain.exceptions import InvalidPricingRequestError
from pricing_engine.domain.models import PricingContext, PricingRecommendation
from pricing_engine.domain.policies import CandidatePrice, PricingPolicy


class RecommendPrice:
    """Coordinates policy, features, model inference, and explainability."""

    def __init__(
        self,
        *,
        predictor: DemandPredictor,
        feature_factory: FeatureFactory,
        pricing_policy: PricingPolicy,
        maximum_candidates: int,
    ) -> None:
        self._predictor = predictor
        self._feature_factory = feature_factory
        self._pricing_policy = pricing_policy
        self._maximum_candidates = maximum_candidates

    def execute(self, context: PricingContext) -> PricingRecommendation:
        if context.tenant_id not in self._predictor.tenant_ids:
            raise InvalidPricingRequestError(
                "The approved model is not scoped to the requested tenant. "
                "Train and promote a model whose governed tenant scope includes this tenant."
            )
        if context.currency != self._predictor.currency:
            raise InvalidPricingRequestError(
                f"Model currency is {self._predictor.currency}; received {context.currency}. "
                "Convert monetary inputs before requesting a recommendation."
            )
        prices = self._pricing_policy.feasible_prices(
            current_price=context.current_price,
            constraints=context.constraints,
            maximum_candidates=self._maximum_candidates,
        )
        features = self._feature_factory.for_candidate_prices(
            context, [float(price) for price in prices]
        )
        estimates = self._predictor.predict(features)
        candidates = (
            CandidatePrice(price=price, demand=estimate)
            for price, estimate in zip(prices, estimates, strict=True)
        )
        selected = self._pricing_policy.select(
            candidates,
            min_expected_occupancy=context.constraints.min_expected_occupancy,
        )
        selected_index = prices.index(selected.price)
        selected_features = features.iloc[[selected_index]]

        interval_width = selected.demand.upper_occupancy - selected.demand.lower_occupancy
        uncertainty_score = max(0.0, 1.0 - interval_width)
        confidence = round(
            100 * (0.65 * uncertainty_score + 0.35 * selected.demand.in_distribution_score),
            1,
        )
        constraints_applied = [
            "min_price",
            "max_price",
            "price_increment",
            "max_price_change_pct",
        ]
        if context.constraints.min_expected_occupancy is not None:
            constraints_applied.append("min_expected_occupancy")

        return PricingRecommendation(
            recommended_price=selected.price,
            currency=context.currency,
            expected_occupancy=selected.demand.expected_occupancy,
            occupancy_interval=(
                selected.demand.lower_occupancy,
                selected.demand.upper_occupancy,
            ),
            confidence_score=confidence,
            expected_revenue=selected.expected_revenue.quantize(Decimal("0.01")),
            model_version=self._predictor.version,
            explanation=tuple(self._predictor.explain(selected_features, top_k=5)),
            feature_importance=self._predictor.global_feature_importance(top_k=10),
            constraints_applied=tuple(constraints_applied),
        )
