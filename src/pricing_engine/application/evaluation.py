"""Offline evaluation and deterministic model-promotion gates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pricing_engine.domain.exceptions import PromotionRejectedError
from pricing_engine.domain.models import DemandEstimate


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Metrics that distinguish prediction quality from pricing decision quality."""

    occupancy_mae: float
    occupancy_rmse: float
    occupancy_r2: float
    interval_coverage: float
    mean_interval_width: float
    expected_revenue_mae: float
    sample_size: int

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class PromotionCriteria:
    """Minimum offline evidence required before a model can be a candidate."""

    max_occupancy_mae: float = 0.20
    min_interval_coverage: float = 0.82
    max_interval_coverage: float = 0.98
    max_mean_interval_width: float = 0.60
    min_test_samples: int = 50

    def evaluate(self, metrics: EvaluationMetrics) -> None:
        failures: list[str] = []
        if metrics.sample_size < self.min_test_samples:
            failures.append(
                f"test sample size {metrics.sample_size} is below {self.min_test_samples}"
            )
        if metrics.occupancy_mae > self.max_occupancy_mae:
            failures.append(
                f"occupancy MAE {metrics.occupancy_mae:.4f} exceeds {self.max_occupancy_mae:.4f}"
            )
        coverage_is_valid = (
            self.min_interval_coverage <= metrics.interval_coverage <= self.max_interval_coverage
        )
        if not coverage_is_valid:
            failures.append(
                f"interval coverage {metrics.interval_coverage:.4f} is outside "
                f"[{self.min_interval_coverage:.2f}, {self.max_interval_coverage:.2f}]"
            )
        if metrics.mean_interval_width > self.max_mean_interval_width:
            failures.append(
                f"interval width {metrics.mean_interval_width:.4f} exceeds "
                f"{self.max_mean_interval_width:.4f}"
            )
        if failures:
            raise PromotionRejectedError("; ".join(failures))


def evaluate_demand_predictions(
    *,
    actual_occupancy: Sequence[float],
    estimates: Sequence[DemandEstimate],
    listed_prices: Sequence[float],
) -> EvaluationMetrics:
    """Calculate all validation metrics using a holdout period only."""

    actual = np.asarray(actual_occupancy, dtype=float)
    prices = np.asarray(listed_prices, dtype=float)
    expected = np.asarray([estimate.expected_occupancy for estimate in estimates])
    lower = np.asarray([estimate.lower_occupancy for estimate in estimates])
    upper = np.asarray([estimate.upper_occupancy for estimate in estimates])
    if len(actual) == 0 or len(actual) != len(expected) or len(actual) != len(prices):
        raise ValueError("Evaluation vectors must be non-empty and equal length.")
    actual_revenue = prices * actual
    predicted_revenue = prices * expected
    return EvaluationMetrics(
        occupancy_mae=float(mean_absolute_error(actual, expected)),
        occupancy_rmse=float(mean_squared_error(actual, expected) ** 0.5),
        occupancy_r2=float(r2_score(actual, expected)),
        interval_coverage=float(np.mean((actual >= lower) & (actual <= upper))),
        mean_interval_width=float(np.mean(upper - lower)),
        expected_revenue_mae=float(mean_absolute_error(actual_revenue, predicted_revenue)),
        sample_size=len(actual),
    )
