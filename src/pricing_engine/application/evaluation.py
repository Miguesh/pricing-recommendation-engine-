"""Offline evaluation and deterministic model-promotion gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast
from urllib.parse import unquote

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pricing_engine.domain.exceptions import PromotionRejectedError
from pricing_engine.domain.models import DemandEstimate


@dataclass(frozen=True, slots=True)
class SliceEvaluationMetrics:
    """Quality evidence for one operationally meaningful holdout slice."""

    name: str
    occupancy_mae: float
    interval_coverage: float
    expected_revenue_wape: float
    sample_size: int
    unique_stays: int | None = None
    mean_price_response: float | None = None
    flat_price_response_rate: float | None = None
    upper_price_boundary_rate: float | None = None
    lower_price_boundary_rate: float | None = None
    any_price_boundary_rate: float | None = None
    mean_selected_price_change_pct: float | None = None


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
    unique_stays: int | None = None
    expected_revenue_wape: float | None = None
    mean_price_response: float | None = None
    flat_price_response_rate: float | None = None
    upper_price_boundary_rate: float | None = None
    lower_price_boundary_rate: float | None = None
    any_price_boundary_rate: float | None = None
    mean_selected_price_change_pct: float | None = None
    slices: tuple[SliceEvaluationMetrics, ...] = ()

    def as_dict(self) -> dict[str, float]:
        metrics = {
            "occupancy_mae": self.occupancy_mae,
            "occupancy_rmse": self.occupancy_rmse,
            "occupancy_r2": self.occupancy_r2,
            "interval_coverage": self.interval_coverage,
            "mean_interval_width": self.mean_interval_width,
            "expected_revenue_mae": self.expected_revenue_mae,
            "sample_size": float(self.sample_size),
        }
        if self.unique_stays is not None:
            metrics["unique_stays"] = float(self.unique_stays)
        if self.expected_revenue_wape is not None:
            metrics["expected_revenue_wape"] = self.expected_revenue_wape
        if self.mean_price_response is not None:
            metrics["mean_price_response"] = self.mean_price_response
        if self.flat_price_response_rate is not None:
            metrics["flat_price_response_rate"] = self.flat_price_response_rate
        if self.upper_price_boundary_rate is not None:
            metrics["upper_price_boundary_rate"] = self.upper_price_boundary_rate
        if self.lower_price_boundary_rate is not None:
            metrics["lower_price_boundary_rate"] = self.lower_price_boundary_rate
        if self.any_price_boundary_rate is not None:
            metrics["any_price_boundary_rate"] = self.any_price_boundary_rate
        if self.mean_selected_price_change_pct is not None:
            metrics["mean_selected_price_change_pct"] = self.mean_selected_price_change_pct
        for item in self.slices:
            safe_name = item.name
            prefix = f"slice.{safe_name}"
            metrics[f"{prefix}.occupancy_mae"] = item.occupancy_mae
            metrics[f"{prefix}.interval_coverage"] = item.interval_coverage
            metrics[f"{prefix}.expected_revenue_wape"] = item.expected_revenue_wape
            metrics[f"{prefix}.sample_size"] = float(item.sample_size)
            if item.unique_stays is not None:
                metrics[f"{prefix}.unique_stays"] = float(item.unique_stays)
            for metric_name in (
                "mean_price_response",
                "flat_price_response_rate",
                "upper_price_boundary_rate",
                "lower_price_boundary_rate",
                "any_price_boundary_rate",
                "mean_selected_price_change_pct",
            ):
                value = getattr(item, metric_name)
                if value is not None:
                    metrics[f"{prefix}.{metric_name}"] = value
        return {name: float(value) for name, value in metrics.items()}

    @classmethod
    def from_mapping(cls, metrics: Mapping[str, float]) -> EvaluationMetrics:
        """Rebuild comparable aggregate evidence from an MLflow run."""

        required = (
            "occupancy_mae",
            "occupancy_rmse",
            "occupancy_r2",
            "interval_coverage",
            "mean_interval_width",
            "expected_revenue_mae",
            "sample_size",
        )
        missing = [name for name in required if name not in metrics]
        if missing:
            raise ValueError(f"MLflow run is missing required evaluation metrics: {missing}")
        slice_values: dict[str, dict[str, float]] = {}
        for name, value in metrics.items():
            if not name.startswith("slice."):
                continue
            remainder = name.removeprefix("slice.")
            encoded_name, separator, metric_name = remainder.rpartition(".")
            if not separator:
                continue
            slice_values.setdefault(encoded_name, {})[metric_name] = float(value)
        slices = []
        required_slice_metrics = {
            "occupancy_mae",
            "interval_coverage",
            "expected_revenue_wape",
            "sample_size",
        }
        for encoded_name, values in slice_values.items():
            if not required_slice_metrics.issubset(values):
                continue
            decoded_name = unquote(encoded_name)
            if "%" not in encoded_name and "." in decoded_name:
                decoded_name = decoded_name.replace(".", "/", 1)
            slices.append(
                SliceEvaluationMetrics(
                    name=decoded_name,
                    occupancy_mae=values["occupancy_mae"],
                    interval_coverage=values["interval_coverage"],
                    expected_revenue_wape=values["expected_revenue_wape"],
                    sample_size=int(values["sample_size"]),
                    unique_stays=(
                        int(values["unique_stays"]) if "unique_stays" in values else None
                    ),
                    mean_price_response=values.get("mean_price_response"),
                    flat_price_response_rate=values.get("flat_price_response_rate"),
                    upper_price_boundary_rate=values.get("upper_price_boundary_rate"),
                    lower_price_boundary_rate=values.get("lower_price_boundary_rate"),
                    any_price_boundary_rate=values.get("any_price_boundary_rate"),
                    mean_selected_price_change_pct=values.get("mean_selected_price_change_pct"),
                )
            )
        return cls(
            occupancy_mae=float(metrics["occupancy_mae"]),
            occupancy_rmse=float(metrics["occupancy_rmse"]),
            occupancy_r2=float(metrics["occupancy_r2"]),
            interval_coverage=float(metrics["interval_coverage"]),
            mean_interval_width=float(metrics["mean_interval_width"]),
            expected_revenue_mae=float(metrics["expected_revenue_mae"]),
            sample_size=int(metrics["sample_size"]),
            unique_stays=(int(metrics["unique_stays"]) if "unique_stays" in metrics else None),
            expected_revenue_wape=(
                float(metrics["expected_revenue_wape"])
                if "expected_revenue_wape" in metrics
                else None
            ),
            mean_price_response=(
                float(metrics["mean_price_response"]) if "mean_price_response" in metrics else None
            ),
            flat_price_response_rate=(
                float(metrics["flat_price_response_rate"])
                if "flat_price_response_rate" in metrics
                else None
            ),
            upper_price_boundary_rate=(
                float(metrics["upper_price_boundary_rate"])
                if "upper_price_boundary_rate" in metrics
                else None
            ),
            lower_price_boundary_rate=(
                float(metrics["lower_price_boundary_rate"])
                if "lower_price_boundary_rate" in metrics
                else None
            ),
            any_price_boundary_rate=(
                float(metrics["any_price_boundary_rate"])
                if "any_price_boundary_rate" in metrics
                else None
            ),
            mean_selected_price_change_pct=(
                float(metrics["mean_selected_price_change_pct"])
                if "mean_selected_price_change_pct" in metrics
                else None
            ),
            slices=tuple(sorted(slices, key=lambda item: item.name)),
        )


@dataclass(frozen=True, slots=True)
class PromotionCriteria:
    """Minimum offline evidence required before a model can be a candidate."""

    max_occupancy_mae: float = 0.20
    min_occupancy_r2: float = 0.20
    min_interval_coverage: float = 0.82
    max_interval_coverage: float = 0.995
    max_mean_interval_width: float = 0.60
    min_test_samples: int = 50
    min_unique_stays: int = 50
    max_expected_revenue_wape: float = 0.35
    min_mean_price_response: float = 0.005
    max_flat_price_response_rate: float = 0.80
    max_upper_price_boundary_rate: float = 0.80
    max_lower_price_boundary_rate: float = 0.50
    max_any_price_boundary_rate: float = 0.85
    max_abs_mean_selected_price_change_pct: float = 0.25
    min_slice_samples: int = 20
    min_slice_unique_stays: int = 20
    max_slice_occupancy_mae: float = 0.30
    max_slice_expected_revenue_wape: float = 0.50
    min_slice_interval_coverage: float = 0.70
    min_slice_mean_price_response: float = 0.005
    max_slice_flat_price_response_rate: float = 0.80
    max_slice_upper_price_boundary_rate: float = 0.80
    max_slice_lower_price_boundary_rate: float = 0.50
    max_slice_any_price_boundary_rate: float = 0.85
    max_slice_abs_mean_selected_price_change_pct: float = 0.25
    max_champion_relative_regression: float = 0.05
    max_champion_boundary_rate_regression: float = 0.10
    max_champion_mean_price_change_delta: float = 0.10
    required_slices: tuple[str, ...] = (
        "lead_time/0_7_days",
        "lead_time/8_30_days",
        "lead_time/31_plus_days",
        "holiday/yes",
        "holiday/no",
        "event/none",
        "event/active",
        "location/known",
        "location/unknown",
    )

    def evaluate(
        self,
        metrics: EvaluationMetrics,
        *,
        champion_metrics: EvaluationMetrics | None = None,
    ) -> None:
        failures: list[str] = []
        aggregate_values = {
            "occupancy_mae": metrics.occupancy_mae,
            "occupancy_rmse": metrics.occupancy_rmse,
            "occupancy_r2": metrics.occupancy_r2,
            "interval_coverage": metrics.interval_coverage,
            "mean_interval_width": metrics.mean_interval_width,
            "expected_revenue_mae": metrics.expected_revenue_mae,
        }
        non_finite = [name for name, value in aggregate_values.items() if not np.isfinite(value)]
        if non_finite:
            failures.append(f"non-finite aggregate metrics: {non_finite}")
        if metrics.sample_size < self.min_test_samples:
            failures.append(
                f"test sample size {metrics.sample_size} is below {self.min_test_samples}"
            )
        if metrics.unique_stays is None:
            failures.append("unique stay evidence is missing")
        elif metrics.unique_stays < self.min_unique_stays:
            failures.append(f"unique stays {metrics.unique_stays} is below {self.min_unique_stays}")
        if metrics.occupancy_mae > self.max_occupancy_mae:
            failures.append(
                f"occupancy MAE {metrics.occupancy_mae:.4f} exceeds {self.max_occupancy_mae:.4f}"
            )
        if metrics.occupancy_r2 < self.min_occupancy_r2:
            failures.append(
                f"occupancy R2 {metrics.occupancy_r2:.4f} is below {self.min_occupancy_r2:.4f}"
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
        if metrics.expected_revenue_wape is None:
            failures.append("expected revenue WAPE evidence is missing")
        elif not np.isfinite(metrics.expected_revenue_wape):
            failures.append("expected revenue WAPE is non-finite")
        elif metrics.expected_revenue_wape > self.max_expected_revenue_wape:
            failures.append(
                f"expected revenue WAPE {metrics.expected_revenue_wape:.4f} exceeds "
                f"{self.max_expected_revenue_wape:.4f}"
            )
        if (
            metrics.mean_price_response is None
            or metrics.flat_price_response_rate is None
            or metrics.upper_price_boundary_rate is None
            or metrics.lower_price_boundary_rate is None
            or metrics.any_price_boundary_rate is None
            or metrics.mean_selected_price_change_pct is None
        ):
            failures.append("controlled price-response evidence is missing")
        elif not all(
            np.isfinite(value)
            for value in (
                metrics.mean_price_response,
                metrics.flat_price_response_rate,
                metrics.upper_price_boundary_rate,
                metrics.lower_price_boundary_rate,
                metrics.any_price_boundary_rate,
                metrics.mean_selected_price_change_pct,
            )
        ):
            failures.append("controlled price-response evidence contains non-finite values")
        else:
            if metrics.mean_price_response < self.min_mean_price_response:
                failures.append(
                    f"mean price response {metrics.mean_price_response:.4f} is below "
                    f"{self.min_mean_price_response:.4f}"
                )
            if metrics.flat_price_response_rate > self.max_flat_price_response_rate:
                failures.append(
                    f"flat price-response rate {metrics.flat_price_response_rate:.2%} exceeds "
                    f"{self.max_flat_price_response_rate:.2%}"
                )
            if metrics.upper_price_boundary_rate > self.max_upper_price_boundary_rate:
                failures.append(
                    f"upper price-boundary rate {metrics.upper_price_boundary_rate:.2%} exceeds "
                    f"{self.max_upper_price_boundary_rate:.2%}"
                )
            if metrics.lower_price_boundary_rate > self.max_lower_price_boundary_rate:
                failures.append(
                    f"lower price-boundary rate {metrics.lower_price_boundary_rate:.2%} exceeds "
                    f"{self.max_lower_price_boundary_rate:.2%}"
                )
            if metrics.any_price_boundary_rate > self.max_any_price_boundary_rate:
                failures.append(
                    f"any price-boundary rate {metrics.any_price_boundary_rate:.2%} exceeds "
                    f"{self.max_any_price_boundary_rate:.2%}"
                )
            if (
                abs(metrics.mean_selected_price_change_pct)
                > self.max_abs_mean_selected_price_change_pct
            ):
                failures.append(
                    f"mean selected price change {metrics.mean_selected_price_change_pct:.2%} "
                    f"exceeds +/-{self.max_abs_mean_selected_price_change_pct:.2%}"
                )
        slices_by_name = {item.name: item for item in metrics.slices}
        for required_slice in self.required_slices:
            item = slices_by_name.get(required_slice)
            if item is None:
                failures.append(f"required slice {required_slice} is missing")
                continue
            independent_evidence = item.unique_stays
            if (
                item.sample_size < self.min_slice_samples
                or independent_evidence is None
                or independent_evidence < self.min_slice_unique_stays
            ):
                failures.append(
                    f"required slice {required_slice} has insufficient evidence: "
                    f"rows={item.sample_size}, unique_stays={independent_evidence}"
                )
        for item in metrics.slices:
            if not all(
                np.isfinite(value)
                for value in (
                    item.occupancy_mae,
                    item.interval_coverage,
                    item.expected_revenue_wape,
                )
            ):
                failures.append(f"slice {item.name} contains non-finite metrics")
                continue
            independent_slice_evidence = (
                item.unique_stays if item.unique_stays is not None else item.sample_size
            )
            if (
                item.sample_size < self.min_slice_samples
                or independent_slice_evidence < self.min_slice_unique_stays
            ):
                continue
            if item.occupancy_mae > self.max_slice_occupancy_mae:
                failures.append(
                    f"slice {item.name} occupancy MAE {item.occupancy_mae:.4f} exceeds "
                    f"{self.max_slice_occupancy_mae:.4f}"
                )
            if item.expected_revenue_wape > self.max_slice_expected_revenue_wape:
                failures.append(
                    f"slice {item.name} expected revenue WAPE "
                    f"{item.expected_revenue_wape:.4f} exceeds "
                    f"{self.max_slice_expected_revenue_wape:.4f}"
                )
            if item.interval_coverage < self.min_slice_interval_coverage:
                failures.append(
                    f"slice {item.name} interval coverage {item.interval_coverage:.4f} is below "
                    f"{self.min_slice_interval_coverage:.4f}"
                )
            behavior_values = {
                "mean price response": item.mean_price_response,
                "flat price-response rate": item.flat_price_response_rate,
                "upper price-boundary rate": item.upper_price_boundary_rate,
                "lower price-boundary rate": item.lower_price_boundary_rate,
                "any price-boundary rate": item.any_price_boundary_rate,
                "mean selected price change": item.mean_selected_price_change_pct,
            }
            if any(value is None for value in behavior_values.values()):
                failures.append(f"slice {item.name} controlled price-response evidence is missing")
                continue
            if not all(
                np.isfinite(value) for value in behavior_values.values() if value is not None
            ):
                failures.append(
                    f"slice {item.name} controlled price-response evidence contains "
                    "non-finite values"
                )
                continue
            mean_response = cast(float, item.mean_price_response)
            flat_rate = cast(float, item.flat_price_response_rate)
            upper_boundary_rate = cast(float, item.upper_price_boundary_rate)
            lower_boundary_rate = cast(float, item.lower_price_boundary_rate)
            any_boundary_rate = cast(float, item.any_price_boundary_rate)
            mean_price_change = cast(float, item.mean_selected_price_change_pct)
            if mean_response < self.min_slice_mean_price_response:
                failures.append(
                    f"slice {item.name} mean price response {mean_response:.4f} "
                    "is below "
                    f"{self.min_slice_mean_price_response:.4f}"
                )
            if flat_rate > self.max_slice_flat_price_response_rate:
                failures.append(
                    f"slice {item.name} flat price-response rate "
                    f"{flat_rate:.2%} exceeds "
                    f"{self.max_slice_flat_price_response_rate:.2%}"
                )
            if upper_boundary_rate > self.max_slice_upper_price_boundary_rate:
                failures.append(
                    f"slice {item.name} upper price-boundary rate "
                    f"{upper_boundary_rate:.2%} exceeds "
                    f"{self.max_slice_upper_price_boundary_rate:.2%}"
                )
            if lower_boundary_rate > self.max_slice_lower_price_boundary_rate:
                failures.append(
                    f"slice {item.name} lower price-boundary rate "
                    f"{lower_boundary_rate:.2%} exceeds "
                    f"{self.max_slice_lower_price_boundary_rate:.2%}"
                )
            if any_boundary_rate > self.max_slice_any_price_boundary_rate:
                failures.append(
                    f"slice {item.name} any price-boundary rate "
                    f"{any_boundary_rate:.2%} exceeds "
                    f"{self.max_slice_any_price_boundary_rate:.2%}"
                )
            if abs(mean_price_change) > self.max_slice_abs_mean_selected_price_change_pct:
                failures.append(
                    f"slice {item.name} mean selected price change "
                    f"{mean_price_change:.2%} exceeds "
                    f"+/-{self.max_slice_abs_mean_selected_price_change_pct:.2%}"
                )
        if champion_metrics is not None:
            self._compare_with_champion(metrics, champion_metrics, failures)
        if failures:
            raise PromotionRejectedError("; ".join(failures))

    def _compare_with_champion(
        self,
        candidate: EvaluationMetrics,
        champion: EvaluationMetrics,
        failures: list[str],
    ) -> None:
        occupancy_limit = champion.occupancy_mae * (1.0 + self.max_champion_relative_regression)
        if candidate.occupancy_mae > occupancy_limit:
            failures.append(
                f"occupancy MAE {candidate.occupancy_mae:.4f} regresses more than "
                f"{self.max_champion_relative_regression:.0%} versus champion "
                f"{champion.occupancy_mae:.4f}"
            )
        if (
            candidate.expected_revenue_wape is not None
            and champion.expected_revenue_wape is not None
        ):
            revenue_limit = champion.expected_revenue_wape * (
                1.0 + self.max_champion_relative_regression
            )
            if candidate.expected_revenue_wape > revenue_limit:
                failures.append(
                    f"expected revenue WAPE {candidate.expected_revenue_wape:.4f} regresses "
                    f"more than {self.max_champion_relative_regression:.0%} versus champion "
                    f"{champion.expected_revenue_wape:.4f}"
                )
        for metric_name in (
            "upper_price_boundary_rate",
            "lower_price_boundary_rate",
            "any_price_boundary_rate",
        ):
            candidate_value = getattr(candidate, metric_name)
            champion_value = getattr(champion, metric_name)
            if (
                candidate_value is not None
                and champion_value is not None
                and candidate_value > champion_value + self.max_champion_boundary_rate_regression
            ):
                failures.append(
                    f"{metric_name} {candidate_value:.2%} regresses more than "
                    f"{self.max_champion_boundary_rate_regression:.0%} versus champion "
                    f"{champion_value:.2%}"
                )
        if (
            candidate.mean_selected_price_change_pct is not None
            and champion.mean_selected_price_change_pct is not None
            and abs(
                candidate.mean_selected_price_change_pct - champion.mean_selected_price_change_pct
            )
            > self.max_champion_mean_price_change_delta
        ):
            failures.append(
                "mean_selected_price_change_pct "
                f"{candidate.mean_selected_price_change_pct:.2%} differs by more than "
                f"{self.max_champion_mean_price_change_delta:.0%} versus champion "
                f"{champion.mean_selected_price_change_pct:.2%}"
            )


def evaluate_demand_predictions(
    *,
    actual_occupancy: Sequence[float],
    estimates: Sequence[DemandEstimate],
    listed_prices: Sequence[float],
    group_keys: Sequence[object] | None = None,
    slice_memberships: Mapping[str, Sequence[bool]] | None = None,
    price_response_deltas: Sequence[float] | None = None,
    selected_price_change_pcts: Sequence[float] | None = None,
) -> EvaluationMetrics:
    """Calculate all validation metrics using a holdout period only."""

    actual = np.asarray(actual_occupancy, dtype=float)
    prices = np.asarray(listed_prices, dtype=float)
    expected = np.asarray([estimate.expected_occupancy for estimate in estimates])
    lower = np.asarray([estimate.lower_occupancy for estimate in estimates])
    upper = np.asarray([estimate.upper_occupancy for estimate in estimates])
    if len(actual) == 0 or len(actual) != len(expected) or len(actual) != len(prices):
        raise ValueError("Evaluation vectors must be non-empty and equal length.")
    if group_keys is not None and len(group_keys) != len(actual):
        raise ValueError("Evaluation group keys must match the prediction vector length.")
    if price_response_deltas is not None and len(price_response_deltas) != len(actual):
        raise ValueError("Price-response deltas must match the prediction vector length.")
    if selected_price_change_pcts is not None and len(selected_price_change_pcts) != len(actual):
        raise ValueError("Selected price changes must match the prediction vector length.")
    normalized_group_keys = list(group_keys) if group_keys is not None else list(range(len(actual)))
    sample_weights = _equal_group_weights(normalized_group_keys)
    actual_revenue = prices * actual
    predicted_revenue = prices * expected
    revenue_mae = float(
        mean_absolute_error(actual_revenue, predicted_revenue, sample_weight=sample_weights)
    )
    revenue_wape = _weighted_absolute_percentage_error(
        actual_revenue,
        predicted_revenue,
        sample_weights=sample_weights,
    )
    response = (
        np.asarray(price_response_deltas, dtype=float)
        if price_response_deltas is not None
        else None
    )
    selected_changes = (
        np.asarray(selected_price_change_pcts, dtype=float)
        if selected_price_change_pcts is not None
        else None
    )
    slices = tuple(
        _evaluate_slice(
            name=name,
            membership=membership,
            actual=actual,
            expected=expected,
            lower=lower,
            upper=upper,
            actual_revenue=actual_revenue,
            predicted_revenue=predicted_revenue,
            group_keys=normalized_group_keys,
            price_response_deltas=response,
            selected_price_change_pcts=selected_changes,
        )
        for name, membership in (slice_memberships or {}).items()
    )
    return EvaluationMetrics(
        occupancy_mae=float(mean_absolute_error(actual, expected, sample_weight=sample_weights)),
        occupancy_rmse=float(
            mean_squared_error(actual, expected, sample_weight=sample_weights) ** 0.5
        ),
        occupancy_r2=float(r2_score(actual, expected, sample_weight=sample_weights)),
        interval_coverage=_group_simultaneous_coverage(
            (actual >= lower) & (actual <= upper),
            normalized_group_keys,
        ),
        mean_interval_width=float(np.average(upper - lower, weights=sample_weights)),
        expected_revenue_mae=revenue_mae,
        sample_size=len(actual),
        unique_stays=len(set(normalized_group_keys)),
        expected_revenue_wape=revenue_wape,
        mean_price_response=(
            float(np.average(response, weights=sample_weights)) if response is not None else None
        ),
        flat_price_response_rate=(
            float(np.average(response < 0.002, weights=sample_weights))
            if response is not None
            else None
        ),
        upper_price_boundary_rate=(
            float(np.average(selected_changes >= 0.35 - 1e-12, weights=sample_weights))
            if selected_changes is not None
            else None
        ),
        lower_price_boundary_rate=(
            float(np.average(selected_changes <= -0.35 + 1e-12, weights=sample_weights))
            if selected_changes is not None
            else None
        ),
        any_price_boundary_rate=(
            float(
                np.average(
                    np.abs(selected_changes) >= 0.35 - 1e-12,
                    weights=sample_weights,
                )
            )
            if selected_changes is not None
            else None
        ),
        mean_selected_price_change_pct=(
            float(np.average(selected_changes, weights=sample_weights))
            if selected_changes is not None
            else None
        ),
        slices=slices,
    )


def _equal_group_weights(group_keys: Sequence[object]) -> np.ndarray:
    """Give every independent stay equal total influence regardless of snapshot count."""

    counts: dict[object, int] = {}
    for key in group_keys:
        counts[key] = counts.get(key, 0) + 1
    return np.asarray([1.0 / counts[key] for key in group_keys], dtype=float)


def _group_simultaneous_coverage(
    covered: np.ndarray,
    group_keys: Sequence[object],
) -> float:
    """Measure the share of stays whose every decision snapshot is covered."""

    by_group: dict[object, bool] = {}
    for key, row_is_covered in zip(group_keys, covered, strict=True):
        by_group[key] = by_group.get(key, True) and bool(row_is_covered)
    return float(np.mean(list(by_group.values())))


def _weighted_absolute_percentage_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    sample_weights: np.ndarray,
) -> float:
    denominator = float(np.sum(sample_weights * np.abs(actual)))
    if denominator <= 1e-12:
        return 0.0 if np.allclose(actual, predicted) else 1.0
    return float(np.sum(sample_weights * np.abs(actual - predicted)) / denominator)


def _evaluate_slice(
    *,
    name: str,
    membership: Sequence[bool],
    actual: np.ndarray,
    expected: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    actual_revenue: np.ndarray,
    predicted_revenue: np.ndarray,
    group_keys: Sequence[object],
    price_response_deltas: np.ndarray | None,
    selected_price_change_pcts: np.ndarray | None,
) -> SliceEvaluationMetrics:
    mask = np.asarray(membership, dtype=bool)
    if len(mask) != len(actual):
        raise ValueError(f"Slice {name!r} membership length differs from evaluation vectors.")
    if not mask.any():
        raise ValueError(f"Slice {name!r} must contain at least one observation.")
    slice_group_keys = [key for key, included in zip(group_keys, mask, strict=True) if included]
    sample_weights = _equal_group_weights(slice_group_keys)
    return SliceEvaluationMetrics(
        name=name,
        occupancy_mae=float(
            mean_absolute_error(actual[mask], expected[mask], sample_weight=sample_weights)
        ),
        interval_coverage=_group_simultaneous_coverage(
            (actual[mask] >= lower[mask]) & (actual[mask] <= upper[mask]),
            slice_group_keys,
        ),
        expected_revenue_wape=_weighted_absolute_percentage_error(
            actual_revenue[mask],
            predicted_revenue[mask],
            sample_weights=sample_weights,
        ),
        sample_size=int(mask.sum()),
        unique_stays=len(set(slice_group_keys)),
        mean_price_response=(
            float(np.average(price_response_deltas[mask], weights=sample_weights))
            if price_response_deltas is not None
            else None
        ),
        flat_price_response_rate=(
            float(np.average(price_response_deltas[mask] < 0.002, weights=sample_weights))
            if price_response_deltas is not None
            else None
        ),
        upper_price_boundary_rate=(
            float(
                np.average(
                    selected_price_change_pcts[mask] >= 0.35 - 1e-12,
                    weights=sample_weights,
                )
            )
            if selected_price_change_pcts is not None
            else None
        ),
        lower_price_boundary_rate=(
            float(
                np.average(
                    selected_price_change_pcts[mask] <= -0.35 + 1e-12,
                    weights=sample_weights,
                )
            )
            if selected_price_change_pcts is not None
            else None
        ),
        any_price_boundary_rate=(
            float(
                np.average(
                    np.abs(selected_price_change_pcts[mask]) >= 0.35 - 1e-12,
                    weights=sample_weights,
                )
            )
            if selected_price_change_pcts is not None
            else None
        ),
        mean_selected_price_change_pct=(
            float(np.average(selected_price_change_pcts[mask], weights=sample_weights))
            if selected_price_change_pcts is not None
            else None
        ),
    )
