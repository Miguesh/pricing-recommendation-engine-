"""Data-quality and feature-drift calculations for retraining decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

MINIMUM_PSI_SAMPLES = 30


class DriftStatus(StrEnum):
    """Operational state for a feature-level drift calculation."""

    OK = "ok"
    DRIFTED = "drifted"
    INSUFFICIENT_REFERENCE = "insufficient_reference"
    INSUFFICIENT_CURRENT = "insufficient_current"
    CONSTANT_REFERENCE = "constant_reference"


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    feature: str
    population_stability_index: float
    reference_missing_rate: float
    current_missing_rate: float
    status: DriftStatus = DriftStatus.OK


@dataclass(frozen=True, slots=True)
class DriftReport:
    features: tuple[FeatureDrift, ...]
    threshold: float

    @property
    def drifted_features(self) -> tuple[FeatureDrift, ...]:
        return tuple(item for item in self.features if item.status is DriftStatus.DRIFTED)

    @property
    def non_computable_features(self) -> tuple[FeatureDrift, ...]:
        return tuple(
            item
            for item in self.features
            if item.status
            not in {
                DriftStatus.OK,
                DriftStatus.DRIFTED,
            }
        )

    @property
    def requires_retraining_review(self) -> bool:
        # Insufficient evidence is review-worthy; it must never be interpreted
        # as proof that a feature is stable.
        return bool(self.drifted_features or self.non_computable_features)


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
) -> float:
    """Calculate PSI or return NaN when the evidence is not computable.

    Callers that need the reason should use ``build_drift_report``, whose
    feature findings carry an explicit ``DriftStatus``.
    """

    value, _ = _calculate_population_stability_index(reference, current, bins=bins)
    return value


def _calculate_population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int,
) -> tuple[float, DriftStatus | None]:
    if bins < 2:
        raise ValueError("bins must be at least 2.")

    baseline = reference.dropna().to_numpy(dtype=float)
    observed = current.dropna().to_numpy(dtype=float)
    if len(baseline) < 2:
        return float("nan"), DriftStatus.INSUFFICIENT_REFERENCE
    if len(observed) < 2:
        return float("nan"), DriftStatus.INSUFFICIENT_CURRENT
    unique_baseline = np.unique(baseline)
    if len(unique_baseline) < 2:
        return float("nan"), DriftStatus.CONSTANT_REFERENCE
    if len(baseline) < MINIMUM_PSI_SAMPLES:
        return float("nan"), DriftStatus.INSUFFICIENT_REFERENCE
    if len(observed) < MINIMUM_PSI_SAMPLES:
        return float("nan"), DriftStatus.INSUFFICIENT_CURRENT

    if len(unique_baseline) <= bins:
        # Low-cardinality numeric features (holiday, month, location-known)
        # are categorical for drift purposes. Quantile edges would collapse a
        # binary baseline into one [-inf, +inf] bin and hide material shifts.
        categories = np.union1d(unique_baseline, np.unique(observed))
        reference_counts = np.asarray([(baseline == value).sum() for value in categories])
        current_counts = np.asarray([(observed == value).sum() for value in categories])
        return _psi_from_counts(reference_counts, current_counts), None

    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return float("nan"), DriftStatus.CONSTANT_REFERENCE
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(observed, bins=edges)
    return _psi_from_counts(reference_counts, current_counts), None


def _psi_from_counts(reference_counts: np.ndarray, current_counts: np.ndarray) -> float:
    """Calculate PSI from aligned counts with normalized additive smoothing."""

    epsilon = 1e-6
    reference_share = (reference_counts.astype(float) + epsilon) / (
        float(reference_counts.sum()) + epsilon * len(reference_counts)
    )
    current_share = (current_counts.astype(float) + epsilon) / (
        float(current_counts.sum()) + epsilon * len(current_counts)
    )
    value = float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )
    return value


def build_drift_report(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    *,
    threshold: float = 0.20,
) -> DriftReport:
    """Compare aligned numeric feature frames and retain a reviewable report."""

    if tuple(reference_features.columns) != tuple(current_features.columns):
        raise ValueError("Reference and current feature contracts must match.")
    findings = tuple(
        _build_feature_drift(
            feature=column,
            reference=reference_features[column],
            current=current_features[column],
            threshold=threshold,
        )
        for column in reference_features.columns
    )
    return DriftReport(features=findings, threshold=threshold)


def _build_feature_drift(
    *,
    feature: str,
    reference: pd.Series,
    current: pd.Series,
    threshold: float,
) -> FeatureDrift:
    value, non_computable_status = _calculate_population_stability_index(
        reference,
        current,
        bins=10,
    )
    status = non_computable_status or (
        DriftStatus.DRIFTED if value >= threshold else DriftStatus.OK
    )
    return FeatureDrift(
        feature=feature,
        population_stability_index=value,
        reference_missing_rate=float(reference.isna().mean()),
        current_missing_rate=float(current.isna().mean()),
        status=status,
    )
