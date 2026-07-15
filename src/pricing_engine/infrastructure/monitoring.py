"""Data-quality and feature-drift calculations for retraining decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    feature: str
    population_stability_index: float
    reference_missing_rate: float
    current_missing_rate: float


@dataclass(frozen=True, slots=True)
class DriftReport:
    features: tuple[FeatureDrift, ...]
    threshold: float

    @property
    def drifted_features(self) -> tuple[FeatureDrift, ...]:
        return tuple(
            item for item in self.features if item.population_stability_index >= self.threshold
        )

    @property
    def requires_retraining_review(self) -> bool:
        return bool(self.drifted_features)


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
) -> float:
    """Calculate PSI with bins derived only from the reference population."""

    baseline = reference.dropna().to_numpy(dtype=float)
    observed = current.dropna().to_numpy(dtype=float)
    if len(baseline) < 2 or len(observed) < 1:
        return 0.0
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(observed, bins=edges)
    epsilon = 1e-6
    reference_share = np.maximum(reference_counts / len(baseline), epsilon)
    current_share = np.maximum(current_counts / len(observed), epsilon)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


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
        FeatureDrift(
            feature=column,
            population_stability_index=population_stability_index(
                reference_features[column], current_features[column]
            ),
            reference_missing_rate=float(reference_features[column].isna().mean()),
            current_missing_rate=float(current_features[column].isna().mean()),
        )
        for column in reference_features.columns
    )
    return DriftReport(features=findings, threshold=threshold)
