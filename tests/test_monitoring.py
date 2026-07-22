"""Tests for feature drift reporting."""

import numpy as np
import pandas as pd

from pricing_engine.infrastructure.monitoring import (
    DriftStatus,
    build_drift_report,
    population_stability_index,
)


def test_population_stability_detects_shift() -> None:
    reference = pd.Series([1.0] * 50 + [2.0] * 50)
    current = pd.Series([10.0] * 100)

    assert population_stability_index(reference, current) > 0.20


def test_population_stability_detects_binary_prevalence_shift() -> None:
    reference = pd.Series([0.0] * 95 + [1.0] * 5)
    current = pd.Series([0.0] * 50 + [1.0] * 50)

    report = build_drift_report(
        pd.DataFrame({"is_holiday": reference}),
        pd.DataFrame({"is_holiday": current}),
    )

    assert population_stability_index(reference, current) > 0.20
    assert report.features[0].status is DriftStatus.DRIFTED


def test_population_stability_requires_a_minimum_current_window() -> None:
    reference = pd.Series(range(100), dtype=float)
    current = pd.Series(range(10), dtype=float)

    report = build_drift_report(
        pd.DataFrame({"price": reference}),
        pd.DataFrame({"price": current}),
    )

    assert np.isnan(report.features[0].population_stability_index)
    assert report.features[0].status is DriftStatus.INSUFFICIENT_CURRENT


def test_drift_report_flags_shifted_feature() -> None:
    reference = pd.DataFrame({"price": list(range(100)), "pace": [0.5] * 100})
    current = pd.DataFrame({"price": list(range(500, 600)), "pace": [0.5] * 100})

    report = build_drift_report(reference, current, threshold=0.20)

    assert report.requires_retraining_review
    assert {item.feature for item in report.drifted_features} == {"price"}
    assert report.features[0].status is DriftStatus.DRIFTED
    assert report.features[1].status is DriftStatus.CONSTANT_REFERENCE


def test_non_computable_psi_is_not_reported_as_false_zero() -> None:
    reference = pd.DataFrame(
        {
            "constant": [0.5] * 20,
            "missing_current": list(range(20)),
        }
    )
    current = pd.DataFrame(
        {
            "constant": [0.5] * 20,
            "missing_current": [np.nan] * 20,
        }
    )

    report = build_drift_report(reference, current)

    assert np.isnan(population_stability_index(reference["constant"], current["constant"]))
    assert {item.status for item in report.non_computable_features} == {
        DriftStatus.CONSTANT_REFERENCE,
        DriftStatus.INSUFFICIENT_CURRENT,
    }
    assert report.requires_retraining_review
