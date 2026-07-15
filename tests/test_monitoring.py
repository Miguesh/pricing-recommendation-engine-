"""Tests for feature drift reporting."""

import pandas as pd

from pricing_engine.infrastructure.monitoring import build_drift_report, population_stability_index


def test_population_stability_detects_shift() -> None:
    reference = pd.Series([1.0] * 50 + [2.0] * 50)
    current = pd.Series([10.0] * 100)

    assert population_stability_index(reference, current) > 0.20


def test_drift_report_flags_shifted_feature() -> None:
    reference = pd.DataFrame({"price": list(range(100)), "pace": [0.5] * 100})
    current = pd.DataFrame({"price": list(range(500, 600)), "pace": [0.5] * 100})

    report = build_drift_report(reference, current, threshold=0.20)

    assert report.requires_retraining_review
    assert {item.feature for item in report.drifted_features} == {"price"}
