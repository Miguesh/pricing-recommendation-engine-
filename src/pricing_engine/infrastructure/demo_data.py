"""Deterministic synthetic observations used only for tests and local demos."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def generate_demo_observations(
    *,
    properties: int = 12,
    decision_days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate observational pricing snapshots with a known demand response.

    This is intentionally isolated from production connectors. It permits a
    reviewer to run the full pipeline without pretending the output is trained
    on market data.
    """

    if properties < 1 or decision_days < 10:
        raise ValueError("properties must be positive and decision_days must be at least 10.")
    rng = np.random.default_rng(seed)
    start = date(2025, 1, 1)
    rows: list[dict[str, object]] = []
    for property_number in range(properties):
        bedrooms = int(rng.integers(1, 5))
        accommodates = bedrooms * 2 + int(rng.integers(0, 3))
        review_score = float(np.round(rng.uniform(3.7, 5.0), 2))
        base_price = 75 + bedrooms * 40 + review_score * 6 + rng.uniform(-10, 10)
        for decision_offset in range(decision_days):
            as_of_date = start + timedelta(days=decision_offset)
            lead_time = int(rng.integers(1, 45))
            stay_date = as_of_date + timedelta(days=lead_time)
            day_of_year = stay_date.timetuple().tm_yday
            season = 1 + 0.18 * np.sin(2 * np.pi * day_of_year / 365.25)
            is_holiday = stay_date.month == 12 and stay_date.day in {24, 25, 31}
            event_intensity = float(rng.choice([0, 0, 0.2, 0.5, 0.8]))
            competitor = base_price * season * (1 + event_intensity * 0.18) * rng.normal(1, 0.06)
            listed_price = competitor * rng.normal(1.0, 0.10)
            historical_occupancy = float(np.clip(rng.normal(0.60, 0.16), 0.05, 0.98))
            booking_pace = float(
                np.clip(
                    historical_occupancy * (45 - lead_time) / 45 + rng.normal(0.08, 0.07),
                    0,
                    1,
                )
            )
            logit = (
                -0.25
                + 1.8 * historical_occupancy
                + 1.2 * booking_pace
                - 1.7 * (listed_price / competitor - 1)
                + 0.55 * float(is_holiday)
                + 0.35 * event_intensity
                - 0.012 * lead_time
                + 0.10 * (review_score - 4.0)
                + rng.normal(0, 0.12)
            )
            occupancy = float(1 / (1 + np.exp(-logit)))
            rows.append(
                {
                    "tenant_id": "demo-tenant",
                    "property_id": f"property-{property_number:03d}",
                    "stay_date": stay_date,
                    "as_of_date": as_of_date,
                    "listed_price": round(float(listed_price), 2),
                    "competitor_price_median": round(float(competitor), 2),
                    "historical_occupancy_7d": historical_occupancy,
                    "booking_pace_7d": booking_pace,
                    "bedrooms": bedrooms,
                    "accommodates": accommodates,
                    "review_score": review_score,
                    "is_holiday": is_holiday,
                    "event_intensity": event_intensity,
                    "observed_occupancy": occupancy,
                }
            )
    return pd.DataFrame(rows)
