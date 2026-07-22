"""Deterministic synthetic observations used only for tests and local demos."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

_DEMO_HOLIDAY_WINDOWS = (
    (1, 1, 5),
    (2, 14, 18),
    (3, 15, 31),
    (5, 24, 31),
    (7, 1, 7),
    (9, 1, 7),
    (11, 20, 30),
    (12, 20, 31),
)


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
    # Property-nights span a full year so seasonality and year-end holidays are
    # represented in the temporal holdout.
    start = date(2025, 1, 1)
    rows: list[dict[str, object]] = []
    market_centers = (
        (25.7617, -80.1918),  # Miami
        (40.7128, -74.0060),  # New York
        (34.0522, -118.2437),  # Los Angeles
    )
    for property_number in range(properties):
        bedrooms = int(rng.integers(1, 5))
        accommodates = bedrooms * 2 + int(rng.integers(0, 3))
        review_score = float(np.round(rng.uniform(3.7, 5.0), 2))
        base_price = 75 + bedrooms * 40 + review_score * 6 + rng.uniform(-10, 10)
        market_latitude, market_longitude = market_centers[property_number % len(market_centers)]
        has_location = property_number % 4 != 0
        latitude = float(market_latitude + rng.normal(0, 0.035)) if has_location else None
        longitude = float(market_longitude + rng.normal(0, 0.035)) if has_location else None
        for decision_offset in range(decision_days):
            # One row represents one unique property-night decision. Spacing
            # stay dates by two days preserves a uniform random 1..44 day lead
            # distribution without creating duplicate target nights.
            stay_date = start + timedelta(days=decision_offset * 2)
            lead_time = int(rng.integers(1, 45))
            as_of_date = stay_date - timedelta(days=lead_time)
            day_of_year = stay_date.timetuple().tm_yday
            season = 1 + 0.18 * np.sin(2 * np.pi * day_of_year / 365.25)
            # Peak travel windows are deliberately wider than the individual
            # federal holiday dates. This models the operational pricing
            # concept (holiday demand period) and leaves enough independent
            # property-nights to evaluate both holiday states in a temporal
            # holdout without duplicating stays.
            is_holiday = any(
                stay_date.month == month and first_day <= stay_date.day <= last_day
                for month, first_day, last_day in _DEMO_HOLIDAY_WINDOWS
            )
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
                - 6.0 * (listed_price / competitor - 1)
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
                    "outcome_available_date": stay_date + timedelta(days=1),
                    "currency": "USD",
                    "listed_price": round(float(listed_price), 2),
                    "competitor_price_median": round(float(competitor), 2),
                    "historical_occupancy_7d": historical_occupancy,
                    "booking_pace_7d": booking_pace,
                    "bedrooms": bedrooms,
                    "accommodates": accommodates,
                    "review_score": review_score,
                    "is_holiday": is_holiday,
                    "event_intensity": event_intensity,
                    "latitude": latitude,
                    "longitude": longitude,
                    "observed_occupancy": occupancy,
                }
            )
    frame = pd.DataFrame(rows)
    stay_keys = ["tenant_id", "property_id", "stay_date"]
    if frame.duplicated(stay_keys).any():
        raise RuntimeError("Synthetic generator produced a duplicate property-night.")
    return frame
