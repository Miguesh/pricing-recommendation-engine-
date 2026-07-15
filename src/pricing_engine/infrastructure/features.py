"""Versioned point-in-time feature engineering."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from pricing_engine.domain.models import PricingContext


class PricingFeatureFactory:
    """Builds the exact feature contract shared by training and serving."""

    VERSION = "1.0.0"
    FEATURE_COLUMNS = (
        "candidate_price",
        "lead_time_days",
        "historical_occupancy_7d",
        "booking_pace_7d",
        "competitor_price_median",
        "price_to_competitor_ratio",
        "bedrooms",
        "accommodates",
        "review_score",
        "is_holiday",
        "event_intensity",
        "stay_day_of_week",
        "stay_month",
        "seasonality_sin",
        "seasonality_cos",
    )

    def build_training_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build model features from only signals available on each as-of date."""

        stay_date = pd.to_datetime(frame["stay_date"])
        as_of_date = pd.to_datetime(frame["as_of_date"])
        lead_time = (stay_date.dt.normalize() - as_of_date.dt.normalize()).dt.days
        if (lead_time < 0).any():
            raise ValueError("Cannot create features with a negative lead time.")
        return self._assemble(
            candidate_price=frame["listed_price"].astype(float),
            lead_time_days=lead_time,
            historical_occupancy_7d=frame["historical_occupancy_7d"],
            booking_pace_7d=frame["booking_pace_7d"],
            competitor_price_median=frame["competitor_price_median"],
            bedrooms=frame["bedrooms"],
            accommodates=frame["accommodates"],
            review_score=frame["review_score"],
            is_holiday=frame["is_holiday"],
            event_intensity=frame["event_intensity"],
            stay_date=stay_date,
        )

    def for_candidate_prices(
        self, context: PricingContext, candidate_prices: Sequence[float]
    ) -> pd.DataFrame:
        """Create one feature row per candidate price without mutating context."""

        length = len(candidate_prices)
        if length == 0:
            raise ValueError("At least one candidate price is required.")
        stay_date = pd.Series([pd.Timestamp(context.stay_date)] * length)
        return self._assemble(
            candidate_price=pd.Series(candidate_prices, dtype=float),
            lead_time_days=pd.Series(
                [(context.stay_date - context.as_of_date).days] * length,
                dtype=float,
            ),
            historical_occupancy_7d=pd.Series([context.historical_occupancy_7d] * length),
            booking_pace_7d=pd.Series([context.booking_pace_7d] * length),
            competitor_price_median=pd.Series(
                [float(context.competitor_price_median)] * length,
                dtype=float,
            ),
            bedrooms=pd.Series([context.bedrooms] * length),
            accommodates=pd.Series([context.accommodates] * length),
            review_score=pd.Series([context.review_score] * length),
            is_holiday=pd.Series([context.is_holiday] * length),
            event_intensity=pd.Series([context.event_intensity] * length),
            stay_date=stay_date,
        )

    def _assemble(
        self,
        *,
        candidate_price: pd.Series,
        lead_time_days: pd.Series,
        historical_occupancy_7d: pd.Series,
        booking_pace_7d: pd.Series,
        competitor_price_median: pd.Series,
        bedrooms: pd.Series,
        accommodates: pd.Series,
        review_score: pd.Series,
        is_holiday: pd.Series,
        event_intensity: pd.Series,
        stay_date: pd.Series,
    ) -> pd.DataFrame:
        day_of_year = stay_date.dt.dayofyear.astype(float)
        output = pd.DataFrame(
            {
                "candidate_price": candidate_price.astype(float),
                "lead_time_days": lead_time_days.astype(float),
                "historical_occupancy_7d": historical_occupancy_7d.astype(float),
                "booking_pace_7d": booking_pace_7d.astype(float),
                "competitor_price_median": competitor_price_median.astype(float),
                "price_to_competitor_ratio": (
                    candidate_price.astype(float) / competitor_price_median.astype(float)
                ),
                "bedrooms": bedrooms.astype(float),
                "accommodates": accommodates.astype(float),
                "review_score": review_score.astype(float),
                "is_holiday": is_holiday.astype(int),
                "event_intensity": event_intensity.astype(float),
                "stay_day_of_week": stay_date.dt.dayofweek.astype(float),
                "stay_month": stay_date.dt.month.astype(float),
                "seasonality_sin": np.sin(2 * np.pi * day_of_year / 365.25),
                "seasonality_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            }
        )
        if not np.isfinite(output.to_numpy(dtype=float)).all():
            raise ValueError("Feature generation produced non-finite values.")
        return cast(pd.DataFrame, output.loc[:, self.FEATURE_COLUMNS])
