"""Versioned point-in-time feature engineering."""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from typing import ClassVar, cast

import numpy as np
import pandas as pd

from pricing_engine.domain.models import PricingContext


class PricingFeatureFactory:
    """Builds the exact feature contract shared by training and serving."""

    VERSION = "2.0.0"
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
        "latitude",
        "longitude",
        "location_known",
        "stay_day_of_week",
        "stay_month",
        "seasonality_sin",
        "seasonality_cos",
    )
    FEATURE_MANIFEST: ClassVar[dict[str, str]] = {
        "candidate_price": "float64|currency:model_currency|source:listed_or_candidate_price",
        "lead_time_days": "float64|days|stay_date-as_of_date_normalized",
        "historical_occupancy_7d": "float64|ratio_0_1|point_in_time_input",
        "booking_pace_7d": "float64|ratio_0_1|point_in_time_input",
        "competitor_price_median": "float64|currency:model_currency|point_in_time_input",
        "price_to_competitor_ratio": "float64|ratio|candidate_price/competitor_price_median",
        "bedrooms": "float64|count_0_20|cast_from_validated_integer",
        "accommodates": "float64|count_1_50|cast_from_validated_integer",
        "review_score": "float64|score_0_5|point_in_time_input",
        "is_holiday": "int64|binary|strict_boolean_cast",
        "event_intensity": "float64|score_0_5|point_in_time_input",
        "latitude": "float64|degrees|missing_sentinel_0_with_location_known",
        "longitude": "float64|degrees|missing_sentinel_0_with_location_known",
        "location_known": "int64|binary|latitude_and_longitude_non_null",
        "stay_day_of_week": "float64|index_0_6|pandas_monday_zero",
        "stay_month": "float64|index_1_12|gregorian_month",
        "seasonality_sin": "float64|cyclic|sin(2*pi*day_of_year/365.25)",
        "seasonality_cos": "float64|cyclic|cos(2*pi*day_of_year/365.25)",
    }
    if tuple(FEATURE_MANIFEST) != FEATURE_COLUMNS:
        raise RuntimeError("Feature manifest order must match FEATURE_COLUMNS.")
    SCHEMA_HASH = sha256(
        json.dumps(
            {
                "version": VERSION,
                "ordered_features": list(FEATURE_COLUMNS),
                "features": FEATURE_MANIFEST,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

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
            latitude=self._optional_numeric_column(frame, "latitude"),
            longitude=self._optional_numeric_column(frame, "longitude"),
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
            latitude=pd.Series([context.latitude] * length, dtype=float),
            longitude=pd.Series([context.longitude] * length, dtype=float),
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
        latitude: pd.Series,
        longitude: pd.Series,
        stay_date: pd.Series,
    ) -> pd.DataFrame:
        day_of_year = stay_date.dt.dayofyear.astype(float)
        location_known = latitude.notna() & longitude.notna()
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
                # Unknown coordinates use a neutral numeric sentinel that is
                # explicitly disambiguated by location_known. This keeps the
                # feature matrix finite and preserves old clients that do not
                # yet send geospatial context.
                "latitude": latitude.fillna(0.0).astype(float),
                "longitude": longitude.fillna(0.0).astype(float),
                "location_known": location_known.astype(int),
                "stay_day_of_week": stay_date.dt.dayofweek.astype(float),
                "stay_month": stay_date.dt.month.astype(float),
                "seasonality_sin": np.sin(2 * np.pi * day_of_year / 365.25),
                "seasonality_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            }
        )
        if not np.isfinite(output.to_numpy(dtype=float)).all():
            raise ValueError("Feature generation produced non-finite values.")
        return cast(pd.DataFrame, output.loc[:, self.FEATURE_COLUMNS])

    @staticmethod
    def _optional_numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
        """Return an optional location column aligned to the source index."""

        if name not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return frame[name].astype(float)
