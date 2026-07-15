"""Validated, point-in-time contracts for offline pricing observations."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import pandera.pandas as pa

from pricing_engine.domain.exceptions import DataContractError

REQUIRED_TRAINING_COLUMNS = (
    "tenant_id",
    "property_id",
    "stay_date",
    "as_of_date",
    "listed_price",
    "competitor_price_median",
    "historical_occupancy_7d",
    "booking_pace_7d",
    "bedrooms",
    "accommodates",
    "review_score",
    "is_holiday",
    "event_intensity",
    "observed_occupancy",
)

TRAINING_SCHEMA = pa.DataFrameSchema(
    {
        "tenant_id": pa.Column(str, nullable=False, checks=pa.Check.str_length(min_value=1)),
        "property_id": pa.Column(str, nullable=False, checks=pa.Check.str_length(min_value=1)),
        "stay_date": pa.Column(pa.DateTime, nullable=False),
        "as_of_date": pa.Column(pa.DateTime, nullable=False),
        "listed_price": pa.Column(float, nullable=False, checks=pa.Check.gt(0)),
        "competitor_price_median": pa.Column(float, nullable=False, checks=pa.Check.gt(0)),
        "historical_occupancy_7d": pa.Column(float, nullable=False, checks=pa.Check.in_range(0, 1)),
        "booking_pace_7d": pa.Column(float, nullable=False, checks=pa.Check.ge(0)),
        "bedrooms": pa.Column(int, nullable=False, checks=pa.Check.ge(0)),
        "accommodates": pa.Column(int, nullable=False, checks=pa.Check.ge(1)),
        "review_score": pa.Column(float, nullable=False, checks=pa.Check.in_range(0, 5)),
        "is_holiday": pa.Column(bool, nullable=False),
        "event_intensity": pa.Column(float, nullable=False, checks=pa.Check.ge(0)),
        "observed_occupancy": pa.Column(float, nullable=False, checks=pa.Check.in_range(0, 1)),
    },
    strict=False,
    coerce=True,
)


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Fail fast with a concise error before a schema validator runs."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DataContractError(f"Training data is missing required columns: {missing}")


def validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of raw historical observations.

    Every observation represents a pricing decision at its as-of timestamp and
    its realized occupancy after the stay date. The explicit ordering check
    prevents a common pricing-model leakage failure.
    """

    require_columns(frame, REQUIRED_TRAINING_COLUMNS)
    candidate = frame.copy()
    candidate["stay_date"] = pd.to_datetime(candidate["stay_date"], utc=False)
    candidate["as_of_date"] = pd.to_datetime(candidate["as_of_date"], utc=False)
    try:
        validated = TRAINING_SCHEMA.validate(candidate, lazy=True)
    except pa.errors.SchemaErrors as error:
        failure_cases = error.failure_cases.head(10).to_dict(orient="records")
        raise DataContractError(f"Training schema validation failed: {failure_cases}") from error
    if (validated["as_of_date"] > validated["stay_date"]).any():
        raise DataContractError("as_of_date must be on or before stay_date for every row.")
    if validated.duplicated(["tenant_id", "property_id", "stay_date", "as_of_date"]).any():
        raise DataContractError(
            "Duplicate tenant/property/stay_date/as_of_date snapshots are not permitted."
        )
    return validated
