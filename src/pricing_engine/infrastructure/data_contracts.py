"""Validated, point-in-time contracts for offline pricing observations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pandera.pandas as pa

from pricing_engine.domain.exceptions import DataContractError

REQUIRED_TRAINING_COLUMNS = (
    "tenant_id",
    "property_id",
    "stay_date",
    "as_of_date",
    "outcome_available_date",
    "currency",
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
        "outcome_available_date": pa.Column(pa.DateTime, nullable=False),
        "currency": pa.Column(
            str,
            nullable=False,
            checks=pa.Check.str_matches(r"^[A-Z]{3}$"),
        ),
        "listed_price": pa.Column(float, nullable=False, checks=pa.Check.gt(0)),
        "competitor_price_median": pa.Column(float, nullable=False, checks=pa.Check.gt(0)),
        "historical_occupancy_7d": pa.Column(float, nullable=False, checks=pa.Check.in_range(0, 1)),
        "booking_pace_7d": pa.Column(float, nullable=False, checks=pa.Check.in_range(0, 1)),
        "bedrooms": pa.Column(int, nullable=False, checks=pa.Check.in_range(0, 20)),
        "accommodates": pa.Column(int, nullable=False, checks=pa.Check.in_range(1, 50)),
        "review_score": pa.Column(float, nullable=False, checks=pa.Check.in_range(0, 5)),
        "is_holiday": pa.Column(bool, nullable=False),
        "event_intensity": pa.Column(float, nullable=False, checks=pa.Check.in_range(0, 5)),
        "latitude": pa.Column(
            float,
            nullable=True,
            required=False,
            checks=pa.Check.in_range(-90, 90),
        ),
        "longitude": pa.Column(
            float,
            nullable=True,
            required=False,
            checks=pa.Check.in_range(-180, 180),
        ),
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
    _normalize_identifiers(candidate)
    _normalize_daily_dates(candidate)
    _normalize_strict_boolean(candidate, "is_holiday")
    _normalize_numeric_columns(candidate)
    has_latitude = "latitude" in candidate.columns
    has_longitude = "longitude" in candidate.columns
    if has_latitude != has_longitude:
        raise DataContractError(
            "latitude and longitude must either both be present or both be absent."
        )
    try:
        validated = TRAINING_SCHEMA.validate(candidate, lazy=True)
    except pa.errors.SchemaErrors as error:
        failure_cases = error.failure_cases.head(10).to_dict(orient="records")
        raise DataContractError(f"Training schema validation failed: {failure_cases}") from error
    if (validated["as_of_date"] > validated["stay_date"]).any():
        raise DataContractError("as_of_date must be on or before stay_date for every row.")
    if (validated["outcome_available_date"] <= validated["stay_date"]).any():
        raise DataContractError("outcome_available_date must be after stay_date for every row.")
    if validated.duplicated(["tenant_id", "property_id", "stay_date", "as_of_date"]).any():
        raise DataContractError(
            "Duplicate tenant/property/stay_date/as_of_date snapshots are not permitted."
        )
    stay_keys = ["tenant_id", "property_id", "stay_date"]
    stay_groups = validated.groupby(stay_keys, sort=False, dropna=False)
    if (stay_groups["observed_occupancy"].nunique(dropna=False) > 1).any():
        raise DataContractError(
            "All snapshots for the same tenant/property/stay_date must share one final outcome."
        )
    if (stay_groups["outcome_available_date"].nunique(dropna=False) > 1).any():
        raise DataContractError(
            "All snapshots for the same tenant/property/stay_date must share one "
            "outcome_available_date."
        )
    if (stay_groups["currency"].nunique(dropna=False) > 1).any():
        raise DataContractError(
            "All snapshots for the same tenant/property/stay_date must share one currency."
        )
    if has_latitude and has_longitude:
        paired_missingness = validated["latitude"].isna() == validated["longitude"].isna()
        if not paired_missingness.all():
            raise DataContractError(
                "latitude and longitude must be populated together for each observation."
            )
    return validated.reset_index(drop=True)


def _normalize_identifiers(frame: pd.DataFrame) -> None:
    """Canonicalize grouping/security identifiers without coercing arbitrary objects."""

    for column in ("tenant_id", "property_id"):
        if not frame[column].map(lambda value: isinstance(value, str)).all():
            raise DataContractError(f"{column} must contain strings, not coerced values.")
        normalized = frame[column].str.strip()
        valid = normalized.map(
            lambda value: (
                bool(value)
                and len(value) <= 100
                and all(
                    character.isprintable() and character not in "\r\n\t" for character in value
                )
            )
        )
        if not valid.all():
            raise DataContractError(
                f"{column} must be a non-blank printable identifier of at most 100 characters."
            )
        frame[column] = normalized

    if not frame["currency"].map(lambda value: isinstance(value, str)).all():
        raise DataContractError("currency must contain three-letter string codes.")
    frame["currency"] = frame["currency"].str.strip().str.upper()


def _normalize_daily_dates(frame: pd.DataFrame) -> None:
    """Enforce timezone-naive midnight timestamps for the nightly data grain."""

    for column in ("stay_date", "as_of_date", "outcome_available_date"):
        raw = frame[column]
        contains_numeric = raw.map(
            lambda value: (
                isinstance(value, (int, float, np.number))
                and not isinstance(value, (bool, np.bool_))
            )
        ).any()
        if contains_numeric:
            raise DataContractError(f"{column} cannot contain numeric timestamp encodings.")
        try:
            parsed = pd.to_datetime(raw, errors="raise", utc=False)
        except (TypeError, ValueError) as error:
            raise DataContractError(f"{column} contains an invalid calendar date.") from error
        if isinstance(parsed.dtype, pd.DatetimeTZDtype) or not pd.api.types.is_datetime64_dtype(
            parsed.dtype
        ):
            raise DataContractError(f"{column} must be timezone-naive and use one date format.")
        normalized = parsed.dt.normalize()
        if not parsed.equals(normalized):
            raise DataContractError(
                f"{column} must represent a nightly date at midnight, without time-of-day."
            )
        frame[column] = normalized


def _normalize_strict_boolean(frame: pd.DataFrame, column: str) -> None:
    """Parse only an explicit boolean vocabulary; never rely on Python truthiness."""

    def parse(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                return True
            if normalized in {"false", "0"}:
                return False
        raise DataContractError(f"{column} accepts only booleans, 0/1, or the strings true/false.")

    frame[column] = frame[column].map(parse)


def _normalize_numeric_columns(frame: pd.DataFrame) -> None:
    """Reject non-finite and lossy numeric coercions before schema conversion."""

    numeric_columns = [
        "listed_price",
        "competitor_price_median",
        "historical_occupancy_7d",
        "booking_pace_7d",
        "bedrooms",
        "accommodates",
        "review_score",
        "event_intensity",
        "observed_occupancy",
    ]
    for optional in ("latitude", "longitude"):
        if optional in frame.columns:
            numeric_columns.append(optional)
    for column in numeric_columns:
        try:
            numeric = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise DataContractError(f"{column} must contain numeric values.") from error
        finite_or_missing = numeric.isna() | np.isfinite(numeric.astype(float))
        if not finite_or_missing.all():
            raise DataContractError(f"{column} must contain only finite values.")
        frame[column] = numeric
    for column in ("bedrooms", "accommodates"):
        if not (frame[column].astype(float) % 1 == 0).all():
            raise DataContractError(f"{column} must contain whole numbers without truncation.")
