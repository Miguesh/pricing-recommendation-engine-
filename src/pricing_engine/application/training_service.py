"""Chronological demand-model training use case."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pricing_engine.application.evaluation import EvaluationMetrics, evaluate_demand_predictions
from pricing_engine.application.ports import (
    BenchmarkStatus,
    DemandModelTrainer,
    DemandPredictor,
    TrainingDataValidator,
    TrainingFeatureFactory,
)
from pricing_engine.domain.exceptions import DataContractError


@dataclass(frozen=True, slots=True)
class TemporalPartitions:
    """Strictly ordered, label-mature feature and target partitions."""

    train_features: pd.DataFrame
    train_target: pd.Series
    train_context: pd.DataFrame
    calibration_features: pd.DataFrame
    calibration_target: pd.Series
    calibration_context: pd.DataFrame
    test_features: pd.DataFrame
    test_target: pd.Series
    test_listed_prices: pd.Series
    test_context: pd.DataFrame
    calibration_start: pd.Timestamp
    test_start: pd.Timestamp
    purged_train_rows: int
    purged_calibration_rows: int


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Result of a complete, reproducible model-training attempt."""

    model: DemandPredictor
    metrics: EvaluationMetrics
    train_rows: int
    calibration_rows: int
    test_rows: int
    purged_train_rows: int
    purged_calibration_rows: int
    feature_version: str
    benchmark_metrics: EvaluationMetrics | None
    benchmark_model_version: str | None
    benchmark_status: BenchmarkStatus


class ChronologicalSplitter:
    """Purged temporal split based on when realized outcomes became available."""

    def __init__(
        self,
        train_fraction: float = 0.70,
        calibration_fraction: float = 0.15,
        embargo_days: int = 0,
        min_train_rows: int = 200,
        min_calibration_rows: int = 30,
        min_test_rows: int = 50,
        min_train_stays: int = 100,
        min_calibration_stays: int = 30,
        min_test_stays: int = 50,
    ) -> None:
        if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
            raise ValueError("Split fractions must be between zero and one.")
        if train_fraction + calibration_fraction >= 1:
            raise ValueError("Train and calibration fractions must leave a test partition.")
        if not isinstance(embargo_days, int) or isinstance(embargo_days, bool) or embargo_days < 0:
            raise ValueError("embargo_days must be a non-negative integer.")
        minimums = {
            "min_train_rows": min_train_rows,
            "min_calibration_rows": min_calibration_rows,
            "min_test_rows": min_test_rows,
            "min_train_stays": min_train_stays,
            "min_calibration_stays": min_calibration_stays,
            "min_test_stays": min_test_stays,
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in minimums.values()
        ):
            raise ValueError("Temporal partition minimums must be positive integers.")
        self._train_fraction = train_fraction
        self._calibration_fraction = calibration_fraction
        self._embargo_days = embargo_days
        self._minimums = minimums

    def split(self, frame: pd.DataFrame, features: pd.DataFrame) -> TemporalPartitions:
        ordered_source = frame.sort_values(["as_of_date", "stay_date", "property_id"])
        ordered_features = features.loc[ordered_source.index].reset_index(drop=True)
        ordered = ordered_source.reset_index(drop=True)
        dates = pd.Index(ordered["as_of_date"].dt.normalize().unique()).sort_values()
        if len(dates) < 10:
            raise ValueError(
                "At least 10 unique as_of_date values are required for temporal splitting."
            )
        train_end = max(1, int(len(dates) * self._train_fraction))
        calibration_end = max(
            train_end + 1, int(len(dates) * (self._train_fraction + self._calibration_fraction))
        )
        if calibration_end >= len(dates):
            calibration_end = len(dates) - 1
        train_dates = dates[:train_end]
        calibration_dates = dates[train_end:calibration_end]
        test_dates = dates[calibration_end:]
        if not len(calibration_dates) or not len(test_dates):
            raise ValueError("Temporal split did not create calibration and test windows.")

        calibration_start = pd.Timestamp(calibration_dates[0])
        test_start = pd.Timestamp(test_dates[0])
        embargo = pd.Timedelta(days=self._embargo_days)
        normalized_as_of = ordered["as_of_date"].dt.normalize()
        outcome_available = ordered["outcome_available_date"].dt.normalize()
        train_window_mask = normalized_as_of.isin(train_dates)
        calibration_window_mask = normalized_as_of.isin(calibration_dates)
        # Outcomes must be available strictly before the next decision window.
        # Rows whose target horizon crosses a boundary are purged, and an
        # optional fixed embargo provides an additional safety gap.
        train_mask = train_window_mask & (outcome_available < calibration_start - embargo)
        calibration_mask = calibration_window_mask & (outcome_available < test_start - embargo)
        test_mask = ordered["as_of_date"].dt.normalize().isin(test_dates)
        purged_train_rows = int(train_window_mask.sum() - train_mask.sum())
        purged_calibration_rows = int(calibration_window_mask.sum() - calibration_mask.sum())
        if (
            not bool(train_mask.any())
            or not bool(calibration_mask.any())
            or not bool(test_mask.any())
        ):
            raise ValueError(
                "Purged temporal split produced an empty partition. Provide a longer history or "
                "reduce the embargo."
            )

        train_context = ordered.loc[train_mask].reset_index(drop=True)
        calibration_context = ordered.loc[calibration_mask].reset_index(drop=True)
        test_context = ordered.loc[test_mask].reset_index(drop=True)
        self._assert_minimum_evidence(
            train_context=train_context,
            calibration_context=calibration_context,
            test_context=test_context,
            purged_train_rows=purged_train_rows,
            purged_calibration_rows=purged_calibration_rows,
        )
        self._assert_partition_invariants(
            train_context=train_context,
            calibration_context=calibration_context,
            test_context=test_context,
            calibration_start=calibration_start,
            test_start=test_start,
            embargo=embargo,
        )
        target = ordered["observed_occupancy"].astype(float)
        return TemporalPartitions(
            train_features=ordered_features.loc[train_mask].reset_index(drop=True),
            train_target=target.loc[train_mask].reset_index(drop=True),
            train_context=train_context,
            calibration_features=ordered_features.loc[calibration_mask].reset_index(drop=True),
            calibration_target=target.loc[calibration_mask].reset_index(drop=True),
            calibration_context=calibration_context,
            test_features=ordered_features.loc[test_mask].reset_index(drop=True),
            test_target=target.loc[test_mask].reset_index(drop=True),
            test_listed_prices=ordered.loc[test_mask, "listed_price"]
            .astype(float)
            .reset_index(drop=True),
            test_context=test_context,
            calibration_start=calibration_start,
            test_start=test_start,
            purged_train_rows=purged_train_rows,
            purged_calibration_rows=purged_calibration_rows,
        )

    def _assert_minimum_evidence(
        self,
        *,
        train_context: pd.DataFrame,
        calibration_context: pd.DataFrame,
        test_context: pd.DataFrame,
        purged_train_rows: int,
        purged_calibration_rows: int,
    ) -> None:
        """Require enough independent stays for fitting, calibration, and gates."""

        group_columns = ["tenant_id", "property_id", "stay_date"]
        evidence = {
            "train_rows": len(train_context),
            "calibration_rows": len(calibration_context),
            "test_rows": len(test_context),
            "train_stays": train_context.groupby(group_columns, sort=False).ngroups,
            "calibration_stays": calibration_context.groupby(group_columns, sort=False).ngroups,
            "test_stays": test_context.groupby(group_columns, sort=False).ngroups,
        }
        failures: list[str] = []
        for name, minimum in self._minimums.items():
            evidence_name = name.removeprefix("min_")
            actual = evidence[evidence_name]
            if actual < minimum:
                failures.append(f"{evidence_name}={actual} (required {minimum})")
        if failures:
            raise ValueError(
                "Insufficient evidence after purged temporal splitting: "
                + ", ".join(failures)
                + f"; purged_train_rows={purged_train_rows}, "
                f"purged_calibration_rows={purged_calibration_rows}."
            )

    @staticmethod
    def _assert_partition_invariants(
        *,
        train_context: pd.DataFrame,
        calibration_context: pd.DataFrame,
        test_context: pd.DataFrame,
        calibration_start: pd.Timestamp,
        test_start: pd.Timestamp,
        embargo: pd.Timedelta,
    ) -> None:
        """Fail closed if point-in-time or stay-group isolation is violated."""

        if not (
            train_context["outcome_available_date"].dt.normalize().max()
            < calibration_start - embargo
            and calibration_context["outcome_available_date"].dt.normalize().max()
            < test_start - embargo
        ):
            raise RuntimeError("Temporal split contains a label that was not available at cutoff.")

        group_columns = ["tenant_id", "property_id", "stay_date"]

        def group_keys(partition: pd.DataFrame) -> set[tuple[object, ...]]:
            return set(partition[group_columns].itertuples(index=False, name=None))

        train_groups = group_keys(train_context)
        calibration_groups = group_keys(calibration_context)
        test_groups = group_keys(test_context)
        if (
            train_groups.intersection(calibration_groups)
            or train_groups.intersection(test_groups)
            or calibration_groups.intersection(test_groups)
        ):
            raise RuntimeError("A tenant/property/stay_date group crosses temporal partitions.")


def _equal_stay_training_weights(context: pd.DataFrame) -> list[float]:
    """Give every realized property-night equal total fitting influence."""

    group_columns = ["tenant_id", "property_id", "stay_date"]
    group_sizes = context.groupby(group_columns, sort=False)["stay_date"].transform("size")
    weights = 1.0 / group_sizes.to_numpy(dtype=float)
    if len(weights) != len(context) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("Unable to construct finite positive stay-level training weights.")
    return [float(value) for value in weights]


class TrainDemandModel:
    """Trains and evaluates a versioned demand model without registry side effects."""

    def __init__(
        self,
        *,
        validator: TrainingDataValidator,
        feature_factory: TrainingFeatureFactory,
        model_trainer: DemandModelTrainer,
        splitter: ChronologicalSplitter | None = None,
    ) -> None:
        self._validator = validator
        self._feature_factory = feature_factory
        self._model_trainer = model_trainer
        self._splitter = splitter or ChronologicalSplitter()

    def execute(
        self,
        raw_frame: pd.DataFrame,
        *,
        benchmark_model: DemandPredictor | None = None,
    ) -> TrainingOutcome:
        validated = self._validator.validate(raw_frame)
        currencies = validated["currency"].drop_duplicates().tolist()
        if len(currencies) != 1:
            raise DataContractError(
                "A demand model must be trained in exactly one normalized currency; "
                f"received {sorted(currencies)}."
            )
        currency = str(currencies[0])
        tenant_ids = tuple(sorted(str(value) for value in validated["tenant_id"].unique()))
        if len(tenant_ids) != 1:
            raise DataContractError(
                "Each model bundle is single-tenant by design; route each tenant to an "
                f"independently trained and registered model. Received {len(tenant_ids)} tenants."
            )
        features = self._feature_factory.build_training_features(validated)
        partitions = self._splitter.split(validated, features)
        model = self._model_trainer.fit(
            train_features=partitions.train_features,
            train_target=partitions.train_target,
            train_sample_weight=_equal_stay_training_weights(partitions.train_context),
            calibration_features=partitions.calibration_features,
            calibration_target=partitions.calibration_target,
            calibration_group_keys=list(
                partitions.calibration_context[
                    ["tenant_id", "property_id", "stay_date"]
                ].itertuples(index=False, name=None)
            ),
            feature_contract_version=self._feature_factory.VERSION,
            feature_schema_hash=self._feature_factory.SCHEMA_HASH,
            currency=currency,
            tenant_ids=tenant_ids,
        )
        metrics = self._evaluate_on_test_partition(
            partitions,
            model,
        )
        benchmark_metrics = None
        benchmark_model_version = None
        benchmark_status: BenchmarkStatus = "not_available"
        if benchmark_model is not None:
            benchmark_model_version = str(getattr(benchmark_model, "version", "legacy-unknown"))
            if (
                getattr(benchmark_model, "currency", None) != currency
                or tuple(sorted(getattr(benchmark_model, "tenant_ids", ()))) != tenant_ids
                or getattr(benchmark_model, "feature_contract_version", None)
                != self._feature_factory.VERSION
                or getattr(benchmark_model, "feature_schema_hash", None)
                != self._feature_factory.SCHEMA_HASH
            ):
                benchmark_status = "incompatible_contract"
            else:
                benchmark_metrics = self._evaluate_on_test_partition(
                    partitions,
                    benchmark_model,
                )
                benchmark_status = "evaluated"
        return TrainingOutcome(
            model=model,
            metrics=metrics,
            train_rows=len(partitions.train_features),
            calibration_rows=len(partitions.calibration_features),
            test_rows=len(partitions.test_features),
            purged_train_rows=partitions.purged_train_rows,
            purged_calibration_rows=partitions.purged_calibration_rows,
            feature_version=self._feature_factory.VERSION,
            benchmark_metrics=benchmark_metrics,
            benchmark_model_version=benchmark_model_version,
            benchmark_status=benchmark_status,
        )

    @classmethod
    def _evaluate_on_test_partition(
        cls,
        partitions: TemporalPartitions,
        predictor: DemandPredictor,
    ) -> EvaluationMetrics:
        """Evaluate any compatible model against one immutable holdout context."""

        # Exercise the same revenue-maximizing behavior used by serving over a
        # canonical full ±35% guardrail, not only a three-point local probe.
        price_multipliers = np.round(np.arange(0.65, 1.351, 0.05), 2)
        grid_frames: list[pd.DataFrame] = []
        for multiplier in price_multipliers:
            grid = partitions.test_features.copy()
            grid["candidate_price"] *= multiplier
            grid["price_to_competitor_ratio"] = (
                grid["candidate_price"] / grid["competitor_price_median"]
            )
            grid_frames.append(grid)
        grid_estimates = predictor.predict(pd.concat(grid_frames, ignore_index=True))
        row_count = len(partitions.test_features)
        occupancy_grid = np.asarray(
            [estimate.expected_occupancy for estimate in grid_estimates],
            dtype=float,
        ).reshape(len(price_multipliers), row_count)
        central_index = int(np.flatnonzero(np.isclose(price_multipliers, 1.0))[0])
        lower_probe_index = int(np.flatnonzero(np.isclose(price_multipliers, 0.9))[0])
        upper_probe_index = int(np.flatnonzero(np.isclose(price_multipliers, 1.1))[0])
        price_response_deltas = (
            occupancy_grid[lower_probe_index] - occupancy_grid[upper_probe_index]
        ).tolist()
        listed_prices = partitions.test_listed_prices.to_numpy(dtype=float)
        revenue_grid = (
            occupancy_grid * price_multipliers[:, np.newaxis] * listed_prices[np.newaxis, :]
        )
        selected_indices = np.argmax(revenue_grid, axis=0)
        selected_price_change_pcts = (price_multipliers[selected_indices] - 1.0).tolist()
        central_estimates = grid_estimates[
            central_index * row_count : (central_index + 1) * row_count
        ]
        return evaluate_demand_predictions(
            actual_occupancy=partitions.test_target.tolist(),
            estimates=central_estimates,
            listed_prices=partitions.test_listed_prices.tolist(),
            group_keys=list(
                partitions.test_context[["tenant_id", "property_id", "stay_date"]].itertuples(
                    index=False,
                    name=None,
                )
            ),
            slice_memberships=cls._build_slice_memberships(partitions.test_context),
            price_response_deltas=price_response_deltas,
            selected_price_change_pcts=selected_price_change_pcts,
        )

    @staticmethod
    def _build_slice_memberships(frame: pd.DataFrame) -> dict[str, list[bool]]:
        """Build stable holdout slices that reveal aggregate-metric blind spots."""

        lead_time = (
            pd.to_datetime(frame["stay_date"]).dt.normalize()
            - pd.to_datetime(frame["as_of_date"]).dt.normalize()
        ).dt.days
        candidates = {
            "lead_time/0_7_days": lead_time <= 7,
            "lead_time/8_30_days": lead_time.between(8, 30),
            "lead_time/31_plus_days": lead_time >= 31,
            "holiday/yes": frame["is_holiday"].astype(bool),
            "holiday/no": ~frame["is_holiday"].astype(bool),
            "event/none": frame["event_intensity"].astype(float) == 0,
            "event/active": frame["event_intensity"].astype(float) > 0,
        }
        if "latitude" in frame.columns and "longitude" in frame.columns:
            location_known = frame["latitude"].notna() & frame["longitude"].notna()
        else:
            location_known = pd.Series(False, index=frame.index)
        candidates["location/known"] = location_known
        candidates["location/unknown"] = ~location_known
        return {
            name: mask.astype(bool).tolist()
            for name, mask in candidates.items()
            if bool(mask.any())
        }
