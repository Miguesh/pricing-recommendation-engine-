"""LightGBM quantile demand model with conformal interval calibration and SHAP."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from pricing_engine.domain.exceptions import ModelUnavailableError
from pricing_engine.domain.models import DemandEstimate, FeatureContribution


@dataclass(frozen=True, slots=True)
class DemandTrainingConfig:
    """Versioned hyperparameters used for each retrain."""

    random_state: int = 42
    # Central and tail objectives have different bias/variance behavior. Keep
    # their capacities independently versioned instead of forcing one value.
    n_estimators: int = 400
    quantile_n_estimators: int = 100
    learning_rate: float = 0.04
    num_leaves: int = 31
    min_child_samples: int = 30
    subsample: float = 0.9
    subsample_freq: int = 1
    colsample_bytree: float = 0.9
    interval_coverage: float = 0.90
    training_num_threads: int = -1
    inference_num_threads: int = 1

    def __post_init__(self) -> None:
        if self.n_estimators < 1 or self.quantile_n_estimators < 1:
            raise ValueError("Estimator counts must be positive.")
        if not 0 < self.interval_coverage < 1:
            raise ValueError("interval_coverage must be between 0 and 1.")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise ValueError("Sampling fractions must be in the interval (0, 1].")
        if self.subsample_freq < 0:
            raise ValueError("subsample_freq must be non-negative.")
        if self.inference_num_threads < 1:
            raise ValueError("inference_num_threads must be positive.")
        if self.training_num_threads == 0:
            raise ValueError("training_num_threads cannot be zero.")


class QuantileLightGBMDemandModel:
    """A serializable LightGBM ensemble implementing the DemandPredictor port.

    Three models estimate lower, central, and upper demand. A held-out
    calibration set expands the interval using split conformal residuals, which
    makes the interval interpretable as an empirical coverage promise rather
    than an uncalibrated tree spread.
    """

    def __init__(
        self,
        *,
        config: DemandTrainingConfig,
        feature_names: Sequence[str],
        lower_model: LGBMRegressor,
        median_model: LGBMRegressor,
        upper_model: LGBMRegressor,
        calibration_radius: float,
        feature_means: dict[str, float],
        feature_stds: dict[str, float],
        model_version: str,
        feature_contract_version: str,
        feature_schema_hash: str,
        currency: str,
        tenant_ids: Sequence[str],
        feature_quantile_lows: dict[str, float] | None = None,
        feature_quantile_highs: dict[str, float] | None = None,
    ) -> None:
        self.config = config
        self.feature_names = tuple(feature_names)
        self.lower_model = lower_model
        self.median_model = median_model
        self.upper_model = upper_model
        self.calibration_radius = calibration_radius
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.feature_quantile_lows = feature_quantile_lows or {}
        self.feature_quantile_highs = feature_quantile_highs or {}
        self.feature_contract_version = feature_contract_version
        self.feature_schema_hash = feature_schema_hash
        self.currency = currency
        self.tenant_ids = tuple(sorted(set(tenant_ids)))
        if len(self.tenant_ids) != 1:
            raise ValueError("A model bundle must declare exactly one tenant in scope.")
        self._version = model_version
        self._shap_explainer: Any | None = None
        self._shap_initialization_failed = False
        self._shap_lock = Lock()

    @property
    def version(self) -> str:
        return self._version

    @property
    def interval_coverage(self) -> float:
        return self.config.interval_coverage

    @property
    def training_parameters(self) -> dict[str, str | int | float | bool]:
        return {str(name): value for name, value in asdict(self.config).items()}

    @classmethod
    def fit(
        cls,
        *,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        train_sample_weight: Sequence[float],
        calibration_features: pd.DataFrame,
        calibration_target: pd.Series,
        calibration_group_keys: Sequence[Hashable],
        feature_contract_version: str,
        feature_schema_hash: str,
        currency: str,
        tenant_ids: Sequence[str],
        config: DemandTrainingConfig | None = None,
    ) -> QuantileLightGBMDemandModel:
        """Fit models only on past data and calibrate only on held-out observations."""

        if train_features.empty or calibration_features.empty:
            raise ValueError("Training and calibration partitions must both be non-empty.")
        sample_weight = np.asarray(train_sample_weight, dtype=float)
        if len(sample_weight) != len(train_target):
            raise ValueError("Training sample weights must match the training target length.")
        if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight <= 0):
            raise ValueError("Training sample weights must be finite and strictly positive.")
        if list(train_features.columns) != list(calibration_features.columns):
            raise ValueError("Train and calibration feature columns differ.")
        if len(calibration_group_keys) != len(calibration_target):
            raise ValueError("Calibration group keys must match the calibration target length.")
        normalized_tenants = tuple(sorted({value.strip() for value in tenant_ids if value.strip()}))
        if len(normalized_tenants) != 1:
            raise ValueError("Exactly one normalized tenant_id is required for training.")
        training_config = config or DemandTrainingConfig()
        feature_names = tuple(train_features.columns)

        common: dict[str, Any] = {
            "learning_rate": training_config.learning_rate,
            "num_leaves": training_config.num_leaves,
            "min_child_samples": training_config.min_child_samples,
            "subsample": training_config.subsample,
            "subsample_freq": training_config.subsample_freq,
            "colsample_bytree": training_config.colsample_bytree,
            "random_state": training_config.random_state,
            "n_jobs": training_config.training_num_threads,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        tail_probability = (1.0 - training_config.interval_coverage) / 2.0
        monotone_constraints = [
            -1 if name in {"candidate_price", "price_to_competitor_ratio"} else 0
            for name in feature_names
        ]
        lower_model = LGBMRegressor(
            objective="quantile",
            alpha=tail_probability,
            n_estimators=training_config.quantile_n_estimators,
            **common,
        )
        median_model = LGBMRegressor(
            # LightGBM rejects monotone constraints for regression_l1. The L2
            # regression objective supports them and is therefore used for the
            # central demand curve; interval tails remain quantile models.
            objective="regression",
            n_estimators=training_config.n_estimators,
            monotone_constraints=monotone_constraints,
            monotone_constraints_method="advanced",
            **common,
        )
        upper_model = LGBMRegressor(
            objective="quantile",
            alpha=1.0 - tail_probability,
            n_estimators=training_config.quantile_n_estimators,
            **common,
        )
        lower_model.fit(train_features, train_target, sample_weight=sample_weight)
        median_model.fit(train_features, train_target, sample_weight=sample_weight)
        upper_model.fit(train_features, train_target, sample_weight=sample_weight)

        lower = np.clip(
            lower_model.predict(
                calibration_features,
                num_threads=training_config.inference_num_threads,
            ),
            0,
            1,
        )
        upper = np.clip(
            upper_model.predict(
                calibration_features,
                num_threads=training_config.inference_num_threads,
            ),
            0,
            1,
        )
        row_residuals = np.maximum.reduce(
            [
                calibration_target.to_numpy(dtype=float) - upper,
                lower - calibration_target.to_numpy(dtype=float),
                np.zeros(len(calibration_target)),
            ]
        )
        # Multiple booking snapshots can share one realized stay outcome. Use
        # the maximum nonconformity per stay so repeated snapshots cannot
        # dominate calibration evidence and coverage is conservative by stay.
        grouped_residuals: dict[Hashable, float] = {}
        for group_key, residual in zip(
            calibration_group_keys,
            row_residuals,
            strict=True,
        ):
            grouped_residuals[group_key] = max(
                grouped_residuals.get(group_key, 0.0),
                float(residual),
            )
        residuals = np.asarray(list(grouped_residuals.values()), dtype=float)
        target_quantile = min(
            1.0,
            np.ceil((len(residuals) + 1) * training_config.interval_coverage) / len(residuals),
        )
        calibration_radius = float(np.quantile(residuals, target_quantile, method="higher"))
        means = {
            str(name): float(value) for name, value in train_features.mean().astype(float).items()
        }
        stds = {
            str(name): float(value)
            for name, value in train_features.std().replace(0, 1.0).astype(float).items()
        }
        quantile_lows = {
            str(name): float(value)
            for name, value in train_features.quantile(0.01).astype(float).items()
        }
        quantile_highs = {
            str(name): float(value)
            for name, value in train_features.quantile(0.99).astype(float).items()
        }
        fitted_state = "|".join(
            (
                repr(asdict(training_config)),
                "|".join(feature_names),
                feature_contract_version,
                feature_schema_hash,
                currency,
                "|".join(normalized_tenants),
                repr(calibration_radius),
                repr(means),
                repr(stds),
                repr(quantile_lows),
                repr(quantile_highs),
                lower_model.booster_.model_to_string(),
                median_model.booster_.model_to_string(),
                upper_model.booster_.model_to_string(),
            )
        )
        fingerprint = sha256(fitted_state.encode()).hexdigest()[:12]
        version = f"lgbm-demand-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{fingerprint}"
        return cls(
            config=training_config,
            feature_names=feature_names,
            lower_model=lower_model,
            median_model=median_model,
            upper_model=upper_model,
            calibration_radius=calibration_radius,
            feature_means=means,
            feature_stds=stds,
            model_version=version,
            feature_contract_version=feature_contract_version,
            feature_schema_hash=feature_schema_hash,
            currency=currency,
            tenant_ids=normalized_tenants,
            feature_quantile_lows=quantile_lows,
            feature_quantile_highs=quantile_highs,
        )

    def predict(self, features: pd.DataFrame) -> Sequence[DemandEstimate]:
        """Estimate calibrated occupancy and feature-space support for each row."""

        matrix = self._validate_features(features)
        prediction_parameters = {"num_threads": self.config.inference_num_threads}
        median_model: Any = self.median_model
        lower_model: Any = self.lower_model
        upper_model: Any = self.upper_model
        central = np.clip(median_model.predict(matrix, **prediction_parameters), 0, 1)
        raw_lower = np.clip(
            lower_model.predict(matrix, **prediction_parameters) - self.calibration_radius,
            0,
            1,
        )
        raw_upper = np.clip(
            upper_model.predict(matrix, **prediction_parameters) + self.calibration_radius,
            0,
            1,
        )
        # Quantile models are fitted independently and can cross. Repair their
        # ordering without moving the central estimate: this guarantees the
        # domain invariant even for out-of-distribution candidate rows.
        lower = np.minimum(raw_lower, central)
        upper = np.maximum(raw_upper, central)
        support = self._in_distribution_score(matrix)
        return [
            DemandEstimate(
                expected_occupancy=float(expected),
                lower_occupancy=float(low),
                upper_occupancy=float(high),
                in_distribution_score=float(score),
            )
            for expected, low, high, score in zip(central, lower, upper, support, strict=True)
        ]

    def explain(self, features: pd.DataFrame, *, top_k: int) -> Sequence[FeatureContribution]:
        """Produce local SHAP attributions for the selected candidate price."""

        matrix = self._validate_features(features)
        if len(matrix) != 1:
            raise ValueError("Local explanation accepts exactly one feature row.")
        try:
            explainer = self._get_shap_explainer()
            shap_values = explainer.shap_values(matrix)
            values = np.asarray(shap_values)[0]
        except Exception as error:
            raise ModelUnavailableError(
                "The model cannot produce its required SHAP explanation."
            ) from error

        ranked = sorted(
            zip(self.feature_names, values, strict=True),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:top_k]
        return [
            FeatureContribution(
                feature=name,
                contribution=round(float(value), 5),
                direction=(
                    "increases_expected_occupancy"
                    if float(value) >= 0
                    else "decreases_expected_occupancy"
                ),
            )
            for name, value in ranked
        ]

    def global_feature_importance(self, *, top_k: int) -> dict[str, float]:
        """Return normalized global gain importance from the central model."""

        raw = np.asarray(
            self.median_model.booster_.feature_importance(importance_type="gain"),
            dtype=float,
        )
        total = raw.sum()
        normalized = raw / total * 100 if total else raw
        ranked = sorted(
            zip(self.feature_names, normalized, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        return {name: round(float(score), 3) for name, score in ranked}

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if tuple(features.columns) != self.feature_names:
            raise ValueError(
                "Feature contract mismatch. "
                f"Expected {self.feature_names}, got {tuple(features.columns)}"
            )
        if features.empty:
            raise ValueError("Cannot predict an empty feature frame.")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Inference features contain non-finite values.")
        return features

    def _in_distribution_score(self, features: pd.DataFrame) -> np.ndarray:
        means = np.asarray([self.feature_means[name] for name in self.feature_names])
        stds = np.asarray([max(self.feature_stds[name], 1e-6) for name in self.feature_names])
        values = features.to_numpy(dtype=float)
        standardized = np.abs((values - means) / stds)

        lows_by_name = getattr(self, "feature_quantile_lows", {})
        highs_by_name = getattr(self, "feature_quantile_highs", {})
        if not lows_by_name or not highs_by_name:
            # Compatibility path for bundles trained before feature-contract v2.
            mean_distance = np.clip(standardized, 0, 5).mean(axis=1)
            return np.asarray(np.exp(-mean_distance / 3), dtype=float)

        lows = np.asarray([lows_by_name[name] for name in self.feature_names])
        highs = np.asarray([highs_by_name[name] for name in self.feature_names])
        ranges = np.maximum(highs - lows, stds)
        marginal_excess = np.maximum.reduce([lows - values, values - highs, np.zeros_like(values)])
        normalized_excess = marginal_excess / ranges

        # This is an operational support score, not a calibrated probability.
        # It penalizes both marginal extrapolation and extreme standardized
        # distance while remaining deterministic and bounded.
        extrapolation_penalty = 2.0 * normalized_excess.mean(axis=1) + normalized_excess.max(axis=1)
        distance_penalty = np.clip(standardized, 0, 8).mean(axis=1) / 8.0
        return np.asarray(
            np.clip(np.exp(-(extrapolation_penalty + distance_penalty)), 0, 1),
            dtype=float,
        )

    def _get_shap_explainer(self) -> Any:
        """Lazily construct and cache the thread-safe read-only explainer."""

        if self._shap_explainer is not None:
            return self._shap_explainer
        if self._shap_initialization_failed:
            raise RuntimeError("SHAP explainer initialization previously failed.")
        with self._shap_lock:
            cached = self._read_shap_cache()
            if cached is not None:
                return cached
            try:
                import shap

                self._shap_explainer = shap.TreeExplainer(self.median_model)
            except Exception:
                self._shap_initialization_failed = True
                raise
        return self._shap_explainer

    def _read_shap_cache(self) -> Any | None:
        """Read cached state through a method to preserve double-checked locking."""

        return self._shap_explainer

    def __getstate__(self) -> dict[str, Any]:
        """Exclude process-local SHAP state and locks from portable bundles."""

        state = self.__dict__.copy()
        state.pop("_shap_explainer", None)
        state.pop("_shap_lock", None)
        state["_shap_initialization_failed"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._shap_explainer = None
        self._shap_initialization_failed = False
        self._shap_lock = Lock()
