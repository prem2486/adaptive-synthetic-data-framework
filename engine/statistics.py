
"""Statistical evaluation metrics for synthetic tabular datasets.

Compares a reference dataset with generated synthetic data.
Designed to be reusable across all generator implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import ks_2samp, wasserstein_distance


def _validate_datasets(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> None:
    """Validate reference and synthetic datasets."""
    if not isinstance(real_data, pd.DataFrame):
        raise TypeError("real_data must be a pandas DataFrame.")

    if not isinstance(synthetic_data, pd.DataFrame):
        raise TypeError(
            "synthetic_data must be a pandas DataFrame."
        )

    if real_data.empty or synthetic_data.empty:
        raise ValueError(
            "Reference and synthetic datasets must contain rows."
        )

    if real_data.columns.has_duplicates:
        raise ValueError(
            "Reference dataset contains duplicate column names."
        )

    if synthetic_data.columns.has_duplicates:
        raise ValueError(
            "Synthetic dataset contains duplicate column names."
        )

    if list(real_data.columns) != list(synthetic_data.columns):
        raise ValueError(
            "Reference and synthetic datasets must have identical "
            "column names in the same order."
        )


def _numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric columns, excluding Boolean columns."""
    return [
        column
        for column in data.columns
        if is_numeric_dtype(data[column].dtype)
        and not is_bool_dtype(data[column].dtype)
    ]


def _categorical_columns(data: pd.DataFrame) -> list[str]:
    """Return non-numeric columns, including Boolean columns."""
    return [
        column
        for column in data.columns
        if column not in _numeric_columns(data)
    ]


def _finite_numeric_values(
    series: pd.Series,
    column_name: str,
    dataset_name: str,
) -> np.ndarray:
    """Extract finite numerical values, excluding missing values."""
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"Column '{column_name}' in {dataset_name} contains "
            "infinite values. Clean the data before evaluation."
        )

    return values


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Calculate a ratio without dividing by zero."""
    if denominator == 0:
        return 0.0 if numerator == 0 else 1.0

    return float(numerator / denominator)


def calculate_ks_statistic(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> dict[str, float]:
    """Calculate the two-sample KS statistic for each numeric column.

    Returns:
        Mapping of column name to KS statistic.
    """
    results = {}

    for column in _numeric_columns(real_data):
        real = _finite_numeric_values(
            real_data[column], column, "reference data"
        )
        synthetic = _finite_numeric_values(
            synthetic_data[column], column, "synthetic data"
        )

        if len(real) == 0 or len(synthetic) == 0:
            continue

        results[column] = float(
            ks_2samp(real, synthetic).statistic
        )

    return results


def calculate_wasserstein_distance(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> dict[str, float]:
    """Calculate Wasserstein distance for each numeric column."""
    results = {}

    for column in _numeric_columns(real_data):
        real = _finite_numeric_values(
            real_data[column], column, "reference data"
        )
        synthetic = _finite_numeric_values(
            synthetic_data[column], column, "synthetic data"
        )

        if len(real) == 0 or len(synthetic) == 0:
            continue

        results[column] = float(
            wasserstein_distance(real, synthetic)
        )

    return results


def calculate_mean_std_gap(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Calculate normalized mean and standard-deviation gaps.

    Each gap is normalized by the reference column's standard
    deviation. If that deviation is zero, a zero gap is returned
    when the corresponding statistics match; otherwise the gap
    is reported as 1.0.

    Returns:
        Mapping of column name to mean_gap and std_gap.
    """
    results = {}

    for column in _numeric_columns(real_data):
        real = _finite_numeric_values(
            real_data[column], column, "reference data"
        )
        synthetic = _finite_numeric_values(
            synthetic_data[column], column, "synthetic data"
        )

        if len(real) == 0 or len(synthetic) == 0:
            continue

        real_mean = float(np.mean(real))
        synthetic_mean = float(np.mean(synthetic))

        real_std = float(np.std(real, ddof=0))
        synthetic_std = float(np.std(synthetic, ddof=0))

        mean_gap = _safe_ratio(
            abs(real_mean - synthetic_mean),
            real_std,
        )

        std_gap = _safe_ratio(
            abs(real_std - synthetic_std),
            real_std,
        )

        results[column] = {
            "mean_gap": mean_gap,
            "std_gap": std_gap,
        }

    return results


def calculate_correlation_mae(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> float:
    """Calculate MAE between reference and synthetic correlations.

    Uses the common numerical columns with at least two valid,
    non-constant observations in both datasets.

    Returns:
        Mean absolute correlation difference, or 0.0 if fewer
        than two usable numerical columns are available.
    """
    common_numeric = _numeric_columns(real_data)

    usable_columns = []

    for column in common_numeric:
        real = _finite_numeric_values(
            real_data[column], column, "reference data"
        )
        synthetic = _finite_numeric_values(
            synthetic_data[column], column, "synthetic data"
        )

        if (
            len(real) >= 2
            and len(synthetic) >= 2
            and np.std(real) > 0
            and np.std(synthetic) > 0
        ):
            usable_columns.append(column)

    if len(usable_columns) < 2:
        return 0.0

    real_frame = real_data[usable_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    synthetic_frame = synthetic_data[usable_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Infinite values were already rejected above.
    real_corr = real_frame.corr().fillna(0.0)
    synthetic_corr = synthetic_frame.corr().fillna(0.0)

    difference = np.abs(
        real_corr.to_numpy()
        - synthetic_corr.to_numpy()
    )

    # Exclude the diagonal, where correlations are always 1.
    off_diagonal = ~np.eye(
        len(usable_columns),
        dtype=bool,
    )

    return float(difference[off_diagonal].mean())


def calculate_categorical_tv(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> dict[str, float]:
    """Calculate total variation distance for categorical columns.

    TV = 0.5 * sum(abs(p_real(category) - p_synthetic(category))).

    The category support is the union of values observed in either
    dataset. Missing values are excluded from frequency estimates.

    Returns:
        Mapping of column name to total variation distance.
    """
    results = {}

    for column in _categorical_columns(real_data):
        real = real_data[column].dropna()
        synthetic = synthetic_data[column].dropna()

        if len(real) == 0 or len(synthetic) == 0:
            continue

        real_freq = real.value_counts(normalize=True)
        synthetic_freq = synthetic.value_counts(normalize=True)

        categories = real_freq.index.union(
            synthetic_freq.index
        )

        real_probs = real_freq.reindex(
            categories,
            fill_value=0.0,
        )

        synthetic_probs = synthetic_freq.reindex(
            categories,
            fill_value=0.0,
        )

        tv_distance = 0.5 * (
            real_probs - synthetic_probs
        ).abs().sum()

        results[column] = float(tv_distance)

    return results


def evaluate_synthetic_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate the common statistical evaluation metrics.

    Returns:
        A dictionary containing per-column metrics and aggregate
        summary metrics.

    Lower values indicate better statistical similarity for these
    metrics. Metrics have different scales and should be normalized
    before combining them into a model-selection score.
    """
    _validate_datasets(real_data, synthetic_data)

    ks_values = calculate_ks_statistic(
        real_data,
        synthetic_data,
    )

    wasserstein_values = calculate_wasserstein_distance(
        real_data,
        synthetic_data,
    )

    gaps = calculate_mean_std_gap(
        real_data,
        synthetic_data,
    )

    correlation_mae = calculate_correlation_mae(
        real_data,
        synthetic_data,
    )

    categorical_tv = calculate_categorical_tv(
        real_data,
        synthetic_data,
    )

    mean_gaps = [
        item["mean_gap"]
        for item in gaps.values()
    ]

    std_gaps = [
        item["std_gap"]
        for item in gaps.values()
    ]

    all_gaps = mean_gaps + std_gaps

    return {
        "KS Statistic": (
            float(np.mean(list(ks_values.values())))
            if ks_values else None
        ),
        "Wasserstein Distance": (
            float(np.mean(list(wasserstein_values.values())))
            if wasserstein_values else None
        ),
        "Mean/Std Gap": (
            float(np.mean(all_gaps))
            if all_gaps else None
        ),
        "Correlation MAE": correlation_mae,
        "Categorical TV": (
            float(np.mean(list(categorical_tv.values())))
            if categorical_tv else None
        ),
        "per_column": {
            "ks_statistic": ks_values,
            "wasserstein_distance": wasserstein_values,
            "mean_std_gap": gaps,
            "categorical_tv": categorical_tv,
        },
    }
