
"""Dataset profiling utilities for synthetic tabular data generation.

Profiles dynamically supplied pandas DataFrames.
Does not load static datasets, modify input data, or generate plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


def _detect_column_type(series: pd.Series) -> str:
    """Identify the general type of a dataset column."""
    if is_bool_dtype(series.dtype):
        return "boolean"

    if is_datetime64_any_dtype(series.dtype):
        return "datetime"

    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"

    if is_numeric_dtype(series.dtype):
        return "numeric"

    return "categorical"


def _safe_value(value: Any) -> Any:
    """Convert common NumPy scalar values to Python scalars."""
    if isinstance(value, np.generic):
        value = value.item()

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)

    return str(value)


def _profile_numeric(series: pd.Series) -> dict[str, Any]:
    """Calculate descriptive statistics for numeric columns."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = numeric.dropna()

    finite_mask = np.isfinite(
        non_missing.to_numpy(dtype=float)
    )
    finite = non_missing.iloc[
        np.flatnonzero(finite_mask)
    ]

    result = {
        "mean": None,
        "std": None,
        "min": None,
        "q25": None,
        "median": None,
        "q75": None,
        "max": None,
        "infinite_count": int(
            np.isinf(
                non_missing.to_numpy(dtype=float)
            ).sum()
        ),
    }

    if finite.empty:
        return result

    result.update({
        "mean": _safe_value(finite.mean()),
        "std": _safe_value(finite.std()),
        "min": _safe_value(finite.min()),
        "q25": _safe_value(finite.quantile(0.25)),
        "median": _safe_value(finite.median()),
        "q75": _safe_value(finite.quantile(0.75)),
        "max": _safe_value(finite.max()),
    })

    return result


def _profile_categorical(
    series: pd.Series,
    top_n: int,
) -> list[dict[str, Any]]:
    """Return the most frequent non-missing categorical values."""
    frequencies = series.value_counts(
        dropna=True
    ).head(top_n)

    return [
        {
            "value": _safe_value(value),
            "count": int(count),
            "percentage": round(
                float(count) / max(len(series), 1) * 100,
                2,
            ),
        }
        for value, count in frequencies.items()
    ]


def profile_raw_dataset(
    data: pd.DataFrame,
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    """Generate a structured profile of a dynamically supplied dataset.

    Args:
        data: Seed dataset received from Streamlit or another caller.
        top_n: Number of most frequent categorical values to return.

    Returns:
        A dictionary containing dataset summary, column profiles,
        and data-quality warnings.

    Notes:
        - The original DataFrame is never modified.
        - Missing values are reported, not imputed.
        - Numeric and categorical summaries are descriptive only.
        - Datetime columns must already have a datetime dtype to be
          profiled as datetime columns.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Cannot profile a dataset with no rows."
        )

    if len(data.columns) == 0:
        raise ValueError(
            "Cannot profile a dataset with no columns."
        )

    if top_n < 1:
        raise ValueError(
            "top_n must be greater than or equal to 1."
        )

    if data.columns.has_duplicates:
        duplicates = data.columns[
            data.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate column names are not supported: {duplicates}."
        )

    if not all(
        isinstance(column, str)
        for column in data.columns
    ):
        raise ValueError(
            "All column names must be strings."
        )

    row_count, column_count = data.shape
    missing_per_column = data.isna().sum()
    missing_cells = int(missing_per_column.sum())
    total_cells = row_count * column_count

    type_counts = {
        "numeric": 0,
        "categorical": 0,
        "boolean": 0,
        "datetime": 0,
    }

    column_profiles: dict[str, Any] = {}
    warnings: list[str] = []

    for column in data.columns:
        series = data[column]
        column_type = _detect_column_type(series)

        type_counts[column_type] += 1

        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        unique_percentage = round(
            unique_count / max(row_count - missing_count, 1)
            * 100,
            2,
        ) if row_count > missing_count else 0.0

        profile: dict[str, Any] = {
            "dtype": str(series.dtype),
            "detected_type": column_type,
            "row_count": row_count,
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_count / row_count * 100,
                2,
            ),
            "unique_count": unique_count,
            "unique_percentage": unique_percentage,
            "is_constant": unique_count <= 1,
            "sample_values": [
                _safe_value(value)
                for value in series.dropna().head(5).tolist()
            ],
        }

        if missing_count > 0:
            warnings.append(
                f"Column '{column}' contains "
                f"{missing_count} missing value(s)."
            )

        if unique_count == 0:
            warnings.append(
                f"Column '{column}' contains no observed values."
            )
        elif unique_count == 1:
            warnings.append(
                f"Column '{column}' is constant."
            )

        if column_type == "numeric":
            profile["statistics"] = _profile_numeric(series)

            numeric_values = pd.to_numeric(
                series,
                errors="coerce",
            ).dropna()

            if len(numeric_values) > 0:
                if np.isinf(
                    numeric_values.to_numpy(dtype=float)
                ).any():
                    warnings.append(
                        f"Column '{column}' contains infinite values."
                    )

        elif column_type == "categorical":
            profile["top_categories"] = _profile_categorical(
                series,
                top_n,
            )

        elif column_type == "datetime":
            valid_dates = series.dropna()

            profile["statistics"] = {
                "min": (
                    _safe_value(valid_dates.min())
                    if not valid_dates.empty else None
                ),
                "max": (
                    _safe_value(valid_dates.max())
                    if not valid_dates.empty else None
                ),
            }

        column_profiles[column] = profile

    duplicate_rows = int(data.duplicated().sum())

    if duplicate_rows > 0:
        warnings.append(
            f"Dataset contains {duplicate_rows} duplicate row(s)."
        )

    summary = {
        "row_count": int(row_count),
        "column_count": int(column_count),
        "total_cells": int(total_cells),
        "missing_cells": missing_cells,
        "missing_percentage": round(
            missing_cells / total_cells * 100,
            2,
        ),
        "duplicate_rows": duplicate_rows,
        "memory_usage_bytes": int(
            data.memory_usage(
                index=True,
                deep=True,
            ).sum()
        ),
        "column_type_counts": type_counts,
    }

    return {
        "dataset_summary": summary,
        "columns": column_profiles,
        "warnings": warnings,
    }
