"""Shared preprocessing utilities for synthetic tabular data generation.

Datasets are supplied dynamically as pandas DataFrames.
This module does not load any static dataset.

Responsibilities:
- Validate input data.
- Detect column types.
- Learn missing-value imputation rules during fit().
- Reuse learned rules during transform().
- Preserve schema metadata for output restoration.

Encoding and scaling are left to individual generator adapters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


class SeedPreprocessor:
    """Fit reusable preprocessing rules on a seed dataset."""

    def __init__(
        self,
        *,
        missing_values: str = "impute",
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        categorical_fill_value: Any = "__MISSING__",
        all_missing: str = "constant",
    ) -> None:
        if missing_values not in {"impute", "raise"}:
            raise ValueError(
                "missing_values must be 'impute' or 'raise'."
            )
        if numeric_strategy not in {"median", "mean"}:
            raise ValueError(
                "numeric_strategy must be 'median' or 'mean'."
            )
        if categorical_strategy not in {"most_frequent", "constant"}:
            raise ValueError(
                "categorical_strategy must be "
                "'most_frequent' or 'constant'."
            )
        if all_missing not in {"raise", "constant"}:
            raise ValueError(
                "all_missing must be 'raise' or 'constant'."
            )

        self.missing_values = missing_values
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value
        self.all_missing = all_missing

        self.is_fitted_ = False
        self.schema_: dict[str, Any] | None = None

    @staticmethod
    def _validate_dataframe(data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("The dataset contains no rows.")
        if len(data.columns) == 0:
            raise ValueError("The dataset contains no columns.")
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
            raise ValueError("All column names must be strings.")
        if any(not column.strip() for column in data.columns):
            raise ValueError("Column names cannot be empty.")

    @staticmethod
    def _detect_column_type(series: pd.Series) -> str:
        if is_bool_dtype(series.dtype):
            return "boolean"
        if is_datetime64_any_dtype(series.dtype):
            return "datetime"
        if isinstance(series.dtype, pd.CategoricalDtype):
            return "categorical"
        if is_numeric_dtype(series.dtype):
            return "numeric"
        return "categorical"

    def _learn_fill_value(self, series, column_type):
        observed = series.dropna()

        if observed.empty:
            if self.all_missing == "raise":
                raise ValueError(
                    f"Column '{series.name}' contains only missing values."
                )
            if column_type == "numeric":
                return 0
            if column_type == "boolean":
                return False
            if column_type == "datetime":
                timezone = getattr(series.dtype, "tz", None)
                return pd.Timestamp("1970-01-01", tz=timezone)
            return self.categorical_fill_value

        if column_type == "numeric":
            if self.numeric_strategy == "median":
                value = observed.median()
            else:
                value = observed.mean()
            if pd.isna(value):
                raise ValueError(
                    f"Cannot calculate a numeric fill value "
                    f"for '{series.name}'."
                )
            return value

        if self.categorical_strategy == "constant":
            return self.categorical_fill_value

        modes = observed.mode(dropna=True)
        if modes.empty:
            raise ValueError(
                f"Cannot calculate a fill value for '{series.name}'."
            )
        return modes.iloc[0]

    def fit(self, data, *, target_column=None):
        self._validate_dataframe(data)

        if (
            target_column is not None
            and target_column not in data.columns
        ):
            raise ValueError(
                f"Target column '{target_column}' "
                "was not found in the dataset."
            )

        column_types = {
            column: self._detect_column_type(data[column])
            for column in data.columns
        }

        missing_columns = [
            column
            for column in data.columns
            if data[column].isna().any()
        ]
        if missing_columns and self.missing_values == "raise":
            counts = {
                column: int(data[column].isna().sum())
                for column in missing_columns
            }
            raise ValueError(f"Missing values found: {counts}")

        fill_values = {}
        if self.missing_values == "impute":
            for column in data.columns:
                if data[column].isna().any():
                    fill_values[column] = self._learn_fill_value(
                        data[column], column_types[column]
                    )

        categorical_values = {}
        for column in data.columns:
            if column_types[column] == "categorical":
                values = data[column].dropna().unique().tolist()
                if not values and self.missing_values == "impute":
                    values = [self.categorical_fill_value]
                categorical_values[column] = values
            else:
                categorical_values[column] = None

        self.schema_ = {
            "version": 1,
            "columns": list(data.columns),
            "dtypes": {
                column: str(data[column].dtype)
                for column in data.columns
            },
            "column_types": column_types,
            "categorical_values": categorical_values,
            "original_missing_counts": {
                column: int(data[column].isna().sum())
                for column in data.columns
            },
            "imputation_values": fill_values,
            "target_column": target_column,
            "missing_values_policy": self.missing_values,
            "numeric_strategy": self.numeric_strategy,
            "categorical_strategy": self.categorical_strategy,
            "all_missing": self.all_missing,
        }

        self.is_fitted_ = True
        return self

    def transform(self, data):
        if not self.is_fitted_ or self.schema_ is None:
            raise RuntimeError("Call fit() before transform().")

        self._validate_dataframe(data)

        expected = self.schema_["columns"]
        missing = [c for c in expected if c not in data.columns]
        extra = [c for c in data.columns if c not in expected]
        if missing or extra:
            raise ValueError(
                "Dataset columns do not match fitted schema. "
                f"Missing: {missing}; extra: {extra}."
            )

        result = data.loc[:, expected].copy(deep=True)

        for column, fill_value in (
            self.schema_["imputation_values"].items()
        ):
            if isinstance(result[column].dtype, pd.CategoricalDtype):
                if fill_value not in result[column].cat.categories:
                    result[column] = result[column].cat.add_categories(
                        [fill_value]
                    )
            missing_mask = result[column].isna()
            if missing_mask.any():
                result.loc[missing_mask, column] = fill_value

        if (
            self.missing_values == "raise"
            and result.isna().any().any()
        ):
            raise ValueError(
                "Missing values found and missing_values='raise'."
            )
        return result.loc[:, expected]

    def fit_transform(self, data, *, target_column=None):
        self.fit(data, target_column=target_column)
        return self.transform(data)

    def get_schema(self):
        if not self.is_fitted_ or self.schema_ is None:
            raise RuntimeError(
                "Call fit() before requesting schema metadata."
            )
        return dict(self.schema_)


def prepare_seed_dataset(
    data,
    target_column=None,
    config=None,
):
    options = dict(config or {})
    allowed = {
        "missing_values",
        "numeric_strategy",
        "categorical_strategy",
        "categorical_fill_value",
        "all_missing",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(
            f"Unknown preprocessing configuration keys: "
            f"{sorted(unknown)}"
        )

    processor = SeedPreprocessor(**options)
    prepared = processor.fit_transform(
        data, target_column=target_column
    )

    for column in prepared.columns:
        series = prepared[column]
        if (
            pd.api.types.is_string_dtype(series.dtype)
            and not isinstance(series.dtype, pd.CategoricalDtype)
        ):
            prepared[column] = series.astype(object)

    return prepared, processor.get_schema()


def restore_original_schema(synthetic_data, schema):
    if not isinstance(synthetic_data, pd.DataFrame):
        raise TypeError("Synthetic data must be a pandas DataFrame.")

    columns = list(schema["columns"])
    missing_columns = [
        column for column in columns
        if column not in synthetic_data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Synthetic data is missing required columns: "
            f"{missing_columns}"
        )

    restored = synthetic_data.loc[:, columns].copy()

    column_types = schema["column_types"]
    dtypes = schema["dtypes"]
    categorical_values = schema.get("categorical_values", {})

    for column in columns:
        column_type = column_types[column]
        original_dtype = dtypes[column]

        if column_type == "numeric":
            values = pd.to_numeric(restored[column], errors="coerce")
            if values.isna().any():
                raise ValueError(
                    f"Column '{column}' contains invalid "
                    "or missing numeric values."
                )
            if original_dtype.startswith(("int", "uint")):
                values = values.round()
            try:
                restored[column] = values.astype(original_dtype)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Could not restore dtype "
                    f"'{original_dtype}' for '{column}'."
                ) from exc

        elif column_type == "boolean":
            mapping = {
                True: True, False: False,
                1: True, 0: False,
                "True": True, "False": False,
                "true": True, "false": False,
            }
            values = restored[column].map(mapping)
            if values.isna().any():
                raise ValueError(
                    f"Column '{column}' contains values "
                    "that cannot be converted to Boolean."
                )
            restored[column] = values.astype(original_dtype)

        elif column_type == "datetime":
            values = pd.to_datetime(restored[column], errors="coerce")
            if values.isna().any():
                raise ValueError(
                    f"Column '{column}' contains invalid datetime values."
                )
            try:
                restored[column] = values.astype(original_dtype)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Could not restore datetime dtype "
                    f"'{original_dtype}' for '{column}'."
                ) from exc

        elif original_dtype == "category":
            allowed_values = categorical_values.get(column) or []
            unknown_mask = ~restored[column].isin(allowed_values)
            if unknown_mask.any():
                if not allowed_values:
                    raise ValueError(
                        f"No original categories available for '{column}'."
                    )
                counts = pd.Series(allowed_values).value_counts()
                fallback = counts.index[0]
                restored.loc[unknown_mask, column] = fallback
            restored[column] = pd.Categorical(
                restored[column],
                categories=allowed_values,
            )

    return restored
