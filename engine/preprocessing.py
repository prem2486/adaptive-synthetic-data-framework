
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
        all_missing: str = "raise",
    ) -> None:
        if missing_values not in {"impute", "raise"}:
            raise ValueError(
                "missing_values must be 'impute' or 'raise'."
            )

        if numeric_strategy not in {"median", "mean"}:
            raise ValueError(
                "numeric_strategy must be 'median' or 'mean'."
            )

        if categorical_strategy not in {
            "most_frequent",
            "constant",
        }:
            raise ValueError(
                "categorical_strategy must be "
                "'most_frequent' or 'constant'."
            )

        if all_missing != "raise":
            raise ValueError(
                "all_missing currently supports only 'raise'."
            )

        self.missing_values = missing_values
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value

        self.is_fitted_ = False
        self.schema_: dict[str, Any] | None = None

    @staticmethod
    def _validate_dataframe(data: pd.DataFrame) -> None:
        """Validate a dynamically supplied DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "The dataset contains no rows."
            )

        if len(data.columns) == 0:
            raise ValueError(
                "The dataset contains no columns."
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

        if any(
            not column.strip()
            for column in data.columns
        ):
            raise ValueError(
                "Column names cannot be empty."
            )

    @staticmethod
    def _detect_column_type(
        series: pd.Series,
    ) -> str:
        """Identify the general type of a column."""
        if is_bool_dtype(series.dtype):
            return "boolean"

        if is_datetime64_any_dtype(series.dtype):
            return "datetime"

        if isinstance(
            series.dtype,
            pd.CategoricalDtype,
        ):
            return "categorical"

        if is_numeric_dtype(series.dtype):
            return "numeric"

        return "categorical"

    def _learn_fill_value(
        self,
        series: pd.Series,
        column_type: str,
    ) -> Any:
        """Learn an imputation value from observed data."""
        observed = series.dropna()

        if observed.empty:
            raise ValueError(
                f"Column '{series.name}' contains only missing "
                "values. Remove it or supply valid values."
            )

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
                f"Cannot calculate a fill value "
                f"for '{series.name}'."
            )

        return modes.iloc[0]

    def fit(
        self,
        data: pd.DataFrame,
        *,
        target_column: str | None = None,
    ) -> "SeedPreprocessor":
        """Learn schema and imputation values from seed data."""
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

        if (
            missing_columns
            and self.missing_values == "raise"
        ):
            counts = {
                column: int(data[column].isna().sum())
                for column in missing_columns
            }

            raise ValueError(
                f"Missing values found: {counts}"
            )

        fill_values: dict[str, Any] = {}

        if self.missing_values == "impute":
            for column in data.columns:
                if data[column].isna().any():
                    fill_values[column] = (
                        self._learn_fill_value(
                            data[column],
                            column_types[column],
                        )
                    )

        categorical_values = {}

        for column in data.columns:
            if column_types[column] == "categorical":
                categorical_values[column] = (
                    data[column]
                    .dropna()
                    .unique()
                    .tolist()
                )
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
        }

        self.is_fitted_ = True
        return self

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply fitted preprocessing rules without changing input."""
        if not self.is_fitted_ or self.schema_ is None:
            raise RuntimeError(
                "Call fit() before transform()."
            )

        self._validate_dataframe(data)

        expected = self.schema_["columns"]

        missing = [
            column
            for column in expected
            if column not in data.columns
        ]

        extra = [
            column
            for column in data.columns
            if column not in expected
        ]

        if missing or extra:
            raise ValueError(
                "Dataset columns do not match fitted schema. "
                f"Missing: {missing}; extra: {extra}."
            )

        # Accept reordered columns, then restore fitted order.
        result = data.loc[:, expected].copy(deep=True)

        for column, fill_value in (
            self.schema_["imputation_values"].items()
        ):
            result[column] = result[column].fillna(
                fill_value
            )

        if (
            self.missing_values == "raise"
            and result.isna().any().any()
        ):
            raise ValueError(
                "Missing values found and "
                "missing_values='raise'."
            )

        return result.loc[:, expected]

    def fit_transform(
        self,
        data: pd.DataFrame,
        *,
        target_column: str | None = None,
    ) -> pd.DataFrame:
        """Fit on data and return the transformed DataFrame."""
        self.fit(
            data,
            target_column=target_column,
        )

        return self.transform(data)

    def get_schema(self) -> dict[str, Any]:
        """Return the fitted schema metadata."""
        if not self.is_fitted_ or self.schema_ is None:
            raise RuntimeError(
                "Call fit() before requesting schema metadata."
            )

        return dict(self.schema_)


def prepare_seed_dataset(
    data: pd.DataFrame,
    target_column: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare a dynamically uploaded seed dataset.

    Returns:
        prepared_data: A copy with missing values handled.
        schema: Metadata used by generators and restoration.
    """
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
        data,
        target_column=target_column,
    )

    return prepared, processor.get_schema()


def restore_original_schema(
    synthetic_data: pd.DataFrame,
    schema: Mapping[str, Any],
) -> pd.DataFrame:
    """Restore generated data to the original column schema.

    Invalid values raise errors instead of being silently accepted.
    Domain-specific constraints must be checked separately.
    """
    if not isinstance(
        synthetic_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Synthetic data must be a pandas DataFrame."
        )

    columns = list(schema["columns"])

    missing_columns = [
        column
        for column in columns
        if column not in synthetic_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Synthetic data is missing required columns: "
            f"{missing_columns}"
        )

    restored = synthetic_data.loc[
        :, columns
    ].copy()

    column_types = schema["column_types"]
    dtypes = schema["dtypes"]
    categorical_values = schema.get(
        "categorical_values",
        {},
    )

    for column in columns:
        column_type = column_types[column]
        original_dtype = dtypes[column]

        if column_type == "numeric":
            values = pd.to_numeric(
                restored[column],
                errors="coerce",
            )

            if values.isna().any():
                raise ValueError(
                    f"Column '{column}' contains invalid "
                    "or missing numeric values."
                )

            if original_dtype.startswith(("int", "uint")):
                values = values.round()

            try:
                restored[column] = values.astype(
                    original_dtype
                )
            except (
                TypeError,
                ValueError,
                OverflowError,
            ) as exc:
                raise ValueError(
                    f"Could not restore dtype "
                    f"'{original_dtype}' for '{column}'."
                ) from exc

        elif column_type == "boolean":
            mapping = {
                True: True,
                False: False,
                1: True,
                0: False,
                "True": True,
                "False": False,
                "true": True,
                "false": False,
            }

            values = restored[column].map(mapping)

            if values.isna().any():
                raise ValueError(
                    f"Column '{column}' contains values "
                    "that cannot be converted to Boolean."
                )

            restored[column] = values.astype(
                original_dtype
            )

        elif column_type == "datetime":
            values = pd.to_datetime(
                restored[column],
                errors="coerce",
            )

            if values.isna().any():
                raise ValueError(
                    f"Column '{column}' contains invalid "
                    "datetime values."
                )

            try:
                restored[column] = values.astype(
                    original_dtype
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Could not restore datetime dtype "
                    f"'{original_dtype}' for '{column}'."
                ) from exc

        elif original_dtype == "category":
            allowed_values = (
                categorical_values.get(column) or []
            )

            unknown_mask = ~restored[column].isin(
                allowed_values
            )

            if unknown_mask.any():
                if not allowed_values:
                    raise ValueError(
                        f"No original categories available "
                        f"for '{column}'."
                    )

                counts = pd.Series(
                    allowed_values
                ).value_counts()

                fallback = counts.index[0]

                restored.loc[
                    unknown_mask,
                    column,
                ] = fallback

            restored[column] = pd.Categorical(
                restored[column],
                categories=allowed_values,
            )

    return restored
