
"""Gaussian Mixture Model (GMM) synthetic data generator.

Accepts dynamically supplied seed data as a pandas DataFrame.

Numerical columns:
    Standardize the features, fit a Gaussian Mixture Model,
    generate samples, and reverse the scaling.

Categorical, Boolean and datetime columns:
    Sample independently from their observed marginal distributions.

Note:
    This baseline does not preserve categorical-numerical dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _validate_seed_data(data: pd.DataFrame) -> None:
    """Validate the seed dataset."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("reference_df must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Seed dataset contains no rows.")

    if len(data.columns) == 0:
        raise ValueError("Seed dataset contains no columns.")

    if data.columns.has_duplicates:
        raise ValueError("Duplicate column names are not supported.")

    if not all(isinstance(c, str) for c in data.columns):
        raise ValueError("All column names must be strings.")

    if data.isna().any().any():
        raise ValueError(
            "Seed data contains missing values. "
            "Preprocess it before GMM generation."
        )


def _get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric columns, excluding Boolean columns."""
    return [
        column
        for column in data.columns
        if is_numeric_dtype(data[column].dtype)
        and not is_bool_dtype(data[column].dtype)
    ]


def _validate_numeric_data(
    data: pd.DataFrame,
    columns: list[str],
) -> None:
    """Reject infinite or non-finite numeric values."""
    for column in columns:
        values = data[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Numeric column '{column}' contains "
                "infinite or non-finite values."
            )


def _validate_positive_integer(
    value: int,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )


def _sample_numeric_columns(
    reference_df: pd.DataFrame,
    columns: list[str],
    n_samples: int,
    rng: np.random.Generator,
    n_components: int,
    reg_covar: float,
) -> dict[str, np.ndarray]:
    """Generate numeric columns with a Gaussian mixture."""
    generated: dict[str, np.ndarray] = {}

    varying_columns = []
    constant_columns = []

    for column in columns:
        values = reference_df[column].to_numpy(dtype=float)

        if np.all(values == values[0]):
            constant_columns.append(column)
        else:
            varying_columns.append(column)

    # Preserve columns that have no variation.
    for column in constant_columns:
        generated[column] = np.full(
            n_samples,
            reference_df[column].iloc[0],
        )

    if not varying_columns:
        return generated

    if len(reference_df) < 2:
        raise ValueError(
            "GMM requires at least two seed rows for "
            "non-constant numerical data."
        )

    numeric_data = reference_df[
        varying_columns
    ].to_numpy(dtype=float)

    # Scaling improves numerical conditioning when columns
    # have substantially different units.
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Do not request more mixture components than observations.
    effective_components = min(
        n_components,
        len(reference_df),
    )

    model = GaussianMixture(
        n_components=effective_components,
        covariance_type="full",
        reg_covar=reg_covar,
        random_state=int(rng.integers(0, 2**31 - 1)),
        n_init=1,
    )

    model.fit(scaled_data)

    synthetic_scaled, _ = model.sample(n_samples)

    synthetic_numeric = scaler.inverse_transform(
        synthetic_scaled
    )

    for index, column in enumerate(varying_columns):
        values = synthetic_numeric[:, index]

        # Round integer columns. Range and domain constraints
        # should be validated separately.
        if is_integer_dtype(reference_df[column].dtype):
            values = np.rint(values)

        generated[column] = values

    return generated


def generate_gmm(
    reference_df: pd.DataFrame,
    n_samples: int = 1000,
    seed: int = 42,
    n_components: int = 5,
    reg_covar: float = 1e-6,
) -> pd.DataFrame:
    """Generate synthetic tabular data using a GMM.

    Args:
        reference_df: Prepared seed DataFrame supplied dynamically.
        n_samples: Number of synthetic rows requested.
        seed: Random seed for reproducibility.
        n_components: Maximum number of mixture components.
        reg_covar: Non-negative covariance regularization.

    Returns:
        Synthetic DataFrame with the original column names and order.
    """
    _validate_positive_integer(n_samples, "n_samples")
    _validate_positive_integer(n_components, "n_components")

    if (
        isinstance(reg_covar, bool)
        or not isinstance(reg_covar, (int, float, np.integer, np.floating))
        or not np.isfinite(reg_covar)
        or reg_covar < 0
    ):
        raise ValueError(
            "reg_covar must be a finite, non-negative number."
        )

    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError(
            "seed must be a non-negative integer."
        )

    _validate_seed_data(reference_df)

    reference = reference_df.copy(deep=True)
    numeric_columns = _get_numeric_columns(reference)

    _validate_numeric_data(reference, numeric_columns)

    rng = np.random.default_rng(int(seed))
    generated_columns: dict[str, np.ndarray] = {}

    if numeric_columns:
        generated_columns.update(
            _sample_numeric_columns(
                reference_df=reference,
                columns=numeric_columns,
                n_samples=int(n_samples),
                rng=rng,
                n_components=int(n_components),
                reg_covar=float(reg_covar),
            )
        )

    # Sample non-numeric columns independently from their
    # empirical marginal distributions.
    for column in reference.columns:
        if column in numeric_columns:
            continue

        values = reference[column].to_numpy()

        generated_columns[column] = rng.choice(
            values,
            size=int(n_samples),
            replace=True,
        )

    synthetic_df = pd.DataFrame(
        {
            column: generated_columns[column]
            for column in reference.columns
        }
    )

    # Restore compatible data types.
    for column in reference.columns:
        original_dtype = reference[column].dtype

        if column in numeric_columns:
            try:
                synthetic_df[column] = synthetic_df[column].astype(
                    original_dtype
                )
            except (TypeError, ValueError, OverflowError):
                # Keep the generated numerical values if the original
                # dtype cannot represent them safely.
                pass

        elif isinstance(original_dtype, pd.CategoricalDtype):
            synthetic_df[column] = pd.Categorical(
                synthetic_df[column],
                categories=original_dtype.categories,
                ordered=original_dtype.ordered,
            )

        elif is_bool_dtype(original_dtype):
            synthetic_df[column] = synthetic_df[column].astype(
                original_dtype
            )

        elif is_datetime64_any_dtype(original_dtype):
            synthetic_df[column] = pd.to_datetime(
                synthetic_df[column]
            )

    return synthetic_df.loc[:, reference.columns]


def generate_gmm_engine(
    profile=None,
    n_samples: int = 1000,
    seed: int = 42,
    reference_df: pd.DataFrame | None = None,
    n_components: int = 5,
    reg_covar: float = 1e-6,
) -> pd.DataFrame:
    """Compatibility wrapper for the shared generation engine.

    A reference DataFrame is required. A profile dictionary alone
    cannot be used to fit a GMM.
    """
    if reference_df is None and isinstance(profile, pd.DataFrame):
        reference_df = profile

    if reference_df is None:
        raise ValueError(
            "A reference DataFrame is required. Pass "
            "reference_df=prepared_data."
        )

    return generate_gmm(
        reference_df=reference_df,
        n_samples=n_samples,
        seed=seed,
        n_components=n_components,
        reg_covar=reg_covar,
    )
