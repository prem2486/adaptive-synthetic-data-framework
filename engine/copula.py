
"""Gaussian Copula synthetic data generator.

Uses NumPy, pandas and SciPy.
The seed dataset is supplied dynamically as a DataFrame.

Numerical columns:
    Model dependence using rank-based Gaussian transformations.

Categorical, Boolean and datetime columns:
    Sample from their observed marginal distributions independently.

Important:
    This baseline preserves approximate numerical marginals and
    numerical dependence. It does not preserve categorical dependence
    or all mixed-type relationships.
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
from scipy.stats import norm, rankdata


def _validate_seed_data(data: pd.DataFrame) -> None:
    """Validate the dynamically supplied seed dataset."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("reference_df must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Seed dataset contains no rows.")

    if len(data.columns) == 0:
        raise ValueError("Seed dataset contains no columns.")

    if data.columns.has_duplicates:
        raise ValueError("Duplicate column names are not supported.")

    if not all(isinstance(column, str) for column in data.columns):
        raise ValueError("All column names must be strings.")

    if data.isna().any().any():
        raise ValueError(
            "Seed data contains missing values. "
            "Run engine.preprocessing.prepare_seed_dataset() first."
        )


def _numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric columns, excluding Boolean columns."""
    return [
        column
        for column in data.columns
        if is_numeric_dtype(data[column].dtype)
        and not is_bool_dtype(data[column].dtype)
    ]


def _validate_numeric_values(data: pd.DataFrame, columns: list[str]) -> None:
    """Reject infinite or non-finite numeric values."""
    for column in columns:
        values = data[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Numeric column '{column}' contains infinite "
                "or non-finite values."
            )


def _rank_to_normal(values: np.ndarray) -> np.ndarray:
    """Transform observations into approximately standard normal scores."""
    n = len(values)

    ranks = rankdata(values, method="average")
    probabilities = (ranks - 0.5) / n

    return norm.ppf(probabilities)


def _correlation_matrix(
    normal_scores: np.ndarray,
) -> np.ndarray:
    """Calculate and regularize a correlation matrix."""
    correlation = np.corrcoef(
        normal_scores,
        rowvar=False,
    )

    # A single feature produces a scalar correlation result.
    correlation = np.atleast_2d(correlation)

    # Protect against numerical errors and singular matrices.
    correlation = np.nan_to_num(
        correlation,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1e-8)

    regularized = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
    )

    # Normalize back to a correlation matrix.
    scales = np.sqrt(
        np.maximum(np.diag(regularized), 1e-12)
    )

    regularized = regularized / np.outer(scales, scales)
    regularized = (regularized + regularized.T) / 2.0
    np.fill_diagonal(regularized, 1.0)

    return regularized


def _sample_numeric_columns(
    reference_df: pd.DataFrame,
    columns: list[str],
    n_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate numerical columns using a Gaussian Copula."""
    output: dict[str, np.ndarray] = {}

    varying_columns = []
    constant_columns = []

    for column in columns:
        values = reference_df[column].to_numpy(dtype=float)

        if np.all(values == values[0]):
            constant_columns.append(column)
        else:
            varying_columns.append(column)

    # Constant columns remain constant.
    for column in constant_columns:
        output[column] = np.full(
            n_samples,
            reference_df[column].iloc[0],
        )

    if not varying_columns:
        return output

    # Learn the dependence structure from the reference data.
    score_columns = [
        _rank_to_normal(
            reference_df[column].to_numpy(dtype=float)
        )
        for column in varying_columns
    ]

    normal_scores = np.column_stack(score_columns)
    correlation = _correlation_matrix(normal_scores)

    # Eigenvalue decomposition allows sampling even when the
    # estimated correlation matrix is singular or nearly singular.
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1e-8)

    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    independent_normal = rng.standard_normal(
        size=(n_samples, len(varying_columns))
    )

    correlated_normal = independent_normal @ transform.T
    uniform_samples = norm.cdf(correlated_normal)

    # Invert each empirical marginal distribution.
    for index, column in enumerate(varying_columns):
        original_values = reference_df[column].to_numpy(
            dtype=float
        )

        generated_values = np.quantile(
            original_values,
            uniform_samples[:, index],
            method="linear",
        )

        if is_integer_dtype(reference_df[column].dtype):
            generated_values = np.rint(generated_values)

        output[column] = generated_values

    return output


def generate_gaussian_copula(
    profile=None,
    n_samples: int = 1000,
    seed: int = 42,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate synthetic tabular data using a Gaussian Copula.

    Args:
        profile:
            Retained as an optional compatibility argument for existing
            orchestration code. A profile alone is insufficient for
            generation because the original observations are required.
        n_samples:
            Number of synthetic rows to generate.
        seed:
            Random seed for reproducible generation.
        reference_df:
            Prepared seed DataFrame supplied dynamically by the engine.
            If omitted, ``profile`` may itself be a DataFrame.

    Returns:
        Synthetic DataFrame with the original column names and order.

    Notes:
        - Numerical dependencies are modelled jointly.
        - Non-numeric columns are sampled independently from their
          observed marginal distributions.
        - The function does not modify the reference dataset.
        - Domain constraints and privacy are not guaranteed.
    """
    if (
        isinstance(n_samples, bool)
        or not isinstance(n_samples, (int, np.integer))
        or n_samples <= 0
    ):
        raise ValueError("n_samples must be a positive integer.")

    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer.")

    if reference_df is None and isinstance(profile, pd.DataFrame):
        reference_df = profile

    if reference_df is None:
        raise ValueError(
            "A reference DataFrame is required. Pass "
            "reference_df=prepared_data. A profile dictionary alone "
            "does not contain the observations needed for generation."
        )

    _validate_seed_data(reference_df)

    reference = reference_df.copy(deep=True)
    numeric_columns = _numeric_columns(reference)

    _validate_numeric_values(reference, numeric_columns)

    rng = np.random.default_rng(int(seed))
    generated_columns: dict[str, np.ndarray] = {}

    # Generate numerical columns jointly.
    if numeric_columns:
        generated_columns.update(
            _sample_numeric_columns(
                reference_df=reference,
                columns=numeric_columns,
                n_samples=int(n_samples),
                rng=rng,
            )
        )

    # Sample non-numeric columns independently from their
    # empirical distributions.
    for column in reference.columns:
        if column in numeric_columns:
            continue

        values = reference[column].to_numpy()

        sampled = rng.choice(
            values,
            size=int(n_samples),
            replace=True,
        )

        generated_columns[column] = sampled

    # Preserve the original column order.
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
                # Preserve generated values if the original dtype
                # cannot represent them without loss.
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
