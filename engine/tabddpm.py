
"""TabDDPM synthetic tabular data generation using Synthcity."""

from __future__ import annotations

import random
from numbers import Integral
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from synthcity.plugins import Plugins


def _validate_reference_df(reference_df: pd.DataFrame) -> None:
    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError("reference_df must be a pandas DataFrame.")

    if reference_df.empty:
        raise ValueError("reference_df must not be empty.")

    if len(reference_df.columns) == 0:
        raise ValueError("reference_df must contain at least one column.")

    if reference_df.columns.has_duplicates:
        raise ValueError("reference_df must not contain duplicate column names.")

    if reference_df.isna().all(axis=0).any():
        raise ValueError(
            "reference_df contains an entirely missing column. "
            "Remove or impute that column before generation."
        )


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer.")


def _restore_output_dtypes(
    synthetic_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Restore compatible output dtypes without silently clipping values."""
    synthetic_df = synthetic_df.loc[:, reference_df.columns].copy()

    for column in reference_df.columns:
        original_dtype = reference_df[column].dtype

        try:
            if isinstance(original_dtype, pd.CategoricalDtype):
                categories = reference_df[column].cat.categories
                synthetic_df[column] = pd.Categorical(
                    synthetic_df[column],
                    categories=categories,
                    ordered=reference_df[column].cat.ordered,
                )
            elif pd.api.types.is_bool_dtype(original_dtype):
                values = synthetic_df[column]
                if values.isin([True, False, 0, 1]).all():
                    synthetic_df[column] = values.astype(original_dtype)
            elif pd.api.types.is_integer_dtype(original_dtype):
                values = pd.to_numeric(synthetic_df[column], errors="raise")
                if np.isfinite(values).all() and np.equal(
                    values, np.floor(values)
                ).all():
                    synthetic_df[column] = values.astype(original_dtype)
            elif pd.api.types.is_float_dtype(original_dtype):
                synthetic_df[column] = pd.to_numeric(
                    synthetic_df[column], errors="raise"
                ).astype(original_dtype)
            elif pd.api.types.is_datetime64_any_dtype(original_dtype):
                synthetic_df[column] = pd.to_datetime(
                    synthetic_df[column], errors="raise"
                )
            elif pd.api.types.is_object_dtype(original_dtype):
                synthetic_df[column] = synthetic_df[column].astype(object)
        except (TypeError, ValueError, OverflowError):
            # Keep generated values when a safe dtype conversion is not possible.
            pass

    return synthetic_df.reset_index(drop=True)


def generate_tabddpm(
    reference_df: pd.DataFrame,
    n_samples: int = 1000,
    seed: int = 42,
    n_iter: int = 1000,
    batch_size: int = 1024,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Fit Synthcity's TabDDPM plugin and generate synthetic rows.

    Args:
        reference_df: Source DataFrame used for training.
        n_samples: Number of synthetic rows to generate.
        seed: Random seed.
        n_iter: Number of training iterations.
        batch_size: Training batch size.
        device: Training device, normally "cpu" or "cuda".

    Returns:
        Synthetic DataFrame with the original column names and order.
    """
    _validate_reference_df(reference_df)
    _validate_positive_integer(n_samples, "n_samples")
    _validate_positive_integer(n_iter, "n_iter")
    _validate_positive_integer(batch_size, "batch_size")
    _validate_seed(seed)

    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'.")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but it is not available in this environment."
        )

    training_data = reference_df.copy(deep=True)

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    if device == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    plugin = Plugins().get(
        "ddpm",
        n_iter=int(n_iter),
        batch_size=int(batch_size),
        device=torch.device(device),
        random_state=int(seed),
    )

    plugin.fit(training_data)

    generated = plugin.generate(
        count=int(n_samples),
        random_state=int(seed),
    )

    if not hasattr(generated, "dataframe"):
        raise TypeError(
            "Synthcity DDPM generation did not return an object "
            "with a dataframe() method."
        )

    synthetic_df = generated.dataframe()

    if not isinstance(synthetic_df, pd.DataFrame):
        raise TypeError("Synthcity did not return a pandas DataFrame.")

    if len(synthetic_df) != n_samples:
        raise RuntimeError(
            f"Requested {n_samples} rows, but Synthcity returned "
            f"{len(synthetic_df)} rows."
        )

    missing_columns = set(reference_df.columns) - set(synthetic_df.columns)
    extra_columns = set(synthetic_df.columns) - set(reference_df.columns)

    if missing_columns or extra_columns:
        raise ValueError(
            "Generated columns do not match the training DataFrame. "
            f"Missing: {list(missing_columns)}; "
            f"extra: {list(extra_columns)}."
        )

    return _restore_output_dtypes(synthetic_df, reference_df)


def generate_tabddpm_engine(
    profile: Optional[Any] = None,
    n_samples: int = 1000,
    seed: int = 42,
    reference_df: Optional[pd.DataFrame] = None,
    n_iter: int = 1000,
    batch_size: int = 1024,
    device: str = "cpu",
) -> pd.DataFrame:
    """Engine-compatible wrapper for TabDDPM."""
    if reference_df is None and isinstance(profile, pd.DataFrame):
        reference_df = profile

    if reference_df is None:
        raise ValueError(
            "Provide reference_df as a pandas DataFrame. "
            "A static dataset path is not used."
        )

    return generate_tabddpm(
        reference_df=reference_df,
        n_samples=n_samples,
        seed=seed,
        n_iter=n_iter,
        batch_size=batch_size,
        device=device,
    )
