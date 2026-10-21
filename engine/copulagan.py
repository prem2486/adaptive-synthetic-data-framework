
"""CopulaGAN synthetic data generation using SDV."""

import numbers
import random
import warnings

import numpy as np
import pandas as pd


def _validate_seed_data(reference_df):
    """Validate the reference DataFrame."""
    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError("reference_df must be a pandas DataFrame.")

    if reference_df.empty:
        raise ValueError("reference_df must contain at least one row.")

    if len(reference_df.columns) == 0:
        raise ValueError("reference_df must contain at least one column.")

    if reference_df.columns.has_duplicates:
        raise ValueError(
            "reference_df must not contain duplicate column names."
        )

    if reference_df.isnull().any().any():
        raise ValueError(
            "reference_df contains missing values. "
            "Handle missing values before CopulaGAN generation."
        )


def _validate_positive_integer(value, name):
    """Validate a positive integer parameter."""
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")


def _validate_seed(seed):
    """Validate a non-negative integer seed."""
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer.")

    if seed < 0:
        raise ValueError("seed must be non-negative.")


def _seed_all(seed):
    """Seed available random number generators."""
    random.seed(int(seed))
    np.random.seed(int(seed))

    try:
        import torch

        torch.manual_seed(int(seed))

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def _restore_output_dtypes(synthetic_data, reference_df):
    """Restore compatible dtypes and original column ordering."""
    synthetic_data = synthetic_data.copy()
    synthetic_data = synthetic_data.loc[:, reference_df.columns]

    for column in reference_df.columns:
        original_dtype = reference_df[column].dtype

        try:
            if isinstance(original_dtype, pd.CategoricalDtype):
                synthetic_data[column] = pd.Categorical(
                    synthetic_data[column],
                    categories=reference_df[column].cat.categories,
                    ordered=reference_df[column].cat.ordered,
                )

            elif pd.api.types.is_bool_dtype(original_dtype):
                synthetic_data[column] = synthetic_data[column].map(
                    lambda value: (
                        value
                        if isinstance(value, (bool, np.bool_))
                        else bool(value)
                    )
                ).astype(original_dtype)

            elif pd.api.types.is_integer_dtype(original_dtype):
                values = pd.to_numeric(
                    synthetic_data[column],
                    errors="raise",
                )
                rounded = values.round()

                # Never silently truncate fractional generated values.
                if np.allclose(values.to_numpy(), rounded.to_numpy()):
                    synthetic_data[column] = rounded.astype(original_dtype)

            elif pd.api.types.is_float_dtype(original_dtype):
                synthetic_data[column] = pd.to_numeric(
                    synthetic_data[column],
                    errors="raise",
                ).astype(original_dtype)

            elif pd.api.types.is_datetime64_any_dtype(original_dtype):
                synthetic_data[column] = pd.to_datetime(
                    synthetic_data[column],
                    errors="raise",
                )

            elif pd.api.types.is_string_dtype(original_dtype):
                synthetic_data[column] = synthetic_data[column].astype(
                    original_dtype
                )

        except (TypeError, ValueError, OverflowError):
            # Retain generated values if conversion is not safe.
            pass

    return synthetic_data


def _summarize_numeric_columns(df):
    """Build diagnostic summaries without modifying the input data."""
    summaries = []

    for column in df.select_dtypes(include="number").columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()

        summaries.append(
            {
                "column": str(column),
                "dtype": str(df[column].dtype),
                "unique": int(values.nunique()),
                "min": float(values.min()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
                "constant": bool(values.nunique() <= 1),
            }
        )

    return summaries


def generate_copulagan(
    reference_df,
    n_samples=1000,
    seed=42,
    epochs=300,
    verbose=False,
):
    """Generate synthetic tabular data using SDV CopulaGAN."""
    _validate_seed_data(reference_df)
    _validate_positive_integer(n_samples, "n_samples")
    _validate_positive_integer(epochs, "epochs")
    _validate_seed(seed)

    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CopulaGANSynthesizer
    except ImportError as exc:
        raise ImportError(
            "CopulaGAN requires SDV. Install the configured SDV version "
            "in the active virtual environment."
        ) from exc

    training_data = reference_df.copy(deep=True)

    # Diagnose numerical columns before fitting. Do not silently remove them.
    numeric_summary = _summarize_numeric_columns(training_data)
    constant_numeric = [
        item["column"]
        for item in numeric_summary
        if item["constant"]
    ]

    if constant_numeric:
        warnings.warn(
            "CopulaGAN input contains constant numeric columns: "
            + ", ".join(constant_numeric),
            UserWarning,
        )

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=training_data)

    _seed_all(seed)

    synthesizer = CopulaGANSynthesizer(
        metadata=metadata,
        epochs=int(epochs),
        verbose=bool(verbose),
        numerical_distributions={},
        default_distribution="norm",
    )

    set_random_state = getattr(synthesizer, "set_random_state", None)
    if callable(set_random_state):
        set_random_state(seed=int(seed))

    try:
        synthesizer.fit(training_data)
    except ValueError as exc:
        message = str(exc)

        if "beta distribution" in message.lower():
            raise ValueError(
                "CopulaGAN failed while fitting an SDV data distribution. "
                "The error indicates that a beta distribution fit produced "
                "parameters outside the valid data range. "
                f"Numeric column diagnostics: {numeric_summary}. "
                "Inspect the affected columns and SDV's transformer settings "
                "before changing or removing any source data."
            ) from exc

        raise

    _seed_all(seed)

    if callable(set_random_state):
        set_random_state(seed=int(seed))

    synthetic_data = synthesizer.sample(num_rows=int(n_samples))

    if not isinstance(synthetic_data, pd.DataFrame):
        raise TypeError("CopulaGAN did not return a pandas DataFrame.")

    if len(synthetic_data) != int(n_samples):
        raise RuntimeError(
            f"Requested {n_samples} rows, but CopulaGAN returned "
            f"{len(synthetic_data)} rows."
        )

    if (
        len(synthetic_data.columns) != len(reference_df.columns)
        or set(synthetic_data.columns) != set(reference_df.columns)
    ):
        raise RuntimeError(
            "CopulaGAN output columns do not match the reference dataset."
        )

    synthetic_data = _restore_output_dtypes(
        synthetic_data,
        reference_df,
    )

    return synthetic_data.reset_index(drop=True)


def generate_copulagan_engine(
    profile=None,
    n_samples=1000,
    seed=42,
    reference_df=None,
    epochs=300,
    verbose=False,
    **kwargs,
):
    """Compatibility wrapper for the framework's generator interface."""
    if reference_df is None:
        if isinstance(profile, pd.DataFrame):
            reference_df = profile
        else:
            raise ValueError("Reference DataFrame is required.")

    return generate_copulagan(
        reference_df=reference_df,
        n_samples=n_samples,
        seed=seed,
        epochs=epochs,
        verbose=verbose,
    )
