"""CTGAN synthetic data generation using SDV.

High-cardinality identifiers and date fields are reconstructed after sampling
instead of being learned as CTGAN categories. The final output preserves the
reference schema and column order.
"""

import numbers
import random

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
        raise ValueError("reference_df must not contain duplicate column names.")

    if reference_df.isnull().any().any():
        raise ValueError(
            "reference_df contains missing values. "
            "Handle missing values before CTGAN generation."
        )


def _validate_positive_integer(value, name):
    """Require a positive integer."""
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")


def _validate_seed(seed):
    """Require a non-negative integer seed."""
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
    """Restore compatible dtypes and the original column order."""
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
                        else str(value).strip().lower() in {"true", "1", "yes"}
                    )
                ).astype(original_dtype)

            elif pd.api.types.is_integer_dtype(original_dtype):
                values = pd.to_numeric(synthetic_data[column], errors="raise")
                rounded = values.round()
                if np.allclose(values.to_numpy(), rounded.to_numpy()):
                    synthetic_data[column] = rounded.astype(original_dtype)

            elif pd.api.types.is_float_dtype(original_dtype):
                synthetic_data[column] = pd.to_numeric(
                    synthetic_data[column], errors="raise"
                ).astype(original_dtype)

            elif pd.api.types.is_datetime64_any_dtype(original_dtype):
                synthetic_data[column] = pd.to_datetime(
                    synthetic_data[column], errors="raise"
                )

            elif pd.api.types.is_string_dtype(original_dtype):
                synthetic_data[column] = synthetic_data[column].astype(
                    original_dtype
                )

        except (TypeError, ValueError, OverflowError):
            # Keep generated values if exact dtype restoration is not possible.
            pass

    return synthetic_data


def _reconstruct_excluded_columns(
    synthetic_data,
    reference_df,
    excluded_columns,
    seed,
    n_samples,
):
    """Recreate excluded identifier/date fields without training CTGAN on them."""
    synthetic_data = synthetic_data.copy()
    rng = np.random.default_rng(int(seed))

    if "ship_id" in excluded_columns:
        # New unique identifiers; these are not learned from reference IDs.
        synthetic_data["ship_id"] = [
            f"SYN_{int(seed)}_{i:06d}"
            for i in range(1, int(n_samples) + 1)
        ]

    if "date" in excluded_columns:
        source_dates = reference_df["date"].dropna()

        if source_dates.empty:
            raise ValueError(
                "Cannot reconstruct 'date': reference date column is empty."
            )

        # Preserve the observed date distribution by sampling existing date
        # values with replacement. This does not preserve joint date/operation
        # relationships, so treat date reconstruction as a separate step.
        sampled_dates = rng.choice(
            source_dates.to_numpy(),
            size=int(n_samples),
            replace=True,
        )
        synthetic_data["date"] = sampled_dates

    return synthetic_data


def generate_ctgan(
    reference_df,
    n_samples=1000,
    seed=42,
    epochs=300,
    verbose=False,
):
    """
    Generate synthetic tabular data using SDV CTGAN.

    Parameters
    ----------
    reference_df : pandas.DataFrame
        Seed/reference dataset supplied by the caller.
    n_samples : int
        Number of synthetic rows to generate.
    seed : int
        Non-negative random seed.
    epochs : int
        Number of CTGAN training epochs.
    verbose : bool
        Whether to display training progress.

    Returns
    -------
    pandas.DataFrame
        Generated synthetic data with the same columns and column order.
    """
    _validate_seed_data(reference_df)
    _validate_positive_integer(n_samples, "n_samples")
    _validate_positive_integer(epochs, "epochs")
    _validate_seed(seed)

    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer
    except ImportError as exc:
        raise ImportError(
            "CTGAN requires SDV. Install the project's configured "
            "SDV version in the active virtual environment."
        ) from exc

    reference_df = reference_df.copy(deep=True)
    original_columns = list(reference_df.columns)

    # IDs have no useful generalizable signal. Dates can create hundreds of
    # categorical levels, so reconstruct these after model sampling.
    excluded_columns = [
        column
        for column in ("ship_id", "date")
        if column in reference_df.columns
    ]

    training_data = reference_df.drop(
        columns=excluded_columns
    ).copy(deep=True)

    if training_data.empty or training_data.shape[1] == 0:
        raise ValueError("No columns remain for CTGAN training.")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=training_data)

    _seed_all(seed)

    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=int(epochs),
        verbose=bool(verbose),
    )

    set_random_state = getattr(synthesizer, "set_random_state", None)
    if callable(set_random_state):
        try:
            set_random_state(seed=int(seed))
        except TypeError:
            # Accommodate SDV versions whose method has a different signature.
            pass

    synthesizer.fit(training_data)

    _seed_all(seed)
    if callable(set_random_state):
        try:
            set_random_state(seed=int(seed))
        except TypeError:
            pass

    synthetic_data = synthesizer.sample(num_rows=int(n_samples))

    if not isinstance(synthetic_data, pd.DataFrame):
        raise TypeError("CTGAN did not return a pandas DataFrame.")

    if len(synthetic_data) != int(n_samples):
        raise RuntimeError(
            f"Requested {n_samples} rows, but CTGAN returned "
            f"{len(synthetic_data)} rows."
        )

    if (
        len(synthetic_data.columns) != len(training_data.columns)
        or set(synthetic_data.columns) != set(training_data.columns)
    ):
        raise RuntimeError(
            "CTGAN output columns do not match the training schema."
        )

    # Add omitted fields back, then restore exact schema/order and compatible
    # dtypes. No approved output columns are dropped.
    synthetic_data = _reconstruct_excluded_columns(
        synthetic_data=synthetic_data,
        reference_df=reference_df,
        excluded_columns=excluded_columns,
        seed=seed,
        n_samples=n_samples,
    )

    missing = [column for column in original_columns if column not in synthetic_data]
    if missing:
        raise RuntimeError(f"Reconstructed output is missing columns: {missing}")

    synthetic_data = synthetic_data.loc[:, original_columns]
    synthetic_data = _restore_output_dtypes(synthetic_data, reference_df)

    if len(synthetic_data) != int(n_samples):
        raise RuntimeError("Final CTGAN output has an unexpected row count.")

    if synthetic_data.columns.tolist() != original_columns:
        raise RuntimeError("Final CTGAN output schema/order does not match input.")

    return synthetic_data.reset_index(drop=True)


def generate_ctgan_engine(
    profile=None,
    n_samples=1000,
    seed=42,
    reference_df=None,
    epochs=300,
    verbose=False,
    **kwargs,
):
    """Compatibility wrapper for the framework's generator interface."""
    # Existing tests/callers may pass the seed DataFrame as `profile`.
    if reference_df is None:
        if isinstance(profile, pd.DataFrame):
            reference_df = profile
        else:
            raise ValueError("Reference DataFrame is required.")

    return generate_ctgan(
        reference_df=reference_df,
        n_samples=n_samples,
        seed=seed,
        epochs=epochs,
        verbose=verbose,
    )
