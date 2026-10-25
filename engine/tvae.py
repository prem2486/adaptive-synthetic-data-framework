"""TVAE synthetic data generation using SDV.

CUDA: enabled when available. Batch size clamped for small GPUs.
High-cardinality CATEGORICAL columns are auto-dropped to prevent OOM.
Numeric columns are never dropped.
"""

import numbers
import random
import warnings

import numpy as np
import pandas as pd
import torch


warnings.filterwarnings(
    "ignore", message=".*SingleTableMetadata.*", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", message=".*save_to_json.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*does not contain only a single table.*",
    category=UserWarning,
)


def _validate_seed_data(reference_df):
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
            "Handle missing values before TVAE generation."
        )


def _validate_positive_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")


def _validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")


def _seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def _vram_safe_batch_size(requested_batch, n_rows):
    if not torch.cuda.is_available():
        return max(32, min(int(requested_batch), int(n_rows)))

    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    )
    if vram_gb <= 3.5:
        max_safe = 128
    elif vram_gb <= 6:
        max_safe = 256
    else:
        max_safe = 512

    return max(16, min(int(requested_batch), int(n_rows), max_safe))


def _is_categorical_column(series):
    return (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series.dtype)
    )


def _build_metadata(training_data):
    from sdv.metadata import SingleTableMetadata

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=training_data)

    for column in training_data.columns:
        series = training_data[column]
        if (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            try:
                metadata.update_column(
                    column_name=column, sdtype="categorical"
                )
            except Exception:
                pass

    return metadata


def _restore_output_dtypes(synthetic_data, reference_df):
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
                    lambda v: (
                        v if isinstance(v, (bool, np.bool_))
                        else bool(v)
                    )
                ).astype(original_dtype)
            elif pd.api.types.is_integer_dtype(original_dtype):
                values = pd.to_numeric(
                    synthetic_data[column], errors="raise"
                )
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
            pass

    return synthetic_data


def generate_tvae(
    reference_df,
    n_samples=1000,
    seed=42,
    epochs=300,
    verbose=False,
    batch_size=500,
):
    _validate_seed_data(reference_df)
    _validate_positive_integer(n_samples, "n_samples")
    _validate_positive_integer(epochs, "epochs")
    _validate_seed(seed)

    try:
        from sdv.single_table import TVAESynthesizer
    except ImportError as exc:
        raise ImportError(
            "TVAE requires SDV. Install the configured SDV version."
        ) from exc

    reference_df = reference_df.copy(deep=True)
    original_columns = list(reference_df.columns)

    auto_dropped = [
        col
        for col in reference_df.columns
        if _is_categorical_column(reference_df[col])
        and reference_df[col].nunique() >= 300
    ]
    if auto_dropped:
        print(
            f"TVAE: auto-dropping high-cardinality categorical columns "
            f"for VRAM safety: {auto_dropped}"
        )

    training_data = reference_df.drop(columns=auto_dropped).copy(deep=True)

    if training_data.empty or training_data.shape[1] == 0:
        raise ValueError("No columns remain for TVAE training.")

    safe_batch = _vram_safe_batch_size(batch_size, len(training_data))

    metadata = _build_metadata(training_data)

    _seed_all(seed)

    synthesizer = TVAESynthesizer(
        metadata=metadata,
        epochs=int(epochs),
        verbose=bool(verbose),
        cuda=torch.cuda.is_available(),
        batch_size=safe_batch,
    )

    set_random_state = getattr(synthesizer, "set_random_state", None)
    if callable(set_random_state):
        try:
            set_random_state(seed=int(seed))
        except TypeError:
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
        raise TypeError("TVAE did not return a pandas DataFrame.")

    if len(synthetic_data) != int(n_samples):
        raise RuntimeError(
            f"Requested {n_samples} rows, but TVAE returned "
            f"{len(synthetic_data)} rows."
        )

    rng = np.random.default_rng(int(seed))
    for column in auto_dropped:
        synthetic_data[column] = rng.choice(
            reference_df[column].dropna().to_numpy(),
            size=len(synthetic_data),
            replace=True,
        )

    synthetic_data = synthetic_data.loc[:, original_columns]
    synthetic_data = _restore_output_dtypes(synthetic_data, reference_df)

    return synthetic_data.reset_index(drop=True)


def generate_tvae_engine(
    profile=None,
    n_samples=1000,
    seed=42,
    reference_df=None,
    epochs=300,
    verbose=False,
    batch_size=500,
    **kwargs,
):
    if reference_df is None:
        if isinstance(profile, pd.DataFrame):
            reference_df = profile
        else:
            raise ValueError("Reference DataFrame is required.")

    return generate_tvae(
        reference_df=reference_df,
        n_samples=n_samples,
        seed=seed,
        epochs=epochs,
        verbose=verbose,
        batch_size=batch_size,
    )