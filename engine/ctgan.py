"""CTGAN synthetic data generation using SDV.

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
            "Handle missing values before CTGAN generation."
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


def _vram_safe_batch_size(requested_batch, n_rows, cardinality):
    """Clamp batch size for GPU VRAM AND make it a multiple of pac=10.

    CTGAN's PacGAN discriminator asserts `batch_size % pac == 0`.
    We round the batch size down to the nearest multiple of 10.
    """
    PAC = 10

    if not torch.cuda.is_available():
        # CPU path — still respect pac multiple
        safe = max(PAC, min(int(requested_batch), int(n_rows)))
        return (safe // PAC) * PAC

    estimated_per_sample = max(1, cardinality // 1000)
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    )

    if vram_gb <= 3.5:
        max_safe = 64 // estimated_per_sample
    elif vram_gb <= 6:
        max_safe = 128 // estimated_per_sample
    elif vram_gb <= 10:
        max_safe = 256 // estimated_per_sample
    else:
        max_safe = 512 // estimated_per_sample

    safe = max(PAC, min(int(requested_batch), int(n_rows), max_safe))
    # Round down to nearest multiple of pac
    return (safe // PAC) * PAC

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
                        else str(v).strip().lower() in {"true", "1", "yes"}
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


def generate_ctgan(
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
        from sdv.single_table import CTGANSynthesizer
    except ImportError as exc:
        raise ImportError(
            "CTGAN requires SDV. Install the configured SDV version."
        ) from exc

    reference_df = reference_df.copy(deep=True)
    original_columns = list(reference_df.columns)

    excluded_columns = [
        column
        for column in ("ship_id", "date")
        if column in reference_df.columns
    ]

    auto_dropped = []
    for column in reference_df.columns:
        if column in excluded_columns:
            continue
        series = reference_df[column]
        if _is_categorical_column(series) and series.nunique() >= 300:
            auto_dropped.append(column)

    if auto_dropped:
        print(
            f"CTGAN: auto-dropping high-cardinality categorical columns "
            f"for VRAM safety: {auto_dropped}"
        )

    drop_columns = list(set(excluded_columns + auto_dropped))
    training_data = reference_df.drop(columns=drop_columns).copy(deep=True)

    if training_data.empty or training_data.shape[1] == 0:
        raise ValueError("No columns remain for CTGAN training.")

    total_cardinality = sum(
        int(training_data[col].nunique())
        for col in training_data.columns
        if _is_categorical_column(training_data[col])
    )
    safe_batch = _vram_safe_batch_size(
        batch_size, len(training_data), total_cardinality
    )

    metadata = _build_metadata(training_data)

    _seed_all(seed)

    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=int(epochs),
        verbose=bool(verbose),
        cuda=torch.cuda.is_available(),
        batch_size=safe_batch,
        discriminator_steps=2,
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
        raise TypeError("CTGAN did not return a pandas DataFrame.")

    if len(synthetic_data) != int(n_samples):
        raise RuntimeError(
            f"Requested {n_samples} rows, but CTGAN returned "
            f"{len(synthetic_data)} rows."
        )

    rng = np.random.default_rng(int(seed))
    for column in drop_columns:
        synthetic_data[column] = rng.choice(
            reference_df[column].dropna().to_numpy(),
            size=len(synthetic_data),
            replace=True,
        )

    synthetic_data = synthetic_data.loc[:, original_columns]
    synthetic_data = _restore_output_dtypes(synthetic_data, reference_df)

    return synthetic_data.reset_index(drop=True)


def generate_ctgan_engine(
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

    return generate_ctgan(
        reference_df=reference_df,
        n_samples=n_samples,
        seed=seed,
        epochs=epochs,
        verbose=verbose,
        batch_size=batch_size,
    )