"""TabDDPM synthetic data generation using Synthcity.

CUDA: device="auto" uses CUDA when available. Batch size clamped for
small GPUs. High-cardinality CATEGORICAL columns are auto-dropped to
prevent OOM. Numeric columns are never dropped.
"""

from __future__ import annotations

import logging
import random
from numbers import Integral
from typing import Any, Optional

import numpy as np
import pandas as pd

logging.getLogger("synthcity").setLevel(logging.ERROR)
logging.getLogger("synthcity.plugins").setLevel(logging.ERROR)
logging.getLogger("synthcity.plugins.core").setLevel(logging.ERROR)
logging.getLogger("synthcity.utils.logging").setLevel(logging.ERROR)

import torch

try:
    from synthcity.plugins import Plugins
except Exception as exc:
    raise ImportError(
        "TabDDPM requires Synthcity. Install it with "
        "'pip install synthcity==0.2.12'."
    ) from exc


def _validate_reference_df(reference_df: pd.DataFrame) -> None:
    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError("reference_df must be a pandas DataFrame.")
    if reference_df.empty:
        raise ValueError("reference_df must not be empty.")
    if len(reference_df.columns) == 0:
        raise ValueError("reference_df must contain at least one column.")
    if reference_df.columns.has_duplicates:
        raise ValueError(
            "reference_df must not contain duplicate column names."
        )
    if reference_df.isna().all(axis=0).any():
        empty_cols = reference_df.columns[
            reference_df.isna().all(axis=0)
        ].tolist()
        raise ValueError(
            "reference_df contains entirely missing column(s): "
            f"{empty_cols}. Impute or drop them before generation."
        )


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer.")


def _is_categorical_column(series):
    return (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series.dtype)
    )


def _prepare_for_synthcity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    for column in out.columns:
        series = out[column]

        if pd.api.types.is_numeric_dtype(series.dtype):
            values = pd.to_numeric(series, errors="coerce").astype(float)
            if not np.isfinite(values.to_numpy()).all():
                raise ValueError(
                    f"TabDDPM cannot fit on numeric column '{column}' "
                    "because it contains infinite values."
                )
            out[column] = values

        elif pd.api.types.is_string_dtype(series.dtype) and not isinstance(
            series.dtype, pd.CategoricalDtype
        ):
            out[column] = series.astype(object)

        elif pd.api.types.is_bool_dtype(series.dtype):
            out[column] = series.astype("int8")

        elif isinstance(series.dtype, pd.CategoricalDtype):
            out[column] = series.astype(object)

    return out


def _restore_output_dtypes(
    synthetic_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
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
                values = pd.to_numeric(
                    synthetic_df[column], errors="coerce"
                )
                if np.isfinite(values).all() and np.equal(
                    values, np.floor(values)
                ).all():
                    synthetic_df[column] = values.astype(original_dtype)
            elif pd.api.types.is_float_dtype(original_dtype):
                synthetic_df[column] = pd.to_numeric(
                    synthetic_df[column], errors="coerce"
                ).astype(original_dtype)
            elif pd.api.types.is_datetime64_any_dtype(original_dtype):
                synthetic_df[column] = pd.to_datetime(
                    synthetic_df[column], errors="coerce"
                )
            elif pd.api.types.is_object_dtype(original_dtype):
                synthetic_df[column] = synthetic_df[column].astype(object)
            elif pd.api.types.is_string_dtype(original_dtype):
                synthetic_df[column] = synthetic_df[column].astype(
                    original_dtype
                )
        except (TypeError, ValueError, OverflowError):
            pass

    return synthetic_df.reset_index(drop=True)


def _vram_safe_batch_size(requested_batch: int, n_rows: int) -> int:
    if not torch.cuda.is_available():
        return max(8, min(int(requested_batch), max(8, n_rows // 2)))

    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    )
    if vram_gb <= 3.5:
        max_batch = 64
    elif vram_gb <= 6:
        max_batch = 128
    elif vram_gb <= 10:
        max_batch = 256
    else:
        max_batch = 512

    return max(8, min(int(requested_batch), max_batch, max(8, n_rows // 2)))


def generate_tabddpm(
    reference_df: pd.DataFrame,
    n_samples: int = 1000,
    seed: int = 42,
    n_iter: int = 1000,
    batch_size: int = 1024,
    device: str = "auto",
) -> pd.DataFrame:
    _validate_reference_df(reference_df)
    _validate_positive_integer(n_samples, "n_samples")
    _validate_positive_integer(n_iter, "n_iter")
    _validate_positive_integer(batch_size, "batch_size")
    _validate_seed(seed)

    if device not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'.")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but it is not available in this environment."
        )

    training_data = _prepare_for_synthcity(reference_df)
    original_columns = list(reference_df.columns)

    # Only drop high-cardinality CATEGORICAL columns.
    auto_dropped = [
        col
        for col in training_data.columns
        if _is_categorical_column(training_data[col])
        and training_data[col].nunique() >= 300
    ]
    if auto_dropped:
        print(
            f"TabDDPM: auto-dropping high-cardinality categorical columns "
            f"for VRAM safety: {auto_dropped}"
        )
        training_data = training_data.drop(columns=auto_dropped)

    n_rows = len(training_data)
    effective_batch = _vram_safe_batch_size(batch_size, n_rows)
    effective_iter = max(int(n_iter), 50)

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    try:
        plugin = Plugins().get(
            "ddpm",
            n_iter=effective_iter,
            batch_size=effective_batch,
            device=torch.device(device),
            random_state=int(seed),
        )
    except Exception as exc:
        raise RuntimeError(
            f"TabDDPM plugin could not be constructed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        plugin.fit(training_data)
    except Exception as exc:
        raise RuntimeError(
            f"TabDDPM fit failed on {n_rows} rows × "
            f"{training_data.shape[1]} columns. "
            f"Synthcity error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        generated = plugin.generate(
            count=int(n_samples),
            random_state=int(seed),
        )
    except Exception as exc:
        raise RuntimeError(
            f"TabDDPM generation failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

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

    rng = np.random.default_rng(int(seed))
    for column in auto_dropped:
        synthetic_df[column] = rng.choice(
            reference_df[column].dropna().to_numpy(),
            size=len(synthetic_df),
            replace=True,
        )

    missing_columns = set(original_columns) - set(synthetic_df.columns)
    extra_columns = set(synthetic_df.columns) - set(original_columns)

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
    device: str = "auto",
) -> pd.DataFrame:
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