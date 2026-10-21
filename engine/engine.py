"""
engine/engine.py

Main orchestration layer for the Adaptive Synthetic Dataset Generation
Framework Using Multi-Algorithm Evaluation and Statistical Validation.

Pipeline:
    1. Validate the uploaded reference dataset.
    2. Profile the dataset.
    3. Generate synthetic candidates.
    4. Evaluate candidate quality.
    5. Rank candidates and select the best model.

Generators:
    Gaussian Copula, CTGAN, TVAE, TabDDPM, GMM, CopulaGAN; manual Gemini CSV evaluation only
"""

from __future__ import annotations

import warnings
import time
import math
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


DEFAULT_SEED = 42

# All metric names must match the columns returned by evaluate_candidate().
# The weights sum to 1.0.
DEFAULT_WEIGHTS = {
    "ks": 0.25,
    "wasserstein": 0.15,
    "mean_std_gap": 0.10,
    "correlation_mae": 0.15,
    "tstr_weighted_f1": 0.20,
    "categorical_tv": 0.10,
    "constraint_violation_rate": 0.05,
}

ALGORITHMIC_METHODS = [
    "Gaussian Copula",
    "CTGAN",
    "TVAE",
    "TabDDPM",
    "GMM",
    "CopulaGAN",
]

# Browser-generated Gemini CSVs are uploaded manually and evaluated as DataFrames.
# This engine never generates data by calling an LLM API.
ALL_METHODS = list(ALGORITHMIC_METHODS)
ENABLE_LLM_API = False


# ============================================================================
# VALIDATION AND HELPERS
# ============================================================================

def validate_reference_dataset(
    reference_df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """Validate and return a safe copy of the uploaded reference dataset."""

    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError("reference_df must be a pandas DataFrame.")

    if reference_df.empty:
        raise ValueError("The reference dataset is empty.")

    if reference_df.shape[1] < 1:
        raise ValueError("The dataset contains no columns.")

    if reference_df.columns.duplicated().any():
        duplicates = reference_df.columns[
            reference_df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate column names are not allowed: {duplicates}"
        )

    if target_col is not None and target_col not in reference_df.columns:
        raise ValueError(
            f"Target column '{target_col}' does not exist in the dataset."
        )

    result = reference_df.copy().reset_index(drop=True)

    empty_columns = [
        column
        for column in result.columns
        if result[column].isna().all()
    ]

    if empty_columns:
        raise ValueError(
            "These columns contain only missing values: "
            f"{empty_columns}"
        )

    return result


def _canonical_method(method: str) -> Optional[str]:
    """Normalize common generator names and aliases."""

    aliases = {
        "gaussian copula": "Gaussian Copula",
        "gaussian_copula": "Gaussian Copula",
        "gaussiancopula": "Gaussian Copula",
        "copula": "Gaussian Copula",
        "gc": "Gaussian Copula",
        "ctgan": "CTGAN",
        "tvae": "TVAE",
        "tabddpm": "TabDDPM",
        "tab ddpm": "TabDDPM",
        "diffusion": "TabDDPM",
        "gmm": "GMM",
        "gaussian mixture": "GMM",
        "gaussian mixture model": "GMM",
        "copulagan": "CopulaGAN",
        "copula gan": "CopulaGAN",
        "copula_gan": "CopulaGAN",
        "llm": "LLM",
        "llm baseline": "LLM",
        "llm_baseline": "LLM",
        "gemini": "LLM",
        "gemini llm": "LLM",
    }

    return aliases.get(str(method).strip().lower())


def recommend_generation_methods(
    reference_df: pd.DataFrame,
) -> tuple[list[str], dict[str, str]]:
    """
    Recommend generators using a conservative high-cardinality screening rule.

    When a dataset contains high-cardinality text/category columns (for example
    unique ship IDs or hundreds of distinct dates), GAN-based generators are
    not selected automatically because they may expand these columns into many
    encoded dimensions. They remain available in manual mode.
    """
    reference_df = validate_reference_dataset(reference_df)

    n_rows = len(reference_df)
    threshold = max(50, int(math.ceil(0.05 * n_rows)))
    high_cardinality = []

    for column in reference_df.columns:
        series = reference_df[column]
        if (
            pd.api.types.is_object_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            unique_count = int(series.nunique(dropna=True))
            if unique_count >= threshold:
                high_cardinality.append((str(column), unique_count))

    selected = list(ALGORITHMIC_METHODS)
    skipped: dict[str, str] = {}

    if high_cardinality:
        column_summary = ", ".join(
            f"{name} ({count} unique values)"
            for name, count in high_cardinality[:6]
        )
        if len(high_cardinality) > 6:
            column_summary += f", and {len(high_cardinality) - 6} more"

        for method in ("CTGAN", "CopulaGAN"):
            if method in selected:
                selected.remove(method)
            skipped[method] = (
                "Automatically skipped because the reference dataset has "
                f"high-cardinality categorical/text columns: {column_summary}. "
                "Select this model manually if you want to test it."
            )

    return selected, skipped


def _normalize_output(
    generated: Any,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate generated output and enforce reference column order."""

    if generated is None:
        raise ValueError("The generator returned None.")

    if not isinstance(generated, pd.DataFrame):
        generated = pd.DataFrame(generated)

    if generated.empty:
        raise ValueError("The generator returned an empty DataFrame.")

    missing = [
        column
        for column in reference_df.columns
        if column not in generated.columns
    ]

    extra = [
        column
        for column in generated.columns
        if column not in reference_df.columns
    ]

    if missing or extra:
        raise ValueError(
            "Generated columns do not match the reference schema. "
            f"Missing columns: {missing}; extra columns: {extra}"
        )

    generated = generated.loc[:, reference_df.columns].copy()

    return generated.reset_index(drop=True)


def _numeric_columns(
    reference_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> list[str]:
    """Find columns that are numeric in both datasets."""

    return [
        column
        for column in reference_df.columns
        if (
            column in synthetic_df.columns
            and pd.api.types.is_numeric_dtype(reference_df[column])
            and pd.api.types.is_numeric_dtype(synthetic_df[column])
        )
    ]


def _categorical_columns(
    reference_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> list[str]:
    """Find nonnumeric columns shared by both datasets."""

    numeric = set(_numeric_columns(reference_df, synthetic_df))

    return [
        column
        for column in reference_df.columns
        if column in synthetic_df.columns and column not in numeric
    ]


# ============================================================================
# PROFILING
# ============================================================================

def profile_reference_dataset(
    reference_df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a dataset profile using the project's profiling module."""

    reference_df = validate_reference_dataset(reference_df, target_col)

    from engine.profiling import profile_raw_dataset

    profile = profile_raw_dataset(reference_df)

    if not isinstance(profile, dict):
        raise TypeError(
            "profile_raw_dataset() must return a dictionary."
        )

    numeric_columns = list(
        reference_df.select_dtypes(include=[np.number]).columns
    )

    profile["all_columns"] = list(reference_df.columns)
    profile["columns"] = list(reference_df.columns)
    profile["numeric_columns"] = numeric_columns
    profile["categorical_columns"] = [
        column
        for column in reference_df.columns
        if column not in numeric_columns
    ]

    if target_col is not None:
        profile["target"] = target_col
        profile["target_column"] = target_col
        profile["target_info"] = {"name": target_col}

    return profile


# Backward-compatible function name.
profile_dataset = profile_reference_dataset


# ============================================================================
# GENERATION
# ============================================================================

def generate_candidates(
    reference_df: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
    n_samples: int = 1000,
    api_key: Optional[str] = None,
    seed: int = DEFAULT_SEED,
    include_llm: bool = False,
    methods: Optional[Iterable[str]] = None,
    epochs: int = 300,
    diffusion_epochs: int = 35,
    batch_size: int = 500,
    gmm_components: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic candidates using the requested models.

    Failed generators are recorded in generate_candidates.last_errors.
    """

    reference_df = validate_reference_dataset(reference_df)

    n_samples = max(1, int(n_samples))
    seed = int(seed)
    epochs = max(1, int(epochs))
    diffusion_epochs = max(1, int(diffusion_epochs))
    batch_size = max(1, int(batch_size))
    gmm_components = max(1, int(gmm_components))

    if profile is None:
        profile = profile_reference_dataset(reference_df)

    # Hard-disable all API-based LLM generation regardless of caller arguments.
    include_llm = False
    api_key = None

    if methods is None:
        requested, skipped_by_suitability = recommend_generation_methods(
            reference_df
        )
        selection_report = {
            "mode": "automatic",
            "selected": list(requested),
            "skipped": dict(skipped_by_suitability),
        }
    else:
        requested = []
        skipped_by_suitability = {}
        selection_report = {
            "mode": "explicit",
            "selected": [],
            "skipped": {},
        }

        for method in methods:
            canonical = _canonical_method(method)

            if canonical is None:
                warnings.warn(
                    f"Unknown generation method ignored: {method}"
                )
                continue

            if canonical == "LLM":
                warnings.warn(
                    f"Skipping {method!r}: LLM API generation is disabled. "
                    "Upload a Gemini-generated CSV manually instead."
                )
                continue

            if canonical not in requested:
                requested.append(canonical)

    # Defensive safeguard: only the six algorithmic methods are permitted.
    requested = [
        method for method in requested
        if method in ALGORITHMIC_METHODS
    ]
    selection_report["selected"] = list(requested)

    candidates: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}
    timings: Dict[str, float] = {}

    def run_generator(
        name: str,
        generator,
        **kwargs,
    ) -> None:
        started = time.perf_counter()
        try:
            result = generator(**kwargs)
            candidates[name] = _normalize_output(
                result,
                reference_df,
            )
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"{name} generation failed: {exc}")
        finally:
            timings[name] = max(time.perf_counter() - started, 1e-9)

    # Gaussian Copula
    if "Gaussian Copula" in requested:
        started = time.perf_counter()
        try:
            from engine.copula import generate_gaussian_copula

            result = generate_gaussian_copula(
                profile=profile,
                reference_df=reference_df,
                n_samples=n_samples,
                seed=seed,
            )

            candidates["Gaussian Copula"] = _normalize_output(
                result,
                reference_df,
            )

        except Exception as exc:
            errors["Gaussian Copula"] = (
                f"{type(exc).__name__}: {exc}"
            )
            warnings.warn(f"Gaussian Copula generation failed: {exc}")
        finally:
            timings["Gaussian Copula"] = max(
                time.perf_counter() - started,
                1e-9,
            )

    # CTGAN
    if "CTGAN" in requested:
        try:
            from engine.ctgan import generate_ctgan_engine

            run_generator(
                "CTGAN",
                generate_ctgan_engine,
                profile=profile,
                n_samples=n_samples,
                reference_df=reference_df,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                allow_fallback=True,
            )

        except Exception as exc:
            errors["CTGAN"] = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"CTGAN setup failed: {exc}")

    # TVAE
    if "TVAE" in requested:
        try:
            from engine.tvae import generate_tvae_engine

            run_generator(
                "TVAE",
                generate_tvae_engine,
                profile=profile,
                n_samples=n_samples,
                reference_df=reference_df,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                allow_fallback=True,
            )

        except Exception as exc:
            errors["TVAE"] = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"TVAE setup failed: {exc}")

    # TabDDPM
    if "TabDDPM" in requested:
        try:
            from engine.tabddpm import generate_tabddpm_engine

            run_generator(
                "TabDDPM",
                generate_tabddpm_engine,
                profile=profile,
                n_samples=n_samples,
                reference_df=reference_df,
                n_iter=diffusion_epochs,
                batch_size=batch_size,
                seed=seed,
            )


        except Exception as exc:
            errors["TabDDPM"] = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"TabDDPM setup failed: {exc}")

    # GMM
    if "GMM" in requested:
        try:
            from engine.gmm import generate_gmm_engine

            run_generator(
                "GMM",
                generate_gmm_engine,
                profile=profile,
                n_samples=n_samples,
                reference_df=reference_df,
                n_components=gmm_components,
                seed=seed,
            )

        except Exception as exc:
            errors["GMM"] = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"GMM setup failed: {exc}")

    # CopulaGAN
    if "CopulaGAN" in requested:
        try:
            from engine.copulagan import generate_copulagan_engine

            run_generator(
                "CopulaGAN",
                generate_copulagan_engine,
                profile=profile,
                n_samples=n_samples,
                reference_df=reference_df,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                allow_fallback=True,
            )

        except Exception as exc:
            errors["CopulaGAN"] = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"CopulaGAN setup failed: {exc}")


    generate_candidates.last_errors = errors
    generate_candidates.last_timings = timings
    generate_candidates.last_selection_report = selection_report

    return candidates


generate_candidates.last_errors = {}
generate_candidates.last_timings = {}
generate_candidates.last_selection_report = {
    "mode": "automatic",
    "selected": [],
    "skipped": {},
}


# ============================================================================
# STATISTICAL EVALUATION
# ============================================================================

def calculate_ks_statistic(
    real: pd.Series,
    synthetic: pd.Series,
) -> float:
    """Return the Kolmogorov-Smirnov statistic for a numeric column."""

    real_values = (
        pd.to_numeric(real, errors="coerce")
        .dropna()
        .to_numpy()
    )

    synthetic_values = (
        pd.to_numeric(synthetic, errors="coerce")
        .dropna()
        .to_numpy()
    )

    if len(real_values) == 0 or len(synthetic_values) == 0:
        return float("nan")

    return float(
        ks_2samp(real_values, synthetic_values).statistic
    )


def calculate_wasserstein_distance(
    real: pd.Series,
    synthetic: pd.Series,
) -> float:
    """Return a scale-normalized Wasserstein distance."""

    real_values = (
        pd.to_numeric(real, errors="coerce")
        .dropna()
        .to_numpy()
    )

    synthetic_values = (
        pd.to_numeric(synthetic, errors="coerce")
        .dropna()
        .to_numpy()
    )

    if len(real_values) == 0 or len(synthetic_values) == 0:
        return float("nan")

    distance = wasserstein_distance(
        real_values,
        synthetic_values,
    )

    scale = float(np.nanstd(real_values))

    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(
            np.nanmax(real_values) - np.nanmin(real_values)
        )

    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0

    return float(distance / scale)


def calculate_mean_std_gap(
    real: pd.Series,
    synthetic: pd.Series,
) -> float:
    """Calculate the average normalized mean and standard-deviation gap."""

    real_values = pd.to_numeric(
        real,
        errors="coerce",
    ).dropna()

    synthetic_values = pd.to_numeric(
        synthetic,
        errors="coerce",
    ).dropna()

    if real_values.empty or synthetic_values.empty:
        return float("nan")

    mean_scale = max(float(real_values.std()), 1e-12)

    mean_gap = (
        abs(
            float(real_values.mean())
            - float(synthetic_values.mean())
        )
        / mean_scale
    )

    real_std = float(real_values.std())
    synthetic_std = float(synthetic_values.std())

    # If both standard deviations are undefined for a single observation,
    # treat their difference as zero rather than propagating NaN.
    if not np.isfinite(real_std):
        real_std = 0.0

    if not np.isfinite(synthetic_std):
        synthetic_std = 0.0

    std_gap = abs(real_std - synthetic_std) / mean_scale

    return float((mean_gap + std_gap) / 2.0)


def calculate_correlation_mae(
    reference_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> float:
    """Compare numeric correlation matrices using mean absolute error."""

    columns = _numeric_columns(
        reference_df,
        synthetic_df,
    )

    if len(columns) < 2:
        return 0.0

    real_corr = reference_df[columns].corr().to_numpy()
    synth_corr = synthetic_df[columns].corr().to_numpy()

    real_corr = np.nan_to_num(
        real_corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    synth_corr = np.nan_to_num(
        synth_corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return float(np.mean(np.abs(real_corr - synth_corr)))


def calculate_categorical_tv(
    real: pd.Series,
    synthetic: pd.Series,
) -> float:
    """Calculate total variation distance between categorical distributions."""

    real_values = real.astype("string").fillna("<MISSING>")
    synthetic_values = synthetic.astype("string").fillna("<MISSING>")

    categories = set(real_values.unique()) | set(
        synthetic_values.unique()
    )

    if not categories:
        return 0.0

    real_freq = real_values.value_counts(normalize=True)
    synthetic_freq = synthetic_values.value_counts(normalize=True)

    difference = sum(
        abs(
            float(real_freq.get(category, 0.0))
            - float(synthetic_freq.get(category, 0.0))
        )
        for category in categories
    )

    return float(difference / 2.0)


def calculate_constraint_violation_rate(
    reference_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> float:
    """
    Calculate the fraction of numeric synthetic values that violate
    basic finite-value and observed-range constraints.

    This is a diagnostic, not a substitute for domain-specific constraints.
    Each numeric cell is counted at most once.
    """

    total_checks = 0
    violations = 0

    for column in reference_df.columns:
        if column not in synthetic_df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(reference_df[column]):
            continue

        real = pd.to_numeric(
            reference_df[column],
            errors="coerce",
        ).dropna()

        if real.empty:
            continue

        values = pd.to_numeric(
            synthetic_df[column],
            errors="coerce",
        ).to_numpy(dtype=float, na_value=np.nan)

        total_checks += len(values)

        invalid = ~np.isfinite(values)
        below_min = values < float(real.min())
        above_max = values > float(real.max())

        # Combine conditions so one value cannot be counted more than once.
        bad_values = invalid | below_min | above_max

        violations += int(np.sum(bad_values))

    if total_checks == 0:
        return 0.0

    return float(violations / total_checks)


# ============================================================================
# TSTR: TRAIN ON SYNTHETIC, TEST ON REAL
# ============================================================================



def calculate_tstr_weighted_f1(
    reference_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    target_col: Optional[str],
    seed: int = DEFAULT_SEED,
) -> float:
    """
    Train on synthetic data and test on real data.
    Returns NaN when meaningful TSTR evaluation is not possible.
    """
    import logging

    logger = logging.getLogger(__name__)

    if target_col is None:
        logger.warning("TSTR skipped: no target column specified.")
        return float("nan")

    if target_col not in reference_df.columns:
        logger.warning("TSTR skipped: target is absent from reference data.")
        return float("nan")

    if target_col not in synthetic_df.columns:
        logger.warning("TSTR skipped: target is absent from synthetic data.")
        return float("nan")

    real = reference_df.dropna(subset=[target_col]).copy()
    synth = synthetic_df.dropna(subset=[target_col]).copy()

    if real.empty or synth.empty:
        logger.warning("TSTR skipped: no valid target rows.")
        return float("nan")

    if real[target_col].nunique() < 2:
        logger.warning("TSTR skipped: real target has fewer than two classes.")
        return float("nan")

    try:
        X_train = synth.drop(columns=[target_col]).copy()
        y_train = synth[target_col].astype(str)

        X_test = real.drop(columns=[target_col]).copy()
        y_test = real[target_col].astype(str)

        if X_train.empty or X_test.empty:
            logger.warning("TSTR skipped: no predictor columns available.")
            return float("nan")

        # Encode categorical predictors consistently across both datasets.
        X_train = pd.get_dummies(X_train, dummy_na=True)
        X_test = pd.get_dummies(X_test, dummy_na=True)

        X_train, X_test = X_train.align(
            X_test,
            join="outer",
            axis=1,
            fill_value=0,
        )

        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Test only classes present in the synthetic training data.
        known_classes = set(y_train.unique())
        mask = y_test.isin(known_classes)

        X_test = X_test.loc[mask]
        y_test = y_test.loc[mask]

        if X_test.empty:
            logger.warning("TSTR skipped: no overlapping target classes.")
            return float("nan")

        if y_train.nunique() < 2:
            logger.warning("TSTR skipped: synthetic target has fewer than two classes.")
            return float("nan")

        # No class_weight dictionary is needed for weighted F1 evaluation.
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            class_weight=None,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        score = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        if not np.isfinite(score):
            logger.warning("TSTR produced a non-finite F1 score.")
            return float("nan")

        return float(score)


    except Exception as exc:
        logger.exception(
            "TSTR evaluation failed for target=%r, "
                  "reference_shape=%s, synthetic_shape=%s. Error: %s",
            target_col,
            reference_df.shape,
            synthetic_df.shape,
            exc,
        )
        return float("nan")


# ============================================================================
# CANDIDATE EVALUATION
# ============================================================================

def evaluate_candidate(
    reference_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Calculate quality metrics for one synthetic candidate."""

    reference_df = validate_reference_dataset(
        reference_df,
        target_col,
    )

    synthetic_df = _normalize_output(
        synthetic_df,
        reference_df,
    )

    numeric = _numeric_columns(
        reference_df,
        synthetic_df,
    )

    categorical = _categorical_columns(
        reference_df,
        synthetic_df,
    )

    ks_values = [
        calculate_ks_statistic(
            reference_df[column],
            synthetic_df[column],
        )
        for column in numeric
    ]

    wasserstein_values = [
        calculate_wasserstein_distance(
            reference_df[column],
            synthetic_df[column],
        )
        for column in numeric
    ]

    mean_std_values = [
        calculate_mean_std_gap(
            reference_df[column],
            synthetic_df[column],
        )
        for column in numeric
    ]

    categorical_values = [
        calculate_categorical_tv(
            reference_df[column],
            synthetic_df[column],
        )
        for column in categorical
    ]

    def safe_mean(values) -> float:
        valid = [
            float(value)
            for value in values
            if np.isfinite(value)
        ]

        return float(np.mean(valid)) if valid else float("nan")

    return {
        "ks": safe_mean(ks_values),
        "wasserstein": safe_mean(wasserstein_values),
        "mean_std_gap": safe_mean(mean_std_values),
        "correlation_mae": calculate_correlation_mae(
            reference_df,
            synthetic_df,
        ),
        "categorical_tv": safe_mean(categorical_values),
        "tstr_weighted_f1": calculate_tstr_weighted_f1(
            reference_df,
            synthetic_df,
            target_col,
            seed,
        ),
        "constraint_violation_rate": (
            calculate_constraint_violation_rate(
                reference_df,
                synthetic_df,
            )
        ),
    }


# ============================================================================
# MODEL ARBITRATION
# ============================================================================

def _minmax_quality(
    values: pd.Series,
    lower_is_better: bool = True,
) -> pd.Series:
    """
    Normalize a metric to a relative quality score between zero and one.

    This score is relative to the candidate models being compared.
    It is not an absolute synthetic-data quality guarantee.
    """

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=numeric.index,
        dtype=float,
    )

    valid = numeric.dropna()

    if valid.empty:
        return result

    low = float(valid.min())
    high = float(valid.max())

    if abs(high - low) < 1e-12:
        result.loc[valid.index] = 1.0
        return result

    normalized = (numeric - low) / (high - low)

    if lower_is_better:
        normalized = 1.0 - normalized

    result.loc[valid.index] = normalized.loc[valid.index]

    return result


def arbitrate_models(
    evaluation_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Rank candidates by weighted relative quality scores.

    Lower is better for statistical distance and violation metrics.
    Higher is better for TSTR weighted F1.

    Missing metrics are excluded from a candidate's weighted average.
    """

    if evaluation_df is None or evaluation_df.empty:
        return pd.DataFrame(), None

    result = evaluation_df.copy()

    effective_weights = dict(
        DEFAULT_WEIGHTS if weights is None else weights
    )

    metric_config = {
        "ks": True,
        "wasserstein": True,
        "mean_std_gap": True,
        "correlation_mae": True,
        "tstr_weighted_f1": False,
        "categorical_tv": True,
        "constraint_violation_rate": True,
    }

    # Validate weights.
    for metric, weight in effective_weights.items():
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid weight for '{metric}': {weight}"
            ) from exc

        if not np.isfinite(numeric_weight) or numeric_weight < 0:
            raise ValueError(
                f"Weight for '{metric}' must be finite and nonnegative."
            )

        effective_weights[metric] = numeric_weight

    available = [
        metric
        for metric in metric_config
        if metric in result.columns
        and effective_weights.get(metric, 0.0) > 0
    ]

    if not available:
        result["composite_score"] = np.nan
        return result, None

    score_columns = []

    for metric in available:
        score_col = f"{metric}_quality"

        result[score_col] = _minmax_quality(
            result[metric],
            lower_is_better=metric_config[metric],
        )

        score_columns.append(
            (metric, score_col)
        )

    weighted_sum = pd.Series(
        0.0,
        index=result.index,
        dtype=float,
    )

    weight_sum = pd.Series(
        0.0,
        index=result.index,
        dtype=float,
    )

    for metric, score_col in score_columns:
        weight = float(effective_weights.get(metric, 0.0))

        if weight <= 0:
            continue

        valid = result[score_col].notna()

        weighted_sum.loc[valid] += (
            result.loc[valid, score_col] * weight
        )

        weight_sum.loc[valid] += weight

    result["composite_score"] = (
        weighted_sum
        / weight_sum.replace(0, np.nan)
    )

    if "status" in result.columns:
        successful = result[
            result["status"].eq("OK")
        ].copy()

        failed = result[
            ~result["status"].eq("OK")
        ].copy()

    else:
        successful = result.copy()
        failed = result.iloc[0:0].copy()

    # Candidates without a score cannot be selected as the winner.
    successful = successful[
        successful["composite_score"].notna()
    ].copy()

    successful = successful.sort_values(
        "composite_score",
        ascending=False,
        na_position="last",
    )

    ranked = pd.concat(
        [successful, failed],
        ignore_index=True,
    )

    winner = (
        str(successful.iloc[0]["model"])
        if (
            not successful.empty
            and "model" in successful.columns
        )
        else None
    )

    return ranked, winner


def evaluate_candidates(
    reference_df: pd.DataFrame,
    candidate_pool: Dict[str, pd.DataFrame],
    profile: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None,
    seed: int = DEFAULT_SEED,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Evaluate every generated candidate and rank successful models."""

    reference_df = validate_reference_dataset(
        reference_df,
        target_col,
    )

    records = []

    for name, candidate in (candidate_pool or {}).items():
        try:
            metrics = evaluate_candidate(
                reference_df=reference_df,
                synthetic_df=candidate,
                profile=profile,
                target_col=target_col,
                seed=seed,
            )

            metrics["model"] = name
            metrics["status"] = "OK"

            records.append(metrics)

        except Exception as exc:
            records.append({
                "model": name,
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    successful = result[
        result["status"].eq("OK")
    ].copy()

    failed = result[
        ~result["status"].eq("OK")
    ].copy()

    if successful.empty:
        return result

    ranked, _ = arbitrate_models(
        successful,
        weights=weights,
    )

    if not failed.empty:
        ranked = pd.concat(
            [ranked, failed],
            ignore_index=True,
        )

    return ranked


# Backward-compatible alias.
run_evaluation = evaluate_candidates


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def run_pipeline(
    reference_df: pd.DataFrame,
    target_col: Optional[str] = None,
    n_samples: int = 1000,
    api_key: Optional[str] = None,
    seed: int = DEFAULT_SEED,
    methods: Optional[Iterable[str]] = None,
    include_llm: bool = False,
    epochs: int = 300,
    diffusion_epochs: int = 35,
    batch_size: int = 500,
    gmm_components: int = 5,
    weights: Optional[Dict[str, float]] = None,
    profile: Optional[Dict[str, Any]] = None,
    external_candidates: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Generate algorithmic candidates and evaluate external datasets too.

    API-based LLM generation is deliberately disabled. Manually generated
    Gemini CSVs can be supplied through ``external_candidates`` and are
    evaluated with the same metrics as algorithmic candidates.
    """

    include_llm = False
    api_key = None

    reference_df = validate_reference_dataset(
        reference_df,
        target_col,
    )

    if profile is None:
        profile = profile_reference_dataset(
            reference_df,
            target_col,
        )
    elif not isinstance(profile, dict):
        raise TypeError("profile must be a dictionary.")

    candidates = generate_candidates(
        reference_df=reference_df,
        profile=profile,
        n_samples=n_samples,
        api_key=api_key,
        seed=seed,
        include_llm=include_llm,
        methods=methods,
        epochs=epochs,
        diffusion_epochs=diffusion_epochs,
        batch_size=batch_size,
        gmm_components=gmm_components,
    )

    # Preserve per-generator timing before any later final-generation call
    # overwrites generate_candidates.last_timings.
    generation_times = dict(
        getattr(generate_candidates, "last_timings", {})
    )
    selection_report = dict(
        getattr(generate_candidates, "last_selection_report", {})
    )

    # Add manually generated external datasets before evaluation so every
    # candidate is scored exactly once with the same reference profile.
    for candidate_name, candidate_df in (external_candidates or {}).items():
        if not isinstance(candidate_df, pd.DataFrame):
            raise TypeError(
                f"External candidate '{candidate_name}' must be a DataFrame."
            )

        if candidate_df.empty:
            warnings.warn(
                f"External candidate '{candidate_name}' is empty and was skipped."
            )
            continue

        candidates[str(candidate_name)] = candidate_df.copy()

    evaluation = evaluate_candidates(
        reference_df=reference_df,
        candidate_pool=candidates,
        profile=profile,
        target_col=target_col,
        seed=seed,
        weights=weights,
    )

    winner = None

    if (
        not evaluation.empty
        and "composite_score" in evaluation.columns
        and "status" in evaluation.columns
    ):
        successful = evaluation[
            evaluation["status"].eq("OK")
            & evaluation["composite_score"].notna()
            & evaluation["model"].isin(ALGORITHMIC_METHODS)
        ]

        if not successful.empty:
            winner = str(successful.iloc[0]["model"])

    return {
        "profile": profile,
        "candidates": candidates,
        "evaluation": evaluation,
        "winner": winner,
        "generation_errors": dict(
            generate_candidates.last_errors
        ),
        "generation_times": generation_times,
        "selection_report": selection_report,
    }


# ============================================================================
# COMPATIBILITY HELPERS
# ============================================================================

def run_model(
    name: str,
    generator,
    *args,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """Run a single generator without crashing the caller."""

    try:
        result = generator(*args, **kwargs)

        if result is None:
            return None

        return (
            result
            if isinstance(result, pd.DataFrame)
            else pd.DataFrame(result)
        )

    except Exception as exc:
        warnings.warn(f"{name} failed: {exc}")
        return None


def run_arbitration_audit(
    real_df: pd.DataFrame,
    synthetic_dict: Dict[str, pd.DataFrame],
    num_cols: Optional[list[str]] = None,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """Create an evaluation audit for a dictionary of synthetic datasets."""

    real_df = validate_reference_dataset(
        real_df,
        target_col,
    )

    records = []

    for name, synthetic_df in (synthetic_dict or {}).items():
        try:
            metrics = evaluate_candidate(
                reference_df=real_df,
                synthetic_df=synthetic_df,
                target_col=target_col,
            )

            metrics["model"] = name
            metrics["status"] = "OK"

            records.append(metrics)

        except Exception as exc:
            records.append({
                "model": name,
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            })

    return pd.DataFrame(records)


def run_benchmark(
    reference_df: pd.DataFrame,
    target_col: Optional[str] = None,
    methods: Optional[Iterable[str]] = None,
    n_samples: int = 1000,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Benchmark selected methods against a supplied reference DataFrame."""

    result = run_pipeline(
        reference_df=reference_df,
        target_col=target_col,
        methods=methods,
        n_samples=n_samples,
        seed=seed,
        include_llm=False,
    )

    return result["evaluation"]


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_WEIGHTS",
    "ALGORITHMIC_METHODS",
    "ALL_METHODS",
    "validate_reference_dataset",
    "recommend_generation_methods",
    "profile_reference_dataset",
    "profile_dataset",
    "generate_candidates",
    "calculate_ks_statistic",
    "calculate_wasserstein_distance",
    "calculate_mean_std_gap",
    "calculate_correlation_mae",
    "calculate_categorical_tv",
    "calculate_constraint_violation_rate",
    "calculate_tstr_weighted_f1",
    "evaluate_candidate",
    "arbitrate_models",
    "evaluate_candidates",
    "run_evaluation",
    "run_pipeline",
    "run_model",
    "run_arbitration_audit",
    "run_benchmark",
]
