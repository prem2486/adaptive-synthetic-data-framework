
"""
Gemini-only synthetic tabular data generator.

Environment variables:
    GEMINI_API_KEY
    GEMINI_MODEL

Install:
    python -m pip install --upgrade google-genai pandas numpy

Public interface remains compatible with the existing project:
    generate_llm_baseline_engine(
        profile=profile,
        n_samples=1000,
        reference_df=reference_df,
        seed=42,
    )
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_RETRIES = 1
DEFAULT_MAX_OUTPUT_TOKENS = 4096


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

def _validate_reference_df(
    reference_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Validate and copy the reference dataframe."""

    if reference_df is None:
        raise ValueError("reference_df is required.")

    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError("reference_df must be a pandas DataFrame.")

    if reference_df.empty:
        raise ValueError("reference_df must not be empty.")

    if reference_df.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")

    if any(not isinstance(column, str) for column in reference_df.columns):
        raise ValueError("All column names must be strings.")

    all_missing = [
        column
        for column in reference_df.columns
        if reference_df[column].isna().all()
    ]

    if all_missing:
        raise ValueError(
            f"Columns contain only missing values: {all_missing}"
        )

    return reference_df.copy().reset_index(drop=True)


def _safe_number(value: Any) -> Optional[float]:
    """Convert a value to a finite float or None."""

    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _short_text(value: Any, limit: int = 40) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


# ---------------------------------------------------------------------
# DATASET SUMMARY
# ---------------------------------------------------------------------

def _build_dataset_summary(
    reference_df: pd.DataFrame,
) -> dict[str, Any]:
    """Create a compact profile without sending original data rows."""

    columns = []

    for column in reference_df.columns:
        series = reference_df[column]

        info: dict[str, Any] = {
            "name": column,
            "missing": round(float(series.isna().mean()), 3),
        }

        if pd.api.types.is_bool_dtype(series):
            frequencies = (
                series.dropna()
                .astype(str)
                .value_counts(normalize=True)
            )

            info["kind"] = "cat"
            info["values"] = [
                [_short_text(value), round(float(freq), 3)]
                for value, freq in frequencies.head(4).items()
            ]

        elif pd.api.types.is_datetime64_any_dtype(series):
            dates = pd.to_datetime(
                series, errors="coerce"
            ).dropna()

            info["kind"] = "date"
            info["min"] = (
                dates.min().isoformat() if not dates.empty else None
            )
            info["max"] = (
                dates.max().isoformat() if not dates.empty else None
            )

        elif pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(
                series, errors="coerce"
            ).dropna()

            info["kind"] = (
                "int"
                if pd.api.types.is_integer_dtype(series)
                else "num"
            )

            if numeric.empty:
                info["stats"] = []
            else:
                info["stats"] = [
                    _safe_number(numeric.min()),
                    _safe_number(numeric.max()),
                    _safe_number(numeric.mean()),
                    _safe_number(numeric.std()),
                    _safe_number(numeric.median()),
                    _safe_number(numeric.quantile(0.25)),
                    _safe_number(numeric.quantile(0.75)),
                ]

        else:
            values = series.dropna().astype(str)
            frequencies = values.value_counts(normalize=True)

            info["kind"] = "cat"
            info["unique"] = int(values.nunique())
            info["values"] = [
                [_short_text(value), round(float(freq), 3)]
                for value, freq in frequencies.head(4).items()
            ]

        columns.append(info)

    numeric_df = reference_df.select_dtypes(include=[np.number])
    correlations = []

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr().replace(
            [np.inf, -np.inf], np.nan
        )
        names = list(corr.columns)

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                value = corr.iat[i, j]

                if pd.notna(value):
                    correlations.append({
                        "a": str(names[i]),
                        "b": str(names[j]),
                        "r": round(float(value), 2),
                    })

        correlations.sort(
            key=lambda item: abs(item["r"]),
            reverse=True,
        )

    return {
        "n": int(len(reference_df)),
        "p": int(len(reference_df.columns)),
        "columns": columns,
        "corr": correlations[:6],
    }


def _build_prompt(
    summary: dict[str, Any],
    requested_rows: int,
    seed: int,
) -> str:
    """Build the Gemini generation prompt."""

    summary_json = json.dumps(
        summary,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )

    return (
        "Generate synthetic tabular records from the aggregate profile.\n"
        f"Generate exactly {requested_rows} rows.\n"
        "Each row must contain every original column using its exact name. "
        "Do not add or remove columns. Follow numeric ranges, approximate "
        "statistics, categorical frequencies and correlations where "
        "practical. Keep integer columns as integers. Use null for missing "
        "values where appropriate. Do not reproduce actual source rows.\n"
        "Return only a valid JSON object. Use double quotes for all keys "
        "and string values. Do not use Markdown, comments, or trailing "
        "commas.\n"
        f"Seed: {seed}\n"
        "Numeric stats order: min,max,mean,std,median,p25,p75.\n"
        f"Profile: {summary_json}\n"
        'Output format: {"rows":[{"column_name":"value"}]}'
    )


# ---------------------------------------------------------------------
# JSON PARSING
# ---------------------------------------------------------------------

def _parse_response(
    response_text: Optional[str],
) -> list[dict[str, Any]]:
    """Parse and validate Gemini's JSON response."""

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned invalid JSON: "
            f"{exc}. Response preview: {response_text[:500]!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Gemini response must be a JSON object.")

    rows = payload.get("rows")

    if not isinstance(rows, list):
        raise ValueError(
            "Gemini response must contain a 'rows' array."
        )

    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(
            "Every generated row must be a JSON object."
        )

    return rows


# ---------------------------------------------------------------------
# SCHEMA VALIDATION AND TYPE RESTORATION
# ---------------------------------------------------------------------

def _validate_and_restore_output(
    rows: list[dict[str, Any]],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Check columns and restore compatible pandas data types."""

    expected_columns = list(reference_df.columns)

    for row_number, row in enumerate(rows):
        missing = [
            column for column in expected_columns if column not in row
        ]
        extra = [
            column for column in row if column not in expected_columns
        ]

        if missing or extra:
            raise ValueError(
                f"Row {row_number} schema mismatch. "
                f"Missing: {missing}; extra: {extra}"
            )

    result = pd.DataFrame(rows, columns=expected_columns)

    for column in expected_columns:
        original = reference_df[column]

        if pd.api.types.is_bool_dtype(original):

            def parse_bool(value):
                if value is None or pd.isna(value):
                    return np.nan

                if isinstance(value, (bool, np.bool_)):
                    return bool(value)

                text = str(value).strip().lower()

                if text in {"true", "1", "yes"}:
                    return True

                if text in {"false", "0", "no"}:
                    return False

                return np.nan

            result[column] = result[column].map(parse_bool)

        elif pd.api.types.is_numeric_dtype(original):
            numeric = pd.to_numeric(
                result[column], errors="coerce"
            )

            if pd.api.types.is_integer_dtype(original):
                numeric = numeric.round()

            result[column] = numeric

        elif pd.api.types.is_datetime64_any_dtype(original):
            result[column] = pd.to_datetime(
                result[column], errors="coerce"
            )

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------
# GEMINI GENERATION
# ---------------------------------------------------------------------


def _generate_gemini_batch(
    prompt: str,
    model: str,
    api_key: str,
    output_limit: int,
    reference_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Generate rows using a schema built from the reference dataframe."""

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "Install the Gemini SDK with: "
            "python -m pip install --upgrade google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)

    properties = {}

    for column in reference_df.columns:
        series = reference_df[column]

        if pd.api.types.is_bool_dtype(series):
            value_schema = {"type": "boolean"}
        elif pd.api.types.is_integer_dtype(series):
            value_schema = {"type": "integer"}
        elif pd.api.types.is_numeric_dtype(series):
            value_schema = {"type": "number"}
        elif pd.api.types.is_datetime64_any_dtype(series):
            value_schema = {"type": "string"}
        else:
            value_schema = {"type": "string"}

        # Allow null values if the original column contains missing data.
        if series.isna().any():
            value_schema = {
                "anyOf": [
                    value_schema,
                    {"type": "null"},
                ]
            }

        properties[column] = value_schema

    row_schema = {
        "type": "object",
        "properties": properties,
        "required": list(reference_df.columns),
        "additionalProperties": False,
    }

    response_schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": row_schema,
            }
        },
        "required": ["rows"],
        "additionalProperties": False,
    }

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_schema,
            max_output_tokens=output_limit,
            thinking_config=types.ThinkingConfig(
                thinking_level="low"
            ),
        ),
    )

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish_reason = str(
            getattr(candidates[0], "finish_reason", "")
        ).upper()

        if "MAX_TOKENS" in finish_reason:
            raise ValueError(
                "Gemini reached its output token limit. "
                "Reduce batch size or increase the output allowance."
            )

    rows = _parse_response(response.text)

    # Validate the exact column schema before returning.
    _validate_and_restore_output(rows, reference_df)

    logger.info(
        "Gemini returned %d rows with %d columns using %s.",
        len(rows),
        len(reference_df.columns),
        model,
    )

    return rows



# ---------------------------------------------------------------------
# ERROR HANDLING
# ---------------------------------------------------------------------


def _is_temporary_api_error(exc: Exception) -> bool:
    """Identify temporary Gemini API errors that can be retried."""

    message = str(exc).lower()

    # HTTP status codes exposed by the Google GenAI SDK.
    code = getattr(exc, "code", None)
    response = getattr(exc, "response", None)

    if code is None and response is not None:
        code = getattr(response, "status_code", None)

    try:
        code = int(code) if code is not None else None
    except (TypeError, ValueError):
        code = None

    if code in {408, 429, 500, 502, 503, 504}:
        return True

    temporary_indicators = (
        "503",
        "unavailable",
        "high demand",
        "temporarily overloaded",
        "resource_exhausted",
        "rate limit",
        "too many requests",
        "500",
        "502",
        "504",
        "internal server error",
        "service unavailable",
        "gateway timeout",
        "connection reset",
        "timed out",
        "timeout",
    )

    return any(item in message for item in temporary_indicators)


# ---------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------

def generate_llm_baseline_engine(
    profile: Optional[dict[str, Any]] = None,
    n_samples: int = 1000,
    api_key: Optional[str] = None,
    reference_df: Optional[pd.DataFrame] = None,
    seed: int = DEFAULT_SEED,
    model: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    provider: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate synthetic data using Gemini only.

    Legacy arguments are retained for compatibility with existing calls.
    Groq is not used, even if GROQ_API_KEY is configured.
    """

    del profile, groq_api_key

    if provider and provider.strip().lower() not in {"gemini", "auto"}:
        raise ValueError(
            "This module is Gemini-only. Use provider='gemini' or omit it."
        )

    reference_df = _validate_reference_df(reference_df)

    n_samples = int(n_samples)
    batch_size = int(batch_size)
    max_retries = int(max_retries)
    seed = int(seed)

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    # Prefer the dedicated Gemini environment variable over the legacy
    # generic api_key argument, which may previously have referred to Groq.
    key = (
        gemini_api_key
        or os.getenv("GEMINI_API_KEY")
        or api_key
    )

    if not key:
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY in your "
            "environment variables or pass gemini_api_key=..."
        )

    selected_model = (
        model
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_MODEL
    )

    summary = _build_dataset_summary(reference_df)
    generated_rows: list[dict[str, Any]] = []

    while len(generated_rows) < n_samples:
        remaining = n_samples - len(generated_rows)
        requested = min(batch_size, remaining)

        prompt = _build_prompt(
            summary=summary,
            requested_rows=requested,
            seed=seed + len(generated_rows),
        )

        # Use a moderate limit for wide datasets. If a response is
        # truncated, reduce batch size or adjust this limit carefully.
        output_limit = min(
            DEFAULT_MAX_OUTPUT_TOKENS,
            max(
                4096,
                requested * (len(reference_df.columns) * 14 + 120),
            ),
        )

        batch = None
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                candidate_rows = _generate_gemini_batch(
                    prompt=prompt,
                    model=selected_model,
                    api_key=key,
                    output_limit=output_limit,
                    reference_df=reference_df,
                )

                if len(candidate_rows) != requested:
                    raise ValueError(
                        f"Requested {requested} rows; Gemini returned "
                        f"{len(candidate_rows)}."
                    )

                # Validate the full schema before accepting the batch.
                _validate_and_restore_output(
                    candidate_rows,
                    reference_df,
                )

                batch = candidate_rows
                break

            except Exception as exc:
                last_error = exc

                if (
                    not _is_temporary_api_error(exc)
                    or attempt >= max_retries
                ):
                    logger.error(
                        "Gemini batch failed. Model=%s, rows=%d. Error: %s",
                        selected_model,
                        requested,
                        exc,
                    )
                    break

                delay = min(60, 5 * (2 ** attempt))

                logger.warning(
                    "Temporary Gemini error. Retrying in %s seconds: %s",
                    delay,
                    exc,
                )

                time.sleep(delay)

        if batch is None:
            raise RuntimeError(
                f"Gemini generation failed for a batch of {requested} "
                f"row(s). Last error: {last_error}"
            ) from last_error

        generated_rows.extend(batch)

    result = _validate_and_restore_output(
        generated_rows[:n_samples],
        reference_df,
    )

    if len(result) != n_samples:
        raise RuntimeError(
            f"Expected {n_samples} rows; generated {len(result)}."
        )

    return result


# Backward-compatible public alias.
generate_llm_baseline = generate_llm_baseline_engine

__all__ = [
    "generate_llm_baseline_engine",
    "generate_llm_baseline",
    "_validate_reference_df",
    "_build_dataset_summary",
    "_build_prompt",
    "_validate_and_restore_output",
    "_is_temporary_api_error",
]
