"""
Adaptive Synthetic Dataset Generation Framework
Streamlit application with browser-assisted Gemini dataset generation.

Workflow:
1. Upload and profile a reference dataset.
2. Select an optional TSTR target.
3. Prepare a Gemini prompt and open Gemini in a browser.
4. Upload the Gemini-generated CSV.
5. Generate and evaluate algorithmic candidates and Gemini data.
6. Select the best algorithmic generator.
7. Generate, validate and download the final dataset.
"""

from __future__ import annotations

import os

# Set resource limits before importing numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import hashlib
import io
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from engine.engine import (
    generate_candidates,
    profile_reference_dataset,
    recommend_generation_methods,
    run_pipeline,
)

from engine.gemini_prompt import build_gemini_browser_prompt


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Adaptive Synthetic Dataset Generator"

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 50
DEFAULT_DIFFUSION_EPOCHS = 15
DEFAULT_BATCH_SIZE = 128
DEFAULT_GMM_COMPONENTS = 5
DEFAULT_EVALUATION_ROWS = 500
DEFAULT_FINAL_ROWS = 10000

ALGORITHMS = [
    "Gaussian Copula",
    "CTGAN",
    "TVAE",
    "TabDDPM",
    "GMM",
    "CopulaGAN",
]

# These candidates are for comparison, not final algorithm selection.
LLM_NAMES = {
    "llm",
    "llm baseline",
    "gemini browser",
    "gemini",
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Adaptive Synthetic Dataset Generation Framework")

st.caption(
    "Multi-algorithm synthetic data generation, statistical evaluation, "
    "adaptive model selection and final dataset validation."
)

# Browser-assisted Gemini state; no Gemini API key is needed.
st.session_state.setdefault("browser_gemini_df", None)
st.session_state.setdefault("browser_gemini_filename", None)
st.session_state.setdefault("browser_gemini_prompt", "")
st.session_state.setdefault("browser_gemini_prompt_signature", None)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV or Excel dataset."""

    filename = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(content))
        except UnicodeDecodeError:
            return pd.read_csv(
                io.BytesIO(content),
                encoding="latin-1",
            )

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))

    raise ValueError(
        "Unsupported file type. Upload a CSV or Excel dataset."
    )



def normalize_benchmark_table(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an uploaded annual-statistics CSV for user-configured comparison."""
    df = benchmark_df.copy()
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("_", " ")
        if key in {"indicator", "metric", "name", "statistic"}:
            rename[col] = "Indicator"
        elif key in {"official value", "value", "target", "annual value", "reported value"}:
            rename[col] = "Official value"
        elif key in {"unit", "units"}:
            rename[col] = "Unit"
        elif key in {"dataset column", "data column", "column", "source column"}:
            rename[col] = "Dataset column"
        elif key in {"aggregation", "aggregate", "operation"}:
            rename[col] = "Aggregation"
        elif key in {"scale factor", "multiplier", "factor"}:
            rename[col] = "Scale factor"
    df = df.rename(columns=rename)
    if "Indicator" not in df.columns or "Official value" not in df.columns:
        raise ValueError("Annual metrics CSV must contain 'Indicator' and 'Official value' columns.")
    for col, default in [("Unit", ""), ("Dataset column", ""), ("Aggregation", "mean"), ("Scale factor", 1.0)]:
        if col not in df.columns:
            df[col] = default
    df["Indicator"] = df["Indicator"].astype(str).str.strip()
    df["Official value"] = pd.to_numeric(df["Official value"], errors="coerce")
    df["Scale factor"] = pd.to_numeric(df["Scale factor"], errors="coerce").fillna(1.0)
    df["Dataset column"] = df["Dataset column"].fillna("").astype(str)
    df["Aggregation"] = df["Aggregation"].fillna("mean").astype(str).str.lower().str.strip()
    return df


def build_official_port_validation(
    benchmark_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    candidates: Optional[dict] = None,
    final_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compare uploaded annual targets against dynamic summaries of generated datasets.

    Each benchmark row defines the dataset column, aggregation, and scale factor.
    The scale factor converts the dataset summary into the official metric's units.
    """
    benchmark = normalize_benchmark_table(benchmark_df)
    datasets = [("LLM Seed / Reference", reference_df)]
    for name, df in (candidates or {}).items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            datasets.append((str(name), df))
    if isinstance(final_df, pd.DataFrame) and not final_df.empty:
        datasets.append(("Selected Final Output", final_df))

    rows = []
    valid_aggregations = {"mean", "median", "sum", "count", "min", "max"}
    for name, data in datasets:
        metric_errors = []
        row = {"Dataset / Method": name, "Rows": int(len(data))}
        for _, metric in benchmark.iterrows():
            indicator = str(metric["Indicator"])
            official_value = metric["Official value"]
            unit = str(metric.get("Unit", "") or "")
            column = str(metric.get("Dataset column", "") or "").strip()
            aggregation = str(metric.get("Aggregation", "mean") or "mean").lower().strip()
            factor = float(metric.get("Scale factor", 1.0))
            prefix = f"{indicator}"

            generated_value = np.nan
            if column in data.columns and aggregation in valid_aggregations:
                values = pd.to_numeric(data[column], errors="coerce")
                if aggregation == "count":
                    raw_value = float(values.notna().sum())
                elif values.notna().any():
                    raw_value = float(getattr(values.dropna(), aggregation)())
                else:
                    raw_value = np.nan
                if pd.notna(raw_value):
                    generated_value = raw_value * factor

            abs_error = abs(generated_value - official_value) if pd.notna(generated_value) and pd.notna(official_value) else np.nan
            rel_error = (abs_error / abs(official_value) * 100.0) if pd.notna(abs_error) and official_value != 0 else (0.0 if pd.notna(abs_error) and official_value == 0 and abs_error == 0 else np.nan)
            if pd.notna(rel_error):
                metric_errors.append(float(min(max(rel_error, 0.0), 100.0)))

            row[f"{prefix} | Official ({unit})" if unit else f"{prefix} | Official"] = official_value
            row[f"{prefix} | Generated ({unit})" if unit else f"{prefix} | Generated"] = generated_value
            row[f"{prefix} | Absolute error"] = abs_error
            row[f"{prefix} | Relative error (%)"] = rel_error

        row["Annual Statistics Similarity Score (%)"] = (
            100.0 - float(np.mean(metric_errors)) if metric_errors else np.nan
        )
        row["Metrics scored"] = len(metric_errors)
        rows.append(row)
    return pd.DataFrame(rows)


def infer_target_column(df: pd.DataFrame):
    """Suggest a likely classification target column."""

    preferred_names = [
        "target",
        "label",
        "class",
        "outcome",
        "response",
        "churn",
        "default",
        "status",
    ]

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in preferred_names:
        if name in lookup:
            candidate = lookup[name]

            if df[candidate].nunique(dropna=True) >= 2:
                return candidate

    for column in df.columns:
        name = str(column).strip().lower()

        if any(
            token in name
            for token in ("id", "identifier", "timestamp", "date")
        ):
            continue

        series = df[column].dropna()

        if len(series) < 5:
            continue

        unique_count = series.nunique()
        max_classes = min(
            20,
            max(2, int(len(series) * 0.20)),
        )

        if 2 <= unique_count <= max_classes:
            return column

    return None


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def validate_final_dataset(
    reference_df: pd.DataFrame,
    synthetic_df: Optional[pd.DataFrame],
) -> dict:
    """Check output schema, missing values and numeric bounds."""

    if synthetic_df is None:
        return {
            "generated": False,
            "reason": "Final dataset generation failed.",
        }

    missing_columns = [
        column
        for column in reference_df.columns
        if column not in synthetic_df.columns
    ]

    extra_columns = [
        column
        for column in synthetic_df.columns
        if column not in reference_df.columns
    ]

    numeric_violations = 0

    for column in reference_df.select_dtypes(
        include="number"
    ).columns:

        if column not in synthetic_df.columns:
            continue

        reference_values = pd.to_numeric(
            reference_df[column],
            errors="coerce",
        )

        synthetic_values = pd.to_numeric(
            synthetic_df[column],
            errors="coerce",
        )

        lower = reference_values.min()
        upper = reference_values.max()

        if pd.isna(lower) or pd.isna(upper):
            continue

        numeric_violations += int(
            (
                (synthetic_values < lower)
                | (synthetic_values > upper)
            ).sum()
        )

    schema_match = (
        list(reference_df.columns)
        == list(synthetic_df.columns)
    )

    return {
        "generated": True,
        "rows": len(synthetic_df),
        "columns": len(synthetic_df.columns),
        "requested_columns": len(reference_df.columns),
        "schema_match": schema_match,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "missing_values": int(
            synthetic_df.isna().sum().sum()
        ),
        "numeric_bound_violations": numeric_violations,
        "structurally_valid": (
            len(synthetic_df) > 0
            and not missing_columns
            and not extra_columns
            and schema_match
        ),
    }


def select_algorithmic_winner(
    evaluation: pd.DataFrame,
) -> Optional[str]:
    """
    Select the best successful algorithmic candidate.

    LLM and browser-generated Gemini candidates are excluded from
    final algorithm selection. The engine's arbitration ranks higher
    composite scores as better.
    """

    if evaluation is None or evaluation.empty:
        return None

    candidates = evaluation.copy()

    if "status" in candidates.columns:
        candidates = candidates[
            candidates["status"].astype(str).str.upper() == "OK"
        ]

    if "model" not in candidates.columns:
        return None

    candidates = candidates[
        ~candidates["model"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(LLM_NAMES)
    ]

    if "composite_score" not in candidates.columns:
        return None

    candidates["composite_score"] = pd.to_numeric(
        candidates["composite_score"],
        errors="coerce",
    )

    candidates = candidates.dropna(
        subset=["composite_score"]
    )

    if candidates.empty:
        return None

    best = candidates.sort_values(
        "composite_score",
        ascending=False,
    ).iloc[0]

    return str(best["model"])


def evaluation_display_table(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare evaluation metrics for display."""

    preferred_columns = [
        "model",
        "status",
        "ks",
        "wasserstein",
        "mean_std_gap",
        "correlation_mae",
        "tstr_weighted_f1",
        "categorical_tv",
        "constraint_violation_rate",
        "composite_score",
    ]

    available = [
        column
        for column in preferred_columns
        if column in evaluation.columns
    ]

    table = evaluation[available].copy()

    labels = {
        "model": "Model",
        "status": "Status",
        "ks": "KS",
        "wasserstein": "Wasserstein",
        "mean_std_gap": "Mean/Std Gap",
        "correlation_mae": "Correlation MAE",
        "tstr_weighted_f1": "TSTR Weighted F1",
        "categorical_tv": "Categorical TV",
        "constraint_violation_rate": "Constraint Violations",
        "composite_score": "Composite Score",
    }

    table = table.rename(columns=labels)

    for column in table.columns:
        if column not in {"Model", "Status"}:
            table[column] = pd.to_numeric(
                table[column],
                errors="coerce",
            ).round(4)

    return table


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Configuration")
st.sidebar.subheader("Dataset Generation")

final_rows = int(
    st.sidebar.number_input(
        "Final synthetic records",
        min_value=50,
        max_value=100000,
        value=DEFAULT_FINAL_ROWS,
        step=100,
    )
)

evaluation_rows = int(
    st.sidebar.number_input(
        "Rows per candidate for evaluation",
        min_value=100,
        max_value=10000,
        value=DEFAULT_EVALUATION_ROWS,
        step=100,
        help=(
            "Each selected generator creates a candidate of this size "
            "for model comparison. The winner is generated again at "
            "the requested final size."
        ),
    )
)

seed = int(
    st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=DEFAULT_SEED,
    )
)

st.sidebar.subheader("Model Settings")

epochs = int(
    st.sidebar.number_input(
        "CTGAN / TVAE / CopulaGAN epochs",
        min_value=1,
        max_value=500,
        value=DEFAULT_EPOCHS,
    )
)

diffusion_epochs = int(
    st.sidebar.number_input(
        "TabDDPM epochs",
        min_value=1,
        max_value=200,
        value=DEFAULT_DIFFUSION_EPOCHS,
    )
)

batch_size = int(
    st.sidebar.number_input(
        "Batch size",
        min_value=32,
        max_value=1024,
        value=DEFAULT_BATCH_SIZE,
        step=32,
    )
)

gmm_components = int(
    st.sidebar.number_input(
        "GMM components",
        min_value=1,
        max_value=50,
        value=DEFAULT_GMM_COMPONENTS,
    )
)

# LLM API generation is disabled. Gemini is generated manually in the browser.
st.sidebar.info(
    "LLM API generation is disabled. You can upload a manually generated "
    "Gemini CSV for statistical evaluation."
)


# ============================================================
# DATASET UPLOAD
# ============================================================

st.header("1. Upload Reference Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Upload the reference dataset directly.",
)

if uploaded_file is None:
    st.info("Upload a dataset to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()

file_signature = hashlib.sha256(
    file_bytes
).hexdigest()

if st.session_state.get("file_signature") != file_signature:
    try:
        reference_df = read_uploaded_dataset(uploaded_file)

        st.session_state["reference_df"] = reference_df
        st.session_state["file_signature"] = file_signature
        st.session_state["pipeline_result"] = None
        st.session_state["result_signature"] = None

        # Clear browser-generated data when the reference dataset changes.
        st.session_state["browser_gemini_df"] = None
        st.session_state["browser_gemini_filename"] = None
        st.session_state["browser_gemini_prompt"] = ""
        st.session_state["browser_gemini_prompt_signature"] = None

    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

reference_df = st.session_state["reference_df"]

if reference_df.empty:
    st.error("The uploaded dataset contains no rows.")
    st.stop()

if len(reference_df) < 2:
    st.error("Upload a dataset containing at least two rows.")
    st.stop()

if len(reference_df.columns) < 1:
    st.error("The uploaded dataset contains no columns.")
    st.stop()


# ============================================================
# DATASET PROFILE
# ============================================================

st.subheader("Reference Dataset Overview")

numeric_count = len(
    reference_df.select_dtypes(include="number").columns
)

categorical_count = len(
    reference_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns
)

missing_count = int(
    reference_df.isna().sum().sum()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{len(reference_df):,}")
c2.metric("Columns", f"{len(reference_df.columns):,}")
c3.metric("Numeric Columns", numeric_count)
c4.metric("Missing Values", f"{missing_count:,}")

with st.expander("Preview uploaded dataset"):
    st.dataframe(
        reference_df.head(20),
        width="stretch",
    )

try:
    profile = profile_reference_dataset(reference_df)
except Exception as exc:
    st.error(f"Dataset profiling failed: {exc}")
    st.stop()

st.write("**Column profile**")

profile_records = []

for column in reference_df.columns:
    series = reference_df[column]

    profile_records.append({
        "Column": str(column),
        "Data Type": str(series.dtype),
        "Non-null Rows": int(series.notna().sum()),
        "Missing Values": int(series.isna().sum()),
        "Unique Values": int(series.nunique(dropna=True)),
    })

st.dataframe(
    pd.DataFrame(profile_records),
    hide_index=True,
    width="stretch",
)


# ============================================================
# TARGET COLUMN FOR TSTR
# ============================================================

st.header("2. TSTR Evaluation Settings")

suggested_target = infer_target_column(reference_df)
target_options = [None] + list(reference_df.columns)

default_target_index = (
    target_options.index(suggested_target)
    if suggested_target in reference_df.columns
    else 0
)

target_col = st.selectbox(
    "Target column for downstream utility evaluation",
    options=target_options,
    index=default_target_index,
    format_func=lambda value: (
        "None — skip TSTR"
        if value is None
        else str(value)
    ),
    help=(
        "Select the column a classifier should predict. "
        "Change the suggested column if needed."
    ),
)

if target_col is None:
    st.caption(
        "TSTR will be skipped. Other statistical metrics remain available."
    )
else:
    st.caption(f"TSTR target column: **{target_col}**")

# Build the final profile with the selected TSTR target included.
try:
    profile = profile_reference_dataset(reference_df, target_col=target_col)
except Exception as exc:
    st.error(f"Dataset profiling failed: {exc}")
    st.stop()


# ============================================================
# GEMINI BROWSER-ASSISTED GENERATION — NO GEMINI API KEY
# ============================================================

st.header("🌐 Gemini Browser-Assisted Dataset Generation")

st.info(
    "Fair comparison: the reference CSV uploaded in Section 1 is the seed for "
    "the statistical generators. Download that same CSV below and attach it in "
    "Gemini before pasting the generated prompt, so Gemini and the statistical "
    "models use the same source dataset."
)
st.download_button(
    "Download the exact reference seed CSV for Gemini",
    data=dataframe_to_csv_bytes(reference_df),
    file_name="reference_seed_for_gemini.csv",
    mime="text/csv",
    key="download_reference_seed_for_gemini",
    width="stretch",
)

st.write(
    "Prepare a schema-aware prompt, open Gemini in a separate browser tab, "
    "sign in with your Google account, generate a CSV, and upload it here. "
    "This application does not collect or store your Google password."
)

with st.expander(
    "Prepare prompt and upload Gemini output",
    expanded=True,
):
    gemini_rows = int(
        st.number_input(
            "Rows to request from Gemini",
            min_value=10,
            max_value=50000,
            value=min(max(evaluation_rows, 100), 50000),
            step=50,
            key="browser_gemini_rows",
        )
    )

    gemini_seed = int(
        st.number_input(
            "Gemini generation seed",
            min_value=0,
            max_value=2147483647,
            value=seed,
            step=1,
            key="browser_gemini_seed",
            help=(
                "The seed is included as an instruction. Gemini's website "
                "does not guarantee identical output for the same seed."
            ),
        )
    )

    if st.button(
        "📝 Prepare Gemini Prompt",
        key="prepare_browser_gemini_prompt",
    ):
        try:
            base_gemini_prompt = build_gemini_browser_prompt(
                profile=profile,
                n_rows=gemini_rows,
                seed=gemini_seed,
            )
            same_seed_instruction = (
                "\n\nFAIR-COMPARISON REQUIREMENT:\n"
                "I will attach the exact reference CSV used by the statistical generators. "
                "Use that attached CSV as the sole data reference. Generate synthetic rows "
                "with the same columns and compatible data types, preserving the source "
                "schema and realistic relationships. Do not use unrelated external data. "
                "Do not copy rows verbatim; generate synthetic records. The CSV uploaded "
                "to this app as the reference dataset and the CSV attached to this prompt "
                "must be the same file.\n"
            )
            st.session_state["browser_gemini_prompt"] = (
                base_gemini_prompt + same_seed_instruction
            )

            st.session_state["browser_gemini_prompt_signature"] = (
                file_signature,
                gemini_rows,
                gemini_seed,
            )

            st.success(
                "Prompt prepared. Copy it and paste it into Gemini."
            )

        except Exception as exc:
            st.error(f"Could not prepare the Gemini prompt: {exc}")

    gemini_prompt = st.session_state.get(
        "browser_gemini_prompt",
        "",
    )

    current_prompt_signature = (
        file_signature,
        gemini_rows,
        gemini_seed,
    )

    if gemini_prompt:
        if (
            st.session_state.get("browser_gemini_prompt_signature")
            != current_prompt_signature
        ):
            st.warning(
                "The dataset, row count, or seed changed after this prompt "
                "was prepared. Prepare a new prompt before generating "
                "another dataset."
            )

        st.caption("Copy the prompt below and paste it into Gemini.")
        st.code(gemini_prompt, language="text")

        st.link_button(
            "🔓 Open Gemini",
            "https://gemini.google.com/",
        )

        st.download_button(
            "⬇️ Download prompt as TXT",
            data=gemini_prompt.encode("utf-8"),
            file_name=f"gemini_prompt_seed_{gemini_seed}.txt",
            mime="text/plain",
            key="download_browser_gemini_prompt",
        )

    st.divider()
    st.subheader("Upload Gemini-generated CSV")

    st.caption(
        "In Gemini, use its data-analysis or code-execution capability "
        "to create and download the CSV. Upload the downloaded file here."
    )

    gemini_elapsed_seconds = float(
        st.number_input(
            "Gemini generation time (seconds; optional)",
            min_value=0.0,
            max_value=86400.0,
            value=0.0,
            step=1.0,
            key="browser_gemini_elapsed_seconds",
            help=(
                "Enter the approximate time Gemini took to generate the CSV. "
                "Leave 0 if unknown; throughput will then be shown as unavailable."
            ),
        )
    )

    gemini_upload = st.file_uploader(
        "Gemini-generated CSV",
        type=["csv"],
        key=f"browser_gemini_upload_{file_signature[:12]}",
    )

    if gemini_upload is not None:
        try:
            gemini_df = pd.read_csv(
                io.BytesIO(gemini_upload.getvalue())
            )

            if gemini_df.empty:
                st.error("The Gemini CSV contains no data rows.")

            else:
                expected_columns = list(reference_df.columns)
                actual_columns = list(gemini_df.columns)

                missing_columns = [
                    column
                    for column in expected_columns
                    if column not in actual_columns
                ]

                extra_columns = [
                    column
                    for column in actual_columns
                    if column not in expected_columns
                ]

                if missing_columns or extra_columns:
                    st.error(
                        "The Gemini CSV schema does not match "
                        "the reference dataset."
                    )

                    if missing_columns:
                        st.write("Missing columns:", missing_columns)

                    if extra_columns:
                        st.write("Unexpected columns:", extra_columns)

                    st.session_state["browser_gemini_df"] = None
                    st.session_state["browser_gemini_filename"] = None

                else:
                    # Align Gemini columns to the reference order.
                    gemini_df = gemini_df[expected_columns].copy()
                    conversion_errors = []

                    # Preserve numeric column types where possible.
                    for column in reference_df.select_dtypes(
                        include="number"
                    ).columns:
                        before = gemini_df[column].notna()

                        converted = pd.to_numeric(
                            gemini_df[column],
                            errors="coerce",
                        )

                        bad_values = int(
                            (before & converted.isna()).sum()
                        )

                        if bad_values:
                            conversion_errors.append(
                                f"{column}: {bad_values} non-numeric value(s)"
                            )

                        gemini_df[column] = converted

                    if conversion_errors:
                        st.error(
                            "Some numeric columns contain non-numeric "
                            "values. Correct the CSV and upload it again."
                        )
                        st.write(conversion_errors)

                        st.session_state["browser_gemini_df"] = None
                        st.session_state["browser_gemini_filename"] = None

                    else:
                        st.session_state["browser_gemini_df"] = gemini_df
                        st.session_state["browser_gemini_filename"] = (
                            gemini_upload.name
                        )

                        st.success(
                            f"Loaded Gemini dataset: {len(gemini_df):,} rows × "
                            f"{len(gemini_df.columns):,} columns."
                        )

                        missing_count_gemini = int(
                            gemini_df.isna().sum().sum()
                        )

                        if missing_count_gemini:
                            st.warning(
                                f"The Gemini dataset contains "
                                f"{missing_count_gemini:,} missing value(s). "
                                "Review them before interpreting its scores."
                            )

                        st.dataframe(
                            gemini_df.head(10),
                            hide_index=True,
                            width="stretch",
                        )

        except Exception as exc:
            st.error(f"Could not read the Gemini CSV: {exc}")
            st.session_state["browser_gemini_df"] = None
            st.session_state["browser_gemini_filename"] = None

if st.session_state.get("browser_gemini_df") is not None:
    st.info(
        "Gemini Browser dataset is ready. It will be included as a separate "
        "evaluation candidate, but it cannot be selected as the final "
        "algorithmic generator."
    )


# ============================================================
# GENERATION METHODS
# ============================================================

st.header("3. Generation Methods")

selection_mode = st.radio(
    "Model selection mode",
    options=["Automatic suitability screening", "Manual selection"],
    index=0,
    help=(
        "Automatic mode screens high-cardinality text/category columns and "
        "skips GAN-based models when they are likely to trigger expensive "
        "one-hot encoding. Manual mode lets you override that recommendation."
    ),
    horizontal=True,
)

auto_skipped_models = {}
if selection_mode == "Automatic suitability screening":
    selected_algorithms, auto_skipped_models = recommend_generation_methods(
        reference_df
    )
    st.success(
        "Automatically selected models: "
        + ", ".join(selected_algorithms)
    )
    if auto_skipped_models:
        st.warning(
            "Some models were skipped automatically because this dataset has "
            "high-cardinality categorical/text columns. They are not disabled; "
            "switch to Manual selection to run them."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Model": model, "Reason": reason}
                    for model, reason in auto_skipped_models.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
else:
    selected_algorithms = st.multiselect(
        "Algorithms to compare",
        options=ALGORITHMS,
        default=[],
        key="manual_algorithm_selection",
        help=(
            "Choose exactly which generators to run. No model is selected "
            "automatically in manual mode."
        ),
    )

if not selected_algorithms:
    st.warning("No algorithms are selected. Choose at least one model.")

st.write("**Run summary**")

st.write({
    "Selection mode": selection_mode,
    "Selected algorithms": selected_algorithms,
    "Automatically skipped models": list(auto_skipped_models),
    "Final output rows": final_rows,
    "Candidate evaluation rows": evaluation_rows,
    "TSTR target": (
        str(target_col)
        if target_col is not None
        else "Disabled"
    ),
    "LLM API generation": "Disabled",
    "Gemini browser dataset": (
        "Uploaded — included in evaluation"
        if isinstance(
            st.session_state.get("browser_gemini_df"),
            pd.DataFrame,
        )
        else "Not uploaded"
    ),
})


# ============================================================
# RUN PIPELINE
# ============================================================

st.header("4. Generate, Evaluate and Select")

config_signature = (
    file_signature,
    str(target_col),
    selection_mode,
    tuple(selected_algorithms),
    final_rows,
    evaluation_rows,
    seed,
    epochs,
    diffusion_epochs,
    batch_size,
    gmm_components,
    st.session_state.get("browser_gemini_filename"),
    float(st.session_state.get("browser_gemini_elapsed_seconds", 0.0)),
)

run_button = st.button(
    "🚀 Generate and Evaluate",
    type="primary",
    width="stretch",
    disabled=not selected_algorithms,
)

if run_button:
    started = time.time()

    try:
        with st.status(
            "Running the synthetic-data pipeline...",
            expanded=True,
        ) as status:

            st.write("Step 1: Profiling reference data.")

            st.write(
                "Step 2: Generating candidate datasets: "
                + ", ".join(selected_algorithms)
            )

            st.write(
                "Step 3: Evaluating algorithmic candidates and the uploaded "
                "Gemini dataset using the same statistical metrics."
            )

            # Add Gemini before evaluation so every candidate is scored once.
            external_candidates = {}
            browser_gemini_df = st.session_state.get("browser_gemini_df")

            if (
                isinstance(browser_gemini_df, pd.DataFrame)
                and not browser_gemini_df.empty
            ):
                gemini_candidate = browser_gemini_df.copy()
                expected_columns = list(reference_df.columns)

                if list(gemini_candidate.columns) == expected_columns:
                    if len(gemini_candidate) > evaluation_rows:
                        gemini_candidate = gemini_candidate.sample(
                            n=evaluation_rows,
                            random_state=seed,
                        ).reset_index(drop=True)

                    external_candidates["Gemini Browser"] = gemini_candidate
                else:
                    st.warning(
                        "The Gemini dataset schema does not match the "
                        "reference dataset, so it was excluded from evaluation."
                    )

            result = run_pipeline(
                reference_df=reference_df,
                target_col=target_col,
                n_samples=evaluation_rows,
                seed=seed,
                methods=list(selected_algorithms),
                include_llm=False,
                epochs=epochs,
                diffusion_epochs=diffusion_epochs,
                batch_size=batch_size,
                gmm_components=gmm_components,
                profile=profile,
                external_candidates=external_candidates,
            )

            if external_candidates:
                result["browser_gemini_filename"] = (
                    st.session_state.get("browser_gemini_filename")
                )
                result["browser_gemini_seed"] = gemini_seed
                result["gemini_uploaded_rows"] = len(browser_gemini_df)
                result["gemini_elapsed_seconds"] = float(
                    st.session_state.get(
                        "browser_gemini_elapsed_seconds", 0.0
                    )
                )

            result["model_selection_mode"] = selection_mode
            result["auto_skipped_models"] = dict(auto_skipped_models)

            evaluation = result.get("evaluation")

            if evaluation is None or evaluation.empty:
                raise RuntimeError(
                    "No candidate evaluation results were produced. "
                    "Check the generation diagnostics."
                )

            st.write("Step 4: Selecting the best algorithmic candidate.")

            winner = select_algorithmic_winner(evaluation)

            final_df = None
            final_error = None
            final_generation_errors = {}

            if winner is not None:
                st.write(
                    f"Step 5: Generating {final_rows:,} fresh rows "
                    f"using {winner}."
                )

                final_pool = generate_candidates(
                    reference_df=reference_df,
                    profile=result.get("profile", profile),
                    n_samples=final_rows,
                    seed=seed + 1,
                    include_llm=False,
                    methods=[winner],
                    epochs=epochs,
                    diffusion_epochs=diffusion_epochs,
                    batch_size=batch_size,
                    gmm_components=gmm_components,
                )

                final_df = final_pool.get(winner)

                final_generation_errors = dict(
                    getattr(
                        generate_candidates,
                        "last_errors",
                        {},
                    )
                )

                if final_df is None:
                    final_error = final_generation_errors.get(
                        winner,
                        "The selected generator returned no dataset.",
                    )

            else:
                final_error = (
                    "No successful algorithmic candidate had a valid "
                    "composite score. No final dataset was generated."
                )

            result["winner"] = winner
            result["final_df"] = final_df
            result["final_error"] = final_error
            result["final_generation_errors"] = final_generation_errors

            st.session_state["pipeline_result"] = result
            st.session_state["result_signature"] = config_signature

            status.update(
                label=(
                    f"Pipeline finished in "
                    f"{time.time() - started:.1f} seconds."
                ),
                state="complete",
                expanded=False,
            )

    except Exception as exc:
        st.error("The pipeline could not complete.")
        st.exception(exc)


# ============================================================
# DISPLAY RESULTS
# ============================================================

result = st.session_state.get("pipeline_result")

if result is None:
    st.info(
        "Configure the dataset and generation methods, "
        "then click Generate and Evaluate."
    )
    st.stop()

if st.session_state.get("result_signature") != config_signature:
    st.info(
        "Your settings have changed. Click Generate and Evaluate "
        "to produce results for the current settings."
    )
    st.stop()


# ============================================================
# EVALUATION RESULTS
# ============================================================

st.header("5. Model Evaluation Results")

evaluation = result.get("evaluation")

if evaluation is None or evaluation.empty:
    st.error("No evaluation results are available.")
    st.stop()

st.dataframe(
    evaluation_display_table(evaluation),
    hide_index=True,
    width="stretch",
)

# ============================================================
# FIDELITY / UTILITY / THROUGHPUT COMPARISON DASHBOARD
# ============================================================

st.header("6. Fidelity and Throughput Comparison")
st.caption(
    "Synthetic candidates are compared against the uploaded reference/seed dataset. "
    "Lower distance/error metrics indicate better fidelity; higher TSTR F1 indicates "
    "better downstream utility. The reference dataset is a benchmark, not a competing model."
)

comparison = evaluation.copy()
comparison["model"] = comparison["model"].astype(str)
comparison = comparison[comparison["status"].astype(str).str.upper().eq("OK")]

reference_metrics = {
    "model": "Seed / Reference (self-baseline)",
    "status": "REFERENCE",
    "ks": 0.0,
    "wasserstein": 0.0,
    "mean_std_gap": 0.0,
    "correlation_mae": 0.0,
    "categorical_tv": 0.0,
    "constraint_violation_rate": 0.0,
    "tstr_weighted_f1": np.nan,
    "composite_score": np.nan,
}

comparison_rows = [reference_metrics]
for model_name in ["Gaussian Copula", "Gemini Browser"]:
    match = comparison[comparison["model"].str.casefold().eq(model_name.casefold())]
    if not match.empty:
        comparison_rows.append(match.iloc[0].to_dict())

comparison_table = pd.DataFrame(comparison_rows)
metric_columns = [
    ("ks", "KS statistic"),
    ("wasserstein", "Wasserstein distance"),
    ("mean_std_gap", "Mean/std gap"),
    ("correlation_mae", "Correlation MAE"),
    ("categorical_tv", "Categorical TV"),
    ("constraint_violation_rate", "Constraint violation rate"),
]

# Count which synthetic candidate has the lower value on each fidelity metric.
model_rows = comparison_table[
    comparison_table["model"].isin(["Gaussian Copula", "Gemini Browser"])
]
fidelity_wins = {"Gaussian Copula": 0, "Gemini Browser": 0}
for metric, _label in metric_columns:
    if metric not in model_rows.columns:
        continue
    valid = model_rows[["model", metric]].copy()
    valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
    valid = valid.dropna()
    if len(valid) >= 2:
        winner_row = valid.loc[valid[metric].idxmin()]
        fidelity_wins[str(winner_row["model"])] += 1

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Best statistical fidelity")
        fidelity_winner = max(fidelity_wins, key=fidelity_wins.get)
        total_fidelity_comparisons = sum(fidelity_wins.values())
        if total_fidelity_comparisons > 0:
            st.metric(
                "Lower-error metrics won",
                f"{fidelity_wins[fidelity_winner]}/{total_fidelity_comparisons}",
                help="Counts metrics where the candidate has the lower distance/error value.",
            )
            st.write(f"**Candidate:** {fidelity_winner}")
        else:
            st.info("Upload both Gaussian Copula and Gemini results to compare fidelity.")
with right:
    with st.container(border=True):
        st.subheader("Composite score and downstream utility")
        gc_row = comparison[comparison["model"].str.casefold().eq("gaussian copula")]
        gem_row = comparison[comparison["model"].str.casefold().eq("gemini browser")]
        if not gc_row.empty and not gem_row.empty:
            gc_score = pd.to_numeric(gc_row.iloc[0].get("composite_score"), errors="coerce")
            gem_score = pd.to_numeric(gem_row.iloc[0].get("composite_score"), errors="coerce")
            gc_f1 = pd.to_numeric(gc_row.iloc[0].get("tstr_weighted_f1"), errors="coerce")
            gem_f1 = pd.to_numeric(gem_row.iloc[0].get("tstr_weighted_f1"), errors="coerce")
            score_cols = st.columns(2)
            score_cols[0].metric("Gaussian Copula score", f"{gc_score:.4f}" if pd.notna(gc_score) else "N/A")
            score_cols[1].metric("Gemini score", f"{gem_score:.4f}" if pd.notna(gem_score) else "N/A")
            utility_cols = st.columns(2)
            utility_cols[0].metric("Gaussian Copula TSTR F1", f"{gc_f1:.4f}" if pd.notna(gc_f1) else "N/A")
            utility_cols[1].metric("Gemini TSTR F1", f"{gem_f1:.4f}" if pd.notna(gem_f1) else "N/A")
        else:
            st.info("Gaussian Copula and Gemini Browser must both be evaluated to show their score comparison.")

st.subheader("Fidelity metrics versus the seed/reference dataset")
st.caption("For these distance/error metrics, lower is better. The seed/reference self-baseline is zero by definition.")
fidelity_chart_columns = [metric for metric, _label in metric_columns if metric in comparison_table.columns]
if fidelity_chart_columns:
    fidelity_chart = comparison_table.set_index("model")[fidelity_chart_columns].apply(pd.to_numeric, errors="coerce")
    fidelity_chart = fidelity_chart.rename(columns=dict(metric_columns))
    st.bar_chart(fidelity_chart, width="stretch")

st.subheader("Downstream utility and composite score")
st.caption("Higher is better for TSTR weighted F1 and the current composite score. Composite score is relative to the candidate pool, not an absolute fidelity measure.")
utility_columns = [column for column in ["tstr_weighted_f1", "composite_score"] if column in comparison_table.columns]
if utility_columns:
    utility_chart = comparison_table.set_index("model")[utility_columns].apply(pd.to_numeric, errors="coerce")
    utility_chart = utility_chart.rename(columns={"tstr_weighted_f1": "TSTR weighted F1", "composite_score": "Composite score"})
    st.bar_chart(utility_chart, width="stretch")

st.subheader("Candidate generation throughput")
generation_times = result.get("generation_times", {}) or {}
throughput_rows = []
for model_name, elapsed in generation_times.items():
    candidate_df = (result.get("candidates") or {}).get(model_name)
    if candidate_df is not None and elapsed and float(elapsed) > 0:
        throughput_rows.append({
            "Model": model_name,
            "Rows generated": len(candidate_df),
            "Generation time (seconds)": round(float(elapsed), 3),
            "Throughput (rows/second)": round(len(candidate_df) / float(elapsed), 2),
        })

gemini_seconds = float(result.get("gemini_elapsed_seconds", 0.0) or 0.0)
gemini_rows = int(result.get("gemini_uploaded_rows", 0) or 0)
if gemini_seconds > 0 and gemini_rows > 0:
    throughput_rows.append({
        "Model": "Gemini Browser (user-reported time)",
        "Rows generated": gemini_rows,
        "Generation time (seconds)": round(gemini_seconds, 3),
        "Throughput (rows/second)": round(gemini_rows / gemini_seconds, 2),
    })

if throughput_rows:
    throughput_table = pd.DataFrame(throughput_rows).sort_values(
        "Throughput (rows/second)", ascending=False
    )
    fastest = throughput_table.iloc[0]
    with st.container(border=True):
        st.metric(
            "Highest measured throughput",
            str(fastest["Model"]),
            f"{fastest['Throughput (rows/second)']:.2f} rows/sec",
        )
        st.dataframe(throughput_table, hide_index=True, width="stretch")
    throughput_chart = throughput_table.set_index("Model")[["Throughput (rows/second)"]]
    st.bar_chart(throughput_chart, width="stretch")
else:
    st.info(
        "Algorithmic generation time is measured automatically. To compare Gemini throughput, "
        "enter its approximate generation time above the Gemini CSV uploader and run evaluation again."
    )

st.caption(
    "Throughput means generated rows per second. Gaussian Copula timing is measured by the app; "
    "Gemini timing is user-reported and includes any waiting or manual steps, so compare it as an "
    "approximate end-to-end rate rather than a controlled benchmark. The seed dataset is a reference "
    "baseline and does not have generation throughput."
)

if (
    "Gemini Browser"
    in set(evaluation.get("model", pd.Series(dtype=str)).astype(str))
):
    st.caption(
        "Gemini Browser was evaluated as an externally generated candidate. "
        "The seed is prompt metadata only; Gemini web output is not guaranteed "
        "to be deterministic. Gemini is not eligible to become the final "
        "algorithmic generator."
    )


# ============================================================
# OFFICIAL / EXTERNAL ANNUAL STATISTICS VALIDATION
# ============================================================

st.header("7. Validation Against Annual / Official Statistics")
st.caption(
    "Upload the annual statistics CSV you want to use as the external reference. "
    "Official values and metric definitions are supplied by the user; this section does not assume a fixed port or year."
)

with st.expander("Annual metrics CSV format and mapping guide", expanded=True):
    st.markdown(
        "Required columns: **Indicator**, **Official value**. Optional columns: **Unit**, **Dataset column**, "
        "**Aggregation**, and **Scale factor**. The generated summary is calculated from the chosen dataset column, "
        "then multiplied by Scale factor to convert it into the official metric's units."
    )
    st.markdown(
        "Aggregation options: `mean`, `median`, `sum`, `count`, `min`, `max`. "
        "Example: to estimate annual cargo in million tonnes from a `cargo_volume_t` column, use `mean`, "
        "and set the scale factor to `official vessel count / 1,000,000`. Only use such extrapolations when the "
        "assumption is justified and document it in your report."
    )
    template_df = pd.DataFrame([
        {"Indicator": "Average waiting time", "Official value": 0.92, "Unit": "hours", "Dataset column": "waiting_time_h", "Aggregation": "mean", "Scale factor": 1.0},
        {"Indicator": "Average turnaround (proxy)", "Official value": 69.19, "Unit": "hours", "Dataset column": "dwell_time_h", "Aggregation": "mean", "Scale factor": 1.0},
    ])
    st.download_button(
        "Download example annual metrics CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="annual_metrics_template.csv",
        mime="text/csv",
    )

benchmark_upload = st.file_uploader(
    "Upload annual statistics / benchmark CSV",
    type=["csv"],
    key="annual_benchmark_upload",
    help="The CSV must contain Indicator and Official value. Mapping columns are optional and can be configured below.",
)

official_comparison = pd.DataFrame()
official_targets_df = pd.DataFrame()
if benchmark_upload is None:
    st.info("Upload your annual statistics CSV to calculate the LLM-seed and selected-model alignment scores dynamically.")
else:
    try:
        raw_benchmark = pd.read_csv(benchmark_upload)
        official_targets_df = normalize_benchmark_table(raw_benchmark)
        available_columns = list(dict.fromkeys(
            list(reference_df.columns) + [
                col for candidate in (result.get("candidates", {}) or {}).values()
                if isinstance(candidate, pd.DataFrame) for col in candidate.columns
            ] + (list(result.get("final_df").columns) if isinstance(result.get("final_df"), pd.DataFrame) else [])
        ))
        if not available_columns:
            available_columns = list(reference_df.columns)

        st.subheader("Map annual indicators to generated-data summaries")
        st.caption("Check the indicator-to-column mapping and scale factor. Errors and the single-value score recalculate from these inputs.")
        edited_benchmark = st.data_editor(
            official_targets_df,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            column_config={
                "Dataset column": st.column_config.SelectboxColumn(
                    "Dataset column", options=[""] + available_columns, required=False
                ),
                "Aggregation": st.column_config.SelectboxColumn(
                    "Aggregation", options=["mean", "median", "sum", "count", "min", "max"], required=True
                ),
                "Scale factor": st.column_config.NumberColumn("Scale factor", min_value=0.0, step=0.1, format="%.8f"),
                "Official value": st.column_config.NumberColumn("Official value", format="%.6f"),
            },
            key="annual_benchmark_editor",
        )
        official_targets_df = normalize_benchmark_table(edited_benchmark)
        candidate_pool_for_official = result.get("candidates", {}) or {}
        official_comparison = build_official_port_validation(
            benchmark_df=official_targets_df,
            reference_df=reference_df,
            candidates=candidate_pool_for_official,
            final_df=result.get("final_df"),
        )

        if official_comparison.empty:
            st.info("No seed or candidate datasets are available for annual-statistics comparison.")
        else:
            st.subheader("Single-value comparison against uploaded annual statistics")
            seed_rows = official_comparison[official_comparison["Dataset / Method"] == "LLM Seed / Reference"]
            final_rows = official_comparison[official_comparison["Dataset / Method"] == "Selected Final Output"]
            score_cols = st.columns(2)
            seed_score = np.nan
            final_score = np.nan
            if not seed_rows.empty:
                seed_score = pd.to_numeric(seed_rows.iloc[0]["Annual Statistics Similarity Score (%)"], errors="coerce")
                with score_cols[0]:
                    st.metric("LLM Seed vs Annual Statistics", f"{seed_score:.2f}%" if pd.notna(seed_score) else "N/A")
                    st.caption("Alignment index across the uploaded metrics that could be calculated.")
            if not final_rows.empty:
                final_score = pd.to_numeric(final_rows.iloc[0]["Annual Statistics Similarity Score (%)"], errors="coerce")
                with score_cols[1]:
                    st.metric("Best/Selected Output vs Annual Statistics", f"{final_score:.2f}%" if pd.notna(final_score) else "N/A")
                    if pd.notna(seed_score) and pd.notna(final_score):
                        delta = final_score - seed_score
                        st.caption(f"{abs(delta):.2f} percentage points {'higher' if delta >= 0 else 'lower'} than the LLM seed.")

            st.caption(
                "Score formula: 100 minus the mean of the available metric relative errors, with each error capped at 100%. "
                "This is a similarity index, not a probability or a claim of 100% accuracy. Metrics without a valid column mapping are excluded."
            )
            st.dataframe(official_comparison.round(4), hide_index=True, width="stretch")

            error_columns = [c for c in official_comparison.columns if c.endswith("| Relative error (%)")]
            if error_columns:
                chart_df = official_comparison.set_index("Dataset / Method")[error_columns].apply(pd.to_numeric, errors="coerce")
                st.subheader("Relative error by annual indicator")
                st.bar_chart(chart_df, width="stretch")

            st.download_button(
                "Download annual-statistics comparison (CSV)",
                data=official_comparison.to_csv(index=False).encode("utf-8"),
                file_name="annual_statistics_comparison.csv",
                mime="text/csv",
                width="stretch",
            )
    except Exception as exc:
        st.error(f"Could not read or compare the annual metrics CSV: {exc}")
        st.caption("Check that the CSV has columns named Indicator and Official value, or equivalent common names.")


generation_errors = result.get("generation_errors", {})

if generation_errors:
    with st.expander("Candidate generation diagnostics"):
        for model_name, error in generation_errors.items():
            st.warning(f"**{model_name}:** {error}")


# ============================================================
# WINNER
# ============================================================

winner = result.get("winner")

st.header("8. Adaptive Model Selection")

if winner:
    st.success(f"Selected algorithm: {winner}")

    st.caption(
        "The LLM and browser-assisted Gemini candidates are excluded from "
        "final algorithm selection. The selected algorithm is the best-scoring "
        "successful algorithmic candidate in the evaluation results."
    )
else:
    st.error(
        "No algorithmic winner could be selected. "
        "Review the evaluation results and generation diagnostics."
    )


# ============================================================
# FINAL DATASET
# ============================================================

st.header("9. Final Synthetic Dataset")

final_df = result.get("final_df")
final_error = result.get("final_error")

if final_df is None:
    st.error(
        "A final synthetic dataset is not available. "
        + str(final_error or "Unknown generation error.")
    )

    final_errors = result.get("final_generation_errors", {})

    if final_errors:
        with st.expander("Final generation diagnostics"):
            st.json(final_errors)

    st.stop()

validation = validate_final_dataset(
    reference_df,
    final_df,
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Generated Rows", f"{len(final_df):,}")
m2.metric("Columns", len(final_df.columns))
m3.metric("Missing Values", f"{validation['missing_values']:,}")
m4.metric(
    "Schema Match",
    "Yes" if validation["schema_match"] else "No",
)

if validation["structurally_valid"]:
    st.success("The generated dataset passed structural validation.")
else:
    st.warning(
        "The generated dataset has structural issues. "
        "Review the validation details before using it."
    )

if validation["numeric_bound_violations"] > 0:
    st.warning(
        f"{validation['numeric_bound_violations']:,} numeric values "
        "fall outside their corresponding reference-data ranges."
    )

with st.expander("Preview final synthetic data", expanded=True):
    st.dataframe(
        final_df.head(50),
        width="stretch",
    )

with st.expander("Final dataset validation details"):
    st.json(validation)


# ============================================================
# DOWNLOADS
# ============================================================

st.header("10. Download Results")

csv_bytes = dataframe_to_csv_bytes(final_df)

st.download_button(
    label="⬇️ Download Final Synthetic Dataset (CSV)",
    data=csv_bytes,
    file_name="adaptive_synthetic_dataset.csv",
    mime="text/csv",
    width="stretch",
)

try:
    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl",
    ) as writer:

        final_df.to_excel(
            writer,
            index=False,
            sheet_name="Synthetic Data",
        )

        evaluation.to_excel(
            writer,
            index=False,
            sheet_name="Model Evaluation",
        )

        if "official_comparison" in globals() and isinstance(official_comparison, pd.DataFrame):
            official_comparison.to_excel(
                writer,
                index=False,
                sheet_name="Official Port Comparison",
            )

        if "official_targets_df" in globals() and isinstance(official_targets_df, pd.DataFrame):
            official_targets_df.to_excel(
                writer,
                index=False,
                sheet_name="Official Port Targets",
            )

        pd.DataFrame(
            [validation]
        ).to_excel(
            writer,
            index=False,
            sheet_name="Validation",
        )

    st.download_button(
        label="⬇️ Download Complete Results (Excel)",
        data=excel_buffer.getvalue(),
        file_name="adaptive_synthetic_results.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        width="stretch",
    )

except Exception as exc:
    st.info(
        "Excel download is unavailable. CSV download remains available. "
        f"Details: {exc}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Synthetic data generation does not by itself guarantee privacy, "
    "statistical equivalence or downstream predictive utility. "
    "Review evaluation metrics and validation results before use."
)
