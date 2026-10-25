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

Fixes applied:
- to_arrow_safe() normalizes mixed-type object columns before every
  st.dataframe() call, eliminating pyarrow ArrowTypeError.
- Object columns are coerced to pandas string dtype at upload time.
- Engine winner is authoritative; local selection is only a fallback.
- use_container_width=True replaces width="stretch" for Streamlit 1.40.0.
- No st.expander inside st.status (fixes StreamlitAPIException).
- Section 7 shows generation diagnostics for all selected models,
  including those that failed silently.
"""

from __future__ import annotations

import os

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
    ALGORITHMIC_METHODS,
    generate_candidates,
    profile_reference_dataset,
    recommend_generation_methods,
    run_pipeline,
)

from engine.gemini_prompt import build_gemini_browser_prompt


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Universal Adaptive Synthetic Dataset Generator"

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

st.session_state.setdefault("browser_gemini_df", None)
st.session_state.setdefault("browser_gemini_filename", None)
st.session_state.setdefault("browser_gemini_prompt", "")
st.session_state.setdefault("browser_gemini_prompt_signature", None)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def to_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe for Arrow serialization (Streamlit st.dataframe)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    safe = df.copy()

    for column in safe.columns:
        series = safe[column]

        if (
            pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_datetime64_any_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue

        try:
            safe[column] = series.astype("string")
        except (TypeError, ValueError):
            safe[column] = series.map(
                lambda v: None if pd.isna(v) else str(v)
            ).astype("string")

    return safe


def read_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV or Excel dataset and normalize object columns."""
    filename = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(content))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError(
            "Unsupported file type. Upload a CSV or Excel dataset."
        )

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]):
            try:
                df[column] = df[column].astype("string")
            except (TypeError, ValueError):
                df[column] = df[column].map(
                    lambda v: None if pd.isna(v) else str(v)
                ).astype("string")

    return df


def infer_target_column(df: pd.DataFrame):
    """Suggest a likely classification target column."""
    preferred_names = [
        "target", "label", "class", "outcome",
        "response", "churn", "default", "status",
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
        max_classes = min(20, max(2, int(len(series) * 0.20)))

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

    for column in reference_df.select_dtypes(include="number").columns:
        if column not in synthetic_df.columns:
            continue

        reference_values = pd.to_numeric(
            reference_df[column], errors="coerce",
        )
        synthetic_values = pd.to_numeric(
            synthetic_df[column], errors="coerce",
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
        list(reference_df.columns) == list(synthetic_df.columns)
    )

    return {
        "generated": True,
        "rows": len(synthetic_df),
        "columns": len(synthetic_df.columns),
        "requested_columns": len(reference_df.columns),
        "schema_match": schema_match,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "missing_values": int(synthetic_df.isna().sum().sum()),
        "numeric_bound_violations": numeric_violations,
        "structurally_valid": (
            len(synthetic_df) > 0
            and not missing_columns
            and not extra_columns
            and schema_match
        ),
    }


def select_algorithmic_winner(evaluation: pd.DataFrame) -> Optional[str]:
    """Select the best successful algorithmic candidate."""
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
        .astype(str).str.strip().str.lower()
        .isin(LLM_NAMES)
    ]

    if "composite_score" not in candidates.columns:
        return None

    candidates["composite_score"] = pd.to_numeric(
        candidates["composite_score"], errors="coerce",
    )

    candidates = candidates.dropna(subset=["composite_score"])

    if candidates.empty:
        return None

    best = candidates.sort_values(
        "composite_score", ascending=False,
    ).iloc[0]

    return str(best["model"])


def evaluation_display_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        "model", "status", "ks", "wasserstein", "mean_std_gap",
        "correlation_mae", "tstr_weighted_f1", "categorical_tv",
        "constraint_violation_rate", "composite_score",
    ]

    available = [
        column for column in preferred_columns
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
                table[column], errors="coerce",
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
file_signature = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.get("file_signature") != file_signature:
    try:
        reference_df = read_uploaded_dataset(uploaded_file)

        st.session_state["reference_df"] = reference_df
        st.session_state["file_signature"] = file_signature
        st.session_state["pipeline_result"] = None
        st.session_state["result_signature"] = None

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
        include=["object", "category", "bool", "string"]
    ).columns
)

missing_count = int(reference_df.isna().sum().sum())

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{len(reference_df):,}")
c2.metric("Columns", f"{len(reference_df.columns):,}")
c3.metric("Numeric Columns", numeric_count)
c4.metric("Missing Values", f"{missing_count:,}")

with st.expander("Preview uploaded dataset"):
    st.dataframe(
        to_arrow_safe(reference_df.head(20)),
        use_container_width=True,
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
    to_arrow_safe(pd.DataFrame(profile_records)),
    hide_index=True,
    use_container_width=True,
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
        "None — skip TSTR" if value is None else str(value)
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

try:
    profile = profile_reference_dataset(reference_df, target_col=target_col)
except Exception as exc:
    st.error(f"Dataset profiling failed: {exc}")
    st.stop()


# ============================================================
# GEMINI BROWSER-ASSISTED GENERATION
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
    use_container_width=True,
)

st.write(
    "Prepare a schema-aware prompt, open Gemini in a separate browser tab, "
    "sign in with your Google account, generate a CSV, and upload it here. "
    "This application does not collect or store your Google password."
)

with st.expander("Prepare prompt and upload Gemini output", expanded=True):
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

    if st.button("📝 Prepare Gemini Prompt", key="prepare_browser_gemini_prompt"):
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

            st.success("Prompt prepared. Copy it and paste it into Gemini.")

        except Exception as exc:
            st.error(f"Could not prepare the Gemini prompt: {exc}")

    gemini_prompt = st.session_state.get("browser_gemini_prompt", "")

    current_prompt_signature = (file_signature, gemini_rows, gemini_seed)

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

        st.link_button("🔓 Open Gemini", "https://gemini.google.com/")

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
            gemini_df = pd.read_csv(io.BytesIO(gemini_upload.getvalue()))

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
                    gemini_df = gemini_df[expected_columns].copy()
                    conversion_errors = []

                    for column in reference_df.select_dtypes(
                        include="number"
                    ).columns:
                        before = gemini_df[column].notna()
                        converted = pd.to_numeric(
                            gemini_df[column], errors="coerce",
                        )
                        bad_values = int((before & converted.isna()).sum())
                        if bad_values:
                            conversion_errors.append(
                                f"{column}: {bad_values} non-numeric value(s)"
                            )
                        gemini_df[column] = converted

                    for column in gemini_df.columns:
                        if pd.api.types.is_object_dtype(gemini_df[column]):
                            try:
                                gemini_df[column] = (
                                    gemini_df[column].astype("string")
                                )
                            except (TypeError, ValueError):
                                gemini_df[column] = gemini_df[column].map(
                                    lambda v: (
                                        None if pd.isna(v) else str(v)
                                    )
                                ).astype("string")

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
                            to_arrow_safe(gemini_df.head(10)),
                            hide_index=True,
                            use_container_width=True,
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
            to_arrow_safe(pd.DataFrame([
                {"Model": model, "Reason": reason}
                for model, reason in auto_skipped_models.items()
            ])),
            hide_index=True,
            use_container_width=True,
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
        str(target_col) if target_col is not None else "Disabled"
    ),
    "LLM API generation": "Disabled",
    "Gemini browser dataset": (
        "Uploaded — included in evaluation"
        if isinstance(st.session_state.get("browser_gemini_df"), pd.DataFrame)
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
    use_container_width=True,
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
            result["requested_algorithms"] = list(selected_algorithms)

            evaluation = result.get("evaluation")

            if evaluation is None or evaluation.empty:
                raise RuntimeError(
                    "No candidate evaluation results were produced. "
                    "Check the generation diagnostics in Section 7."
                )

            st.write("Step 4: Selecting the best algorithmic candidate.")

            winner = result.get("winner") or select_algorithmic_winner(
                evaluation
            )

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
                    getattr(generate_candidates, "last_errors", {})
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
    to_arrow_safe(evaluation_display_table(evaluation)),
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# UNIVERSAL FIDELITY / UTILITY / THROUGHPUT DASHBOARD
# ============================================================

st.header("6. Fidelity, Utility and Throughput Comparison")
st.caption(
    "All successful candidates are compared against the uploaded reference dataset. "
    "Lower distance/error metrics are better; higher TSTR weighted F1 and composite "
    "score are better. TSTR is only meaningful when a suitable target is selected."
)

comparison = evaluation.copy()
if "model" not in comparison.columns:
    st.warning("Evaluation output does not contain a model column.")
    comparison = pd.DataFrame()
else:
    comparison["model"] = comparison["model"].astype(str)
    if "status" in comparison.columns:
        comparison = comparison[
            comparison["status"].astype(str).str.upper().eq("OK")
        ]

if comparison.empty:
    st.info("No successful candidates are available for comparison.")
else:
    if "composite_score" in comparison.columns:
        comparison["composite_score"] = pd.to_numeric(
            comparison["composite_score"], errors="coerce"
        )
        ranked = comparison.sort_values(
            "composite_score", ascending=False, na_position="last"
        )
    else:
        ranked = comparison.copy()

    display_cols = [
        c for c in [
            "model", "status", "ks", "wasserstein", "mean_std_gap",
            "correlation_mae", "categorical_tv", "constraint_violation_rate",
            "tstr_weighted_f1", "composite_score",
        ] if c in ranked.columns
    ]

    st.subheader("All successful candidates — ranked by composite score")
    ranked_display = ranked[display_cols].copy()
    ranked_display = ranked_display.rename(columns={
        "model": "Model", "status": "Status", "ks": "KS statistic",
        "wasserstein": "Wasserstein distance", "mean_std_gap": "Mean/std gap",
        "correlation_mae": "Correlation MAE", "categorical_tv": "Categorical TV",
        "constraint_violation_rate": "Constraint violation rate",
        "tstr_weighted_f1": "TSTR weighted F1", "composite_score": "Composite score",
    })
    for col in ranked_display.columns:
        if col not in {"Model", "Status"}:
            ranked_display[col] = pd.to_numeric(
                ranked_display[col], errors="coerce"
            ).round(4)

    st.dataframe(
        to_arrow_safe(ranked_display),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Best candidate by metric")
    metric_specs = [
        ("ks", "KS statistic", "min"),
        ("wasserstein", "Wasserstein distance", "min"),
        ("mean_std_gap", "Mean/std gap", "min"),
        ("correlation_mae", "Correlation MAE", "min"),
        ("categorical_tv", "Categorical TV", "min"),
        ("constraint_violation_rate", "Constraint violations", "min"),
        ("tstr_weighted_f1", "TSTR weighted F1", "max"),
        ("composite_score", "Composite score", "max"),
    ]
    leader_rows = []
    for metric, label, direction in metric_specs:
        if metric not in comparison.columns:
            continue
        vals = pd.to_numeric(comparison[metric], errors="coerce")
        valid = vals.dropna()
        if valid.empty:
            continue
        idx = valid.idxmin() if direction == "min" else valid.idxmax()
        leader_rows.append({
            "Metric": label,
            "Best candidate": str(comparison.loc[idx, "model"]),
            "Value": round(float(vals.loc[idx]), 4),
            "Direction": (
                "Lower is better" if direction == "min"
                else "Higher is better"
            ),
        })
    if leader_rows:
        st.dataframe(
            to_arrow_safe(pd.DataFrame(leader_rows)),
            hide_index=True,
            use_container_width=True,
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Statistical fidelity")
        lower_metrics = [
            m for m, _, d in metric_specs
            if d == "min" and m in comparison.columns
        ]
        if lower_metrics:
            chart = comparison.set_index("model")[lower_metrics].apply(
                pd.to_numeric, errors="coerce"
            )
            chart = chart.rename(
                columns={m: label for m, label, _ in metric_specs}
            )
            st.bar_chart(chart, use_container_width=True)
        else:
            st.info("No numeric fidelity metrics are available.")
    with right:
        st.subheader("Utility and composite score")
        higher_metrics = [
            m for m in ["tstr_weighted_f1", "composite_score"]
            if m in comparison.columns
        ]
        if higher_metrics:
            chart = comparison.set_index("model")[higher_metrics].apply(
                pd.to_numeric, errors="coerce"
            )
            chart = chart.rename(columns={
                "tstr_weighted_f1": "TSTR weighted F1",
                "composite_score": "Composite score",
            })
            st.bar_chart(chart, use_container_width=True)
        else:
            st.info("No utility/composite metrics are available.")

    st.caption(
        "Composite score is a relative ranking produced by the evaluation engine; "
        "it is not an absolute guarantee of fidelity. Gemini Browser is shown as a "
        "candidate when uploaded, but remains excluded from final algorithmic selection."
    )

    st.subheader("Candidate generation throughput")
    generation_times = result.get("generation_times", {}) or {}
    throughput_rows = []

    for model_name, elapsed in generation_times.items():
        candidate_df = (result.get("candidates") or {}).get(model_name)
        try:
            seconds = float(elapsed)
        except (TypeError, ValueError):
            seconds = 0.0

        if candidate_df is not None and seconds > 0:
            throughput_rows.append({
                "Model": model_name,
                "Rows generated": len(candidate_df),
                "Generation time (seconds)": round(seconds, 3),
                "Throughput (rows/second)": round(
                    len(candidate_df) / seconds, 2
                ),
            })

    gemini_seconds = float(
        result.get("gemini_elapsed_seconds", 0.0) or 0.0
    )
    gemini_count = int(result.get("gemini_uploaded_rows", 0) or 0)

    if gemini_seconds > 0 and gemini_count > 0:
        throughput_rows.append({
            "Model": "Gemini Browser (user-reported time)",
            "Rows generated": gemini_count,
            "Generation time (seconds)": round(gemini_seconds, 3),
            "Throughput (rows/second)": round(
                gemini_count / gemini_seconds, 2
            ),
        })

    if throughput_rows:
        throughput_table = pd.DataFrame(throughput_rows).sort_values(
            "Throughput (rows/second)", ascending=False
        )
        st.dataframe(
            to_arrow_safe(throughput_table),
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(
            throughput_table.set_index("Model")[["Throughput (rows/second)"]],
            use_container_width=True,
        )
    else:
        st.info(
            "Throughput is unavailable until generation timings are recorded."
        )


# ============================================================
# GENERATION DIAGNOSTICS  ← failures shown here, at top level
# ============================================================

st.header("7. Generation Diagnostics")

requested = result.get("requested_algorithms", [])
gen_errors = result.get("generation_errors", {}) or {}
gen_times = result.get("generation_times", {}) or {}
candidates_dict = result.get("candidates", {}) or {}
selection_report = result.get("selection_report", {}) or {}

algorithmic_requested = [
    m for m in requested if m in ALGORITHMIC_METHODS
]

succeeded = [
    m for m in algorithmic_requested if m in candidates_dict
]
failed = [
    m for m in algorithmic_requested if m in gen_errors
]

d1, d2, d3 = st.columns(3)
d1.metric("Requested", len(algorithmic_requested))
d2.metric("Succeeded", len(succeeded))
d3.metric("Failed", len(failed))

st.write(
    "**Requested algorithms:** "
    + (", ".join(algorithmic_requested) or "None")
)

if succeeded:
    st.success("**Succeeded:** " + ", ".join(sorted(succeeded)))

if failed:
    st.error("**Failed:** " + ", ".join(sorted(failed)))

    with st.expander("Failure details", expanded=True):
        for model in sorted(failed):
            st.markdown(f"### ❌ {model}")
            st.code(gen_errors[model], language="text")

if selection_report.get("skipped"):
    st.warning("**Automatically skipped by suitability screening**")
    with st.expander("Skipped model details"):
        st.json(selection_report["skipped"])

if selection_report.get("fallback"):
    st.info("**Fallback was used**")
    with st.expander("Fallback details"):
        st.json(selection_report["fallback"])

if gen_times:
    with st.expander("Generation timings (seconds)", expanded=False):
        timing_table = pd.DataFrame([
            {"Model": model, "Seconds": round(seconds, 3)}
            for model, seconds in sorted(gen_times.items())
        ])
        st.dataframe(
            to_arrow_safe(timing_table),
            hide_index=True,
            use_container_width=True,
        )


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

validation = validate_final_dataset(reference_df, final_df)

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
        to_arrow_safe(final_df.head(50)),
        use_container_width=True,
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
    use_container_width=True,
)

try:
    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:

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

        pd.DataFrame([validation]).to_excel(
            writer,
            index=False,
            sheet_name="Validation",
        )

        diag_records = []
        for model in algorithmic_requested:
            status = (
                "OK" if model in candidates_dict
                else "FAILED" if model in gen_errors
                else "NOT RUN"
            )
            diag_records.append({
                "Model": model,
                "Status": status,
                "Error": gen_errors.get(model, ""),
                "Seconds": round(float(gen_times.get(model, 0.0)), 3),
            })

        if diag_records:
            pd.DataFrame(diag_records).to_excel(
                writer,
                index=False,
                sheet_name="Generation Diagnostics",
            )

    st.download_button(
        label="⬇️ Download Complete Results (Excel)",
        data=excel_buffer.getvalue(),
        file_name="adaptive_synthetic_results.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
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