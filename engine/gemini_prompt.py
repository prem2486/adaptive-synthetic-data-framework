
"""Build prompts for browser-assisted Gemini dataset generation.

This module does not call Gemini or require an API key.
It prepares instructions for the user to submit through Gemini's website.
"""

import json


def build_gemini_browser_prompt(
    profile,
    n_rows,
    seed=42,
    domain_context="",
    government_context="",
    user_summary="",
):
    """Create a prompt from an approved statistical profile."""

    if not isinstance(profile, dict):
        raise TypeError("profile must be a dictionary.")

    if not profile.get("all_columns"):
        raise ValueError("The profile must contain all_columns.")

    n_rows = int(n_rows)

    if n_rows < 1:
        raise ValueError("n_rows must be at least 1.")

    numeric_cols = profile.get("numeric_cols", []) or []
    categorical_cols = profile.get("categorical_cols", []) or []
    means = profile.get("means", {}) or {}
    stds = profile.get("stds", {}) or {}
    bounds = profile.get("bounds", {}) or {}
    categories = profile.get("categories", {}) or {}
    probabilities = profile.get("category_probabilities", {}) or {}
    provenance = profile.get("provenance", {}) or {}

    schema = []

    for name in profile["all_columns"]:
        details = {
            "name": name,
            "type": (
                "numeric"
                if name in numeric_cols
                else "categorical"
                if name in categorical_cols
                else "unknown"
            ),
        }

        if name in numeric_cols:
            details["mean_or_value"] = means.get(name)
            details["standard_deviation"] = stds.get(name)
            details["allowed_range"] = bounds.get(name)

        if name in categorical_cols:
            details["allowed_categories"] = categories.get(name, [])
            details["category_probabilities"] = probabilities.get(name, {})

        if name in provenance:
            details["metadata"] = provenance[name]

        schema.append(details)

    statistical_profile = {
        "domain": profile.get("domain", ""),
        "columns": schema,
        "numeric_correlations": profile.get("corr_matrix", {}),
        "target_column": (
            profile.get("target_column")
            or profile.get("target")
        ),
        "notes": profile.get("notes", ""),
    }

    profile_json = json.dumps(
        statistical_profile,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    return f"""
You are generating a synthetic tabular reference dataset for a
research project on adaptive synthetic dataset generation.

TASK
Create exactly {n_rows} synthetic data rows.

REPRODUCIBILITY INSTRUCTION
Use random seed {int(seed)} when initializing your data-generation
code or random-number generator. Treat this as a reproducibility
instruction; do not claim that the website guarantees identical
results across separate runs.

DOMAIN
{domain_context or profile.get("domain", "Not specified")}

ADDITIONAL USER SUMMARY
{user_summary or "None supplied"}

GOVERNMENT / PUBLIC SOURCE CONTEXT
{government_context or "None supplied"}

APPROVED SCHEMA AND STATISTICAL PROFILE
{profile_json}

REQUIREMENTS
1. Preserve every approved column name exactly, including spelling
   and capitalization.
2. Include every approved column, in the supplied order.
3. Generate exactly {n_rows} data rows, excluding the header.
4. Respect numeric ranges, units, category lists and supplied
   statistical constraints as closely as possible.
5. Preserve plausible relationships between related attributes.
6. Avoid impossible values and contradictory combinations.
7. Do not invent government statistics or describe estimates as
   verified observations.
8. Generate synthetic records, not copied real people's records.
9. Do not add an ID, index, explanation or extra column unless it
   exists in the approved schema.
10. Use the supplied seed in the generation code where possible.

OUTPUT
Use Gemini's data-analysis or code-execution capability to create
a downloadable CSV file named gemini_reference_seed_{int(seed)}.csv.

The CSV must contain the header and data rows only. Ensure the file
can be downloaded. Do not substitute a Markdown table for the CSV.

After creating the file, briefly report the row count, column count,
and filename. Do not claim the dataset passed statistical validation;
the application will perform that evaluation separately.
""".strip()
