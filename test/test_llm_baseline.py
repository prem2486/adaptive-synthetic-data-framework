
import os

import pandas as pd
import pytest

from engine.llm_baseline import generate_llm_baseline_engine


@pytest.mark.live
def test_live_gemini_generation():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        pytest.skip("Set GEMINI_API_KEY before running this live test.")

    df = pd.DataFrame({
        "age": [21, 25, 30, 35, 40, 28],
        "department": ["IT", "HR", "IT", "Sales", "HR", "Sales"],
        "salary": [25000, 30000, 45000, 40000, 35000, 32000],
    })

    result = generate_llm_baseline_engine(
        reference_df=df,
        n_samples=5,
        batch_size=1,
        gemini_api_key=api_key,
        provider="gemini",
        seed=42,
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        max_retries=1,
    )

    assert result.shape == (5, 3)
    assert list(result.columns) == list(df.columns)
    assert result["age"].notna().all()
    assert result["salary"].notna().all()
    assert result["department"].notna().all()

    print(result)
