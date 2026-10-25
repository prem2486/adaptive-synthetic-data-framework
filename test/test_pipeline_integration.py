import pandas as pd

from engine.engine import run_pipeline


def test_pipeline_with_gmm_only():
    reference_df = pd.DataFrame({
        "age": [21, 25, 30, 35, 40, 28, 32, 45, 27, 38],
        "salary": [
            25000, 30000, 45000, 40000, 35000,
            32000, 42000, 50000, 28000, 46000,
        ],
        "department": [
            "IT", "HR", "IT", "Sales", "HR",
            "Sales", "IT", "Sales", "HR", "IT",
        ],
    })

    result = run_pipeline(
        reference_df=reference_df,
        n_samples=20,
        methods=["GMM"],
        include_llm=False,
        seed=42,
        gmm_components=2,
    )

    assert "profile" in result
    assert "candidates" in result
    assert "evaluation" in result
    assert "winner" in result
    assert "generation_errors" in result

    assert "GMM" in result["candidates"]
    assert len(result["candidates"]["GMM"]) == 20
    assert list(result["candidates"]["GMM"].columns) == list(
        reference_df.columns
    )

    assert not result["evaluation"].empty
    assert result["winner"] == "GMM"

    print("Winner:", result["winner"])
    print(result["evaluation"])
    print("PIPELINE INTEGRATION TEST PASSED")