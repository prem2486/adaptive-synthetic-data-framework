
import pandas as pd
import pytest

from engine.profiling import profile_raw_dataset


def test_profile_dynamic_dataset():
    data = pd.DataFrame({
        "waiting_time_h": [2.0, 4.0, None],
        "queue": ["Low", "High", "Low"],
        "is_congested": [False, True, False],
    })

    original = data.copy(deep=True)

    profile = profile_raw_dataset(data)

    summary = profile["dataset_summary"]

    assert summary["row_count"] == 3
    assert summary["column_count"] == 3
    assert summary["missing_cells"] == 1

    assert (
        profile["columns"]["waiting_time_h"]["detected_type"]
        == "numeric"
    )

    assert (
        profile["columns"]["queue"]["detected_type"]
        == "categorical"
    )

    assert (
        profile["columns"]["is_congested"]["detected_type"]
        == "boolean"
    )

    assert profile["columns"]["queue"]["unique_count"] == 2

    # Profiling must not modify the input.
    pd.testing.assert_frame_equal(data, original)


def test_numeric_statistics():
    data = pd.DataFrame({
        "value": [10.0, 20.0, 30.0]
    })

    profile = profile_raw_dataset(data)
    stats = profile["columns"]["value"]["statistics"]

    assert stats["mean"] == 20.0
    assert stats["min"] == 10.0
    assert stats["median"] == 20.0
    assert stats["max"] == 30.0


def test_duplicate_rows():
    data = pd.DataFrame({
        "value": [1, 1, 2]
    })

    profile = profile_raw_dataset(data)

    assert profile["dataset_summary"]["duplicate_rows"] == 1


def test_invalid_input():
    with pytest.raises(ValueError):
        profile_raw_dataset(pd.DataFrame())

    with pytest.raises(ValueError):
        profile_raw_dataset(
            pd.DataFrame({"value": [1, 2]}),
            top_n=0,
        )

    with pytest.raises(TypeError):
        profile_raw_dataset([1, 2, 3])
