
import pandas as pd
import pytest

from engine.preprocessing import (
    prepare_seed_dataset,
    restore_original_schema,
)


def test_prepare_dynamic_dataset():
    seed = pd.DataFrame({
        "waiting_time_h": [2.0, None, 5.0],
        "queue": ["Low", "High", "Low"],
        "is_congested": [False, True, False],
    })

    original = seed.copy(deep=True)

    prepared, schema = prepare_seed_dataset(seed)

    assert prepared.shape == seed.shape
    assert prepared.isna().sum().sum() == 0
    assert schema["columns"] == list(seed.columns)
    assert schema["column_types"]["waiting_time_h"] == "numeric"
    assert schema["column_types"]["queue"] == "categorical"
    assert schema["column_types"]["is_congested"] == "boolean"

    # Preprocessing must not modify the uploaded dataset.
    pd.testing.assert_frame_equal(seed, original)


def test_restore_schema():
    seed = pd.DataFrame({
        "waiting_time_h": [2.0, 5.0],
        "queue": ["Low", "High"],
    })

    _, schema = prepare_seed_dataset(seed)

    generated = pd.DataFrame({
        "waiting_time_h": [3.2, 4.8],
        "queue": ["High", "Low"],
        "extra_column": [100, 200],
    })

    restored = restore_original_schema(generated, schema)

    assert list(restored.columns) == list(seed.columns)
    assert len(restored) == 2
    assert "extra_column" not in restored.columns


def test_invalid_input():
    with pytest.raises(ValueError):
        prepare_seed_dataset(pd.DataFrame())

    with pytest.raises(ValueError):
        prepare_seed_dataset(
            pd.DataFrame({"value": [1, 2]}),
            target_column="missing_target",
        )


def test_all_missing_column():
    seed = pd.DataFrame({"value": [None, None]})

    with pytest.raises(ValueError, match="only missing values"):
        prepare_seed_dataset(seed)
