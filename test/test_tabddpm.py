
import pandas as pd
import pytest

from engine.tabddpm import (
    generate_tabddpm,
    generate_tabddpm_engine,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [22, 35, 41, 28, 50, 31],
            "income": [25000.0, 42000.0, 51000.0, 32000.0, 68000.0, 39000.0],
            "category": ["A", "B", "A", "C", "B", "A"],
        }
    )


def test_rejects_non_dataframe():
    with pytest.raises(TypeError):
        generate_tabddpm([1, 2, 3])


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        generate_tabddpm(pd.DataFrame())


def test_rejects_all_missing_column():
    df = pd.DataFrame({"age": [20, 30], "empty": [None, None]})
    with pytest.raises(ValueError, match="entirely missing"):
        generate_tabddpm(df)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 0},
        {"n_samples": -1},
        {"n_samples": True},
        {"n_iter": 0},
        {"batch_size": 0},
    ],
)
def test_rejects_invalid_positive_parameters(sample_df, kwargs):
    with pytest.raises(ValueError):
        generate_tabddpm(sample_df, **kwargs)


def test_rejects_invalid_seed(sample_df):
    with pytest.raises(ValueError, match="seed"):
        generate_tabddpm(sample_df, seed=1.5)


def test_rejects_invalid_device(sample_df):
    with pytest.raises(ValueError, match="device"):
        generate_tabddpm(sample_df, device="tpu")


def test_engine_requires_dataframe():
    with pytest.raises(ValueError, match="reference_df"):
        generate_tabddpm_engine()
