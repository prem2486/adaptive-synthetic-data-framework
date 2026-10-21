
import numpy as np
import pandas as pd
import pytest

from engine.ctgan import generate_ctgan, generate_ctgan_engine


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(123)

    size = 40
    age = rng.integers(20, 65, size=size)

    return pd.DataFrame({
        "age": age,
        "income": age * 1200 + rng.normal(0, 5000, size=size),
        "department": rng.choice(
            ["Production", "Quality", "Maintenance"],
            size=size,
        ),
        "active": rng.choice([True, False], size=size),
    })


def test_generate_requested_sample_size(sample_data):
    result = generate_ctgan(
        sample_data, n_samples=12, seed=10, epochs=2
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 12


def test_output_schema_and_column_order(sample_data):
    result = generate_ctgan(
        sample_data, n_samples=10, seed=11, epochs=2
    )

    assert list(result.columns) == list(sample_data.columns)
    assert len(result) == 10


def test_categorical_values_are_valid(sample_data):
    result = generate_ctgan(
        sample_data, n_samples=10, seed=12, epochs=2
    )

    assert set(result["department"].dropna()).issubset(
        set(sample_data["department"])
    )


def test_reproducibility(sample_data):
    first = generate_ctgan(
        sample_data, n_samples=8, seed=25, epochs=2
    )
    second = generate_ctgan(
        sample_data, n_samples=8, seed=25, epochs=2
    )

    pd.testing.assert_frame_equal(first, second)


def test_invalid_sample_size(sample_data):
    with pytest.raises(ValueError):
        generate_ctgan(sample_data, n_samples=0, epochs=2)


def test_missing_values_rejected(sample_data):
    data = sample_data.copy()
    data.loc[0, "income"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        generate_ctgan(data, n_samples=5, epochs=2)


def test_reference_data_not_modified(sample_data):
    original = sample_data.copy(deep=True)

    generate_ctgan(sample_data, n_samples=5, seed=13, epochs=2)

    pd.testing.assert_frame_equal(sample_data, original)


def test_engine_compatibility_wrapper(sample_data):
    result = generate_ctgan_engine(
        profile=sample_data,
        n_samples=6,
        seed=14,
        epochs=2,
    )

    assert len(result) == 6
    assert list(result.columns) == list(sample_data.columns)


def test_missing_reference_rejected():
    with pytest.raises(ValueError, match="reference DataFrame is required"):
        generate_ctgan_engine(n_samples=5, epochs=2)
