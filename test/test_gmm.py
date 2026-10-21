
import pandas as pd
import pytest

from engine.gmm import generate_gmm, generate_gmm_engine


@pytest.fixture
def seed_data():
    return pd.DataFrame({
        "waiting_time_h": [2.0, 3.0, 5.0, 7.0, 8.0, 9.0],
        "berth_time_h": [4.0, 6.0, 10.0, 14.0, 16.0, 18.0],
        "queue": ["Low", "Low", "Medium", "High", "High", "Medium"],
        "is_congested": [False, False, True, True, True, True],
    })


def test_generate_requested_sample_size(seed_data):
    generated = generate_gmm(
        seed_data,
        n_samples=100,
        seed=42,
    )

    assert len(generated) == 100
    assert list(generated.columns) == list(seed_data.columns)


def test_reproducibility(seed_data):
    first = generate_gmm(
        seed_data,
        n_samples=30,
        seed=42,
    )

    second = generate_gmm(
        seed_data,
        n_samples=30,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_output_categories_and_boolean(seed_data):
    generated = generate_gmm(
        seed_data,
        n_samples=50,
        seed=42,
    )

    assert set(generated["queue"]).issubset(
        set(seed_data["queue"])
    )

    assert set(generated["is_congested"].unique()).issubset(
        {True, False}
    )


def test_constant_numeric_column():
    data = pd.DataFrame({
        "constant": [5.0, 5.0, 5.0, 5.0],
        "value": [1.0, 2.0, 3.0, 4.0],
    })

    generated = generate_gmm(
        data,
        n_samples=25,
        seed=42,
    )

    assert (generated["constant"] == 5.0).all()


def test_invalid_sample_size(seed_data):
    with pytest.raises(ValueError):
        generate_gmm(seed_data, n_samples=0)


def test_missing_values_rejected():
    data = pd.DataFrame({
        "value": [1.0, None, 3.0]
    })

    with pytest.raises(ValueError, match="missing values"):
        generate_gmm(data, n_samples=10)


def test_reference_data_not_modified(seed_data):
    original = seed_data.copy(deep=True)

    generate_gmm(seed_data, n_samples=20)

    pd.testing.assert_frame_equal(seed_data, original)


def test_engine_compatibility_wrapper(seed_data):
    generated = generate_gmm_engine(
        reference_df=seed_data,
        n_samples=15,
        seed=42,
    )

    assert generated.shape == (15, len(seed_data.columns))


def test_missing_reference_rejected():
    with pytest.raises(ValueError, match="reference DataFrame"):
        generate_gmm_engine(n_samples=10)
