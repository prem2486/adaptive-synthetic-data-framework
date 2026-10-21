
import pandas as pd
import pytest

from engine.statistics import (
    calculate_categorical_tv,
    calculate_correlation_mae,
    calculate_ks_statistic,
    calculate_mean_std_gap,
    calculate_wasserstein_distance,
    evaluate_synthetic_data,
)


@pytest.fixture
def reference_data():
    return pd.DataFrame({
        "waiting_time": [1.0, 2.0, 3.0, 4.0, 5.0],
        "berth_time": [2.0, 4.0, 6.0, 8.0, 10.0],
        "queue": ["Low", "Low", "Medium", "High", "High"],
    })


def test_identical_data(reference_data):
    results = evaluate_synthetic_data(
        reference_data,
        reference_data.copy(),
    )

    assert results["KS Statistic"] == pytest.approx(0.0)
    assert results["Wasserstein Distance"] == pytest.approx(0.0)
    assert results["Mean/Std Gap"] == pytest.approx(0.0)
    assert results["Correlation MAE"] == pytest.approx(0.0)
    assert results["Categorical TV"] == pytest.approx(0.0)


def test_ks_statistic(reference_data):
    synthetic = reference_data.copy()
    synthetic["waiting_time"] += 100

    results = calculate_ks_statistic(
        reference_data,
        synthetic,
    )

    assert results["waiting_time"] > 0
    assert 0 <= results["waiting_time"] <= 1


def test_wasserstein_distance(reference_data):
    synthetic = reference_data.copy()
    synthetic["waiting_time"] += 10

    results = calculate_wasserstein_distance(
        reference_data,
        synthetic,
    )

    assert results["waiting_time"] == pytest.approx(10.0)


def test_mean_std_gap(reference_data):
    synthetic = reference_data.copy()
    synthetic["waiting_time"] += 5

    results = calculate_mean_std_gap(
        reference_data,
        synthetic,
    )

    assert results["waiting_time"]["mean_gap"] > 0
    assert results["waiting_time"]["std_gap"] == pytest.approx(0.0)


def test_correlation_mae(reference_data):
    identical = calculate_correlation_mae(
        reference_data,
        reference_data.copy(),
    )

    assert identical == pytest.approx(0.0)


def test_categorical_tv(reference_data):
    synthetic = reference_data.copy()
    synthetic["queue"] = ["Low", "Low", "Low", "Low", "Low"]

    results = calculate_categorical_tv(
        reference_data,
        synthetic,
    )

    assert results["queue"] > 0
    assert 0 <= results["queue"] <= 1


def test_evaluation_structure(reference_data):
    results = evaluate_synthetic_data(
        reference_data,
        reference_data.copy(),
    )

    assert "KS Statistic" in results
    assert "Wasserstein Distance" in results
    assert "Mean/Std Gap" in results
    assert "Correlation MAE" in results
    assert "Categorical TV" in results
    assert "per_column" in results


def test_mismatched_columns_rejected(reference_data):
    synthetic = reference_data.rename(
        columns={"queue": "queue_type"}
    )

    with pytest.raises(ValueError):
        evaluate_synthetic_data(
            reference_data,
            synthetic,
        )
