import numpy as np
import pandas as pd
import pytest

from snlda import metrics
from snlda.grid import nearest_grid_index


def test_rmse_known_value():
    obs = np.array([1.0, 2.0, 3.0])
    sim = np.array([1.0, 2.0, 5.0])
    # errors are 0, 0, 2 -> sqrt(4 / 3)
    assert metrics.root_mean_squared_error(obs, sim) == pytest.approx(np.sqrt(4 / 3))


def test_mae_and_mse_known_values():
    obs = np.array([0.0, 0.0, 0.0, 0.0])
    sim = np.array([1.0, -1.0, 2.0, -2.0])
    assert metrics.mean_absolute_error(obs, sim) == pytest.approx(1.5)
    assert metrics.mean_squared_error(obs, sim) == pytest.approx(2.5)


def test_perfect_match():
    x = np.random.default_rng(42).random(200)
    assert metrics.mean_absolute_error(x, x) == 0
    assert metrics.root_mean_squared_error(x, x) == 0
    assert metrics.correlation(x, x) == pytest.approx(1.0)


def test_correlation_of_scaled_series_is_one():
    x = np.random.default_rng(1).random(100)
    assert metrics.correlation(x, 3 * x + 2) == pytest.approx(1.0)
    assert metrics.correlation(x, -x) == pytest.approx(-1.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        metrics.mean_squared_error(np.array([1, 2, 3]), np.array([1, 2]))


def test_mutual_information_independent_is_small_and_identical_is_large():
    rng = np.random.default_rng(0)
    x = rng.random(5000)
    y = rng.random(5000)
    assert metrics.mutual_information(x, y) < 0.05
    assert metrics.mutual_information(x, x) > 2.0  # close to log(20) = 3.0
    assert metrics.mutual_information(x, y) >= 0


def test_mutual_information_matches_scikit_learn_if_available():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    x = rng.gamma(2.0, 2.0, 2000)
    y = x + rng.normal(0, 1, 2000)
    contingency = np.histogram2d(x, y, 20)[0]
    expected = sklearn_metrics.mutual_info_score(None, None, contingency=contingency)
    assert metrics.mutual_information(x, y) == pytest.approx(expected)


def test_cross_entropy_known_value():
    targets = np.array([1.0, 1.0])
    predictions = np.array([0.5, 0.5])
    assert metrics.cross_entropy(targets, predictions) == pytest.approx(-np.log(0.5) * 2 / 2)


def test_evaluate_pair_ignores_nan_and_handles_too_few_points():
    obs = np.array([1.0, 2.0, np.nan, 4.0])
    sim = np.array([1.0, 2.0, 3.0, 4.0])
    res = metrics.evaluate_pair(obs, sim)
    assert res["RMSE"] == 0
    assert set(res) == set(metrics.METRIC_NAMES)
    empty = metrics.evaluate_pair([np.nan, 1.0], [1.0, np.nan])
    assert all(np.isnan(v) for v in empty.values())


def test_nearest_grid_index_ascending_and_descending():
    asc = np.array([0.0, 0.1, 0.2, 0.3])
    desc = asc[::-1]
    assert nearest_grid_index(asc, 0.14) == 1
    assert nearest_grid_index(desc, 0.14) == 2
    assert nearest_grid_index(asc, 0.0) == 0  # first cell is reachable
    assert nearest_grid_index(asc, 0.3) == 3  # last cell is reachable


def test_nearest_grid_index_outside_grid_raises():
    with pytest.raises(ValueError):
        nearest_grid_index(np.array([0.0, 0.1, 0.2]), 5.0)


def test_station_metrics_uses_the_cell_containing_the_station():
    lat = np.array([2.0, 1.0, 0.0])  # descending, like ERA5
    lon = np.array([10.0, 11.0, 12.0])
    time = pd.date_range("2000-01-01", periods=6)
    data = np.zeros((6, 3, 3))
    truth = np.array([0.0, 2.0, 0.0, 5.0, 1.0, 0.0])
    data[:, 1, 2] = truth  # lat 1.0, lon 12.0
    stations = pd.DataFrame({"A": truth}, index=time)
    coords = pd.DataFrame(
        {"Lat": [1.02], "Long": [11.97], "Length": [6]}, index=pd.Index(["A"], name="Station")
    )
    out = metrics.station_metrics(stations, coords, data, lat, lon)
    assert out.loc["A", "RMSE"] == 0
    assert out.loc["A", "Correlation"] == pytest.approx(1.0)
    assert "Length" not in out.columns


def test_pixel_metrics_respects_mask():
    rng = np.random.default_rng(5)
    ref = rng.random((20, 2, 2))
    sim = ref.copy()
    sim[:, 0, 0] += 1.0
    mask = np.array([[1.0, np.nan], [1.0, 1.0]])
    res = metrics.pixel_metrics(ref, sim, mask)
    assert res["MAE"][0, 0] == pytest.approx(1.0)
    assert res["MAE"][1, 1] == 0
    assert np.isnan(res["MAE"][0, 1])
