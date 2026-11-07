import numpy as np
import pytest

from snlda import grid
from snlda.io import basin_mask_from_array


def test_lon_to_360_shifts_only_when_negative_values_exist():
    assert list(grid.lon_to_360([-75.0, -74.0])) == [285.0, 286.0]
    assert list(grid.lon_to_360([285.0, 286.0])) == [285.0, 286.0]


def test_index_to_coord_interpolates():
    coords = [10.0, 20.0, 30.0]
    assert grid.index_to_coord(1.5, coords) == pytest.approx(25.0)
    assert grid.index_to_coord(2, coords) == pytest.approx(30.0)


def test_select_on_grid_matches_and_fills_missing_with_nan():
    data = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    lat = np.array([2.0, 1.0, 0.0])
    lon = np.array([10.0, 11.0, 12.0])
    out = grid.select_on_grid(data, lat, lon, target_lat=[1.0, 5.0], target_lon=[12.0, 10.0])
    assert out.shape == (2, 2, 2)
    assert out[0, 0, 0] == data[0, 1, 2]
    assert out[1, 0, 1] == data[1, 1, 0]
    assert np.isnan(out[:, 1, :]).all()


def test_basin_mask_from_array_flips_rows_and_marks_outside_as_nan():
    raw = np.zeros((3, 3))
    raw[2, 0] = 7.0  # basin cell in the last row
    mask = basin_mask_from_array(raw)
    assert mask[0, 0] == 1.0  # flipped to the first row
    assert np.isnan(mask[1:, :]).all()


def test_crop_to_basin_applies_mask_and_orders_latitudes_like_the_mask():
    lat = np.array([3.0, 2.0, 1.0, 0.0])
    lon = np.array([10.0, 11.0, 12.0, 13.0])
    data = np.ones((2, 4, 4))
    mask_lat = np.array([2.0, 1.0])
    mask_lon = np.array([11.0, 12.0])
    mask = np.array([[1.0, np.nan], [1.0, 1.0]])
    out = grid.crop_to_basin(data, lat, lon, mask, mask_lat, mask_lon)
    assert out.data.shape == (2, 2, 2)
    assert np.isnan(out.data[:, 0, 1]).all()
    assert (out.data[:, 1, :] == 1).all()
    assert list(out.lat) == [2.0, 1.0]


def test_crop_to_basin_flips_ascending_datasets_to_match_the_mask():
    lat = np.array([0.0, 1.0, 2.0, 3.0])  # south to north
    lon = np.array([10.0, 11.0])
    data = np.zeros((1, 4, 2))
    data[0, 2, :] = 20.0  # lat 2.0
    data[0, 1, :] = 10.0  # lat 1.0
    mask = np.ones((2, 2))
    out = grid.crop_to_basin(data, lat, lon, mask, np.array([2.0, 1.0]), lon)
    assert list(out.data[0, :, 0]) == [20.0, 10.0]  # north first, like the mask


def test_crop_to_basin_without_overlap_raises():
    with pytest.raises(ValueError):
        grid.crop_to_basin(
            np.ones((1, 2, 2)), [0.0, 1.0], [0.0, 1.0], np.ones((1, 1)), [50.0], [50.0]
        )


def test_elevation_on_grid_converts_geopotential_to_metres():
    z = np.array([[9.80665 * 100, 9.80665 * 200], [9.80665 * 300, 9.80665 * 400]])
    elev = grid.elevation_on_grid(z, [1.0, 0.0], [10.0, 11.0], lat=[0.0, 1.0], lon=[10.0, 11.0])
    assert elev[0, 0] == pytest.approx(300.0)  # lat 0.0, lon 10.0
    assert elev[1, 1] == pytest.approx(200.0)  # lat 1.0, lon 11.0
