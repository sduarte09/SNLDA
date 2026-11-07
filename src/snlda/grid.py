"""Helpers for working with regular latitude/longitude grids.

All functions are pure NumPy so they can be tested without any NetCDF files.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def lon_to_360(lon):
    """Return longitudes in the 0-360 convention.

    If any value is negative the whole array is shifted by +360, which is the
    rule used throughout the original notebooks.
    """
    lon = np.asarray(lon, dtype="float64")
    return lon + 360.0 if (lon < 0).any() else lon


def nearest_grid_index(grid, value, tol_cells=0.5):
    """Index of the grid centre closest to ``value``.

    A station belongs to the cell whose boundaries (midpoints between
    neighbouring centres) contain it, which is the same as the nearest centre.
    Raises ``ValueError`` if ``value`` lies more than ``tol_cells`` cells
    outside the grid.
    """
    grid = np.asarray(grid, dtype="float64")
    idx = int(np.argmin(np.abs(grid - value)))
    if grid.size > 1:
        step = float(np.median(np.abs(np.diff(grid))))
        if abs(grid[idx] - value) > tol_cells * step * (1 + 1e-9):
            raise ValueError(
                f"{value} lies outside the grid range [{grid.min()}, {grid.max()}]"
            )
    return idx


def index_to_coord(index, coords):
    """Linearly interpolate a (fractional) array index to a coordinate value."""
    coords = np.asarray(coords, dtype="float64")
    return np.interp(index, np.arange(coords.size), coords)


def select_on_grid(data, lat, lon, target_lat, target_lon, ndigits=3):
    """Pick the cells of ``data`` that match a target grid.

    ``data`` has shape (time, lat, lon). The result has shape
    (time, len(target_lat), len(target_lon)); target cells that do not exist in
    the source grid are filled with NaN. Coordinates are compared after
    rounding to ``ndigits`` decimals to avoid float32/float64 mismatches.
    """
    data = np.asarray(data)
    lat_r = np.round(np.asarray(lat, dtype="float64"), ndigits)
    lon_r = np.round(np.asarray(lon, dtype="float64"), ndigits)
    lat_pos = {v: i for i, v in enumerate(lat_r)}
    lon_pos = {v: i for i, v in enumerate(lon_r)}
    t_lat = np.round(np.asarray(target_lat, dtype="float64"), ndigits)
    t_lon = np.round(np.asarray(target_lon, dtype="float64"), ndigits)
    ilat = np.array([lat_pos.get(v, -1) for v in t_lat], dtype=int)
    ilon = np.array([lon_pos.get(v, -1) for v in t_lon], dtype=int)
    out = np.full((data.shape[0], t_lat.size, t_lon.size), np.nan)
    ok_lat, ok_lon = np.flatnonzero(ilat >= 0), np.flatnonzero(ilon >= 0)
    out[:, ok_lat[:, None], ok_lon[None, :]] = data[:, ilat[ok_lat][:, None], ilon[ok_lon][None, :]]
    return out


@dataclass
class CroppedGrid:
    """Dataset cropped to the extent of a basin mask."""

    data: np.ndarray  # (time, lat, lon); NaN outside the basin
    mask: np.ndarray  # (lat, lon); 1 inside the basin, NaN outside
    lat: np.ndarray  # same order as the mask rows (north to south)
    lon: np.ndarray  # same order as the mask columns, 0-360 convention


def crop_to_basin(data, lat, lon, mask, mask_lat, mask_lon, ndigits=1):
    """Crop ``data`` to the part of the grid shared with the basin mask.

    The result is multiplied by the mask, so cells outside the basin are NaN.
    The latitude axis of the result follows the mask rows. If the dataset runs
    south to north while the mask runs north to south, the dataset is flipped,
    so data and mask always line up.
    """
    data = np.asarray(data)
    lat_r = np.round(np.asarray(lat, dtype="float64"), ndigits)
    lon_r = np.round(np.asarray(lon, dtype="float64"), ndigits)
    mlat_r = np.round(np.asarray(mask_lat, dtype="float64"), ndigits)
    mlon_r = np.round(np.asarray(mask_lon, dtype="float64"), ndigits)

    common_lat = np.intersect1d(mlat_r, lat_r)
    common_lon = np.intersect1d(mlon_r, lon_r)
    if common_lat.size == 0 or common_lon.size == 0:
        raise ValueError("Dataset and basin mask do not overlap")

    mrows = np.flatnonzero(np.isin(mlat_r, common_lat))
    mcols = np.flatnonzero(np.isin(mlon_r, common_lon))
    drows = np.flatnonzero(np.isin(lat_r, common_lat))
    dcols = np.flatnonzero(np.isin(lon_r, common_lon))

    sub = data[:, drows.min() : drows.max() + 1, dcols.min() : dcols.max() + 1]
    sub_lat = lat_r[drows.min() : drows.max() + 1]
    mask_sub = mask[mrows.min() : mrows.max() + 1, mcols.min() : mcols.max() + 1]
    out_lat = mlat_r[mrows.min() : mrows.max() + 1]
    out_lon = mlon_r[mcols.min() : mcols.max() + 1]

    if (
        sub_lat.size > 1
        and out_lat.size > 1
        and (sub_lat[0] > sub_lat[-1]) != (out_lat[0] > out_lat[-1])
    ):
        sub = sub[:, ::-1, :]

    if sub.shape[1:] != mask_sub.shape:
        raise ValueError(
            f"Grid mismatch after cropping: data {sub.shape[1:]} vs mask {mask_sub.shape}"
        )
    return CroppedGrid(data=sub * mask_sub, mask=mask_sub, lat=out_lat, lon=out_lon)


def elevation_on_grid(geopotential, z_lat, z_lon, lat, lon, g=9.80665):
    """Elevation (m) on the grid defined by ``lat``/``lon``.

    ``geopotential`` is a 2-D (lat, lon) field in m2 s-2, e.g. ERA5 ``z``.
    Each target cell takes the nearest geopotential cell.
    """
    z_lon = lon_to_360(z_lon)
    ilat = [nearest_grid_index(z_lat, v) for v in lat]
    ilon = [nearest_grid_index(z_lon, v) for v in lon]
    return np.asarray(geopotential, dtype="float64")[np.ix_(ilat, ilon)] / g
