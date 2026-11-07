"""Reading and writing the files used by SNLDA (CSV gauges, NetCDF grids)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .grid import lon_to_360

MISSING_VALUE = -9999.0


@dataclass
class GriddedData:
    """A (time, lat, lon) variable read from a NetCDF file."""

    data: np.ndarray
    time: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray  # 0-360 convention
    lat_units: str = "degrees_north"
    lon_units: str = "degrees_east"


@dataclass
class BasinMask:
    """Basin mask on its own grid.

    ``mask`` is 1 inside the basin and NaN outside. Rows follow ``lat``, which
    is the latitude axis of the file flipped (north to south), matching the
    orientation of ERA5 grids. ``lon`` is in the 0-360 convention.
    """

    mask: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


def read_station_data(data_path, coords_path):
    """Read daily gauge data and station coordinates.

    Returns ``(stations, coords)``. ``stations`` has one column per station
    indexed by date (``Fecha`` column). ``coords`` is indexed by station name
    and has longitudes in the 0-360 convention.
    """
    stations = pd.read_csv(data_path, sep=",", index_col="Fecha", parse_dates=True)
    coords = pd.read_csv(coords_path, sep=";", index_col="Station")
    coords["Long"] = lon_to_360(coords["Long"])
    return stations, coords


def read_netcdf_grid(path, var_name=None, fill_value=np.nan):
    """Read a (time, lat, lon) NetCDF variable.

    ``var_name`` defaults to the last variable in the file, as in the original
    notebooks. Masked cells are replaced by ``fill_value`` (NaN by default).
    """
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        tvar = ds.variables["time"]
        dates = nc.num2date(
            tvar[:],
            units=tvar.units,
            calendar=getattr(tvar, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        time = pd.DatetimeIndex(
            [dt.datetime(d.year, d.month, d.day, d.hour, d.minute) for d in dates]
        )
        if not time.is_monotonic_increasing:
            raise ValueError(f"Time axis of {path} is not increasing")
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        lon = lon_to_360(np.asarray(ds.variables["longitude"][:], dtype="float64"))
        name = var_name or list(ds.variables.keys())[-1]
        data = np.ma.filled(ds.variables[name][:, :, :].astype("float64"), fill_value)
        return GriddedData(
            data=data,
            time=time,
            lat=lat,
            lon=lon,
            lat_units=getattr(ds.variables["latitude"], "units", "degrees_north"),
            lon_units=getattr(ds.variables["longitude"], "units", "degrees_east"),
        )


def basin_mask_from_array(values):
    """Turn a raw basin grid into a mask: 1 inside, NaN outside, rows flipped.

    Cells equal to the value at position ``[1, 1]`` (the background) are
    outside the basin.
    """
    values = np.asarray(values)
    inside = (values != values[1, 1]).astype("float64")
    mask = np.flip(inside, axis=0)
    mask[mask == 0] = np.nan
    return mask


def read_basin_mask(path):
    """Read a basin extent NetCDF (variables ``lat``, ``lon`` and one grid)."""
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        lat = np.ma.getdata(ds.variables["lat"][:]).astype("float64")
        lon = np.ma.getdata(ds.variables["lon"][:]).astype("float64")
        grid_name = list(ds.variables.keys())[0]
        values = np.ma.getdata(ds.variables[grid_name][:, :])
    return BasinMask(mask=basin_mask_from_array(values), lat=np.flip(lat), lon=lon_to_360(lon))


def read_oni(path, sep=";"):
    """Read the Oceanic Nino Index table (columns Year, Month, ONI)."""
    return pd.read_csv(path, sep=sep)


def read_geopotential(path):
    """First time step of the ERA5 geopotential ``z`` as ``(z, lat, lon)``."""
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        lat = np.asarray(ds.variables["latitude"][:], dtype="float64")
        lon = np.asarray(ds.variables["longitude"][:], dtype="float64")
        z = np.asarray(ds.variables["z"][0, :, :], dtype="float64")
    return z, lat, lon


def write_netcdf_grid(path, variables, lat, lon, title="", lat_units="degrees_north", lon_units="degrees_east"):
    """Write 2-D (lat, lon) arrays to a NetCDF file.

    ``variables`` maps variable name to array. NaN is stored as ``-9999`` and
    declared through the ``missing_value`` attribute.
    """
    import netCDF4 as nc

    with nc.Dataset(path, mode="w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.title = title
        ds.createDimension("latitude", len(lat))
        ds.createDimension("longitude", len(lon))
        v_lat = ds.createVariable("latitude", np.float32, ("latitude",))
        v_lat.units = lat_units
        v_lat[:] = lat
        v_lon = ds.createVariable("longitude", np.float32, ("longitude",))
        v_lon.units = lon_units
        v_lon[:] = lon
        for name, array in variables.items():
            var = ds.createVariable(name.replace(" ", "_"), np.float64, ("latitude", "longitude"))
            var.missing_value = MISSING_VALUE
            var[:, :] = np.where(np.isnan(array), MISSING_VALUE, array)
