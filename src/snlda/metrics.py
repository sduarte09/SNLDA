"""Standard performance metrics between observed and simulated rainfall."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import nearest_grid_index

METRIC_NAMES = ("MAE", "MSE", "RMSE", "Correlation", "Mutual Information", "Cross Entropy")


def mean_absolute_error(obs, sim):
    obs, sim = _pair(obs, sim)
    return float(np.mean(np.abs(obs - sim)))


def mean_squared_error(obs, sim):
    obs, sim = _pair(obs, sim)
    return float(np.mean((obs - sim) ** 2))


def root_mean_squared_error(obs, sim):
    return float(np.sqrt(mean_squared_error(obs, sim)))


def correlation(obs, sim):
    """Pearson correlation coefficient (NaN for constant series)."""
    obs, sim = _pair(obs, sim)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(obs, sim)[0, 1])


def mutual_information(x, y, bins=20):
    """Mutual information (nats) from a 2-D histogram of ``x`` and ``y``."""
    x, y = _pair(x, y)
    counts = np.histogram2d(x, y, bins)[0]
    pxy = counts / counts.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))
    return max(mi, 0.0)


def cross_entropy(targets, predictions, epsilon=1e-12):
    """Cross entropy as defined in the original notebooks.

    ``predictions`` are clipped to [epsilon, 1 - epsilon]. Note that this
    treats the values as probabilities, so rainfall in mm is clipped to just
    below 1; interpret the result with that in mind.
    """
    targets = np.asarray(targets, dtype="float64")
    predictions = np.clip(np.asarray(predictions, dtype="float64"), epsilon, 1.0 - epsilon)
    return float(-np.sum(targets * np.log(predictions)) / predictions.shape[0])


def evaluate_pair(obs, sim, bins=20):
    """All metrics for one pair of series, ignoring time steps with NaN.

    Returns a dict keyed by ``METRIC_NAMES``; every value is NaN if fewer than
    two valid pairs remain.
    """
    obs = np.asarray(obs, dtype="float64").ravel()
    sim = np.asarray(sim, dtype="float64").ravel()
    ok = np.isfinite(obs) & np.isfinite(sim)
    if ok.sum() < 2:
        return {name: np.nan for name in METRIC_NAMES}
    o, s = obs[ok], sim[ok]
    return {
        "MAE": mean_absolute_error(o, s),
        "MSE": mean_squared_error(o, s),
        "RMSE": root_mean_squared_error(o, s),
        "Correlation": correlation(o, s),
        "Mutual Information": mutual_information(o, s, bins),
        "Cross Entropy": cross_entropy(o, s),
    }


def station_metrics(stations, coords, data, lat, lon, lat_col="Lat", lon_col="Long"):
    """Compare gauge series with the grid cell that contains each station.

    Parameters
    ----------
    stations : DataFrame, one column per station, indexed by date
    coords : DataFrame indexed by station name with latitude/longitude columns
        (longitudes in the same convention as ``lon``)
    data : array (time, lat, lon) with the gridded product
    """
    rows = {}
    for name in coords.index:
        i = nearest_grid_index(lat, coords.loc[name, lat_col])
        j = nearest_grid_index(lon, coords.loc[name, lon_col])
        rows[name] = evaluate_pair(stations[name].to_numpy(), data[:, i, j])
    out = coords.drop(columns=["Length"], errors="ignore").copy()
    result = pd.DataFrame.from_dict(rows, orient="index", columns=list(METRIC_NAMES))
    return out.join(result)


def pixel_metrics(ref, sim, mask=None):
    """Metrics for every pixel of two gridded datasets on the same grid.

    ``ref`` and ``sim`` have shape (time, lat, lon). ``mask`` (lat, lon) is 1
    inside the area of interest and NaN outside; pixels outside it stay NaN.
    Returns a dict mapping each metric name to a (lat, lon) array.
    """
    ref = np.asarray(ref, dtype="float64")
    sim = np.asarray(sim, dtype="float64")
    if ref.shape != sim.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {sim.shape}")
    _, n_lat, n_lon = ref.shape
    out = {name: np.full((n_lat, n_lon), np.nan) for name in METRIC_NAMES}
    for i in range(n_lat):
        for j in range(n_lon):
            if mask is not None and mask[i, j] != 1:
                continue
            res = evaluate_pair(ref[:, i, j], sim[:, i, j])
            for name in METRIC_NAMES:
                out[name][i, j] = res[name]
    return out


def _pair(a, b):
    a = np.asarray(a, dtype="float64").ravel()
    b = np.asarray(b, dtype="float64").ravel()
    if a.shape != b.shape:
        raise ValueError(f"Series must have the same length, got {a.size} and {b.size}")
    return a, b
