"""Non-linear dynamics of a time series: delay, embedding, Lyapunov and Hurst.

The false-nearest-neighbour code is adapted from ``pypsr`` by H. Sharrison
(MIT licence, https://github.com/hsharrison/pypsr).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .metrics import mutual_information

#: Value written when no embedding dimension satisfies the FNN threshold.
NO_DIMENSION = -99


class DynamicsResult(NamedTuple):
    lag: int
    embedding_dim: int
    lyapunov: np.ndarray
    hurst: float


def autocorrelation_delay(x):
    """First lag (>= 1) at which the autocorrelation function changes sign."""
    x = _series(x)
    n = x.size
    var = x.var()
    if var == 0:
        raise ValueError("Constant series: autocorrelation is undefined")
    y = x - x.mean()
    r = np.correlate(y, y, mode="full")[-n:]
    auto = r / (var * np.arange(n, 0, -1))
    d = 0
    while d < n - 2 and auto[d] * auto[d + 1] > 0:
        d += 1
    return d + 1


def mutual_information_delay(x, bins=20, max_lag=None):
    """Lag of the first minimum of the mutual information with the lagged series."""
    x = _series(x)
    max_lag = x.size - 1 if max_lag is None else max_lag
    lag = 0
    current = mutual_information(x, np.roll(x, 0), bins)
    while lag < max_lag:
        nxt = mutual_information(x, np.roll(x, lag + 1), bins)
        if current >= nxt:
            lag += 1
            current = nxt
        else:
            break
    return lag


def optimal_delay(x, bins=20):
    """Time delay: the smaller of the autocorrelation and mutual-information delays."""
    return min(autocorrelation_delay(x), mutual_information_delay(x, bins))


def reconstruct(x, lag, n_dims):
    """Delay-embedding matrix of shape (len(x) - lag * (n_dims - 1), n_dims)."""
    x = _series(x)
    span = lag * (n_dims - 1)
    n = x.size - span
    if n <= 0:
        raise ValueError("Series too short for this lag and dimension")
    return np.column_stack([x[i * lag : i * lag + n] for i in range(n_dims)])


def deconstruct(embedded, lag, n_dims):
    """Inverse of :func:`reconstruct`: rebuild the series from the embedding."""
    n = len(embedded)
    out = np.full(n + lag * (n_dims - 1), np.nan)
    out[:n] = embedded[:, 0]
    if n_dims > 1:
        out[n:] = embedded[-lag * (n_dims - 1) :, -1]
    return out


def false_nearest_neighbors(
    x, lag, min_dims=1, max_dims=15, relative_distance_cutoff=15, relative_radius_cutoff=2
):
    """Fraction of false nearest neighbours for each embedding dimension.

    Returns ``(dimensions, false_neighbor_fractions)``.

    Neighbours at distance zero (repeated points, common in rainfall series
    with many dry days) give 0/0 or x/0 and are therefore counted as *false*
    neighbours. This reproduces the original notebook behaviour.
    """
    x = _series(x)
    dims = np.arange(min_dims, max_dims + 1)
    fractions = np.array(
        [
            _false_neighbor_fraction(
                x, lag, d, relative_distance_cutoff, relative_radius_cutoff
            )
            for d in dims
        ]
    )
    return dims, fractions


def _false_neighbor_fraction(x, lag, n_dims, rtol, atol):
    offset = lag * n_dims
    radius = np.sqrt(((x - x.mean()) ** 2).mean())
    embedded = reconstruct(x, lag, n_dims)
    dist, idx = cKDTree(embedded).query(embedded, k=2)
    nearest = dist[:, 1]
    valid = (idx + offset < x.size).all(axis=1)
    if not valid.any():
        return np.nan
    idx, nearest = idx[valid], nearest[valid]
    increase = np.abs(x[idx[:, 0] + offset] - x[idx[:, 1] + offset])
    with np.errstate(divide="ignore", invalid="ignore"):
        is_true = (increase / nearest < rtol) & (increase / radius < atol)
    return float(np.mean(~is_true))


def dynamics_analysis(x, fnn_threshold=0.2, dimension_offset=0):
    """Delay, embedding dimension, Lyapunov spectrum and Hurst exponent.

    Parameters
    ----------
    fnn_threshold : float
        An embedding dimension is accepted when its false-neighbour fraction
        is at or below this value.
    dimension_offset : int
        The original notebooks used the 0-based *index* of the selected
        dimension in the FNN array as the embedding dimension, i.e. one less
        than the dimension itself. The default (0) reproduces that. Use 1 to
        use the actual dimension.

    Returns a :class:`DynamicsResult`. If no dimension passes the threshold,
    ``embedding_dim`` is -99 and ``lyapunov`` is ``[-99]``.
    """
    nolds = _import_nolds()

    x = _series(x)
    lag = optimal_delay(x)
    max_dim = 2 * int(math.floor(2 * math.log10(x.size))) + 1
    _, fnn = false_nearest_neighbors(x, lag, 1, max_dim)

    if np.any(fnn <= fnn_threshold):
        if np.any(fnn == 0):
            index = int(np.flatnonzero(fnn == 0)[0])
        else:
            index = int(np.flatnonzero(fnn <= fnn_threshold)[0])
        dim = index + dimension_offset
        lyapunov = np.asarray(nolds.lyap_e(x, emb_dim=dim, matrix_dim=dim, tau=lag))
    else:
        dim = NO_DIMENSION
        lyapunov = np.array([NO_DIMENSION], dtype="float64")
    hurst = float(nolds.hurst_rs(x))
    return DynamicsResult(int(lag), int(dim), lyapunov, hurst)


def dynamics_summary(x, **kwargs):
    """Scalar summary of :func:`dynamics_analysis`, as stored in the result tables."""
    res = dynamics_analysis(x, **kwargs)
    return {
        "delay": res.lag,
        "dim_fnn": res.embedding_dim,
        "hurst": res.hurst,
        "lyapunov_max": float(np.max(res.lyapunov)),
        "lyapunov_sum": float(np.sum(res.lyapunov)),
    }


def dynamics_table(series, on_error="nan", progress=None, **kwargs):
    """Dynamics summary for every column of a DataFrame (e.g. one per station).

    ``on_error="nan"`` records NaN for series that fail (constant series,
    too short, ...); ``"raise"`` propagates the exception.
    """
    columns = ["delay", "dim_fnn", "hurst", "lyapunov_max", "lyapunov_sum"]
    rows = {}
    for k, name in enumerate(series.columns):
        if progress is not None:
            progress(k, len(series.columns), name)
        try:
            rows[name] = dynamics_summary(series[name].to_numpy(), **kwargs)
        except Exception:
            if on_error == "raise":
                raise
            rows[name] = {c: np.nan for c in columns}
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def dynamics_on_grid(data, mask=None, on_error="nan", progress=None, **kwargs):
    """Dynamics summary for every pixel of a (time, lat, lon) array.

    Returns a dict mapping each property to a (lat, lon) array. Pixels outside
    ``mask`` (1 inside, NaN outside) or containing NaN stay NaN.
    """
    data = np.asarray(data, dtype="float64")
    _, n_lat, n_lon = data.shape
    names = ["delay", "dim_fnn", "hurst", "lyapunov_max", "lyapunov_sum"]
    out = {n: np.full((n_lat, n_lon), np.nan) for n in names}
    pixels = [
        (i, j)
        for i in range(n_lat)
        for j in range(n_lon)
        if (mask is None or mask[i, j] == 1) and np.isfinite(data[:, i, j]).all()
    ]
    for k, (i, j) in enumerate(pixels):
        if progress is not None:
            progress(k, len(pixels), (i, j))
        try:
            res = dynamics_summary(data[:, i, j], **kwargs)
        except Exception:
            if on_error == "raise":
                raise
            continue
        for n in names:
            out[n][i, j] = res[n]
    return out


def _import_nolds():
    """Import ``nolds`` lazily, so the rest of the package works without it."""
    try:
        import nolds
    except (ImportError, TypeError) as error:
        raise ImportError(
            "nolds could not be imported in this environment "
            f"({type(error).__name__}: {error}). Some nolds releases fail to import "
            "on older Python versions; try a newer Python or a different nolds release."
        ) from error
    return nolds


def _series(x):
    x = np.squeeze(np.asarray(x, dtype="float64"))
    if x.ndim != 1:
        raise ValueError("x(t) must be a 1-dimensional signal")
    return x
