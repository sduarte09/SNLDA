![tests](https://github.com/sduarte09/SNLDA/actions/workflows/tests.yml/badge.svg)
# SNLDA
**SNLDA (Spatiotemporal Non-Linear Dynamics Assessment)** is a Python framework for evaluating gridded precipitation datasets against rain gauges. It has three components, which can be used on their own or combined:

1. **Standard performance metrics** (`snlda.metrics`): MAE, MSE, RMSE, correlation, mutual information and cross entropy, per station and per pixel.
2. **Rainfall spatiotemporal objects** (`snlda.objects`): rainfall events as 3-D (time, latitude, longitude) connected objects, with volumes, peak rainfall, start/end centroids and elevations, filtered by ENSO phase.
3. **Non-linear dynamics complexity** (`snlda.dynamics`): time delay, embedding dimension (false nearest neighbours), Lyapunov exponents and the Hurst exponent, which describe the attractor of each rainfall series.

It was developed for the validation of ERA5-Land daily precipitation in the Magdalena River Basin (Colombia), using 558 rain gauges (1980-2020) and an inverse-distance-weighted (IDW) interpolation to compare across spatial scales. The framework is described in:

> Duarte, S., Corzo, G., Solomatine, D., Uijlenhoet, R. (2026). Spatiotemporal Non-Linear Dynamics Assessment (SNLDA) of ERA5-Land precipitation in the Magdalena River Basin. *Journal of Hydrology: Regional Studies*, 63, 103083. [https://doi.org/10.1016/j.ejrh.2025.103083](https://doi.org/10.1016/j.ejrh.2025.103083)

## Installation

SNLDA needs **Python 3.12 or newer**: the `nolds` release it depends on does not import on Python 3.10.

```bash
git clone https://github.com/sduarte09/SNLDA.git
cd SNLDA
pip install -e ".[test]"          # package + test tools
# or, with conda:
conda env create -f environment.yml && conda activate snlda
```

## Repository layout

```
src/snlda/
    metrics.py    standard metrics (station and per-pixel)
    objects.py    rainfall objects and ENSO event properties
    dynamics.py   delay, false nearest neighbours, Lyapunov, Hurst
    grid.py       grid helpers (nearest cell, cropping to a basin, elevation)
    io.py         CSV and NetCDF readers/writers
notebooks/
    standard_errors_daily.ipynb
    attractor_properties_daily.ipynb
    volume_aggregation_daily.ipynb
tests/            pytest tests on small synthetic examples
legacy/           the original notebooks, kept unchanged for reproducibility
```

The notebooks only load data, call the package and save results. All calculations are in `src/snlda`, so they can be imported, tested and reused.

## Quick start

```python
import numpy as np
from snlda import dynamics, metrics, objects

# 1. Metrics between two series
metrics.evaluate_pair(gauge_series, grid_cell_series)

# 2. Rainfall objects in a (time, lat, lon) array
rain, labels, n = objects.rain_objects(precip, threshold=1.0)

# 3. Dynamical properties of a series
dynamics.dynamics_summary(series)
```

Each notebook shows the full workflow for the Magdalena data.

## Expected data layout

The data are not part of this repository. The notebooks expect:

```
Data/Local CSV/        daily gauge series (CSV, column "Fecha") and station coordinates (";"-separated)
Data/Local IDW/        IDW-interpolated gauges, NetCDF, 0.1 degrees
Data/ERALand NC/       ERA5-Land daily precipitation, NetCDF, 0.1 degrees
Data/Magdalena/        basin mask, ONI table, ERA5 geopotential
```

Paths are set in the first cell of each notebook.

## Tests

```bash
pytest
```

The tests use small synthetic arrays with known answers and need no data files.

## Differences from the first notebook version

The original notebooks (folder `legacy/`) are the scripts referred to in the paper's data statement. The refactoring changed a few behaviours on purpose. If you compare results with the earlier notebooks, check these points:

- **Rainfall objects are labelled on the binary rain mask.** The notebooks passed the raw rainfall amounts to `skimage.measure.label`, which connects neighbouring cells only if their values are identical. `rain_objects(..., binary=False)` reproduces that old behaviour.
- **NaN cells are no rain.** Cells outside the basin were NaN and were labelled as foreground objects. The old `drop(0)` line removed the first such object; it is now optional (`drop_first_event`, off by default).
- **Station-to-cell matching** uses the nearest grid cell. The old search loops could not select the first cell and could run past the last one.
- **Embedding dimension.** The old code used the 0-based index of the FNN array as the dimension (one less than the real dimension). The default `dimension_offset=0` keeps this; `dimension_offset=1` uses the actual dimension.
- **Elevation** is matched to the cropped grid by coordinates.
- The largest `max_events=1000` events are kept (the old `.loc[:1000]` kept 1001).
- `numba`, `toolz` and `nolitsa` are no longer needed; the GPU code path was removed. `scikit-learn` is still installed because `nolds` needs it for the RANSAC line fit used in the Hurst exponent; without it `nolds` silently switches to a different fit and the Hurst values change.

## Citation

If you use SNLDA, please cite the paper above; `CITATION.cff` has the same reference and a "Cite this repository" entry for the software. The false-nearest-neighbour routine is adapted from [pypsr](https://github.com/hsharrison/pypsr) (MIT).

## License

MIT, see `LICENSE`.
