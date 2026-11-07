# SNLDA

**SNLDA (Spatiotemporal Non-Linear Dynamics Assessment)** is a Python framework for evaluating gridded precipitation datasets against rain gauges. It has three components, which can be used on their own or combined:

1. **Standard performance metrics** (`snlda.metrics`): MAE, MSE, RMSE, correlation, mutual information and cross entropy, per station and per pixel.
2. **Rainfall spatiotemporal objects** (`snlda.objects`): rainfall events as 3-D (time, latitude, longitude) connected objects, with volumes, peak rainfall, start/end centroids and elevations, filtered by ENSO phase.
3. **Non-linear dynamics complexity** (`snlda.dynamics`): time delay, embedding dimension (false nearest neighbours), Lyapunov exponents and the Hurst exponent, which describe the attractor of each rainfall series.

It was developed for the validation of ERA5-Land daily precipitation in the Magdalena River Basin (Colombia), using 558 rain gauges (1980-2020) and an inverse-distance-weighted (IDW) interpolation to compare across spatial scales. The framework is described in:

> Duarte, S., Corzo, G., Solomatine, D., Uijlenhoet, R. (2026). Spatiotemporal Non-Linear Dynamics Assessment (SNLDA) of ERA5-Land precipitation in the Magdalena River Basin. *Journal of Hydrology: Regional Studies*, 63, 103083. [https://doi.org/10.1016/j.ejrh.2025.103083](https://doi.org/10.1016/j.ejrh.2025.103083)

## Installation

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

Each notebook shows the full workflow for the basin data.

## Expected data layout

The full data is not part of this repository, however a small portion is shared. The notebooks expect:

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

## Citation

If you use SNLDA, please cite the paper above; `CITATION.cff` has the same reference and a "Cite this repository" entry for the software. The false-nearest-neighbour routine is adapted from [pypsr](https://github.com/hsharrison/pypsr) (MIT).

## License

MIT, see `LICENSE`.
