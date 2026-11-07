"""SNLDA: Spatiotemporal Non-Linear Dynamics Assessment of precipitation datasets.

Three components, one module each:

* :mod:`snlda.metrics`   - standard performance metrics (MAE, RMSE, correlation, ...)
* :mod:`snlda.objects`   - rainfall spatiotemporal objects (volumes, centroids, ENSO phases)
* :mod:`snlda.dynamics`  - non-linear dynamics (delay, embedding, Lyapunov, Hurst)

:mod:`snlda.grid` and :mod:`snlda.io` hold the shared grid and file helpers.
"""
from . import dynamics, grid, io, metrics, objects

__version__ = "0.1.0"

__all__ = ["dynamics", "grid", "io", "metrics", "objects", "__version__"]
