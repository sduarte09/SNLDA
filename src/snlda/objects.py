"""Rainfall spatiotemporal objects: labelling, ENSO filtering and event properties."""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure

from .grid import index_to_coord


def rain_objects(data, threshold=1.0, connectivity=2, binary=True):
    """Label 3-D (time, lat, lon) connected components of rainfall.

    Parameters
    ----------
    data : array (time, lat, lon); NaN is treated as "no rain"
    threshold : minimum rainfall (same units as ``data``) to count as rain
    connectivity : 1 = face neighbours only, 2 = faces and edges (3-D: up to 3)
    binary : if True (default) neighbouring rainy cells are always connected.
        If False, ``skimage`` connects only cells with *identical* values,
        which is what happened when the notebooks labelled the raw rainfall
        amounts; this option exists only to reproduce that behaviour.

    Returns ``(rain, labels, n_labels)`` where ``rain`` is ``data`` with cells
    below the threshold (and NaN) set to 0.
    """
    clean = np.nan_to_num(np.asarray(data, dtype="float64"), nan=0.0)
    rain = np.where(clean >= threshold, clean, 0.0)
    image = rain > 0 if binary else rain
    labels, n = measure.label(image, connectivity=connectivity, return_num=True, background=0)
    return rain, labels, n


def object_table(labels, time):
    """Basic properties (area, centroid, bounding box, duration, dates) per object."""
    props = pd.DataFrame(
        measure.regionprops_table(labels, properties=["label", "area", "centroid", "bbox"])
    )
    time = pd.DatetimeIndex(time)
    props["Total time"] = props["bbox-3"] - props["bbox-0"]
    props["start date"] = time[props["bbox-0"].to_numpy()]
    props["end date"] = time[props["bbox-3"].to_numpy() - 1]
    return props


def enso_months(oni, phase):
    """Set of (year, month) pairs in an ENSO phase.

    ``oni`` needs columns ``Year``, ``Month`` and ``ONI``. ``phase`` is
    ``"nina"`` (ONI <= -0.5) or ``"nino"`` (ONI >= 0.5).
    """
    if phase == "nina":
        sel = oni[oni["ONI"] <= -0.5]
    elif phase == "nino":
        sel = oni[oni["ONI"] >= 0.5]
    else:
        raise ValueError("phase must be 'nina' or 'nino'")
    return set(zip(sel["Year"].astype(int), sel["Month"].astype(int)))


def in_enso_phase(events, months):
    """Boolean Series: does an event start or end in a month of the phase?"""
    flags = [
        (s.year, s.month) in months or (e.year, e.month) in months
        for s, e in zip(events["start date"], events["end date"])
    ]
    return pd.Series(flags, index=events.index)


def event_properties(
    events,
    labels,
    rain,
    time,
    lat,
    lon,
    elevation=None,
    cell_size_deg=0.1,
    km_per_degree=111.0,
    lon_offset=360.0,
):
    """Add volume, peak, centroid and elevation properties to an events table.

    ``events`` comes from :func:`object_table`. ``lat``/``lon`` are the grid
    coordinates matching the last two axes of ``labels`` (``lon`` in the 0-360
    convention; ``lon_offset`` is subtracted in the output). Volumes are in
    million cubic metres, assuming rainfall in mm and a cell area of
    ``(km_per_degree * cell_size_deg)`` squared.
    """
    time = pd.DatetimeIndex(time)
    cell_area_m2 = (km_per_degree * 1000.0 * cell_size_deg) ** 2
    records = []
    for _, ev in events.iterrows():
        lo = [int(ev[f"bbox-{k}"]) for k in range(3)]
        hi = [int(ev[f"bbox-{k}"]) for k in range(3, 6)]
        sub = labels[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] == ev["label"]
        t, i, j = np.nonzero(sub)
        precip = rain[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]][sub]

        rec = {
            "Precipitation volume": (cell_area_m2 * precip.sum() / 1000.0) / 10**6,
            "Precipitation max magnitude": precip.max(),
            "Precipitation max date": time[lo[0] + t[np.argmax(precip)]],
            "Average Centroid lat": index_to_coord(ev["centroid-1"], lat),
            "Average Centroid long": index_to_coord(ev["centroid-2"], lon) - lon_offset,
        }
        for prefix, slice_t in (("Start", lo[0]), ("Finish", hi[0] - 1)):
            footprint = sub[slice_t - lo[0]]
            ci, cj = np.argwhere(footprint).mean(axis=0) + np.array([lo[1], lo[2]])
            rec[f"{prefix} Centroid lat"] = index_to_coord(ci, lat)
            rec[f"{prefix} Centroid long"] = index_to_coord(cj, lon) - lon_offset
            if elevation is not None:
                full = np.zeros(labels.shape[1:], dtype=bool)
                full[lo[1] : hi[1], lo[2] : hi[2]] = footprint
                rec[f"{prefix} Centroid elev"] = elevation[int(np.round(ci)), int(np.round(cj))]
                rec[f"{prefix} average elev"] = elevation[full].mean()
        records.append(rec)
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def enso_event_table(
    labels,
    rain,
    time,
    lat,
    lon,
    months,
    elevation=None,
    min_duration=1,
    max_events=1000,
    drop_first_event=False,
    **kwargs,
):
    """Largest rainfall events that touch an ENSO phase, with their properties.

    Events lasting at most ``min_duration`` time steps are discarded, as are
    events that neither start nor end in a month of ``months``. The
    ``max_events`` largest (by volume in cells) are kept and described with
    :func:`event_properties`.

    ``drop_first_event`` removes the first remaining event (lowest label)
    before ranking. The original notebooks did this unconditionally; it is off
    by default because it only made sense when the first label was the
    region outside the basin (see :func:`rain_objects`).
    """
    events = object_table(labels, time)
    events = events[events["Total time"] > min_duration]
    events = events[in_enso_phase(events, months)]
    if drop_first_event:
        events = events.iloc[1:]
    events = events.sort_values("area", ascending=False).head(max_events)
    return event_properties(events, labels, rain, time, lat, lon, elevation, **kwargs)
