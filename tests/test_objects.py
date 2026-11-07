import numpy as np
import pandas as pd
import pytest

from snlda import objects


def _time(n):
    return pd.date_range("2000-01-01", periods=n, freq="D")


def test_rain_objects_connects_neighbouring_cells_with_different_values():
    data = np.zeros((1, 2, 3))
    data[0, 0, 0], data[0, 0, 1], data[0, 1, 1] = 2.0, 3.0, 5.0
    _, labels, n = objects.rain_objects(data, threshold=1.0)
    assert n == 1
    assert labels.max() == 1


def test_rain_objects_legacy_mode_only_connects_identical_values():
    data = np.zeros((1, 2, 3))
    data[0, 0, 0], data[0, 0, 1], data[0, 1, 1] = 2.0, 3.0, 3.0
    _, _, n_binary = objects.rain_objects(data, binary=True)
    _, _, n_legacy = objects.rain_objects(data, binary=False)
    assert n_binary == 1
    assert n_legacy == 2


def test_rain_objects_ignores_cells_below_threshold_and_nan():
    data = np.zeros((2, 3, 3))
    data[0, 0, 0] = 0.5  # below threshold
    data[1, 2, :] = np.nan  # outside the basin
    data[1, 1, 1] = 4.0
    rain, _, n = objects.rain_objects(data, threshold=1.0)
    assert n == 1
    assert rain[0, 0, 0] == 0
    assert rain[1, 1, 1] == 4.0
    assert not np.isnan(rain).any()


def test_object_table_duration_and_dates():
    time = _time(5)
    labels = np.zeros((5, 2, 2), dtype=int)
    labels[1:4, 0, 0] = 1  # days 1, 2, 3
    table = objects.object_table(labels, time)
    row = table.iloc[0]
    assert row["Total time"] == 3
    assert row["start date"] == time[1]
    assert row["end date"] == time[3]
    assert row["area"] == 3


def test_enso_months_and_phase_membership():
    oni = pd.DataFrame({"Year": [2000, 2000, 2001], "Month": [1, 2, 1], "ONI": [-0.8, 0.1, 0.9]})
    assert objects.enso_months(oni, "nina") == {(2000, 1)}
    assert objects.enso_months(oni, "nino") == {(2001, 1)}
    with pytest.raises(ValueError):
        objects.enso_months(oni, "neutral")
    events = pd.DataFrame(
        {
            "start date": pd.to_datetime(["2000-01-20", "2000-03-01"]),
            "end date": pd.to_datetime(["2000-02-05", "2000-03-10"]),
        }
    )
    flags = objects.in_enso_phase(events, {(2000, 1)})
    assert list(flags) == [True, False]  # first event starts in January 2000


def test_event_properties_volume_peak_and_centroids():
    time = _time(4)
    lat = np.array([2.0, 1.0, 0.0])  # north to south
    lon = np.array([290.0, 291.0, 292.0])
    data = np.zeros((4, 3, 3))
    data[0, 0, 0] = 10.0  # day 0, lat 2.0, lon 290
    data[1, 0, 0] = 20.0  # day 1
    data[1, 0, 1] = 30.0  # day 1, lon 291
    data[2, 0, 1] = 40.0  # day 2
    rain, labels, n = objects.rain_objects(data, threshold=1.0)
    assert n == 1
    events = objects.object_table(labels, time)
    out = objects.event_properties(
        events, labels, rain, time, lat, lon, cell_size_deg=1.0, km_per_degree=1.0, lon_offset=360.0
    )
    row = out.iloc[0]
    # cell area is (1 km)^2 = 1e6 m2; total rainfall 100 mm -> 1e6 * 0.1 m = 1e5 m3 = 0.1 Mm3
    assert row["Precipitation volume"] == pytest.approx(0.1)
    assert row["Precipitation max magnitude"] == 40.0
    assert row["Precipitation max date"] == time[2]
    assert row["Start Centroid lat"] == pytest.approx(2.0)
    assert row["Start Centroid long"] == pytest.approx(290.0 - 360.0)
    assert row["Finish Centroid long"] == pytest.approx(291.0 - 360.0)  # last day only has lon 291
    assert row["Average Centroid lat"] == pytest.approx(2.0)


def test_event_properties_elevation_at_start_and_finish():
    time = _time(2)
    lat = np.array([1.0, 0.0])
    lon = np.array([0.0, 1.0])
    data = np.zeros((2, 2, 2))
    data[0, 0, 0] = 5.0
    data[1, 1, 1] = 5.0
    data[0, 1, 1] = 5.0  # connects the two days diagonally in time
    rain, labels, _ = objects.rain_objects(data)
    elevation = np.array([[100.0, 200.0], [300.0, 400.0]])
    events = objects.object_table(labels, time)
    out = objects.event_properties(events, labels, rain, time, lat, lon, elevation=elevation)
    row = out.iloc[0]
    # day 0 footprint: cells (0,0) and (1,1); day 1 footprint: cell (1,1)
    assert row["Start average elev"] == pytest.approx(250.0)
    assert row["Finish average elev"] == pytest.approx(400.0)
    assert row["Finish Centroid elev"] == pytest.approx(400.0)


def test_enso_event_table_filters_by_duration_phase_and_size():
    time = pd.date_range("2000-01-01", periods=40, freq="D")
    lat = np.array([1.0, 0.0])
    lon = np.array([0.0, 1.0])
    data = np.zeros((40, 2, 2))
    data[0:3, 0, 0] = 5.0  # 3 days, January 2000 (in phase)
    data[5:7, 0, 0] = 5.0  # 2 days, January 2000; two diagonal cells form ONE event of area 4
    data[5:7, 1, 1] = 5.0
    data[10, 1, 0] = 5.0  # single day: too short
    data[35:38, 0, 1] = 5.0  # February 2000: out of phase
    rain, labels, _ = objects.rain_objects(data)
    out = objects.enso_event_table(
        labels, rain, time, lat, lon, months={(2000, 1)}, min_duration=1, max_events=1
    )
    assert len(out) == 1
    assert out.iloc[0]["area"] == 4  # the 2-day, 2-cell event is larger than the 3-day, 1-cell event
    all_events = objects.enso_event_table(labels, rain, time, lat, lon, months={(2000, 1)})
    assert len(all_events) == 2  # two events of >1 day touching January
    skipped = objects.enso_event_table(
        labels, rain, time, lat, lon, months={(2000, 1)}, drop_first_event=True
    )
    assert len(skipped) == 1
