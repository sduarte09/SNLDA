import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from snlda import dynamics


def _sine(n=2000, period=50):
    return np.sin(2 * np.pi * np.arange(n) / period)


def test_reconstruct_shape_and_values():
    x = np.arange(10.0)
    emb = dynamics.reconstruct(x, lag=2, n_dims=3)
    assert emb.shape == (6, 3)
    assert list(emb[0]) == [0.0, 2.0, 4.0]
    assert list(emb[-1]) == [5.0, 7.0, 9.0]


def test_reconstruct_one_dimension_returns_the_series():
    x = np.arange(5.0)
    assert dynamics.reconstruct(x, lag=3, n_dims=1).ravel().tolist() == x.tolist()


def test_reconstruct_then_deconstruct_round_trips():
    x = np.random.default_rng(0).random(30)
    emb = dynamics.reconstruct(x, lag=3, n_dims=4)
    np.testing.assert_allclose(dynamics.deconstruct(emb, lag=3, n_dims=4), x)


def test_reconstruct_too_short_raises():
    with pytest.raises(ValueError):
        dynamics.reconstruct(np.arange(5.0), lag=3, n_dims=4)


def test_autocorrelation_delay_of_sine_is_about_a_quarter_period():
    assert 11 <= dynamics.autocorrelation_delay(_sine(period=50)) <= 14


def test_constant_series_raises():
    with pytest.raises(ValueError):
        dynamics.autocorrelation_delay(np.ones(100))


def test_series_must_be_one_dimensional():
    with pytest.raises(ValueError):
        dynamics.reconstruct(np.ones((3, 3)), lag=1, n_dims=2)


def test_optimal_delay_is_at_most_the_autocorrelation_delay():
    x = _sine(period=50) + np.random.default_rng(1).normal(0, 0.05, 2000)
    assert dynamics.optimal_delay(x) <= dynamics.autocorrelation_delay(x)


def test_false_neighbors_drop_when_the_attractor_is_unfolded():
    x = _sine(1500, period=50)
    dims, frac = dynamics.false_nearest_neighbors(x, lag=12, min_dims=1, max_dims=4)
    assert list(dims) == [1, 2, 3, 4]
    assert frac[0] > frac[1]  # a circle needs two dimensions
    assert frac[1] < 0.2


def _fake_nolds():
    fake = mock.MagicMock()
    fake.lyap_e.return_value = np.array([0.3, -0.1])
    fake.hurst_rs.return_value = 0.6
    return fake


def test_dynamics_analysis_uses_fnn_index_as_dimension_by_default():
    fake = _fake_nolds()
    with mock.patch.dict(sys.modules, {"nolds": fake}):
        res = dynamics.dynamics_analysis(_sine(1500, period=50))
        fnn_dims, fnn = dynamics.false_nearest_neighbors(
            _sine(1500, period=50), res.lag, 1, 2 * int(np.floor(2 * np.log10(1500))) + 1
        )
    expected_index = int(np.flatnonzero(fnn == 0)[0]) if np.any(fnn == 0) else int(
        np.flatnonzero(fnn <= 0.2)[0]
    )
    assert fake.lyap_e.call_args.kwargs["emb_dim"] == expected_index
    assert res.embedding_dim == expected_index
    assert res.hurst == 0.6


def test_dimension_offset_one_gives_the_real_dimension():
    fake = _fake_nolds()
    with mock.patch.dict(sys.modules, {"nolds": fake}):
        base = dynamics.dynamics_analysis(_sine(1500, period=50))
        shifted = dynamics.dynamics_analysis(_sine(1500, period=50), dimension_offset=1)
    assert shifted.embedding_dim == base.embedding_dim + 1


def test_dynamics_analysis_returns_sentinel_when_no_dimension_passes():
    fake = _fake_nolds()
    noise = np.random.default_rng(2).random(400)
    with mock.patch.dict(sys.modules, {"nolds": fake}):
        res = dynamics.dynamics_analysis(noise, fnn_threshold=-1.0)
    assert res.embedding_dim == dynamics.NO_DIMENSION
    assert list(res.lyapunov) == [dynamics.NO_DIMENSION]
    fake.lyap_e.assert_not_called()


def test_dynamics_summary_and_table_with_failing_series():
    fake = _fake_nolds()
    df = pd.DataFrame({"ok": _sine(800, period=40), "flat": np.ones(800)})
    with mock.patch.dict(sys.modules, {"nolds": fake}):
        table = dynamics.dynamics_table(df)
    assert table.loc["ok", "hurst"] == 0.6
    assert table.loc["ok", "lyapunov_max"] == pytest.approx(0.3)
    assert table.loc["ok", "lyapunov_sum"] == pytest.approx(0.2)
    assert table.loc["flat"].isna().all()


def test_dynamics_on_grid_skips_masked_and_nan_pixels():
    fake = _fake_nolds()
    data = np.stack([_sine(600, 40), _sine(600, 40), _sine(600, 40)], axis=1)[:, :, None]
    data[0, 2, 0] = np.nan
    mask = np.array([[1.0], [np.nan], [1.0]])
    with mock.patch.dict(sys.modules, {"nolds": fake}):
        out = dynamics.dynamics_on_grid(data, mask)
    assert out["hurst"][0, 0] == 0.6
    assert np.isnan(out["hurst"][1, 0])  # outside the mask
    assert np.isnan(out["hurst"][2, 0])  # contains NaN


def test_hurst_exponent_of_white_noise_is_about_one_half():
    try:
        import nolds
    except (ImportError, TypeError) as error:  # some nolds releases do not import on old Pythons
        pytest.skip(f"nolds cannot be imported in this environment: {error}")
    x = np.random.default_rng(0).normal(size=4000)
    assert 0.4 < nolds.hurst_rs(x) < 0.7


def test_clear_error_when_nolds_cannot_be_imported():
    # setting the entry to None makes ``import nolds`` fail with ImportError
    with mock.patch.dict(sys.modules, {"nolds": None}):
        with pytest.raises(ImportError):
            dynamics.dynamics_analysis(_sine(600, 40))
