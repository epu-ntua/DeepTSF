"""Unit tests of resolution inference and regularization of irregular series
(utils.infer_resolution / utils.regularize, same code in utils_backend).

These run in milliseconds and need no servers; the end-to-end behaviour on
irregular inputs is covered by the "irregular" cases of test_pipeline.py.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module", params=["utils", "utils_backend"])
def helpers(request):
    import importlib
    return importlib.import_module(request.param)


def _series(index):
    return pd.Series(np.arange(len(index), dtype=float), index=pd.DatetimeIndex(index, name="Datetime"),
                     name="Value")


@pytest.mark.parametrize("freq, expected", [("15min", "15min"), ("1h", "1h"), ("1D", "1d"), ("7D", "7d")])
def test_regular_series_keep_their_step(helpers, freq, expected):
    idx = pd.date_range("2024-01-01", periods=50, freq=freq)
    assert helpers.infer_resolution(idx) == expected


def test_missing_timestamps_do_not_change_the_resolution(helpers):
    idx = pd.date_range("2024-01-01", periods=100, freq="1h").delete([5, 6, 7, 40])
    assert helpers.infer_resolution(idx) == "1h"


def test_jitter_uses_the_typical_step(helpers):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=200, freq="1h") + pd.to_timedelta(rng.integers(-90, 91, 200), "s")
    assert helpers.infer_resolution(idx) == "1h"


def test_stray_reading_does_not_set_the_resolution(helpers):
    idx = pd.date_range("2024-01-01", periods=96, freq="15min").append(pd.DatetimeIndex(["2024-01-01 03:07"]))
    assert helpers.infer_resolution(idx.sort_values()) == "15min"


def test_regular_series_is_only_asfreq(helpers):
    idx = pd.date_range("2024-01-01", periods=10, freq="1h").delete(4)
    out = helpers.regularize(_series(idx), "1h")
    assert len(out) == 10 and out.isna().sum() == 1
    assert out.dropna().tolist() == _series(idx).tolist()


def test_jittered_readings_land_on_their_nearest_step(helpers):
    base = pd.date_range("2024-01-01", periods=48, freq="1h")
    shift = pd.to_timedelta(np.tile([-40, 25, -5, 50], 12), "s")     # both sides of the hour
    out = helpers.regularize(_series(base + shift), "1h")
    assert list(out.index) == list(base)
    assert out.tolist() == list(range(48)), "each reading must keep its own step"
    assert out.index.name == "Datetime"


def test_weekly_readings_on_different_weekdays(helpers):
    base = pd.date_range("2024-01-03", periods=30, freq="7D")        # Wednesdays
    shift = pd.to_timedelta(np.tile([0, 1, 2, -1], 8)[:30], "D")
    out = helpers.regularize(_series(base + shift), "7d")
    assert len(out) == 30 and out.notna().all()


def test_dataframes_are_regularized_too(helpers):
    idx = pd.date_range("2024-01-01", periods=24, freq="1h") + pd.Timedelta("10s")
    df = _series(idx).to_frame()
    df.index = df.index.insert(3, pd.Timestamp("2024-01-01 02:31"))[:-1]  # one reading off the grid
    out = helpers.regularize(df, "1h")
    assert list(out.columns) == ["Value"] and out.index.freqstr in ("h", "H")
