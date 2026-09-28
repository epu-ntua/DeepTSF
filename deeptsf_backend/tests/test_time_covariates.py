"""Unit tests of the calendar features used as future covariates (time_covs).

The ETL builds them for training and darts_flavor rebuilds them at inference;
both must produce the same components, in the same order, or a served model sees
different inputs than it was trained on. Models trained with time_covs and served
through the API are covered end to end by the "time_covariates" cases.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def modules(services):
    # etl.py creates its MinIO client at import, so the local servers must be up first
    import importlib
    return {name: importlib.import_module(name) for name in ("utils", "utils_backend", "dagster_deeptsf.etl")}


@pytest.mark.parametrize("freq", ["15min", "1h", "1D", "7D"])
def test_etl_and_inference_build_the_same_features(modules, freq):
    index = pd.date_range("2024-03-25", periods=200, freq=freq)
    etl_frames, etl_ids, _ = modules["dagster_deeptsf.etl"].get_time_covariates(
        pd.Series(np.ones(len(index)), index=index), "PT", "A")
    etl = np.column_stack([f.to_numpy().ravel() for f in etl_frames])
    for name in ("utils", "utils_backend"):
        built = modules[name].time_covariates(index, "PT")
        assert built.n_components == len(modules[name].TIME_COVARIATE_NAMES) == len(etl_ids)
        np.testing.assert_allclose(built.values(), etl, err_msg=f"{name} differs from the ETL")
    assert etl_ids == modules["utils"].TIME_COVARIATE_NAMES == modules["utils_backend"].TIME_COVARIATE_NAMES


def test_holiday_calendar_falls_back_to_the_configured_country(modules):
    index = pd.date_range("2024-12-24", periods=3, freq="1D")        # Christmas in the middle
    for name in ("utils", "utils_backend"):
        by_id = modules[name].time_covariates_for(index, "GR", "PT")   # the id is a country: used
        fallback = modules[name].time_covariates_for(index, "A", "PT")  # not a country: PT
        assert by_id["holidays"].values().ravel().tolist() == [0, 1, 1]  # Greece: 25 and 26 Dec
        assert fallback["holidays"].values().ravel().tolist() == [0, 1, 0]  # Portugal: 25 Dec
