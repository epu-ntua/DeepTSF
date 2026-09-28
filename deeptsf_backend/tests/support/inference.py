"""Serving the models produced by the pipeline cases through the real API.

``api_client()`` imports api.py (with USE_AUTH off, so no Keycloak) and wraps
the FastAPI app in a TestClient; ``build_request()`` turns the data a case was
trained on into a /serving/get_result request body, the way the frontend sends
one. Weather covariates come from open-meteo in production; ``fake_open_meteo``
answers those requests locally with the same JSON layout.
"""
import json
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pandas as pd

from . import data
from .cases import Case

_CLIENT = None


def api_client():
    global _CLIENT
    if _CLIENT is None:
        from fastapi.testclient import TestClient

        import api
        _CLIENT = TestClient(api.app)
    return _CLIENT


def _as_long_records(df: pd.DataFrame) -> dict:
    out = df.reset_index(drop=True).copy()
    out["Datetime"] = out["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_dict()


def _at_training_resolution(df: pd.DataFrame, case: Case) -> pd.DataFrame:
    """Resampling cases train on a coarser resolution than the raw data; serve at that one."""
    if not case.train_resolution:
        return df
    return (df.set_index("Datetime").groupby(["ID", "Timeseries ID"])["Value"]
              .resample(case.train_resolution).mean().reset_index()[["Datetime", "ID", "Timeseries ID", "Value"]])


def _served_series(run, weather: bool) -> pd.DataFrame:
    """The history sent with the request. open-meteo only has weather from today on,
    so for weather requests the history is moved to end just after today's start,
    far enough in that the models' input windows are covered by weather data."""
    series = _at_training_resolution(run.inputs.series, run.case)
    if weather:
        spec = run.case.spec
        end = pd.Timestamp.today().normalize() + (spec.input_length + 2) * pd.Timedelta(spec.freq)
        series = series.assign(Datetime=series["Datetime"] + (end - series["Datetime"].max()))
    return series


def build_request(run, weather: bool = False) -> dict:
    case, inputs = run.case, run.inputs
    series = _served_series(run, weather)
    body = {
        "run_id": run.train_run.info.run_id,
        "timesteps_ahead": case.spec.horizon,
        "resolution": case.spec.freq,
        "multiple_file_type": case.kind == "multiple",
        "ts_id_pred": data.SERIES_IDS[case.kind][0],
        "weather_covariates": weather,
        "format": "long",
        "roll_size": case.spec.horizon,
        "batch_size": 16,
    }
    if case.kind == "multiple":
        body["series"] = _as_long_records(series)
    else:
        single = series.set_index("Datetime")[["Value"]]
        single.index = single.index.strftime("%Y-%m-%d %H:%M:%S")
        body["series"] = single.to_dict()
    for key, frame in (("past_covariates", inputs.past_covs), ("future_covariates", inputs.future_covs)):
        if frame is not None and not (weather and key == "future_covariates"):
            body[key] = _as_long_records(_at_training_resolution(frame, case))
    return body


def expected_forecast_index(run, weather: bool = False) -> pd.DatetimeIndex:
    """Where the forecast must start: one step after the last input timestamp."""
    case = run.case
    last = _served_series(run, weather)["Datetime"].max()
    if case.irregular:
        # the API puts irregular input on a grid starting at midnight of the first day
        # (utils.regularize) and moves every reading to its nearest step
        served = _served_series(run, weather)
        first = served[served["Timeseries ID"] == data.SERIES_IDS[case.kind][0]]["Datetime"].min()
        origin, step = first.floor("D"), pd.Timedelta(case.spec.freq)
        last = origin + round((last - origin) / step) * step
    return pd.date_range(last, periods=case.spec.horizon + 1, freq=case.spec.freq)[1:]


class _OpenMeteoResponse:
    def __init__(self, fields):
        start = pd.Timestamp.today().normalize()
        times = pd.date_range(start, periods=24 * 10, freq="1h")
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        for i, f in enumerate(fields):
            hourly[f] = list(np.round(200 + 100 * np.sin(np.arange(len(times)) * 2 * np.pi / 24 + i), 2))
        self.text = json.dumps({"hourly": hourly})
        self.status_code = 200

    def json(self):
        return json.loads(self.text)


@contextmanager
def fake_open_meteo():
    """Answer open-meteo forecast requests locally (10 days of hourly values from today,
    like the real `forecast_days=10` call); anything else fails loudly."""
    import requests

    real_get = requests.get

    def fake_get(url, *args, **kwargs):
        if "api.open-meteo.com" not in str(url):
            return real_get(url, *args, **kwargs)
        fields = str(url).split("hourly=")[1].split("&")[0].split(",")
        return _OpenMeteoResponse(fields)

    with mock.patch("requests.get", fake_get):
        yield

