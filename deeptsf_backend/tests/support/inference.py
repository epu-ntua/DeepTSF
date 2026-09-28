"""Serving the models produced by the pipeline cases through the real API.

``api_client()`` imports api.py (with USE_AUTH off, so no Keycloak) and wraps
the FastAPI app in a TestClient; ``build_request()`` turns the data a case was
trained on into a /serving/get_result request body, the way the frontend sends
one.
"""
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


def build_request(run) -> dict:
    """The history the case was trained on (and its covariates), asking for one horizon."""
    case, inputs = run.case, run.inputs
    series = _at_training_resolution(inputs.series, case)
    body = {
        "run_id": run.train_run.info.run_id,
        "timesteps_ahead": case.spec.horizon,
        "resolution": case.spec.freq,
        "multiple_file_type": case.kind == "multiple",
        "ts_id_pred": data.SERIES_IDS[case.kind][0],
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
        if frame is not None:
            body[key] = _as_long_records(_at_training_resolution(frame, case))
    return body


def expected_forecast_index(run) -> pd.DatetimeIndex:
    """Where the forecast must start: one step after the last input timestamp."""
    case = run.case
    served = _at_training_resolution(run.inputs.series, case)
    last = served["Datetime"].max()
    if case.irregular:
        # the API puts irregular input on a grid starting at midnight of the first day
        # (utils.regularize) and moves every reading to its nearest step
        first = served[served["Timeseries ID"] == data.SERIES_IDS[case.kind][0]]["Datetime"].min()
        origin, step = first.floor("D"), pd.Timedelta(case.spec.freq)
        last = origin + round((last - origin) / step) * step
    return pd.date_range(last, periods=case.spec.horizon + 1, freq=case.spec.freq)[1:]
