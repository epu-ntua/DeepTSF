"""Tests of the models the pipeline cases produce, and of serving them.

For every successful pipeline case (see test_pipeline.py):

1. ``test_model_artifacts`` checks the model the training step logged to MLflow:
   the pyfunc model folder holds the model, its metadata, the scalers and the
   training series ids, and it loads through MLflow as the expected darts model.
2. ``test_inference_endpoint`` then serves that model through the API's
   ``POST /serving/get_result`` (api.py, the endpoint the frontend uses), with the
   history the model was trained on, and checks the forecast.

The pipeline run itself is shared with test_pipeline.py (support/pipeline.py
caches it), so these tests only add the checks and the API call.
"""
import math

import pandas as pd
import pytest

from support.cases import all_cases
from support.inference import api_client, build_request, expected_forecast_index, fake_open_meteo
from support.pipeline import get_run

CASES = [c for c in all_cases() if not c.expect_error]

# Class of the darts model each DeepTSF model name trains (training.py).
DARTS_CLASS = {
    "Naive": "NaiveSeasonal", "LightGBM": "LightGBMModel", "RandomForest": "RandomForest",
    "ARIMA": "ARIMA", "NBEATS": "NBEATSModel", "NHiTS": "NHiTSModel", "RNN": "RNNModel",
    "BlockRNN": "BlockRNNModel", "TCN": "TCNModel", "Transformer": "TransformerModel",
    "TFT": "TFTModel", "MLP": "MLPModel",
}


def _list_artifacts(client, run_id, path):
    out = []
    for a in client.list_artifacts(run_id, path):
        out += _list_artifacts(client, run_id, a.path) if a.is_dir else [a.path]
    return out


def _successful_run(case, workdir):
    run = get_run(case, workdir)
    if not run.success:
        pytest.skip("the pipeline run of this case failed (reported by test_pipeline)")
    return run


@pytest.mark.inference
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_model_artifacts(case, services, workdir):
    import mlflow
    from mlflow.tracking import MlflowClient

    run = _successful_run(case, workdir)
    train = run.train_run
    folder = train.data.tags["pyfunc_model_folder"]

    # the files darts_flavor._load_pyfunc reads (they sit in the pyfunc model's data/ folder)
    files = {p.split("/")[-1] for p in _list_artifacts(MlflowClient(), train.info.run_id, "pyfunc_model")}
    for name in ("MLmodel", "model_info.yml", "ts_id_l.pkl"):
        assert name in files, f"{name} missing from the model folder: {sorted(files)}"
    model_type = train.data.tags["model_type"]
    if model_type == "pkl":
        assert "_model.pkl" in files, sorted(files)
    else:
        assert any(f.endswith(".pth.tar") for f in files), f"no torch checkpoint in {sorted(files)}"

    loaded = mlflow.pyfunc.load_model(folder)
    wrapper = loaded._model_impl
    assert type(wrapper.model).__name__ == DARTS_CLASS[case.model]
    if case.model != "Naive":            # Naive is trained unscaled
        assert wrapper.transformer is not None, "series scaler was not packaged"
    if case.past_covs:
        assert wrapper.transformer_past_covs is not None, "past covariates scaler was not packaged"
    if case.future_covs:
        assert wrapper.transformer_future_covs is not None, "future covariates scaler was not packaged"


@pytest.mark.inference
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_inference_endpoint(case, services, workdir):
    run = _successful_run(case, workdir)
    body = build_request(run, weather=case.weather)
    with fake_open_meteo():
        response = api_client().post("/serving/get_result", json=body)
    assert response.status_code == 200, response.text

    forecast = pd.DataFrame(response.json())
    forecast.index = pd.to_datetime(forecast.index)
    forecast = forecast.sort_index()
    assert len(forecast) == case.spec.horizon, forecast
    assert list(forecast.index) == list(expected_forecast_index(run, weather=case.weather))
    values = forecast.to_numpy().ravel()
    assert all(math.isfinite(v) for v in values), forecast
