"""Runs a ``Case`` through the real DeepTSF Dagster job, in this process.

The job (dagster_deeptsf.deeptsf_dagster_job.deepTSF_pipeline) is materialized
with dagster's in-process executor instead of celery and an in-memory IO
manager instead of MinIO; everything inside the assets (downloads from S3,
MLflow runs and artifacts, model packaging) runs unchanged against the local
servers from servers.py.

Results are cached per case, so the pipeline test and the inference tests of a
case share one run (conftest.py keeps a case's tests on one xdist worker).
"""
import os
import traceback
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import data
from .cases import WEATHER_FIELD, Case, hyperparameters, optuna_hyperparameters

INPUT_BUCKET = "dataset-storage"


@dataclass
class CaseInputs:
    """What the case feeds the pipeline, kept for building inference requests."""
    series: pd.DataFrame                   # long format, also for single series
    past_covs: Optional[pd.DataFrame]
    future_covs: Optional[pd.DataFrame]
    series_csv: str                        # bucket/key, as the frontend passes it
    past_covs_csv: str = "None"
    future_covs_csv: str = "None"


@dataclass
class PipelineRun:
    case: Case
    inputs: CaseInputs
    success: bool
    error: str = ""
    parent_run: object = None                          # mlflow Run
    children: dict = field(default_factory=dict)       # runName -> mlflow Run

    @property
    def train_run(self):
        return next((r for r in self.children.values() if r.data.tags.get("stage") == "training"
                     or "model_uri" in r.data.tags), None)

    @property
    def eval_run(self):
        return self.children.get("eval")


def _upload(local_path, key: str) -> str:
    import boto3
    s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"], region_name="us-east-1")
    s3.upload_file(str(local_path), INPUT_BUCKET, key)
    return f"{INPUT_BUCKET}/{key}"


def prepare_inputs(case: Case, workdir) -> CaseInputs:
    spec = case.data_spec
    series = data.series_frame(spec, case.kind)
    if case.irregular:
        series = data.jitter(series, spec)
    local = workdir / "inputs" / case.id
    series_csv = _upload(data.write_series_csv(series, case.kind, local / "series.csv"), f"{case.id}/series.csv")
    inputs = CaseInputs(series=series, past_covs=None, future_covs=None, series_csv=series_csv)
    if case.past_covs:
        inputs.past_covs = data.covariates_frame(spec, case.kind, names=("past_1", "past_2"), seed=20)
        inputs.past_covs_csv = _upload(data.write_covariates_csv(inputs.past_covs, local / "past.csv"),
                                       f"{case.id}/past.csv")
    if case.future_covs:
        names = (WEATHER_FIELD,) if case.weather else ("future_1", "future_2")
        inputs.future_covs = data.covariates_frame(spec, case.kind, names=names, seed=30)
        inputs.future_covs_csv = _upload(data.write_covariates_csv(inputs.future_covs, local / "future.csv"),
                                         f"{case.id}/future.csv")
    return inputs


def build_config(case: Case, inputs: CaseInputs) -> dict:
    """The resource config the frontend / API would send for this case (api.py run_all)."""
    spec = case.spec
    return {
        "experiment_name": f"tests-{case.id}",
        "parent_run_name": case.id,
        "trial_name": case.id,
        "series_csv": inputs.series_csv,
        "past_covs_csv": inputs.past_covs_csv,
        "future_covs_csv": inputs.future_covs_csv,
        "resolution": spec.freq,
        "darts_model": case.model,
        "hyperparams_entrypoint": optuna_hyperparameters(case) if case.optuna else hyperparameters(case),
        **data.split_dates(case.data_spec),
        "forecast_horizon": spec.horizon,
        "multiple": case.kind == "multiple",
        "format": "long",
        "device": "cpu",
        "num_workers": 2,
        "time_covs": case.time_covs,
        "opt_test": case.optuna,
        "n_trials": 2,
        "analyze_with_shap": case.shap,
        "shap_data_size": 2,
        "shap_input_length": spec.input_length,
        # SHAP explains one test series; otherwise evaluate every series
        "evaluate_all_ts": not case.shap,
        "eval_series": data.SERIES_IDS[case.kind][0],
        "convert_to_local_tz": False,        # as sent by the API
        "rmv_outliers": True,
        "imputation_method": "linear",
        "m_mase": 1,
        "tenant": "None",
    }


def _failure_message(result) -> str:
    msgs = []
    for event in result.all_events:
        if event.is_step_failure:
            err = event.event_specific_data.error
            msgs.append(f"[{event.step_key}] {err.to_string() if err else event.message}")
    return "\n".join(msgs) or "run failed without a step failure event"


def _collect_mlflow(run: PipelineRun) -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = client.get_experiment_by_name(f"tests-{run.case.id}")
    if exp is None:
        return
    runs = client.search_runs([exp.experiment_id], max_results=1000)
    parents = [r for r in runs if r.data.tags.get("stage") == "main"]
    if parents:
        run.parent_run = parents[0]
        for r in runs:
            if r.data.tags.get("mlflow.parentRunId") == run.parent_run.info.run_id:
                run.children[r.data.tags.get("mlflow.runName", r.info.run_id)] = r


def run_case(case: Case, workdir) -> PipelineRun:
    from dagster import DagsterInstance, materialize, mem_io_manager

    from dagster_deeptsf.deeptsf_dagster_job import DeepTSFConfig, deepTSF_pipeline

    inputs = prepare_inputs(case, workdir)
    config = build_config(case, inputs)
    try:
        with DagsterInstance.ephemeral() as instance:
            result = materialize(
                [deepTSF_pipeline],
                resources={"config": DeepTSFConfig(**config), "io_manager": mem_io_manager},
                instance=instance,
                raise_on_error=False,
            )
            run = PipelineRun(case, inputs, success=result.success,
                              error="" if result.success else _failure_message(result))
    except Exception:
        run = PipelineRun(case, inputs, success=False, error=traceback.format_exc())
    _collect_mlflow(run)
    return run


_CACHE = {}


def get_run(case: Case, workdir) -> PipelineRun:
    if case.id not in _CACHE:
        _CACHE[case.id] = run_case(case, workdir)
    return _CACHE[case.id]
