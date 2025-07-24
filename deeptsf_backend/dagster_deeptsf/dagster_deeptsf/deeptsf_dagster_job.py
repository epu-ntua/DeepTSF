import os

import mlflow
from dagster import op, get_dagster_logger, graph, Definitions, ScheduleDefinition, job
from dagster import Config
from dagster_shell.ops import shell_op
from dagster import multi_asset, AssetIn, AssetOut, MetadataValue, Output, graph_multi_asset, define_asset_job, asset
from dagster import ConfigurableResource, Field
from pydantic import Field
from dagster_mlflow import end_mlflow_on_run_finished, mlflow_tracking
from dagster_deeptsf.assets import start_pipeline_run, training_and_hyperparameter_tuning_asset
from dagster_deeptsf.load_raw_data import load_raw_data_asset
from dagster_deeptsf.etl import etl_asset
from dagster_deeptsf.evaluate_forecasts import evaluation_asset
from typing import Optional
from dagster import ConfigurableResource

class DeepTSFConfig(ConfigurableResource):
    resolution: str = "None"  
    experiment_name: str = "Default"
    parent_run_name: str = "None"
    series_csv: str = "series_csv"
    series_uri: str = "None"
    past_covs_csv: str = "None"
    past_covs_uri: str = "None"
    future_covs_csv: str = "None"
    future_covs_uri: str = "None"
    year_range: str = "None"
    time_covs: bool = False
    hyperparams_entrypoint: dict = {"insert key": "insert value"}
    cut_date_val: str = "None"
    cut_date_test: str = "None"
    test_end_date: str = "None"
    darts_model: str = "None"
    device: str = "gpu"
    forecast_horizon: int = -1
    stride: int = -1
    retrain: bool = False
    ignore_previous_runs: bool = True
    scale: bool = True
    scale_covs: bool = True
    country: str = "PT"
    std_dev: float = 4.5
    max_thr: int = -1
    a: float = 0.3
    wncutoff: float = 0.000694
    ycutoff: float = 3
    ydcutoff: float = 30
    shap_data_size: int = 100
    analyze_with_shap: bool = False
    multiple: bool = False
    eval_series: str = "eval_series"
    n_trials: int = 100
    opt_test: bool = False
    from_database: bool = False
    database_name: str = "rdn_load_data"
    trial_name: str = "Default"
    num_workers: int = 4
    eval_method: str = "ts_ID"
    imputation_method: str = "linear"
    order: int = 1
    rmv_outliers: bool = True
    loss_function: str = "mape"
    evaluate_all_ts: bool = True
    convert_to_local_tz: bool = False
    grid_search: bool = False
    shap_input_length: int = -1
    ts_used_id: str = "None"
    m_mase: int = 1
    min_non_nan_interval: int = 24
    num_samples: int = 1
    resampling_agg_method: str = "averaging"
    pv_ensemble: bool = False
    format: str = "long"

    def to_dict(self):
        return {
            "experiment_name": self.experiment_name,
            "parent_run_name": self.parent_run_name,
            "series_csv": self.series_csv,
            "series_uri": self.series_uri,
            "past_covs_csv": self.past_covs_csv,
            "past_covs_uri": self.past_covs_uri,
            "future_covs_csv": self.future_covs_csv,
            "future_covs_uri": self.future_covs_uri,
            "resolution": self.resolution,
            "year_range": self.year_range,
            "time_covs": self.time_covs,
            "hyperparams_entrypoint": self.hyperparams_entrypoint,
            "cut_date_val": self.cut_date_val,
            "cut_date_test": self.cut_date_test,
            "test_end_date": self.test_end_date,
            "darts_model": self.darts_model,
            "device": self.device,
            "forecast_horizon": self.forecast_horizon,
            "stride": self.stride,
            "retrain": self.retrain,
            "ignore_previous_runs": self.ignore_previous_runs,
            "scale": self.scale,
            "scale_covs": self.scale_covs,
            "country": self.country,
            "std_dev": self.std_dev,
            "max_thr": self.max_thr,
            "a": self.a,
            "wncutoff": self.wncutoff,
            "ycutoff": self.ycutoff,
            "ydcutoff": self.ydcutoff,
            "shap_data_size": self.shap_data_size,
            "analyze_with_shap": self.analyze_with_shap,
            "multiple": self.multiple,
            "eval_series": self.eval_series,
            "n_trials": self.n_trials,
            "opt_test": self.opt_test,
            "from_database": self.from_database,
            "trial_name": self.trial_name,
            "database_name": self.database_name,
            "num_workers": self.num_workers,
            "eval_method": self.eval_method,
            "imputation_method": self.imputation_method,
            "order": self.order,
            "rmv_outliers": self.rmv_outliers,
            "loss_function": self.loss_function,
            "evaluate_all_ts": self.evaluate_all_ts,
            "convert_to_local_tz": self.convert_to_local_tz,
            "grid_search": self.grid_search,
            "shap_input_length": self.shap_input_length,
            "ts_used_id": self.ts_used_id,
            "m_mase": self.m_mase,
            "min_non_nan_interval": self.min_non_nan_interval,
            "num_samples": self.num_samples,
            "resampling_agg_method": self.resampling_agg_method,
            "pv_ensemble": self.pv_ensemble,
            "format": self.format,
        }

@graph_multi_asset(
    name="deepTSF_pipeline",
    group_name='deepTSF_pipeline',
    outs={
        "start_pipeline_run": AssetOut(dagster_type=str),
        "load_raw_data_out": AssetOut(dagster_type=dict),
        "etl_out": AssetOut(dagster_type=dict),
        "training_and_hyperparameter_tuning_out": AssetOut(dagster_type=dict),
        "evaluation_out": AssetOut(dagster_type=dict)})

def deepTSF_pipeline():
    # mlflow.set_experiment("dagster_test")
    # with mlflow.start_run(tags={"mlflow.runName": "darts_model" + '_pipeline'}) as active_run:
    # active_run = context.resources.mlflow.active_run()
    parent_run_id = start_pipeline_run()

    # Prepare data
    load_raw_data_out = load_raw_data_asset(parent_run_id)

    etl_out = etl_asset(parent_run_id, load_raw_data_out)

    training_and_hyperparameter_tuning_out = training_and_hyperparameter_tuning_asset(parent_run_id, etl_out)

    evaluation_out = evaluation_asset(parent_run_id, training_and_hyperparameter_tuning_out)

    return {'start_pipeline_run' : parent_run_id,
            'load_raw_data_out': load_raw_data_out,
            'etl_out': etl_out,
            'training_and_hyperparameter_tuning_out': training_and_hyperparameter_tuning_out,
            'evaluation_out': evaluation_out}

deeptsf_dagster_job = define_asset_job("deeptsf_dagster_job", selection=[deepTSF_pipeline])

# basic_schedule = ScheduleDefinition(job=deeptsf_dagster_job, 
#                                     cron_schedule="0 0 * * *",
#                                     run_config=DeepTSFConfig())