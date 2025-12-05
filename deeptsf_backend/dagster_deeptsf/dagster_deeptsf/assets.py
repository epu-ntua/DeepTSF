import os

import mlflow

from dagster import op, get_dagster_logger, graph, Definitions, ScheduleDefinition, job
from dagster import Config
from dagster_shell.ops import shell_op
from dagster import multi_asset, AssetIn, AssetOut, MetadataValue, Output, graph_multi_asset, define_asset_job, asset
from dagster import ConfigurableResource
from dagster_mlflow import end_mlflow_on_run_finished, mlflow_tracking
from .optuna_search import optuna_search
from .training import train
from dotenv import load_dotenv
load_dotenv()
from minio import Minio
import logging
import sys
sys.path.append('..')
from utils import none_checker, check_mandatory, truth_checker, download_online_file, load_yaml_as_dict

import re
from datetime import datetime
from typing import Optional

def validate_dates(
    cut_date_val: str,
    cut_date_test: str,
    test_end_date: Optional[str] = None,
) -> None:
    """
    Validate experiment‑split dates.

    • Every non‑None date string must be exactly YYYYMMDD and a real calendar date.
    • Always enforce:         cut_date_val < cut_date_test
    • Additionally enforce:   cut_date_test < test_end_date  (if test_end_date is provided)

    Raises
    ------
    ValueError – on the first failure encountered.
    """

    DATE_PATTERN = re.compile(r"\d{8}$")  # exactly eight digits → YYYYMMDD
    FMT = "%Y%m%d"

    # ---------- 1. Format check ------------------------------------------------
    for name, value in {
        "cut_date_val": cut_date_val,
        "cut_date_test": cut_date_test,
        "test_end_date": test_end_date,
    }.items():
        if value is None:
            continue  # optional -> skip
        if not DATE_PATTERN.fullmatch(str(value)):
            raise ValueError(f"{name}='{value}' is not in YYYYMMDD format.")

    # ---------- 2. Parse to datetime (validity) --------------------------------
    try:
        d_val  = datetime.strptime(cut_date_val,  FMT)
        d_test = datetime.strptime(cut_date_test, FMT)
        d_end  = (
            datetime.strptime(test_end_date, FMT)
            if test_end_date is not None else None
        )
    except ValueError as exc:                    # e.g. 20250230
        raise ValueError(f"Invalid calendar date: {exc}")

    # ---------- 3. Chronological ordering --------------------------------------
    if not (d_val < d_test):
        raise ValueError(
            f"Date order must satisfy cut_date_val < cut_date_test "
            f"({cut_date_val} !< {cut_date_test})"
        )
    if d_end is not None and not (d_test < d_end):
        raise ValueError(
            f"Date order must satisfy cut_date_test < test_end_date "
            f"({cut_date_test} !< {test_end_date})"
        )

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MINIO_CLIENT_URL = os.environ.get("MINIO_CLIENT_URL")
MINIO_SSL = truth_checker(os.environ.get("MINIO_SSL"))
client = Minio(MINIO_CLIENT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, secure=MINIO_SSL)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

@asset(
    name="start_pipeline_run",
    group_name='deepTSF_pipeline',
    required_resource_keys={"config"},)

def start_pipeline_run(context):
    config = context.resources.config
    tenant = context.config.tenant
    mlflow_uri = f"http://{tenant}-mlflow:5000"
    mlflow.set_tracking_uri(mlflow_uri)

    experiment_name = config.experiment_name
    darts_model = config.darts_model
    parent_run_name = config.parent_run_name if none_checker(config.parent_run_name) != None else darts_model + '_pipeline'
    series_uri = config.series_uri
    from_database = config.from_database
    series_csv = config.series_csv
    past_covs_csv = config.past_covs_csv
    future_covs_csv = config.future_covs_csv

    resolution = config.resolution
    darts_model = config.darts_model
    hyperparams_entrypoint = config.hyperparams_entrypoint
    cut_date_val = config.cut_date_val
    cut_date_test = config.cut_date_test
    test_end_date = config.test_end_date
    forecast_horizon = config.forecast_horizon
    eval_series = config.eval_series
    multiple = config.multiple
    evaluate_all_ts = config.evaluate_all_ts
    stride = config.stride

    if none_checker(series_uri) is None and not from_database and none_checker(series_uri) is None:
        check_mandatory(series_csv, "series_csv", [["series_uri", "None"], ["from_database", "False"]])

    if none_checker(resolution) is None:
        check_mandatory(resolution, "resolution", [])

    if none_checker(darts_model) is None:
        check_mandatory(darts_model, "darts_model", [])
        
    if none_checker(hyperparams_entrypoint) is None:
        check_mandatory(hyperparams_entrypoint, "hyperparams_entrypoint", [])

    if none_checker(cut_date_val) is None:
        check_mandatory(cut_date_val, "cut_date_val", [])

    if none_checker(cut_date_test) is None:
        check_mandatory(cut_date_test, "cut_date_test", [])

    if none_checker(forecast_horizon) is None or forecast_horizon == -1:
        forecast_horizon = None
        check_mandatory(forecast_horizon, "forecast_horizon", [])

    if none_checker(eval_series) is None and multiple and not evaluate_all_ts:
        check_mandatory(eval_series, "eval_series", [["multiple", "True"], ["evaluate_all_ts", "False"]])

    test_end_date = none_checker(test_end_date)
    validate_dates(cut_date_val, cut_date_test, test_end_date)

    if not (none_checker(stride) is None) and not (stride == -1):
        if stride > forecast_horizon:
            raise ValueError(
                "Stride values greater than the forecast horizon "
                "are not supported currently."
            )

    if none_checker(series_csv):
        download_online_file(client, f'dataset-storage/{series_csv}', dst_dir='dataset-storage', bucket_name='dataset-storage')

    if none_checker(future_covs_csv):
        download_online_file(client, f'dataset-storage/{future_covs_csv}', dst_dir='dataset-storage', bucket_name='dataset-storage')

    if none_checker(past_covs_csv):
        download_online_file(client, f'dataset-storage/{past_covs_csv}', dst_dir='dataset-storage', bucket_name='dataset-storage')


    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(tags={"mlflow.runName": parent_run_name}) as active_run:
        mlflow.set_tag("stage", "main")
        return Output(active_run.info.run_id)

@multi_asset(
    name="training_and_hyperparameter_tuning_asset",
    description="For training the model and / or hyperparameter tuning",
    group_name='deepTSF_pipeline',
    required_resource_keys={"config"},
    ins={'start_pipeline_run': AssetIn(key='start_pipeline_run', dagster_type=str),
         'etl_out': AssetIn(key='etl_out', dagster_type=dict)},
    outs={"training_and_hyperparameter_tuning_out": AssetOut(dagster_type=dict)}
    )

def training_and_hyperparameter_tuning_asset(context, start_pipeline_run, etl_out):
    config = context.resources.config
    tenant = context.config.tenant
    mlflow_uri = f"http://{tenant}-mlflow:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    opt_test = config.opt_test
    experiment_name = config.experiment_name
    darts_model = config.darts_model
    parent_run_name = config.parent_run_name if none_checker(config.parent_run_name) != None else darts_model + '_pipeline'


    if opt_test:
        child_run_id, parameters_dict = optuna_search(context, start_pipeline_run, etl_out)
    else:
        child_run_id, parameters_dict = train(context, start_pipeline_run, etl_out)
    
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(tags={"mlflow.runName": parent_run_name}, run_id=start_pipeline_run) as parent_run:
        completed_run = mlflow.tracking.MlflowClient().get_run(child_run_id)

        for param_name, param_value in parameters_dict.items():
            try:
                mlflow.log_param(param_name, param_value)
            except mlflow.exceptions.RestException:
                pass
            except mlflow.exceptions.MlflowException:
                pass

        if "model_uri" not in completed_run.data.tags:
            print(f'\nHyperparameter tuning did not produce new model. Skipping Evaluation')
            logging.info(f'\nHyperparameter tuning did not produce new model. Skipping Evaluation')
            return Output({"series_uri": None,
                    "past_covariates_uri": None,
                    "future_covariates_uri": None,
                    "model_uri": None,
                    "model_type": None,
                    "scaler_uri": None,
                    "setup_uri": None,
                    "shap_input_length": None,
                    "retrain": False,
                    "cut_date_test": None,
                    "test_end_date": None,
                })

        model_uri = completed_run.data.tags["model_uri"].replace("s3:/", S3_ENDPOINT_URL)
        model_type = completed_run.data.tags["model_type"]
        series_uri = completed_run.data.tags["series_uri"].replace("s3:/", S3_ENDPOINT_URL)
        future_covariates_uri = completed_run.data.tags["future_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
        past_covariates_uri = completed_run.data.tags["past_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
        scaler_uri = completed_run.data.tags["scaler_uri"].replace("s3:/", S3_ENDPOINT_URL)
        setup_uri = completed_run.data.tags["setup_uri"].replace("s3:/", S3_ENDPOINT_URL)
        scaler_past_covariates_uri = completed_run.data.tags["scaler_past_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
        scaler_future_covariates_uri = completed_run.data.tags["scaler_future_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)

        setup_file = download_online_file(
            client, setup_uri, "setup.yml")
        setup = load_yaml_as_dict(setup_file)
        print(f"\nSplit info: {setup} \n")
    
        #CHECK AND CHANGE
        if "input_chunk_length" in completed_run.data.tags:
            shap_input_length = completed_run.data.tags["input_chunk_length"]
        else:
            shap_input_length = config.shap_input_length
            
        # Naive models require retrain=True
        if "naive" in [parameters_dict["darts_model"].lower()]:
            retrain = True
        else:
            retrain = config.retrain
            print("Warning: Switching retrain flag to True as Naive models require...\n")

    return Output({"series_uri": series_uri,
                    "past_covariates_uri": past_covariates_uri,
                    "future_covariates_uri": future_covariates_uri,
                    "model_uri": model_uri,
                    "model_type": model_type,
                    "scaler_uri": scaler_uri,
                    "scaler_past_covariates_uri": scaler_past_covariates_uri,
                    "scaler_future_covariates_uri": scaler_future_covariates_uri,
                    "setup_uri": setup_uri,
                    "shap_input_length": shap_input_length,
                    "retrain": retrain,
                    "cut_date_test": setup['test_start'],
                    "test_end_date": setup['test_end'],
                })
