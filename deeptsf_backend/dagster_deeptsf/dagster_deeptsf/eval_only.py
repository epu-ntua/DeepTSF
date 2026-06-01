"""
Standalone evaluation job.

This module lets you re-run *only* the evaluation stage of the DeepTSF pipeline
against an already-trained model, without re-running ETL / training. It is fully
isolated from the normal ``deeptsf_dagster_job`` flow: it defines its own assets,
its own graph and its own job (``deeptsf_eval_job``).

How it works
------------
The pipeline's ``evaluation_asset`` needs an upstream ``training_and_hyperparameter_tuning_out``
dict (model / series / scaler URIs, test dates, ...). Dagster does not reliably
persist that intermediate output, so for a standalone eval we instead reconstruct
it from MLflow, which is the real source of truth: the training stage writes all
those URIs as tags on the model's MLflow run (and ``split_info.yml`` for the test
dates). The user only has to supply the model's MLflow run id via config
(``eval_model_run_id``).

Parent run handling
-------------------
If ``parent_run_id`` is given in the config, the eval run is nested under that
existing MLflow run. If it is not given (``"None"``), a brand new standalone
MLflow run is created and the eval is logged separately, without touching any
previous pipeline run.
"""

import os
import logging

import mlflow
from dagster import (
    asset,
    multi_asset,
    AssetIn,
    AssetOut,
    Output,
)

from dagster_deeptsf.evaluate_forecasts import run_evaluation

import sys
sys.path.append('..')
from utils import none_checker, download_online_file, load_yaml_as_dict

from minio import Minio
from utils import truth_checker

from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MINIO_CLIENT_URL = os.environ.get("MINIO_CLIENT_URL")
MINIO_SSL = truth_checker(os.environ.get("MINIO_SSL"))
client = Minio(MINIO_CLIENT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, secure=MINIO_SSL)
S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')


@asset(
    name="start_eval_run",
    group_name='deepTSF_eval',
    required_resource_keys={"config"},)
def start_eval_run(context):
    """Resolve the MLflow run the standalone evaluation is logged under.

    * If ``parent_run_id`` is provided in the config, that existing run is used as
      the parent and the eval run is nested under it. In this case the experiment
      is dictated by the parent run (a nested run must live in its parent's
      experiment), so ``experiment_name`` from the config is ignored here.
    * Otherwise a new standalone MLflow run is created in ``experiment_name``
      (created if it does not exist) so the evaluation is logged separately,
      without affecting any previous run.
    """
    config = context.resources.config

    experiment_name = config.experiment_name
    darts_model = config.darts_model
    parent_run_id = none_checker(config.parent_run_id)
    parent_run_name = config.parent_run_name \
        if none_checker(config.parent_run_name) is not None \
        else (none_checker(darts_model) or "model") + '_eval'

    if parent_run_id is not None:
        # Nest the eval under an existing run (e.g. the original pipeline run).
        # The eval will live in that run's experiment; warn if the configured
        # experiment_name does not match, since it cannot be honoured here.
        parent_exp_id = mlflow.tracking.MlflowClient().get_run(parent_run_id).info.experiment_id
        parent_exp_name = mlflow.tracking.MlflowClient().get_experiment(parent_exp_id).name
        if none_checker(experiment_name) is not None and experiment_name != parent_exp_name:
            msg = (f"\nIgnoring experiment_name='{experiment_name}': the eval is nested under "
                   f"parent run {parent_run_id}, so it is logged in that run's experiment "
                   f"'{parent_exp_name}'. To log into a new/other experiment, leave parent_run_id as None.")
            print(msg)
            logging.warning(msg)
        print(f"\nLogging evaluation under existing parent run {parent_run_id} (experiment '{parent_exp_name}')")
        logging.info(f"\nLogging evaluation under existing parent run {parent_run_id} (experiment '{parent_exp_name}')")
        return Output(parent_run_id)

    # No parent run given -> log this evaluation separately as its own run, in the
    # configured experiment (created on the fly if it does not exist yet).
    mlflow.set_experiment(experiment_name)
    print(f"\nNo parent run given. Logging evaluation as a separate run in experiment '{experiment_name}'.")
    logging.info(f"\nNo parent run given. Logging evaluation as a separate run in experiment '{experiment_name}'.")
    with mlflow.start_run(tags={"mlflow.runName": parent_run_name}) as active_run:
        mlflow.set_tag("stage", "eval_standalone")
        return Output(active_run.info.run_id)


@multi_asset(
    name="eval_model_input",
    description="Reconstruct the evaluation input (model / series / scaler URIs) "
                "from the MLflow tags of an already-trained model run.",
    group_name='deepTSF_eval',
    required_resource_keys={"config"},
    outs={"eval_model_input": AssetOut(dagster_type=dict)})
def eval_model_input(context):
    """Rebuild the dict that ``evaluation_asset`` normally receives from training.

    Mirrors the tag-reading logic of
    ``assets.training_and_hyperparameter_tuning_asset`` but sources everything from
    the MLflow run identified by ``config.eval_model_run_id`` instead of a fresh
    training run.
    """
    config = context.resources.config

    model_run_id = none_checker(config.eval_model_run_id)
    if model_run_id is None:
        raise ValueError(
            "eval_model_run_id must be set to the MLflow run id of the trained "
            "model you want to evaluate."
        )

    completed_run = mlflow.tracking.MlflowClient().get_run(model_run_id)

    if "model_uri" not in completed_run.data.tags:
        print(f'\nMLflow run {model_run_id} has no model_uri tag. Skipping Evaluation')
        logging.info(f'\nMLflow run {model_run_id} has no model_uri tag. Skipping Evaluation')
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

    tags = completed_run.data.tags

    model_uri = tags["model_uri"].replace("s3:/", S3_ENDPOINT_URL)
    model_type = tags["model_type"]
    series_uri = tags["series_uri"].replace("s3:/", S3_ENDPOINT_URL)
    future_covariates_uri = tags["future_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
    past_covariates_uri = tags["past_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
    scaler_uri = tags["scaler_uri"].replace("s3:/", S3_ENDPOINT_URL)
    setup_uri = tags["setup_uri"].replace("s3:/", S3_ENDPOINT_URL)
    scaler_past_covariates_uri = tags["scaler_past_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
    scaler_future_covariates_uri = tags["scaler_future_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)

    setup_file = download_online_file(client, setup_uri, "setup.yml")
    setup = load_yaml_as_dict(setup_file)
    print(f"\nSplit info: {setup} \n")

    if "input_chunk_length" in tags:
        shap_input_length = tags["input_chunk_length"]
    else:
        shap_input_length = config.shap_input_length

    # Naive models require retrain=True. Determine the model name from config,
    # falling back to the tags written at training time.
    darts_model = none_checker(config.darts_model) \
        or tags.get("darts_forecasting_model") \
        or model_type \
        or ""
    if "naive" in darts_model.lower():
        retrain = True
        print("Warning: Switching retrain flag to True as Naive models require...\n")
    else:
        retrain = config.retrain

    # Test window: allow the config to override the split stored at training time.
    # If cut_date_test / test_end_date are given in the config, use them; otherwise
    # fall back to the model's split_info.yml (the original training split).
    cut_date_test = none_checker(config.cut_date_test)
    if cut_date_test is None:
        cut_date_test = setup['test_start']
    test_end_date = none_checker(config.test_end_date)
    if test_end_date is None:
        test_end_date = setup['test_end']

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
                   "cut_date_test": cut_date_test,
                   "test_end_date": test_end_date,
                   })


@multi_asset(
    name="evaluation_only_asset",
    description="Standalone evaluation of an already-trained model.",
    group_name='deepTSF_eval',
    required_resource_keys={"config"},
    ins={'start_eval_run': AssetIn(key='start_eval_run', dagster_type=str),
         'eval_model_input': AssetIn(key='eval_model_input', dagster_type=dict)},
    outs={"evaluation_only_out": AssetOut(dagster_type=dict)})
def evaluation_only_asset(context, start_eval_run, eval_model_input):
    # Reuses the exact same evaluation logic as the pipeline's evaluation_asset.
    return Output(run_evaluation(context, start_eval_run, eval_model_input))
