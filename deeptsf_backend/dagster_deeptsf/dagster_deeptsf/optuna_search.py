import pretty_errors
from .preprocessing import scale_covariates, split_dataset, split_nans, filtering
from darts.utils.missing_values import extract_subseries
import string
from functools import reduce
from darts.metrics import mape as mape_darts
from darts.metrics import mase as mase_darts
from darts.metrics import mae as mae_darts
from darts.metrics import rmse as rmse_darts
from darts.metrics import smape as smape_darts
from darts.models import (
    NaiveSeasonal,
)
# the following are used through eval(darts_model + 'Model')
from darts.models import RNNModel, BlockRNNModel, NBEATSModel, TFTModel, NaiveDrift, NaiveSeasonal, TCNModel, NHiTSModel, TransformerModel
from darts.models.forecasting.arima import ARIMA
# from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.lgbm import LightGBMModel
from darts.models.forecasting.random_forest import RandomForest
from darts.utils.likelihood_models import ContinuousBernoulliLikelihood, GaussianLikelihood, DirichletLikelihood, ExponentialLikelihood, GammaLikelihood, GeometricLikelihood

import yaml
import mlflow
import click
import os
import torch
import logging
import pickle
import tempfile
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import shutil
import optuna
import pandas as pd
# Inference requirements to be stored with the darts flavor !!
from sys import version_info
import torch, cloudpickle, darts
import matplotlib.pyplot as plt
import pprint
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import random
from minio import Minio
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)
import sys
sys.path.append('..')
from utils import none_checker, ConfigParser, download_online_file, load_local_csv_or_df_as_darts_timeseries, truth_checker, load_yaml_as_dict, load_model, load_scaler, multiple_dfs_to_ts_file, get_pv_forecast, plot_series, to_seconds
from exceptions import EvalSeriesNotFound

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MINIO_CLIENT_URL = os.environ.get("MINIO_CLIENT_URL")
MINIO_SSL = truth_checker(os.environ.get("MINIO_SSL"))
client = Minio(MINIO_CLIENT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, secure=MINIO_SSL)

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
mlflow_serve_conda_env = {
    'channels': ['defaults'],
    'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': [
                'cloudpickle=={}'.format(cloudpickle.__version__),
                'darts=={}'.format(darts.__version__),
                'pretty_errors=={}'.format(pretty_errors.__version__),
                'torch=={}'.format(torch.__version__),
                'mlflow=={}'.format(mlflow.__version__)
            ],
        },
    ],
    'name': 'darts_infer_pl_env'
}

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
# a period of 5 epochs (`patience`)
def log_optuna(study, 
               opt_tmpdir, 
               hyperparams_entrypoint,
               trial_name,
               mlrun, 
               log_model=False, 
               curr_loss=0, 
               model=None, 
               darts_model=None, 
               scale="False", 
               scalers_dir=None, 
               features_dir=None, 
               opt_all_results=None, 
               past_covariates=None, 
               future_covariates=None, 
               evaluate_all_ts=False,
               scale_covs=True):
    scale = scale
    if evaluate_all_ts: 
        mlflow.log_artifacts(opt_all_results, "optuna_val_results_all_timeseries")
    
    if log_model and (len(study.trials_dataframe()[study.trials_dataframe()["state"] == "COMPLETE"]) < 1 or study.best_trial.values[0] >= curr_loss):
        if darts_model in ['NHiTS', 'NBEATS', 'RNN', 'BlockRNN', 'TFT', 'TCN', 'Transformer']:
            logs_path = f"./darts_logs/{mlrun.info.run_id}"
            model_type = "pl"
        elif darts_model in ['LightGBM', 'RandomForest', 'ARIMA']:
            print('\nStoring the model as pkl to MLflow...')
            logging.info('\nStoring the model as pkl to MLflow...')
            forest_dir = tempfile.mkdtemp()

            pickle.dump(model, open(
                f"{forest_dir}/_model.pkl", "wb"))

            logs_path = forest_dir
            model_type = "pkl"

        if scale or scale_covs:
            source_dir = scalers_dir
            target_dir = logs_path
            file_names = os.listdir(source_dir)
            for file_name in file_names:
                shutil.move(os.path.join(source_dir, file_name),
                target_dir)
            
        ## Create and move model info in logs path
        model_info_dict = {
            "darts_forecasting_model":  model.__class__.__name__,
            "run_id": mlrun.info.run_id,
            "scale": scale,
            "scale_covs": scale_covs,
            "past_covs": past_covariates is not None,
            "future_covs": future_covariates is not None,
            }
        
        with open('model_info.yml', mode='w') as outfile:
            yaml.dump(
                model_info_dict,
                outfile,
                default_flow_style=False)

        
        shutil.move('model_info.yml', logs_path)

        ## Rename logs path to get rid of run name
        if model_type == 'pkl':
            logs_path_new = logs_path.replace(
            forest_dir.split('/')[-1], mlrun.info.run_id)
            os.rename(logs_path, logs_path_new)
        elif model_type == 'pl':
            logs_path_new = logs_path
        
        mlflow_model_root_dir = "pyfunc_model"
            
        ## Log MLflow model and code
        mlflow.pyfunc.log_model(mlflow_model_root_dir,
                            loader_module="darts_flavor",
                            data_path=logs_path_new,
                            code_path=['exceptions.py', 'utils.py', 'inference.py', 'darts_flavor.py'],
                            conda_env=mlflow_serve_conda_env)
            
        shutil.rmtree(logs_path_new)

        print("\nArtifacts are being uploaded to MLflow...")
        logging.info("\nArtifacts are being uploaded to MLflow...")
        mlflow.log_artifacts(features_dir, "features")

        if scale:
            # mlflow.log_artifacts(scalers_dir, f"{mlflow_model_path}/scalers")
            mlflow.set_tag(
                'scaler_uri',
                f'{mlrun.info.artifact_uri}/{mlflow_model_root_dir}/data/{mlrun.info.run_id}/scaler_series.pkl')
        else:
            mlflow.set_tag('scaler_uri', 'None')

        if scale_covs and past_covariates is not None:
            mlflow.set_tag(
                'scaler_past_covariates_uri',
                f'{mlrun.info.artifact_uri}/{mlflow_model_root_dir}/data/{mlrun.info.run_id}/scaler_past_covariates.pkl')
        else:
            mlflow.set_tag('scaler_past_covariates_uri', 'None')

        if scale_covs and future_covariates is not None:
            mlflow.set_tag(
                'scaler_future_covariates_uri',
                f'{mlrun.info.artifact_uri}/{mlflow_model_root_dir}/data/{mlrun.info.run_id}/scaler_future_covariates.pkl')
        else:
            mlflow.set_tag('scaler_future_covariates_uri', 'None')


        mlflow.set_tag(
            'ts_id_l_uri',
            f'{mlrun.info.artifact_uri}/{mlflow_model_root_dir}/data/{mlrun.info.run_id}/ts_id_l.pkl')


        mlflow.set_tag("run_id", mlrun.info.run_id)
        mlflow.set_tag("stage", "optuna_search")
        mlflow.set_tag("model_type", model_type)

        mlflow.set_tag(
            'setup_uri',
            f'{mlrun.info.artifact_uri}/features/split_info.yml')

        mlflow.set_tag("darts_forecasting_model",
            model.__class__.__name__)
            
        if "input_chunk_length" in hyperparams_entrypoint:
            mlflow.set_tag('input_chunk_length', hyperparams_entrypoint["input_chunk_length"])

        # model_uri
        mlflow.set_tag('model_uri', mlflow.get_artifact_uri(
            f"{mlflow_model_root_dir}/data/{mlrun.info.run_id}"))
        # inference_model_uri
        mlflow.set_tag('pyfunc_model_folder', mlflow.get_artifact_uri(
            f"{mlflow_model_root_dir}"))

        mlflow.set_tag('series_uri',
            f'{mlrun.info.artifact_uri}/features/series.csv')

        if future_covariates is not None:
            mlflow.set_tag(
                'future_covariates_uri',
                f'{mlrun.info.artifact_uri}/features/future_covariates_transformed.csv')
        else:
            mlflow.set_tag(
                'future_covariates_uri',
                'None')

        if past_covariates is not None:
            mlflow.set_tag(
                'past_covariates_uri',
                f'{mlrun.info.artifact_uri}/features/past_covariates_transformed.csv')
        else:
            mlflow.set_tag('past_covariates_uri',
                'None')

        print("\nArtifacts uploaded.")
        logging.info("\nArtifacts uploaded.")
    if not log_model:
        ######################
        # Log hyperparameters
        best_params = study.best_params
        if "scale" in best_params:
            best_params["scale_optuna"] = best_params["scale"]
            del best_params["scale"]
        mlflow.log_params(best_params)

        # Log log_metrics
        mlflow.log_metrics(study.best_trial.user_attrs)




    if len(study.trials_dataframe()[study.trials_dataframe()["state"] == "COMPLETE"]) <= 1: return

    plt.close()

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{opt_tmpdir}/plot_optimization_history.html")
    plt.close()

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"{opt_tmpdir}/plot_param_importances.html")
    plt.close()

    fig = optuna.visualization.plot_slice(study)
    fig.write_html(f"{opt_tmpdir}/plot_slice.html")
    plt.close()

    study.trials_dataframe().to_csv(f"{opt_tmpdir}/{trial_name}.csv")

    print("\nUploading optuna plots to MLflow server...")
    logging.info("\nUploading optuna plots to MLflow server...")

    mlflow.log_artifacts(opt_tmpdir, "optuna_results")

def append(x, y):
    """
    Append `y` to `x`, discarding any part of `y` whose timestamps already appear in `x`.

    Parameters
    ----------
    x : TimeSeries
        Leading series.
    y : TimeSeries
        Series to append.

    Returns
    -------
    TimeSeries
        `x` followed by the non‑overlapping portion of `y`.
    """
    last_ts = x.end_time()              # final timestamp in x
    if last_ts > y.start_time():
        y_tail  = y.drop_before(last_ts)    # keeps strictly after last_ts
    else:
        y_tail  = y

    # If y had nothing new, just return x (copy so caller can mutate safely)
    if y_tail.n_timesteps == 0:
        return x.copy()

    return x.append(y_tail)

def objective(series_csv, series_uri, future_covs_csv, future_covs_uri,
             past_covs_csv, past_covs_uri, year_range, resolution,
             darts_model, hyperparams_entrypoint, trial_name, cut_date_val, test_end_date, 
             cut_date_test, device, forecast_horizon, m_mase, stride, retrain, scale, 
             scale_covs, multiple, eval_series, mlrun, trial, study, opt_tmpdir, 
             num_workers, eval_method, loss_function, opt_all_results,
             evaluate_all_ts, num_samples, pv_ensemble, format):

                # hyperparameters = ConfigParser(config_file='../config_opt.yml', config_string=hyperparams_entrypoint).read_hyperparameters(hyperparams_entrypoint)
                hyperparameters = hyperparams_entrypoint
                training_dict = {}
                for param, value in hyperparameters.items():
                    if type(value) == list and value and value[0] == "range":
                        if type(value[1]) == int:
                            if param == "lags_future_covariates":
                                training_dict[param] = trial.suggest_int(param, value[1], value[2], value[3])
                                training_dict[param] = [training_dict[param], 24]
                            else:
                                training_dict[param] = trial.suggest_int(param, value[1], value[2], value[3])
                        else:
                            training_dict[param] = trial.suggest_float(param, value[1], value[2], value[3])
                    elif type(value) == list and value and value[0] == "list":
                        if param == "lags_future_covariates":
                            training_dict[param] = trial.suggest_categorical(param, value[1:])
                            training_dict[param] = [training_dict[param], 24]
                        else:
                            training_dict[param] = trial.suggest_categorical(param, value[1:])
                    elif type(value) == list and value and value[0] == "equal":
                        continue
                    else:
                        training_dict[param] = value
                for param, value in hyperparameters.items():
                    if type(value) == list and value and value[0] == "equal":
                        training_dict[param] = training_dict[value[1]]
                if 'scale' in training_dict:
                     scale = training_dict['scale']
                     del training_dict['scale']

                #TODO: Make it work with csvs also
                model, scaler, train_future_covariates, train_past_covariates, features_dir, scalers_dir = train(
                      series_uri=series_uri,
                      future_covs_uri=future_covs_uri,
                      past_covs_uri=past_covs_uri, # fix that in case REAL Temperatures come -> etl_temp_covs_uri. For forecasts, integrate them into future covariates!!
                      darts_model=darts_model,
                      hyperparams_entrypoint=hyperparams_entrypoint,
                      trial_name=trial_name,
                      cut_date_val=cut_date_val,
                      cut_date_test=cut_date_test,
                      test_end_date=test_end_date,
                      device=device,
                      scale=scale,
                      scale_covs=scale_covs,
                      multiple=multiple,
                      training_dict=training_dict,
                      mlrun=mlrun,
                      num_workers=num_workers,
                      resolution=resolution,
                      trial=trial,
                      pv_ensemble=pv_ensemble,
                      format=format,
                      )
                try:
                    trial.set_user_attr("epochs_trained", model.epochs_trained)
                except:
                    pass
                metrics = validate(
                    series_uri=series_uri,
                    future_covariates=train_future_covariates,
                    past_covariates=train_past_covariates,
                    scaler=scaler,
                    cut_date_test=cut_date_test,
                    test_end_date=test_end_date,#check that again
                    model=model,
                    forecast_horizon=forecast_horizon,
                    m_mase=m_mase,
                    stride=stride,
                    retrain=retrain,
                    multiple=multiple,
                    eval_series=eval_series,
                    cut_date_val=cut_date_val,
                    mlrun=mlrun,
                    resolution=resolution,
                    eval_method=eval_method,
                    opt_all_results=opt_all_results,
                    evaluate_all_ts=evaluate_all_ts,
                    study=study,
                    num_samples=num_samples,
                    pv_ensemble=pv_ensemble,
                    format=format,
                    )
                trial.set_user_attr("mape", float(metrics["mape"]))
                trial.set_user_attr("smape", float(metrics["smape"]))
                trial.set_user_attr("mase", float(metrics["mase"]))
                trial.set_user_attr("mae", float(metrics["mae"]))
                trial.set_user_attr("rmse", float(metrics["rmse"]))
                trial.set_user_attr("nrmse_min_max", float(metrics["nrmse_min_max"]))
                trial.set_user_attr("nrmse_mean", float(metrics["nrmse_mean"]))
                log_optuna(study, opt_tmpdir, hyperparams_entrypoint, trial_name, mlrun, 
                    log_model=True, curr_loss=float(metrics[loss_function]), 
                    model=model, darts_model=darts_model, scale=scale, scalers_dir=scalers_dir, 
                    features_dir=features_dir, opt_all_results=opt_all_results, 
                    past_covariates=train_past_covariates, future_covariates=train_future_covariates, 
                    evaluate_all_ts=evaluate_all_ts, scale_covs=scale_covs)

                return metrics[loss_function]

def train(series_uri, future_covs_uri, past_covs_uri, darts_model,
          hyperparams_entrypoint, trial_name, cut_date_val, cut_date_test,
          test_end_date, device, scale, scale_covs, multiple,
          training_dict, mlrun, num_workers, resolution, trial, pv_ensemble, format):


    # Argument preprocessing

    ## test_end_date
    num_workers = int(num_workers)
    torch.set_num_threads(num_workers)

    my_stopper = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-6,
        mode='min',
    )

    test_end_date = none_checker(test_end_date)

    ## scale or not
    scale = scale
    scale_covs = scale_covs

    multiple = multiple

    ## hyperparameters
    hyperparameters = training_dict

    ## device
    if device == 'gpu' and torch.cuda.is_available():
        device = 'gpu'
        print("\nGPU is available")
    else:
        device = 'cpu'
        print("\nGPU is available")
    ## series and covariates uri and csv
    series_uri = none_checker(series_uri)
    future_covs_uri = none_checker(future_covs_uri)
    past_covs_uri = none_checker(past_covs_uri)

    # redirect to local location of downloaded remote file
    if series_uri is not None:
        download_file_path = download_online_file(client, series_uri, dst_filename="load.csv")
        series_csv = download_file_path.replace('/', os.path.sep).replace("'", "")
    else:
        series_csv = None

    if  future_covs_uri is not None:
        download_file_path = download_online_file(client, future_covs_uri, dst_filename="future.csv")
        future_covs_csv = download_file_path.replace('/', os.path.sep).replace("'", "")
    else:
        future_covs_csv = None

    if  past_covs_uri is not None:
        download_file_path = download_online_file(client, past_covs_uri, dst_filename="past.csv")
        past_covs_csv = download_file_path.replace('/', os.path.sep).replace("'", "")
    else:
        past_covs_csv = None

    ## model
    # TODO: Take care of future covariates (RNN, ...) / past covariates (BlockRNN, NBEATS, ...)
    if darts_model in ["NBEATS", "BlockRNN", "TCN", "NHiTS", "Transformer"]:
        """They do not accept future covariates as they predict blocks all together.
        They won't use initial forecasted values to predict the rest of the block
        So they won't need to additionally feed future covariates during the recurrent process.
        """
        #TODO Concatenate future covs to past??
        #past_covs_csv = future_covs_csv
        future_covs_csv = None

    elif darts_model in ["RNN", "ARIMA"]:
        """Does not accept past covariates as it needs to know future ones to provide chain forecasts
        its input needs to remain in the same feature space while recurring and with no future covariates
        this is not possible. The existence of past_covs is not permitted for the same reason. The
        feature space will change during inference. If for example I have current temperature and during
        the forecast chain I only have time covariates, as I won't know the real temp then a constant \
        architecture like LSTM cannot handle this"""
        past_covs_csv = None
        # TODO: when actual weather comes extend it, now the stage only accepts future covariates as argument.
    #elif: extend for other models!! (time_covariates are always future covariates, but some models can't handle them as so)

    future_covariates = none_checker(future_covs_csv)
    past_covariates = none_checker(past_covs_csv)


    ######################
    # Load series and covariates datasets
    time_col = "Datetime"
    series, id_l, ts_id_l = load_local_csv_or_df_as_darts_timeseries(
                local_path_or_df=series_csv,
                name='series',
                time_col=time_col,
                last_date=test_end_date,
                multiple=multiple,
                resolution=resolution,
                format=format)
    
    if future_covariates is not None:
        future_covariates, id_l_future_covs, ts_id_l_future_covs = load_local_csv_or_df_as_darts_timeseries(
                local_path_or_df=future_covs_csv,
                name='future covariates',
                time_col=time_col,
                last_date=test_end_date,
                multiple=True,
                resolution=resolution,
                format=format)
    else:
        future_covariates, id_l_future_covs, ts_id_l_future_covs = None, None, None
    if past_covariates is not None:
        past_covariates, id_l_past_covs, ts_id_l_past_covs = load_local_csv_or_df_as_darts_timeseries(
                local_path_or_df=past_covs_csv,
                name='past covariates',
                time_col=time_col,
                last_date=test_end_date,
                multiple=True,
                resolution=resolution,
                format=format)
    else:
        past_covariates, id_l_past_covs, ts_id_l_past_covs = None, None, None
    
    if (len(id_l) != 1 or len(id_l[0]) > 1) and darts_model=='ARIMA':
        raise Exception("ARIMA does not support multiple time series") 

    scalers_dir = tempfile.mkdtemp()
    features_dir = tempfile.mkdtemp()

    ######################
    # Train / Test split
    print(
        f"\nTrain / Test split: Validation set starts: {cut_date_val} - Test set starts: {cut_date_test} - Test set end: {test_end_date}")
    logging.info(
        f"\nTrain / Test split: Validation set starts: {cut_date_val} - Test set starts: {cut_date_test} - Test set end: {test_end_date}")

    ## series
    series_split = split_dataset(
            series,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            store_dir=features_dir,
            name='series',
            conf_file_name='split_info.yml',
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l,
            format=format)
        ## future covariates
    future_covariates_split = split_dataset(
            future_covariates,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            # store_dir=features_dir,
            name='future_covariates',
            multiple=True,
            id_l=id_l_future_covs,
            ts_id_l=ts_id_l_future_covs,
            format=format)
        ## past covariates
    past_covariates_split = split_dataset(
            past_covariates,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            # store_dir=features_dir,
            name='past_covariates',
            multiple=True,
            id_l=id_l_past_covs,
            ts_id_l=ts_id_l_past_covs,
            format=format)
    #################
    # Scaling
    print("\nScaling...")
    logging.info("\nScaling...")

    if pv_ensemble:
        for i in range(len(series_split['train'])):
            series_split['train'][i] = series_split['train'][i] + get_pv_forecast(ts_id_l[i], 
                                                                                  start=series_split['train'][i].pd_dataframe().index[0], 
                                                                                  end=series_split['train'][i].pd_dataframe().index[-1], 
                                                                                  inference=False, 
                                                                                  kW=60, 
                                                                                  use_saved=True)
            series_split['val'][i] = series_split['val'][i] + get_pv_forecast(ts_id_l[i], 
                                                                                  start=series_split['val'][i].pd_dataframe().index[0], 
                                                                                  end=series_split['val'][i].pd_dataframe().index[-1], 
                                                                                  inference=False, 
                                                                                  kW=60, 
                                                                                  use_saved=True)


        #plot_series([series_split['train'][0], series_split['val'][0]], ["train", "val"], os.path.join(f"{features_dir}",f'series_train.html'))


    # #TODO Add smoothing

    # savgol_polyorder = 0

    # savgol_window_length = 0

    # #TODO Add parameters to mlflow
    # if 'savgol_window_length' in hyperparameters:
    #         savgol_window_length = hyperparameters['savgol_window_length']
    #         del hyperparameters['savgol_window_length']
        
    # if 'savgol_polyorder' in hyperparameters:
    #         savgol_polyorder = hyperparameters['savgol_polyorder']
    #         del hyperparameters['savgol_polyorder']

    # if savgol_window_length <= savgol_polyorder and not (savgol_window_length == 0 and savgol_polyorder == 0):
    #     raise optuna.TrialPruned()


    # #TODO: Add parameter
    # if savgol_window_length != 0 and savgol_polyorder != 0:
    #     series_split['train'], past_covariates_split['train'], future_covariates_split['train'] = \
    #         filtering(series_split['train'], past_covariates_split['train'], future_covariates_split['train'], savgol_window_length, savgol_polyorder)


    ## scale series
    series_transformed = scale_covariates(
            series_split,
            store_dir=features_dir,
            filename_suffix="series_transformed.csv",
            scale=scale,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l,
            format=format,
            )
    if scale:
        pickle.dump(series_transformed["transformer"], open(f"{scalers_dir}/scaler_series.pkl", "wb"))
        ## scale future covariates
    pickle.dump(ts_id_l, open(f"{scalers_dir}/ts_id_l.pkl", "wb"))
    future_covariates_transformed = scale_covariates(
            future_covariates_split,
            store_dir=features_dir,
            filename_suffix="future_covariates_transformed.csv",
            scale=scale_covs,
            multiple=True,
            id_l=id_l_future_covs,
            ts_id_l=ts_id_l_future_covs,
            format=format,
            )
        ## scale past covariates
    past_covariates_transformed = scale_covariates(
            past_covariates_split,
            store_dir=features_dir,
            filename_suffix="past_covariates_transformed.csv",
            scale=scale_covs,
            multiple=True,
            id_l=id_l_past_covs,
            ts_id_l=ts_id_l_past_covs,
            format=format
            )
    
    if scale_covs and future_covariates is not None:
            pickle.dump(future_covariates_transformed["transformer"], open(f"{scalers_dir}/scaler_future_covariates.pkl", "wb"))

    if scale_covs and past_covariates is not None:
            pickle.dump(past_covariates_transformed["transformer"], open(f"{scalers_dir}/scaler_past_covariates.pkl", "wb"))



    ######################
    # Model training
    print("\nTraining model...")
    logging.info("\nTraining model...")
    pl_trainer_kwargs = {"callbacks": [my_stopper],
                         "accelerator": 'auto',
                        #  "gpus": 1,
                        #  "auto_select_gpus": True,
                         "log_every_n_steps": 10}

    print("\nTraining on series:\n")
    logging.info("\nTraining on series:\n")
    if multiple:
        for i, series in enumerate(series_transformed['train']):
            print(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
            logging.info(f"Timeseries ID: {ts_id_l[i][0]} starting at {series.time_index[0]} and ending at {series.time_index[-1]}")
    else:
        print(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")
        logging.info(f"Series starts at {series_transformed['train'].time_index[0]} and ends at {series_transformed['train'].time_index[-1]}")
    print("")

    #TODO maybe modify print to include split train based on nans
    #TODO make more efficient by also spliting covariates where the nans are split 
    if darts_model not in ['ARIMA']:
        series_transformed['train'], past_covariates_transformed['train'], future_covariates_transformed['train'] = \
            split_nans(series_transformed['train'], past_covariates_transformed['train'], future_covariates_transformed['train'])
    

    ## choose architecture
    if darts_model in ['NBEATS', 'RNN', 'BlockRNN', 'TFT', 'TCN', 'NHiTS', 'Transformer']:
        hparams_to_log = hyperparameters
        if 'learning_rate' in hyperparameters:
            hyperparameters['optimizer_kwargs'] = {'lr': hyperparameters['learning_rate']}
            del hyperparameters['learning_rate']

        if 'likelihood' in hyperparameters:
            hyperparameters['likelihood'] = eval(hyperparameters['likelihood']+"Likelihood"+"()")
        model = eval(darts_model + 'Model')(
            force_reset=True,
            save_checkpoints=True,
            log_tensorboard=False,
            model_name=mlrun.info.run_id,
            pl_trainer_kwargs=pl_trainer_kwargs,
            **hyperparameters
        )
        model.fit(series_transformed['train'],
            future_covariates=future_covariates_transformed['train'],
            past_covariates=past_covariates_transformed['train'],
            val_series=series_transformed['val'],
            val_future_covariates=future_covariates_transformed['val'],
            val_past_covariates=past_covariates_transformed['val'])

        # LightGBM and RandomForest
    elif darts_model in ['LightGBM', 'RandomForest']:

        try:
            if "lags_future_covariates" in hyperparameters:
                if truth_checker(str(hyperparameters["future_covs_as_tuple"])):
                    hyperparameters["lags_future_covariates"] = tuple(
                        hyperparameters["lags_future_covariates"])
                hyperparameters.pop("future_covs_as_tuple")
        except:
            pass

        if future_covariates is None:
            hyperparameters["lags_future_covariates"] = None
        if past_covariates is None:
            hyperparameters["lags_past_covariates"] = None

        hparams_to_log = hyperparameters

        if darts_model == 'RandomForest':
            model = RandomForest(**hyperparameters)
        elif darts_model == 'LightGBM':
            model = LightGBMModel(**hyperparameters)

        print(f'\nTraining {darts_model}...')
        logging.info(f'\nTraining {darts_model}...')
        # for elem in series_transformed['train']:
        #     print(elem)
        # for elem in future_covariates_transformed['train']:
        #     print(elem)

        # for i, series in enumerate(series_transformed['train']):
        #     series.pd_dataframe().to_csv(f"{i}_series_bad")
        model.fit(
            series=series_transformed['train'],
            # val_series=series_transformed['val'],
            future_covariates=future_covariates_transformed['train'],
            past_covariates=past_covariates_transformed['train'],
            # val_future_covariates=future_covariates_transformed['val'],
            # val_past_covariates=past_covariates_transformed['val']
            )
    elif darts_model == 'ARIMA':
        print(f'\nTrained Model: {darts_model}') 

        hparams_to_log = hyperparameters
        model = ARIMA(**hyperparameters)

        print(f'\nTraining {darts_model}...')
        logging.info(f'\nTraining {darts_model}...')

        if type(series_transformed['train']) == list:
            fit_series = series_transformed['train'][-1]
        else:
            fit_series = series_transformed['train']

        model.fit(
            series=fit_series,
            future_covariates=future_covariates_transformed['train'],
            )
        model_type = "pkl"
    
    if scale:
        scaler = series_transformed["transformer"]
    else:
        scaler = None

    if future_covariates is not None:
        return_future_covariates = future_covariates_transformed['all']
    else:
        return_future_covariates = None

    if past_covariates is not None:
        return_past_covariates = past_covariates_transformed['all']
    else:
        return_past_covariates = None
    return model, scaler, return_future_covariates, return_past_covariates, features_dir, scalers_dir

def backtester(model,
               series_transformed,
               test_start_date,
               forecast_horizon,
               stride=None,
               series=None,
               transformer_ts=None,
               retrain=False,
               future_covariates=None,
               past_covariates=None,
               path_to_save_backtest=None,
               m_mase=1,
               num_samples=1,
               pv_ensemble=False,
               resolution="60min"):
               #TODO Add mase
    """ Does the same job with advanced forecast but much more quickly using the darts
    bult-in historical_forecasts method. Use this for evaluation. The other only
    provides pure inference. Provide a unified timeseries test set point based
    on test_start_date. series_transformed does not need to be adjacent to
    training series. if transformer_ts=None then no inverse transform is applied
    to the model predictions.
    """
    # produce the fewest forecasts possible.
    if stride is None:
        stride = forecast_horizon

    #keep last non nan values
    #must be sufficient for historical_forecasts and mase calculation
    #TODO Add check for that in the beggining
    # series = extract_subseries(series, min_gap_size=1, mode='any')[-1]
    # series_transformed = extract_subseries(series_transformed, min_gap_size=1, mode='any')[-1]

    test_start_date = series_transformed.pd_dataframe()[series_transformed.pd_dataframe().index >= pd.Timestamp(test_start_date + " 00:00:00")].index[0]

    # produce list of forecasts
    #print("backtesting starting at", test_start_date, "series:", series_transformed)
    backtest_series_transformed = model.historical_forecasts(series_transformed,
                                                             future_covariates=future_covariates,
                                                             past_covariates=past_covariates,
                                                             start=test_start_date,
                                                             forecast_horizon=forecast_horizon,
                                                             stride=stride,
                                                             retrain=retrain,
                                                             last_points_only=False,
                                                             verbose=False,
                                                             num_samples=num_samples)
    
    # with open("/app/dagster_deeptsf/all_series.txt", "w", encoding="utf-8") as fh:
    #     for i, ts in enumerate(backtest_series_transformed, start=1):
    #         fh.write(f"# ---- series_{i} ----\n")
    #         fh.write(ts.pd_dataframe().to_csv(index=True))  # keep timestamps
    #         fh.write("\n")  # blank line between blocks

    overlap = forecast_horizon - stride          # positive ⇒ overlapping windows

    if overlap > 0:
        print(
            f"\nDetected overlap of {overlap} time steps "
            f"(stride={stride} < forecast_horizon={forecast_horizon}).\n"
            f"→ This will be treated as forecasting "
            f"{1+ forecast_horizon - stride} to {forecast_horizon} "
            f"steps ahead instead of the usual 1 to {forecast_horizon}.\n"
            f"→ Dropping the first {overlap} time steps of the forecasted time series."
        )

    elif overlap == 0:
        # stride and forecast_horizon align perfectly – nothing to do
        pass
    else:
        raise ValueError(
            f"Stride ({stride}) exceeds forecast_horizon ({forecast_horizon})."
            "This is not supported for now."
        )


    # flatten lists of forecasts due to last_points_only=False
    if isinstance(backtest_series_transformed, list):
        backtest_series_transformed = reduce(
            append, backtest_series_transformed)
        
    # Drop the first timesteps equal to the overlap
    backtest_series_transformed = backtest_series_transformed[overlap:]

    # backtest_series_transformed.pd_dataframe().to_csv("/app/dagster_deeptsf/combined_series.csv", index=True)

    # inverse scaling
    if transformer_ts is not None and series is not None:
        backtest_series = transformer_ts.inverse_transform(
            backtest_series_transformed)
    else:
        backtest_series = backtest_series_transformed
        print("\nWarning: Scaler not provided. Ensure model provides normal scale predictions")
        logging.info(
            "\n Warning: Scaler not provided. Ensure model provides normal scale predictions")
    if pv_ensemble:
        print("\nAdding pv forecast back to forecasted series")
        logging.info("\nAdding pv forecast back to forecasted series")
        backtest_series = backtest_series - get_pv_forecast([], 
                                                            start=backtest_series.pd_dataframe().index[0], 
                                                            end=backtest_series.pd_dataframe().index[-1], 
                                                            inference=False, 
                                                            kW=60, 
                                                            use_saved=True)
    # Metrics
    test_series = series.drop_before(pd.Timestamp(test_start_date) + (overlap - 1) * pd.Timedelta(resolution))
    # print("test series start", test_series.pd_dataframe().index[0])
    # print("test series end", test_series.pd_dataframe().index[-1])
    # print("backtest series start", backtest_series.pd_dataframe().index[0])
    # print("backtest series end", backtest_series.pd_dataframe().index[-1])
    # print("insample start", series.drop_after(pd.Timestamp(test_start_date) + overlap * pd.Timedelta(resolution)).pd_dataframe().index[0])
    # print("insample end", series.drop_after(pd.Timestamp(test_start_date) + overlap * pd.Timedelta(resolution)).pd_dataframe().index[-1])
    metrics = {
        "mae": mae_darts(
            test_series,
            backtest_series),
        "rmse": rmse_darts(
            test_series,
            backtest_series),
        "nrmse_min_max": rmse_darts(
            test_series,
            backtest_series) / (
            test_series.pd_dataframe().max().iloc[0]- 
            test_series.pd_dataframe().min().iloc[0]),
        "nrmse_mean": rmse_darts(
            test_series,
            backtest_series) / (
            test_series.pd_dataframe().mean().iloc[0])
    }
    if min(test_series.min(axis=1).values()) > 0 and min(backtest_series.min(axis=1).values()) > 0:
        metrics["mape"] = mape_darts(
            test_series,
            backtest_series)
    else:
        print("\nModel result or testing series not strictly positive. Setting mape to NaN...")
        logging.info("\nModel result or testing series not strictly positive. Setting mape to NaN...")
        metrics["mape"] = np.nan
    try:
        metrics["mase"] = mase_darts(
            test_series,
            backtest_series,
            insample=series.drop_after(pd.Timestamp(test_start_date) + overlap * pd.Timedelta(resolution)),
            m = m_mase)
    except:
        print("\nSeries is periodical. Setting mase to NaN...")
        logging.info("\nModel result or testing series not strictly positive. Setting mape to NaN...")
        metrics["mase"] = np.nan

    try:
        metrics["smape"] = smape_darts(
            test_series,
            backtest_series)
    except:
        print("\nSeries not strictly positive. Setting smape to NaN...")
        logging.info("\nSeries not strictly positive. Setting smape to NaN...")
        metrics["smape"] = np.nan
    
    for key, value in metrics.items():
        print(key, ': ', value)

    return {"metrics": metrics, "backtest_series": backtest_series}


def validate(series_uri, future_covariates, past_covariates, scaler, cut_date_test, test_end_date,
             model, forecast_horizon, m_mase, stride, retrain, multiple, eval_series, cut_date_val, mlrun, 
             resolution, eval_method, opt_all_results, evaluate_all_ts, study, num_samples, pv_ensemble, format, mode='remote'):
    # TODO: modify functions to support models with likelihood != None
    # TODO: Validate evaluation step for all models. It is mainly tailored for the RNNModel for now.

    # Argument processing
    stride = none_checker(stride)
    forecast_horizon = int(forecast_horizon)
    m_mase = int(m_mase)
    stride = int(forecast_horizon) if stride == -1 or stride == None else int(stride)
    retrain = retrain
    multiple = multiple
    num_samples = int(num_samples)
    test_end_date = none_checker(test_end_date)

    # Load model / datasets / scalers from Mlflow server

    ## load series from MLflow
    series_path = download_online_file(
        client, series_uri, "series.csv") if mode == 'remote' else series_uri
    series, id_l, ts_id_l = load_local_csv_or_df_as_darts_timeseries(
        local_path_or_df=series_path,
        last_date=test_end_date,
        multiple=multiple,
        resolution=resolution,
        format=format)
    
    series_transformed=series.copy()

    if pv_ensemble:
        print("\nSubtracting pv forecast from train and val series")
        logging.info("\nSubtracting pv forecast from train and val series")
        for i in range(len(series_transformed)):
            series_transformed[i] = series_transformed[i] + get_pv_forecast(ts_id_l[i], 
                                                                                  start=series_transformed[i].pd_dataframe().index[0], 
                                                                                  end=series_transformed[i].pd_dataframe().index[-1], 
                                                                                  inference=False, 
                                                                                  kW=60, 
                                                                                  use_saved=True)
        
        #plot_series([series_transformed[0]], ["val"], os.path.join(f"{opt_all_results}",'series_val.html'))

    if scaler is not None:
        if not multiple:
            series_transformed = scaler.transform(series_transformed)
        else:
            series_transformed = [scaler[i].transform(series_transformed[i]) for i in range(len(series_transformed))]
    elif not pv_ensemble:
        series_transformed = series

    # Split in the same way as in training
    ## series
    series_split = split_dataset(
            series,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l,
            format=format)


    series_transformed_split = split_dataset(
            series_transformed,
            val_start_date_str=cut_date_val,
            test_start_date_str=cut_date_test,
            test_end_date=test_end_date,
            multiple=multiple,
            id_l=id_l,
            ts_id_l=ts_id_l,
            format=format)
        
    if evaluate_all_ts and multiple:
        eval_results = {}
        ts_n = len(ts_id_l)
        for eval_i in range(ts_n):
            backtest_series = darts.timeseries.concatenate([series_split['train'][eval_i], series_split['val'][eval_i]]) if multiple else \
                            darts.timeseries.concatenate([series_split['train'], series_split['val']])
            backtest_series_transformed = darts.timeseries.concatenate([series_transformed_split['train'][eval_i], series_transformed_split['val'][eval_i]]) if multiple else \
                            darts.timeseries.concatenate([series_transformed_split['train'], series_transformed_split['val']])
            print(f"Validating timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")
            logging.info(f"Validating timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")
            print(f"Validating from {pd.Timestamp(cut_date_val)} to {backtest_series_transformed.time_index[-1]}...")
            logging.info(f"Validating from {pd.Timestamp(cut_date_val)} to {backtest_series_transformed.time_index[-1]}...")
            print("")
            validation_results = backtester(model=model,
                                            series_transformed=backtest_series_transformed,
                                            series=backtest_series,
                                            transformer_ts=scaler if (not multiple or (scaler == None)) else scaler[eval_i],
                                            test_start_date=cut_date_val,
                                            forecast_horizon=forecast_horizon,
                                            stride=stride,
                                            retrain=retrain,
                                            future_covariates=None if future_covariates == None else (future_covariates[0] if not multiple else future_covariates[eval_i]),
                                            past_covariates=None if past_covariates == None else (past_covariates[0] if not multiple else past_covariates[eval_i]),
                                            m_mase=m_mase,
                                            num_samples=num_samples,
                                            pv_ensemble=pv_ensemble,
                                            resolution=resolution)
            
            eval_results[eval_i] = [str(ts_id_l[eval_i][0])] + [validation_results["metrics"]["smape"],
                                                                      validation_results["metrics"]["mase"],
                                                                      validation_results["metrics"]["mae"],
                                                                      validation_results["metrics"]["rmse"],
                                                                      validation_results["metrics"]["mape"],
                                                                      validation_results["metrics"]["nrmse_min_max"],
                                                                      validation_results["metrics"]["nrmse_mean"]]
        print(eval_results)

        eval_results = pd.DataFrame.from_dict(eval_results, orient='index', columns=["Timeseries ID", "smape", "mase", "mae", "rmse", "mape", "nrmse_min_max", "nrmse_mean"])
        trial_num = len(study.trials_dataframe()) - 1
        save_path = f"{opt_all_results}/trial_{trial_num}.csv"
        if os.path.exists(save_path):
            print(f"Path {opt_all_results}/trial_{trial_num}.csv already exists. Creating extra file for Trial {trial_num}...")
            logging.info(f"Path {opt_all_results}/trial_{trial_num}.csv already exists. Creating extra file for Trial {trial_num}...")
            all_letters = string.ascii_lowercase
            save_path = save_path.split(".")[0] + ''.join(random.choice(all_letters) for i in range(20)) + ".csv"
        eval_results.to_csv(save_path)
        return eval_results.mean(axis=0, numeric_only=True).to_dict()
    else:
        if multiple:
            eval_i = -1
            if eval_method == "ts_ID":
                for i, comps in enumerate(ts_id_l):
                    for comp in comps:
                        if eval_series == str(comp):
                            eval_i = i
            else:
                for i, comps in enumerate(id_l):
                    for comp in comps:
                        if eval_series == str(comp):
                            eval_i = i
        else:
            eval_i = 0

        if eval_i == -1:
            raise EvalSeriesNotFound(eval_series)
        # Evaluate Model
        backtest_series = darts.timeseries.concatenate([series_split['train'][eval_i], series_split['val'][eval_i]]) if multiple else \
                          darts.timeseries.concatenate([series_split['train'], series_split['val']])
        backtest_series_transformed = darts.timeseries.concatenate([series_transformed_split['train'][eval_i], series_transformed_split['val'][eval_i]]) if multiple else \
                                      darts.timeseries.concatenate([series_transformed_split['train'], series_transformed_split['val']])
        #print("testing on", eval_i, backtest_series_transformed)
        if multiple:
            print(f"Validating timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")
            logging.info(f"Validating timeseries number {eval_i} with Timeseries ID {ts_id_l[eval_i][0]} and ID of first component {id_l[eval_i][0]}...")

        print(f"Validating from {pd.Timestamp(cut_date_val)} to {backtest_series_transformed.time_index[-1]}...")
        logging.info(f"Validating from {pd.Timestamp(cut_date_val)} to {backtest_series_transformed.time_index[-1]}...")
        print("")

        validation_results = backtester(model=model,
                                        series_transformed=backtest_series_transformed,
                                        series=backtest_series,
                                        transformer_ts=scaler if (not multiple or (scaler == None)) else scaler[eval_i],
                                        test_start_date=cut_date_val,
                                        forecast_horizon=forecast_horizon,
                                        stride=stride,
                                        retrain=retrain,
                                        future_covariates=None if future_covariates == None else (future_covariates[0] if not multiple else future_covariates[eval_i]),
                                        past_covariates=None if past_covariates == None else (past_covariates[0] if not multiple else past_covariates[eval_i]),
                                        m_mase=m_mase,
                                        num_samples=num_samples,
                                        pv_ensemble=pv_ensemble,
                                        resolution=resolution)

        return validation_results["metrics"]

def optuna_search(context, start_pipeline_run, etl_out):
    
    past_covs_uri = etl_out["past_covs_uri"]
    future_covs_uri = etl_out["future_covs_uri"]
    series_uri = etl_out["series_uri"]


    config = context.resources.config

        
    series_csv = config.series_csv
    future_covs_csv = config.future_covs_csv
    past_covs_csv = config.past_covs_csv
    year_range = config.year_range
    resolution = config.resolution
    darts_model = config.darts_model
    hyperparams_entrypoint = config.hyperparams_entrypoint
    trial_name = config.trial_name
    cut_date_val = config.cut_date_val
    cut_date_test = config.cut_date_test
    test_end_date = config.test_end_date
    device = config.device
    forecast_horizon = config.forecast_horizon
    stride = config.stride
    retrain = config.retrain
    scale = config.scale
    scale_covs = config.scale_covs
    multiple = config.multiple
    eval_series = config.eval_series
    n_trials = config.n_trials
    num_workers = config.num_workers
    m_mase = config.m_mase
    eval_method = config.eval_method
    loss_function = config.loss_function
    evaluate_all_ts = config.evaluate_all_ts
    grid_search = config.grid_search
    num_samples = config.num_samples
    pv_ensemble = config.pv_ensemble
    format = config.format
    experiment_name = config.experiment_name
    parent_run_name = config.parent_run_name if none_checker(config.parent_run_name) != None else darts_model + '_pipeline'

    parameters_dict = {
        "past_covs_uri": past_covs_uri,
        "future_covs_uri": future_covs_uri,
        "series_uri": series_uri,
        "series_csv": series_csv,
        "future_covs_csv": future_covs_csv,
        "past_covs_csv": past_covs_csv,
        "year_range": year_range,
        "resolution": resolution,
        "darts_model": darts_model,
        "hyperparams_entrypoint": hyperparams_entrypoint,
        "trial_name": trial_name,
        "cut_date_val": cut_date_val,
        "cut_date_test": cut_date_test,
        "test_end_date": test_end_date,
        "device": device,
        "forecast_horizon": forecast_horizon,
        "stride": stride,
        "retrain": retrain,
        "scale": scale,
        "scale_covs": scale_covs,
        "multiple": multiple,
        "eval_series": eval_series,
        "n_trials": n_trials,
        "num_workers": num_workers,
        "eval_method": eval_method,
        "loss_function": loss_function,
        "evaluate_all_ts": evaluate_all_ts,
        "grid_search": grid_search,
        "num_samples": num_samples,
        "pv_ensemble": pv_ensemble,
        "format": format,
        "experiment_name": experiment_name,
        "parent_run_name": parent_run_name
        if none_checker(parent_run_name) is not None
        else darts_model + "_pipeline",
    }



    n_trials = none_checker(n_trials)
    n_trials = int(n_trials)
    evaluate_all_ts = evaluate_all_ts
    pv_ensemble = pv_ensemble
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(tags={"mlflow.runName": parent_run_name}, run_id=start_pipeline_run) as parent_run:
        with mlflow.start_run(tags={"mlflow.runName": f'optuna_test_{darts_model}'}, nested=True) as mlrun:
            if grid_search:
                # hyperparameters = ConfigParser(config_file='../config_opt.yml', config_string=hyperparams_entrypoint).read_hyperparameters(hyperparams_entrypoint)
                hyperparameters = hyperparams_entrypoint
                training_dict = {}
                for param, value in hyperparameters.items():
                    if type(value) == list and value and value[0] == "range":
                        if type(value[1]) == int:
                            training_dict[param] = list(range(value[1], value[2], value[3]))
                        else:
                            training_dict[param] = list(range(value[1], value[2], value[3]))
                    elif type(value) == list and value and value[0] == "list":
                        training_dict[param] = value[1:]
                study = optuna.create_study(storage="sqlite:///memory.db", study_name=trial_name, load_if_exists=True, sampler=optuna.samplers.GridSampler(training_dict))
            else:
                study = optuna.create_study(storage="sqlite:///memory.db", study_name=trial_name, load_if_exists=True)

            opt_tmpdir = tempfile.mkdtemp()
            curr_run_id = mlrun.info.run_id

            if evaluate_all_ts:
                opt_all_results = tempfile.mkdtemp()
            else:
                opt_all_results = None
            study.optimize(lambda trial: objective(series_csv, series_uri, future_covs_csv, future_covs_uri, past_covs_csv, past_covs_uri, year_range, resolution,
                        darts_model, hyperparams_entrypoint, trial_name, cut_date_val, test_end_date, cut_date_test, device,
                        forecast_horizon, m_mase, stride, retrain, scale, scale_covs,
                        multiple, eval_series, mlrun, trial, study, opt_tmpdir, num_workers, eval_method, 
                        loss_function, opt_all_results, evaluate_all_ts, num_samples, pv_ensemble, format),
                        n_trials=n_trials, n_jobs = 1)

            log_optuna(study, opt_tmpdir, hyperparams_entrypoint, trial_name, mlrun, opt_all_results=opt_all_results, evaluate_all_ts=evaluate_all_ts, scale_covs=scale_covs)

        return curr_run_id, parameters_dict