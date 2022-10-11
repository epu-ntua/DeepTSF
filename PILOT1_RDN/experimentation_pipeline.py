"""
Downloads the REN dataset, ETLs (cleansing, resampling) it together with time covariates,
trains a darts model, and evaluates the model.
"""
import matplotlib.pyplot as plt
import tempfile
import pretty_errors
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, BlockRNNModel, NBEATSModel, LightGBMModel, RandomForest, TFTModel, TCNModel
import mlflow
import click
import os
import pretty_errors
from utils import download_online_file
# from darts.utils.likelihood_models import ContinuousBernoulliLikelihood, GaussianLikelihood, DirichletLikelihood, ExponentialLikelihood, GammaLikelihood, GeometricLikelihood
import pretty_errors
import click
import os
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id
from utils import truth_checker, load_yaml_as_dict, download_online_file, ConfigParser, save_dict_as_yaml, none_checker
import optuna
import logging

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
# os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().mlflow_tracking_uri
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

def objective(series_csv, series_uri, year_range, resolution, time_covs,
             darts_model, hyperparams_entrypoint, cut_date_val, test_end_date, cut_date_test, device,
             forecast_horizon, stride, retrain, ignore_previous_runs, scale, scale_covs, day_first,
             country, std_dev, max_thr, a, wncutoff, ycutoff, ydcutoff, shap_data_size, analyze_with_shap,
             multiple, eval_country, etl_series_uri, etl_time_covariates_uri, git_commit, trial):
                print("CUUUUUUT", cut_date_val)
                hyperparameters = ConfigParser().read_hyperparameters(hyperparams_entrypoint)
                training_dict = {}
                for key, value in hyperparameters.items():
                    if key.split("_")[-2:] == ["opt", "test"]:
                        param = "_".join(key.split("_")[:-2])
                        print(param, value)
                        if value[0] == "range":
                            if type(value[1]) == int:
                                print(param)
                                training_dict[param] = trial.suggest_int(param, value[1], value[2], value[3])
                            else:
                                print(param)
                                training_dict[param] = trial.suggest_float(param, value[1], value[2], value[3])
                        else:
                            print(param)
                            training_dict[param] = trial.suggest_categorical(param, value[1:])
                    else:
                        print(key)
                        training_dict[key] = value

                filedir = tempfile.mkdtemp()
                filepath = os.path.join(filedir, "dict.yml")
                save_dict_as_yaml(filepath, training_dict)

                train_params = {
                    "series_uri": etl_series_uri,
                    "future_covs_uri": etl_time_covariates_uri,
                    "past_covs_uri": None, # fix that in case REAL Temperatures come -> etl_temp_covs_uri. For forecasts, integrate them into future covariates!!
                    "darts_model": darts_model,
                    "hyperparams_entrypoint": hyperparams_entrypoint,
                    "cut_date_val": cut_date_val,
                    "cut_date_test": cut_date_test,
                    "test_end_date": test_end_date,
                    "device": device,
                    "scale": scale,
                    "scale_covs": scale_covs,
                    "multiple": multiple,
                    "opt_test": True,
                    "training_dict": filepath,
                    }
                train_run = _get_or_run("train", train_params, git_commit, ignore_previous_runs)

                # Log train params (mainly for logging hyperparams to father run)
                for param_name, param_value in train_run.data.params.items():
                    try:
                        mlflow.log_param(param_name, param_value)
                    except mlflow.exceptions.RestException:
                        pass
                    except mlflow.exceptions.MlflowException:
                        pass

                train_model_uri = train_run.data.tags["model_uri"].replace("s3:/", S3_ENDPOINT_URL)
                train_model_type = train_run.data.tags["model_type"]
                train_series_uri = train_run.data.tags["series_uri"].replace("s3:/", S3_ENDPOINT_URL)
                train_future_covariates_uri = train_run.data.tags["future_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
                train_past_covariates_uri = train_run.data.tags["past_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
                train_scaler_uri = train_run.data.tags["scaler_uri"].replace("s3:/", S3_ENDPOINT_URL)
                train_setup_uri = train_run.data.tags["setup_uri"].replace("s3:/", S3_ENDPOINT_URL)

                # 4. Evaluation
                ## load setup file
                setup_file = download_online_file(
                    train_setup_uri, "setup.yml")
                setup = load_yaml_as_dict(setup_file)
                print(f"\nSplit info: {setup} \n")
                eval_params = {
                    "series_uri": train_series_uri,
                    "future_covs_uri": train_future_covariates_uri,
                    "past_covs_uri": train_past_covariates_uri,
                    "scaler_uri": train_scaler_uri,
                    "cut_date_test": cut_date_test,
                    "test_end_date": test_end_date,#check that again
                    "model_uri": train_model_uri,
                    "model_type": train_model_type,
                    "forecast_horizon": forecast_horizon,
                    "stride": stride,
                    "retrain": retrain,
                    "input_chunk_length" : None,
                    "size" : shap_data_size,
                    "analyze_with_shap" : analyze_with_shap,
                    "multiple": multiple,
                    "eval_country": eval_country,
                    "cut_date_val": cut_date_val,
                    "opt_test": True,
                    }

                if "input_chunk_length" in train_run.data.params:
                    eval_params["input_chunk_length"] = train_run.data.params["input_chunk_length"]

                eval_run = _get_or_run("eval", eval_params, git_commit)

                # Log eval metrics to father run for consistency and clear results
                mlflow.log_metrics(eval_run.data.metrics)

                return eval_run.data.metrics["mape"]


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping " "(run_id=%s, status=%s)")
                % (run_info.run_id, run_info.status)
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(
                (
                    "Run matched, but has a different source code version, so skipping "
                    "(found=%s, expected=%s)"
                )
                % (previous_version, git_commit)
            )
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependent steps
def _get_or_run(entrypoint, parameters, git_commit, ignore_previous_run=True, use_cache=True):
    # TODO: this was removed to always run the pipeline from the beginning.
    if not ignore_previous_run:
        existing_run = _already_ran(entrypoint, parameters, git_commit)
        if use_cache and existing_run:
            print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
            return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, env_manager="local")
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


@click.command()
# load arguments
@click.option("--series-csv",
    type=str,
    default="../../RDN/Load_Data/2009-2021-global-load.csv",
    help="Local timeseries file"
    )
@click.option("--series-uri",
    default="online_artifact",
    help="Online link to download the series from"
    )
# etl arguments
@click.option("--resolution",
    default="15",
    type=str,
    help="Change the resolution of the dataset (minutes)."
)
@click.option('--year-range',
    default="2009-2019",
    type=str,
    help='Choose year range to include in the dataset.'
)
@click.option(
    "--time-covs",
    default="PT",
    type=click.Choice(["None", "PT"]),
    help="Optionally add time covariates to the timeseries."
)
# training arguments
@click.option("--darts-model",
              type=click.Choice(
                  ['NBEATS',
                   'RNN',
                   'TCN',
                   'BlockRNN',
                   'TFT',
                   'LightGBM',
                   'RandomForest',
                   'Naive',
                   'AutoARIMA']),
              multiple=False,
              default='RNN',
              help="The base architecture of the model to be trained"
              )
@click.option("--hyperparams-entrypoint", "-h",
              type=str,
              default='LSTM1',
              help=""" The entry point of config.yml under the 'hyperparams'
              one containing the desired hyperparameters for the selected model"""
              )
@click.option("--cut-date-val",
              type=str,
              default='20190101',
              help="Validation set start date [str: 'YYYYMMDD']"
              )
@click.option("--cut-date-test",
              type=str,
              default='20200101',
              help="Test set start date [str: 'YYYYMMDD']",
              )
@click.option("--test-end-date",
              type=str,
              default='None',
              help="Test set ending date [str: 'YYYYMMDD']",
              )
@click.option("--device",
              type=click.Choice(
                  ['gpu',
                   'cpu']),
              multiple=False,
              default='gpu',
              )
# eval
@click.option("--forecast-horizon",
              type=str,
              default="96")
@click.option("--stride",
              type=str,
              default="None")
@click.option("--retrain",
              type=str,
              default="false",
              help="Whether to retrain model during backtesting")
@click.option("--ignore-previous-runs",
              type=str,
              default="true",
              help="Whether to ignore previous step runs while running the pipeline")
@click.option("--scale",
              type=str,
              default="true",
              help="Whether to scale the target series")
@click.option("--scale-covs",
              type=str,
              default="true",
              help="Whether to scale the covariates")
@click.option("--day-first",
              type=str,
              default="true",
              help="Whether the date has the day before the month")
@click.option("--country",
              type=str,
              default="Portugal",
              help="The country this dataset belongs to")

@click.option("--std-dev",
              type=str,
              default="4.5",
              help="The number to be multiplied with the standard deviation of \
                    each 1 month  period of the dataframe. The result is then used as \
                    a cut-off value as described above")

@click.option("--max-thr",
              type=str,
              default="48",
              help="If there is a consecutive subseries of NaNs longer than max_thr, \
                    then it is not imputed and returned with NaN values")

@click.option("--a",
              type=str,
              default="0.3",
              help="The weight that shows how quickly simple interpolation's weight decreases as \
                    the distacne to the nearest non NaN value increases")

@click.option("--wncutoff",
              type=str,
              default="0.000694",
              help="Historical data will only take into account dates that have at most wncutoff distance \
                    from the current null value's WN(Week Number)")

@click.option("--ycutoff",
             type=str,
             default="3",
             help="Historical data will only take into account dates that have at most ycutoff distance \
                   from the current null value's year")

@click.option("--ydcutoff",
             type=str,
             default="30",
             help="Historical data will only take into account dates that have at most ydcutoff distance \
                   from the current null value's yearday")

@click.option("--shap-data-size",
             type=str,
             default="10",
             help="Size of shap dataset in samples")

@click.option("--analyze-with-shap",
             type=str,
             default="False",
             help="Whether to do SHAP analysis on the model. Only global forecasting models are supported")
@click.option("--multiple",
    type=str,
    default="false",
    help="Whether to train on multiple timeseries")

@click.option("--eval-country",
    type=str,
    default="Portugal",
    help="On which country to run the backtesting. Only for multiple timeseries")

@click.option("--n-trials",
    type=str,
    default="100",
    help="How many trials optuna will run")


def workflow(series_csv, series_uri, year_range, resolution, time_covs,
             darts_model, hyperparams_entrypoint, cut_date_val, test_end_date, cut_date_test, device,
             forecast_horizon, stride, retrain, ignore_previous_runs, scale, scale_covs, day_first,
             country, std_dev, max_thr, a, wncutoff, ycutoff, ydcutoff, shap_data_size, analyze_with_shap,
             multiple, eval_country, n_trials):

    # Argument preprocessing
    ignore_previous_runs = truth_checker(ignore_previous_runs)


    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run(run_name=darts_model + '_pipeline') as active_run:
        mlflow.set_tag("stage", "main")

        # 1.Load Data
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)

        load_raw_data_params = {"series_csv": series_csv, "series_uri": series_uri, "day_first": day_first, "multiple": multiple}
        load_raw_data_run = _get_or_run("load_raw_data", load_raw_data_params, git_commit, ignore_previous_runs)
        # series_uri = f"{load_raw_data_run.info.artifact_uri}/raw_data/series.csv" \
        #                 .replace("s3:/", S3_ENDPOINT_URL)
        load_data_series_uri = load_raw_data_run.data.tags['dataset_uri'].replace("s3:/", S3_ENDPOINT_URL)

        # 2. ETL
        etl_params = {"series_uri": load_data_series_uri,
                      "year_range": year_range,
                      "resolution": resolution,
                      "time_covs": time_covs,
                      "day_first": day_first,
                      "country": country,
                      "std_dev": std_dev,
                      "max_thr": max_thr,
                      "a": a,
                      "wncutoff": wncutoff,
                      "ycutoff": ycutoff,
                      "ydcutoff": ydcutoff,
                      "multiple": multiple}

        etl_run = _get_or_run("etl", etl_params, git_commit, ignore_previous_runs)

        etl_series_uri =  etl_run.data.tags["series_uri"].replace("s3:/", S3_ENDPOINT_URL)
        etl_time_covariates_uri =  etl_run.data.tags["time_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)

        # weather_covariates_uri = ...

        # 3. Training
        if hyperparams_entrypoint.split("_")[-2:] == ["opt", "test"]:
            n_trials = none_checker(n_trials)
            n_trials = int(n_trials)
            study = optuna.create_study(storage="sqlite:///memory.db", study_name=hyperparams_entrypoint, load_if_exists=True)

            study.optimize(lambda trial: objective(series_csv, series_uri, year_range, resolution, time_covs,
                           darts_model, hyperparams_entrypoint, cut_date_val, test_end_date, cut_date_test, device,
                           forecast_horizon, stride, retrain, ignore_previous_runs, scale, scale_covs, day_first,
                           country, std_dev, max_thr, a, wncutoff, ycutoff, ydcutoff, shap_data_size, analyze_with_shap,
                           multiple, eval_country, etl_series_uri, etl_time_covariates_uri, git_commit, trial), n_trials=n_trials, n_jobs = 1)

            opt_tmpdir = tempfile.mkdtemp()
            plt.close()

            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig(f"{opt_tmpdir}/plot_optimization_history.png")
            plt.close()

            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(f"{opt_tmpdir}/plot_param_importances.png")
            plt.close()

            fig = optuna.visualization.matplotlib.plot_slice(study)
            plt.savefig(f"{opt_tmpdir}/plot_slice.png")
            plt.close()

            study.trials_dataframe().to_csv(f"{opt_tmpdir}/{hyperparams_entrypoint}.csv")

            print("\nUploading optuna plots to MLflow server...")
            logging.info("\nUploading optuna plots to MLflow server...")

            mlflow.log_artifacts(opt_tmpdir, "optuna_results")



        else:
            train_params = {
                "series_uri": etl_series_uri,
                "future_covs_uri": etl_time_covariates_uri,
                "past_covs_uri": None, # fix that in case REAL Temperatures come -> etl_temp_covs_uri. For forecasts, integrate them into future covariates!!
                "darts_model": darts_model,
                "hyperparams_entrypoint": hyperparams_entrypoint,
                "cut_date_val": cut_date_val,
                "cut_date_test": cut_date_test,
                "test_end_date": test_end_date,
                "device": device,
                "scale": scale,
                "scale_covs": scale_covs,
                "multiple": multiple,
                "opt_test": False,
                "training_dict": None,
            }
            train_run = _get_or_run("train", train_params, git_commit, ignore_previous_runs)

            # Log train params (mainly for logging hyperparams to father run)
            for param_name, param_value in train_run.data.params.items():
                try:
                    mlflow.log_param(param_name, param_value)
                except mlflow.exceptions.RestException:
                    pass
                except mlflow.exceptions.MlflowException:
                    pass

            train_model_uri = train_run.data.tags["model_uri"].replace("s3:/", S3_ENDPOINT_URL)
            train_model_type = train_run.data.tags["model_type"]
            train_series_uri = train_run.data.tags["series_uri"].replace("s3:/", S3_ENDPOINT_URL)
            train_future_covariates_uri = train_run.data.tags["future_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
            train_past_covariates_uri = train_run.data.tags["past_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
            train_scaler_uri = train_run.data.tags["scaler_uri"].replace("s3:/", S3_ENDPOINT_URL)
            train_setup_uri = train_run.data.tags["setup_uri"].replace("s3:/", S3_ENDPOINT_URL)

            # 4. Evaluation
            ## load setup file
            setup_file = download_online_file(
                train_setup_uri, "setup.yml")
            setup = load_yaml_as_dict(setup_file)
            print(f"\nSplit info: {setup} \n")

            eval_params = {
                "series_uri": train_series_uri,
                "future_covs_uri": train_future_covariates_uri,
                "past_covs_uri": train_past_covariates_uri,
                "scaler_uri": train_scaler_uri,
                "cut_date_test": setup['test_start'],
                "test_end_date": setup['test_end'],
                "model_uri": train_model_uri,
                "model_type": train_model_type,
                "forecast_horizon": forecast_horizon,
                "stride": stride,
                "retrain": retrain,
                "input_chunk_length" : None,
                "size" : shap_data_size,
                "analyze_with_shap" : analyze_with_shap,
                "multiple": multiple,
                "eval_country": eval_country,
            }

            if "input_chunk_length" in train_run.data.params:
                eval_params["input_chunk_length"] = train_run.data.params["input_chunk_length"]

            eval_run = _get_or_run("eval", eval_params, git_commit)

            # Log eval metrics to father run for consistency and clear results
            mlflow.log_metrics(eval_run.data.metrics)

if __name__ == "__main__":
    workflow()
