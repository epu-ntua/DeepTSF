# launch_deeptsf.py
import yaml
from dagster_graphql import DagsterGraphQLClient, DagsterGraphQLClientError

HOST = "localhost"          # or the container’s DNS name in Docker Compose
PORT = 8006                 # bind-port from step 1

JOB_NAME = "deeptsf_dagster_job"

# -------------------------------------------------------------------
#  1) baseline single-series – LightGBM, no SHAP, no covariates
# -------------------------------------------------------------------
cfg1 = {
    "a": 0.3,
    "analyze_with_shap": False,
    "convert_to_local_tz": True,
    "country": "PT",
    "cut_date_test": "20210101",
    "cut_date_val": "20200101",
    "darts_model": "LightGBM",
    "database_name": "rdn_load_data",
    "device": "gpu",
    "eval_method": "ts_ID",
    "eval_series": "eval_series",
    "evaluate_all_ts": True,
    "experiment_name": "dagster_test",
    "forecast_horizon": 24,
    "format": "long",
    "from_database": False,
    "future_covs_csv": "None",
    "future_covs_uri": "None",
    "grid_search": False,
    "hyperparams_entrypoint": {"lags": [-1, -2, -14]},
    "ignore_previous_runs": True,
    "imputation_method": "linear",
    "loss_function": "mape",
    "m_mase": 1,
    "max_thr": -1,
    "min_non_nan_interval": 24,
    "multiple": False,
    "n_trials": 100,
    "num_samples": 1,
    "num_workers": 4,
    "opt_test": False,
    "order": 1,
    "parent_run_name": "dagster_test",
    "past_covs_csv": "None",
    "past_covs_uri": "None",
    "pv_ensemble": False,
    "resampling_agg_method": "averaging",
    "resolution": "1h",
    "retrain": False,
    "rmv_outliers": True,
    "scale": True,
    "scale_covs": True,
    "series_csv": "dataset-storage/Italy.csv",
    "series_uri": "None",
    "shap_data_size": 100,
    "shap_input_length": -1,
    "std_dev": 4.5,
    "stride": -1,
    "test_end_date": "None",
    "time_covs": False,
    "ts_used_id": "None",
    "wncutoff": 0.000694,
    "ycutoff": 3,
    "ydcutoff": 30,
    "year_range": "None",
}

# -------------------------------------------------------------------
#  2) multi-series with SHAP, LightGBM
# -------------------------------------------------------------------
cfg2 = {
    "a": 0.3,
    "analyze_with_shap": True,
    "convert_to_local_tz": True,
    "country": "PT",
    "cut_date_test": "20150429",
    "cut_date_val": "20150419",
    "darts_model": "LightGBM",
    "database_name": "rdn_load_data",
    "device": "gpu",
    "eval_method": "ts_ID",
    "eval_series": "1",
    "evaluate_all_ts": False,
    "experiment_name": "dagster_test",
    "forecast_horizon": 24,
    "format": "long",
    "from_database": False,
    "future_covs_csv": "None",
    "future_covs_uri": "None",
    "grid_search": False,
    "hyperparams_entrypoint": {"lags": [-1, -2, -14]},
    "ignore_previous_runs": True,
    "imputation_method": "linear",
    "loss_function": "mape",
    "m_mase": 1,
    "max_thr": -1,
    "min_non_nan_interval": 24,
    "multiple": True,
    "n_trials": 100,
    "num_samples": 1,
    "num_workers": 4,
    "opt_test": False,
    "order": 1,
    "parent_run_name": "dagster_test",
    "past_covs_csv": "None",
    "past_covs_uri": "None",
    "pv_ensemble": False,
    "resampling_agg_method": "averaging",
    "resolution": "1h",
    "retrain": False,
    "rmv_outliers": True,
    "scale": True,
    "scale_covs": True,
    "series_csv": "dataset-storage/multiple_and_multivariate_sample_series_long.csv",
    "series_uri": "None",
    "shap_data_size": 4,
    "shap_input_length": 24,
    "std_dev": 4.5,
    "stride": -1,
    "test_end_date": "None",
    "time_covs": False,
    "ts_used_id": "None",
    "wncutoff": 0.000694,
    "ycutoff": 3,
    "ydcutoff": 30,
    "year_range": "None",
}

# -------------------------------------------------------------------
#  3) multi-series, past+future covariates, no SHAP
# -------------------------------------------------------------------
cfg3 = {
    **cfg2,  # inherit cfg2 then override what differs
    "analyze_with_shap": False,
    "evaluate_all_ts": True,
    "future_covs_csv": "dataset-storage/future_covs_multiple_long.csv",
    "past_covs_csv": "dataset-storage/future_covs_multiple_long.csv",
    "hyperparams_entrypoint": {
        "lags": [-1, -2, -14],
        "lags_past_covariates": [-1, -5],
        "lags_future_covariates": [-1, -5],
    },
    "shap_data_size": 100,
    "shap_input_length": -1,
}

# -------------------------------------------------------------------
#  4) grid-search on multi-series with covariates + SHAP
# -------------------------------------------------------------------
cfg4 = {
    **cfg3,
    "analyze_with_shap": True,
    "grid_search": True,
    "shap_data_size": 3,
    "opt_test": True,
}

# -------------------------------------------------------------------
#  5) single-series, country IT, SHAP + grid-search
# -------------------------------------------------------------------
cfg5 = {
    **cfg1,
    "analyze_with_shap": True,
    "country": "IT",
    "grid_search": True,
    "shap_data_size": 2,
    "opt_test": True,
    "hyperparams_entrypoint": {"lags": ["list", 1, 2, 14]},
    "trial_name": "Default",
}

# -------------------------------------------------------------------
#  6) multi-series, datasets & covariates on S3 URIs
# -------------------------------------------------------------------
cfg6 = {
    **cfg2,
    "future_covs_uri": "s3://mlflow-bucket/12/8d84031093e74243a2ec26e344ac4ee4/artifacts/features/past_covariates_transformed.csv",
    "past_covs_uri": "s3://mlflow-bucket/12/8d84031093e74243a2ec26e344ac4ee4/artifacts/features/past_covariates_transformed.csv",
    "series_csv": "None",
    "series_uri": "s3://mlflow-bucket/12/8d84031093e74243a2ec26e344ac4ee4/artifacts/features/series.csv",
}

# -------------------------------------------------------------------
#  7) multi-series SHAP + covariates again but smaller SHAP sample
# -------------------------------------------------------------------
cfg7 = {
    **cfg4,
    "shap_data_size": 4,
}

# -------------------------------------------------------------------
#  8) multi-series, SHAP, *time_covs* enabled
# -------------------------------------------------------------------
cfg8 = {
    **cfg4,
    "time_covs": True,
    "shap_data_size": 2,
}

# -------------------------------------------------------------------
#  9) single-series NBEATS (generic arch) – SHAP
# -------------------------------------------------------------------
cfg9 = {
    **cfg1,
    "darts_model": "NBEATS",
    "hyperparams_entrypoint": {
        "input_chunk_length": 384,
        "output_chunk_length": 96,
        "num_stacks": 25,
        "num_blocks": 1,
        "num_layers": 4,
        "generic_architecture": True,
        "layer_widths": 128,
        "expansion_coefficient_dim": 5,
        "n_epochs": 3,
        "random_state": 0,
        "nr_epochs_val_period": 2,
        "batch_size": 1024,
    },
    "time_covs": True,
    "shap_input_length": -1,
    "shap_data_size": 2,
}

# -------------------------------------------------------------------
# 10) multi-series NBEATS, grid-search, small epochs
# -------------------------------------------------------------------
cfg10 = {
    **cfg2,
    "darts_model": "NBEATS",
    "forecast_horizon": 12,
    "grid_search": True,
    "opt_test": True,
    "time_covs": True,
    "hyperparams_entrypoint": {
        "input_chunk_length": 200,
        "output_chunk_length": 12,
        "num_stacks": 25,
        "num_blocks": 1,
        "num_layers": 4,
        "generic_architecture": True,
        "layer_widths": 128,
        "expansion_coefficient_dim": 5,
        "n_epochs": 2,
        "random_state": 0,
        "nr_epochs_val_period": 2,
        "batch_size": 1024,
    },
}

# -------------------------------------------------------------------
# 11) multi-series RNN (LSTM) grid-search, Gaussian likelihood
# -------------------------------------------------------------------
cfg11 = {
    **cfg2,
    "darts_model": "RNN",
    "forecast_horizon": 12,
    "grid_search": True,
    "opt_test": True,
    "loss_function": "mase",
    "trial_name": "test22",
    "hyperparams_entrypoint": {
        "model": "LSTM",
        "n_rnn_layers": 1,
        "input_chunk_length": 24,
        "output_chunk_length": 24,
        "hidden_dim": ["range", 24, 72, 24],
        "n_epochs": 2,
        "random_state": 0,
        "nr_epochs_val_period": 2,
        "dropout": 0,
        "learning_rate": 0.001,
        "batch_size": 1024,
        "likelihood": "Gaussian",
        "training_length": 24,
    },
}

# -------------------------------------------------------------------
# 12) single-series LightGBM, shap_input_length=None (str) example
# -------------------------------------------------------------------
cfg12 = {
    **cfg1,
    "forecast_horizon": "24",          # kept as str to match original
    "shap_input_length": "None",       # string literal “None”
}

# -------------------------------------------------------------------
# 13) single-series LightGBM with explicit lags & output_chunk_length
# -------------------------------------------------------------------
cfg13 = {
    **cfg1,
    "hyperparams_entrypoint": {
        "lags": 168,
        "output_chunk_length": 24,
        "random_state": 42,
    },
}

run_config = {
    "resources": {
        "config": {
            "config": {
                "a": 0.3,
                "analyze_with_shap": False,
                "convert_to_local_tz": True,
                "country": "PT",
                "cut_date_test": "20210101",
                "cut_date_val": "20200101",
                "darts_model": "LightGBM",
                "database_name": "rdn_load_data",
                "device": "gpu",
                "eval_method": "ts_ID",
                "eval_series": "eval_series",
                "evaluate_all_ts": True,
                "experiment_name": "dagster_test",
                "forecast_horizon": 24,
                "format": "long",
                "from_database": False,
                "future_covs_csv": "None",
                "future_covs_uri": "None",
                "grid_search": False,
                "hyperparams_entrypoint": {
                    "lags": [-1, -2, -14],
                },
                "ignore_previous_runs": True,
                "imputation_method": "linear",
                "loss_function": "mape",
                "m_mase": 1,
                "max_thr": -1,
                "min_non_nan_interval": 24,
                "multiple": False,
                "n_trials": 100,
                "num_samples": 1,
                "num_workers": 4,
                "opt_test": False,
                "order": 1,
                "parent_run_name": "dagster_test",
                "past_covs_csv": "None",
                "past_covs_uri": "None",
                "pv_ensemble": False,
                "resampling_agg_method": "averaging",
                "resolution": "1h",
                "retrain": False,
                "rmv_outliers": True,
                "scale": True,
                "scale_covs": True,
                "series_csv": "dataset-storage/Italy.csv",
                "series_uri": "None",
                "shap_data_size": 100,
                "shap_input_length": -1,
                "std_dev": 4.5,
                "stride": -1,
                "test_end_date": "None",
                "time_covs": False,
                "ts_used_id": "None",
                "wncutoff": 0.000694,
                "ycutoff": 3,
                "ydcutoff": 30,
                "year_range": "None",
            }
        }
    }
}

# 2  connect to Dagster’s GraphQL endpoint
client = DagsterGraphQLClient(HOST, port_number=PORT)

# 3  submit an asynchronous run
try:
    run_id = client.submit_job_execution(
        JOB_NAME,
        run_config=run_config,
    )
    print(f"Launched Dagster run {run_id}")
except DagsterGraphQLClientError as exc:          # handy for surfacing schema errors
    print(f"Dagster rejected the launch: {exc}")
    raise