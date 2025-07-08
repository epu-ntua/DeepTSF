# launch_deeptsf.py
import yaml
from dagster_graphql import DagsterGraphQLClient, DagsterGraphQLClientError

HOST = "localhost"          # or the container’s DNS name in Docker Compose
PORT = 8006                 # bind-port from step 1

JOB_NAME = "deeptsf_dagster_job"

# 1  read the YAML file into a dict
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