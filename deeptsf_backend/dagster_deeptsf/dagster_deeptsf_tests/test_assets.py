import importlib
from typing import Dict, Any

import pytest
from dagster import Definitions

# ---------------------------------------------------------------------------
# Adjust this to the *actual* module name that contains your Dagster objects.
# ---------------------------------------------------------------------------
MODULE_UNDER_TEST = "deeptsf_pipeline"  # <- change if your file is named differently

pipeline = importlib.import_module(MODULE_UNDER_TEST)

# ---------------------------------------------------------------------------
# Light‑weight stubs to ensure the pipeline runs fast in unit‑tests.
# ---------------------------------------------------------------------------

# def _stub_start_pipeline_run() -> str:
#     """Pretend we launched a parent MLflow run and return its ID."""
#     return "test_parent_run"

# def _stub_load_raw_data_asset(parent_run_id: str) -> Dict[str, Any]:
#     assert parent_run_id == "test_parent_run"
#     return {"rows": 10}

# def _stub_etl_asset(parent_run_id: str, load_raw_data_out: Dict[str, Any]) -> Dict[str, Any]:
#     # Show that the ETL step receives the previous output
#     assert load_raw_data_out["rows"] == 10
#     return {"clean_rows": 10}

# def _stub_training_and_hyperparameter_tuning_asset(parent_run_id: str, etl_out: Dict[str, Any]) -> Dict[str, Any]:
#     return {"best_score": 0.12}

# def _stub_evaluation_asset(parent_run_id: str, training_and_hyperparameter_tuning_out: Dict[str, Any]) -> Dict[str, Any]:
#     return {"mape": 0.15}

# ---------------------------------------------------------------------------
# Example configuration taken from the prompt, converted to Python dict.
# You can add additional configs to the list below to expand coverage.
# ---------------------------------------------------------------------------
example_config = {
    "a": 0.3,
    "analyze_with_shap": True,
    "convert_to_local_tz": True,
    "country": "IT",
    "cut_date_test": "20210101",
    "cut_date_val": "20200101",
    "darts_model": "LightGBM",
    "database_name": "rdn_load_data",
    "device": "gpu",
    "eval_method": "ts_ID",
    "eval_series": "eval_series",
    "evaluate_all_ts": True,
    "experiment_name": "deeptsf_demo",
    "forecast_horizon": 24,
    "format": "long",
    "from_database": False,
    "future_covs_csv": "None",
    "future_covs_uri": "None",
    "grid_search": True,
    "hyperparams_entrypoint": {"lags": ["list", 1, 2, 12, 24]},
    "trial_name": "lgbm_italy",
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
    "opt_test": True,
    "order": 1,
    "parent_run_name": "deeptsf_demo",
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
    "shap_data_size": 50,
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

quick_config = {
    "country": "ES",
    "darts_model": "LinearRegression",
    "multiple": True,
    "forecast_horizon": 12,
    "resolution": "30min",
}

minimal_config: Dict[str, Any] = {}  # Relies entirely on DeepTSFConfig defaults


# @pytest.fixture(autouse=True)
# def _patch_external_dependencies(monkeypatch):
#     """Replace heavyweight asset functions with light stubs so tests stay fast & deterministic."""
#     monkeypatch.setattr(pipeline, "start_pipeline_run", _stub_start_pipeline_run)
#     monkeypatch.setattr(pipeline, "load_raw_data_asset", _stub_load_raw_data_asset)
#     monkeypatch.setattr(pipeline, "etl_asset", _stub_etl_asset)
#     monkeypatch.setattr(pipeline, "training_and_hyperparameter_tuning_asset", _stub_training_and_hyperparameter_tuning_asset)
#     monkeypatch.setattr(pipeline, "evaluation_asset", _stub_evaluation_asset)


# ---------------------------------------------------------------------------
# The actual parametrised test.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cfg_overrides",
    [example_config],
    ids=["example_from_prompt"]
)
def test_deeptsf_job_executes_successfully(cfg_overrides):
    """Ensure the Dagster job runs to completion for a variety of resource configurations."""

    # Instantiate the ConfigurableResource with overrides (falls back to defaults for everything else)
    deep_cfg = pipeline.DeepTSFConfig(**cfg_overrides)

    # Build a Definitions object on‑the‑fly so that we can feed our resource in
    defs = Definitions(
        assets=[pipeline.deepTSF_pipeline],
        jobs=[pipeline.deeptsf_dagster_job],
        resources={"config": deep_cfg},
    )

    # Retrieve the job from the Definitions bundle and execute it in‑process
    job = defs.get_job("deeptsf_dagster_job")
    result = job.execute_in_process(raise_on_error=True)

    assert result.success, "Job did not finish successfully"

    # Optional sanity checks on the outputs produced by our stub assets
    materialized_asset_keys = {mat.asset_key.to_string() for mat in result.asset_materializations}
    expected_asset_keys = {
        "deepTSF_pipeline.start_pipeline_run",
        "deepTSF_pipeline.load_raw_data_out",
        "deepTSF_pipeline.etl_out",
        "deepTSF_pipeline.training_and_hyperparameter_tuning_out",
        "deepTSF_pipeline.evaluation_out",
    }
    # All expected keys should be present in the run (order is irrelevant)
    assert expected_asset_keys.issubset(materialized_asset_keys)
