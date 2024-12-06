from dagster import Definitions, load_assets_from_modules
import os

from . import assets, load_raw_data, etl, evaluate_forecasts
from dagster_deeptsf.job_uc2 import uc2_mlflow_cli_job, basic_schedule, deeptsf_dagster_job, DeepTSFConfig

all_assets = load_assets_from_modules([load_raw_data, etl, assets, evaluate_forecasts])

defs = Definitions(
    assets=all_assets,
    jobs=[uc2_mlflow_cli_job, deeptsf_dagster_job],
    schedules=[basic_schedule],
    resources={
        "config": DeepTSFConfig(),
    }
)