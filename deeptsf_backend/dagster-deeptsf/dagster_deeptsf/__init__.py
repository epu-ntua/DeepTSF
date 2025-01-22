from dagster import Definitions, load_assets_from_modules
import os

from . import assets, load_raw_data, etl, evaluate_forecasts
from dagster_deeptsf.deeptsf_dagster_job import deeptsf_dagster_job, DeepTSFConfig

all_assets = load_assets_from_modules([load_raw_data, etl, assets, evaluate_forecasts])

defs = Definitions(
    assets=all_assets,
    jobs=[deeptsf_dagster_job],
    resources={
        "config": DeepTSFConfig(),
    }
)