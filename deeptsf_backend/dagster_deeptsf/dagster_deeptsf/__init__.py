from dagster import Definitions, load_assets_from_modules
import os

from dagster_deeptsf import assets, load_raw_data, etl, evaluate_forecasts
from dagster_deeptsf.deeptsf_dagster_job import deeptsf_dagster_job, DeepTSFConfig
from dagster_celery import celery_executor
from dagster_aws.s3 import s3_pickle_io_manager, s3_resource

all_assets = load_assets_from_modules([load_raw_data, etl, assets, evaluate_forecasts])

# defs = Definitions(
#     assets=all_assets,
#     jobs=[deeptsf_dagster_job],
#     # schedules=[basic_schedule],
#     schedules=[],
#     executor=celery_executor,
#     resources={
#         "config": DeepTSFConfig(),
#         "io_manager": s3_pickle_io_manager.configured({
#             "s3_bucket": "dagster-storage",
#             "s3_prefix": "dagster-data/io-manager"
#         }),
#         "s3": s3_resource.configured({
#             "endpoint_url": "http://s3:9000"
#         }),
#     }
# )

# Route dagster's celery dispatch through the same broker the worker listens on
# (dagster-celery otherwise defaults to pyamqp://guest@localhost//).
celery_executor_env = celery_executor.configured(
    {
        "broker": {"env": "CELERY_BROKER_URL"},
        "backend": {"env": "CELERY_RESULT_BACKEND"},
    },
    name="celery",
)

defs = Definitions(
    assets=all_assets,
    jobs=[deeptsf_dagster_job],
    # schedules=[basic_schedule],
    schedules=[],
    executor=celery_executor_env,
    resources={
        "config": DeepTSFConfig(),
    }
)