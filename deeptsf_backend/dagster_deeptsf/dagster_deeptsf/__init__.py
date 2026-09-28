from dagster import Definitions, EnvVar, load_assets_from_modules
import os

from dagster_deeptsf import assets, load_raw_data, etl, evaluate_forecasts
from dagster_deeptsf.deeptsf_dagster_job import deeptsf_dagster_job, DeepTSFConfig
from dagster_celery import celery_executor
from dagster_aws.s3 import s3_pickle_io_manager, s3_resource, S3Resource
from dagster_deeptsf.io_managers import RunScopedS3PickleIOManager

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
        # Step outputs go to MinIO, kept per run so concurrent runs cannot read each other's.
        "io_manager": RunScopedS3PickleIOManager(
            s3_resource=S3Resource(
                endpoint_url=EnvVar("MLFLOW_S3_ENDPOINT_URL"),
                aws_access_key_id=EnvVar("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=EnvVar("AWS_SECRET_ACCESS_KEY"),
                verify=False,
            ),
            s3_bucket="dagster-storage",
            s3_prefix=os.environ.get("DAGSTER_IO_MANAGER_PREFIX", "dagster-io-manager"),
        ),
    }
)