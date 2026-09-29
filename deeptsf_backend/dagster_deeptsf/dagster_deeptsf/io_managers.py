from dagster._utils.cached_method import cached_method
from dagster_aws.s3 import S3PickleIOManager
from dagster_aws.s3.io_manager import PickledObjectS3IOManager


class RunScopedPickledObjectS3IOManager(PickledObjectS3IOManager):
    """Pickles step outputs to S3 under ``<prefix>/storage/<run_id>/<step_key>/<output>``.

    Dagster's pickle IO managers (fs and S3 alike) store *asset* outputs under the
    asset key only (``<prefix>/start_pipeline_run``, ``<prefix>/etl_out``, ...), so
    every run of deeptsf_dagster_job shares one object per asset. With concurrent
    runs a later run overwrites those objects before an earlier run's downstream
    steps read them, e.g. a step loads another run's ``start_pipeline_run`` and
    nests its MLflow child run under the wrong parent. Scoping by run id, as dagster
    already does for plain op outputs, keeps runs isolated. Re-execution still works
    since ``get_identifier`` resolves to the parent run for steps that were not re-run.
    """

    def get_asset_relative_path(self, context):
        return self.get_op_output_relative_path(context)


class RunScopedS3PickleIOManager(S3PickleIOManager):
    """S3PickleIOManager that keeps each run's asset outputs separate (see above)."""

    @classmethod
    def _is_dagster_maintained(cls) -> bool:
        return False

    @cached_method
    def inner_io_manager(self) -> RunScopedPickledObjectS3IOManager:
        return RunScopedPickledObjectS3IOManager(
            s3_bucket=self.s3_bucket,
            s3_session=self.s3_resource.get_client(),
            s3_prefix=self.s3_prefix,
        )
