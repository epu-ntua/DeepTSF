"""pytest setup shared by all DeepTSF tests. See README.md.

* ``pytest_configure`` isolates each process (controller and every xdist worker)
  from the deployment before any DeepTSF module is imported.
* ``services`` starts the local S3 (MinIO stand-in) and MLflow servers once per
  worker and points DeepTSF at them.
* Tests of the same case are grouped onto one xdist worker, so the inference
  tests reuse the pipeline run of their case instead of re-running it.
"""
import datetime
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import environment  # noqa: E402


def _run_uid() -> str:
    return (os.environ.get("DEEPTSF_TEST_RUN_ID")
            or os.environ.get("PYTEST_XDIST_TESTRUNUID", "")[:12]
            or datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def pytest_configure(config):
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    config._deeptsf_workdir = environment.worker_dir(_run_uid(), worker)
    environment.isolate(config._deeptsf_workdir)


def pytest_collection_finish(session):
    environment.enter(session.config._deeptsf_workdir)


def pytest_collection_modifyitems(config, items):
    for item in items:
        case = getattr(item, "callspec", None) and item.callspec.params.get("case")
        if case is not None:
            item.add_marker(pytest.mark.xdist_group(case.id))


@pytest.fixture(scope="session")
def workdir(pytestconfig) -> Path:
    return pytestconfig._deeptsf_workdir


@pytest.fixture(scope="session")
def services(workdir):
    """Local S3 + MLflow servers for this worker, torn down at the end of the session."""
    from support.servers import MlflowServer, S3Server, point_deeptsf_at

    s3 = S3Server().start()
    mlflow_server = MlflowServer(workdir / "mlflow", s3.endpoint)
    try:
        mlflow_server.start()
        point_deeptsf_at(s3, mlflow_server)
        yield {"s3": s3, "mlflow": mlflow_server}
    finally:
        mlflow_server.stop()
        s3.stop()
        if os.environ.get("DEEPTSF_KEEP_TEST_FILES", "0") != "1":
            # temp files and checkpoints; the MLflow db and server logs stay for debugging
            for name in ("tmp", "darts_logs", "dataset-storage", "inputs", "mlruns"):
                shutil.rmtree(workdir / name, ignore_errors=True)
