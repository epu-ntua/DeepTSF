"""Process-level setup that has to happen before any DeepTSF module is imported.

DeepTSF reads its configuration from environment variables at import time and
calls ``load_dotenv()``, which walks up from each module to
``deeptsf_backend/.env``: the deployment's MinIO, MLflow and Keycloak settings.
The tests must never pick those up, so ``isolate()`` disables ``.env`` loading
and pins every variable the code reads to a test value. The endpoints of the
local S3 and MLflow servers are filled in later by ``servers.py``.
"""
import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = TESTS_DIR.parent

# Everything the tests write goes below this directory (see README.md).
TEST_HOME = Path(os.environ.get("DEEPTSF_TEST_HOME", TESTS_DIR / ".work")).resolve()

# Files the training step packages into every MLflow model (log_model's
# code_path, relative to the working directory, which is /app in the worker).
MODEL_CODE_FILES = ["exceptions.py", "utils.py", "inference.py", "darts_flavor.py"]

TEST_ENV = {
    "USE_AUTH": "None",                       # no Keycloak / JWT, no per-run MLflow auth
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_DEFAULT_REGION": "us-east-1",
    "MINIO_ACCESS_KEY": "testing",
    "MINIO_SECRET_KEY": "testing",
    "MINIO_SSL": "False",
    "GIT_PYTHON_REFRESH": "quiet",
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
    "CELERY_BROKER_URL": "memory://",
    "CELERY_RESULT_BACKEND": "cache+memory://",
    "PYTHONWARNINGS": "ignore",
    "TQDM_DISABLE": "1",
}


def worker_dir(run_uid: str, worker_id: str) -> Path:
    return TEST_HOME / "runs" / run_uid / worker_id


def isolate(workdir: Path) -> None:
    """Prepare this process: env vars and sys.path."""
    import dotenv
    import dotenv.main

    def _no_dotenv(*args, **kwargs):
        return False

    dotenv.load_dotenv = _no_dotenv
    dotenv.main.load_dotenv = _no_dotenv

    workdir.mkdir(parents=True, exist_ok=True)
    tmp = workdir / "tmp"
    tmp.mkdir(exist_ok=True)
    os.environ.update(TEST_ENV)
    os.environ["TMPDIR"] = str(tmp)
    os.environ["DAGSTER_HOME"] = str(workdir / "dagster_home")
    (workdir / "dagster_home").mkdir(exist_ok=True)
    import tempfile
    tempfile.tempdir = str(tmp)

    # The dagster package lives in dagster_deeptsf/dagster_deeptsf and its modules
    # import the shared helpers (utils, exceptions, ...) from deeptsf_backend.
    # dagster_deeptsf/ must come first: deeptsf_backend/ also contains a
    # "dagster_deeptsf" folder (the project root), which would shadow the package.
    for p in (str(BACKEND_DIR), str(BACKEND_DIR / "dagster_deeptsf")):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)



def enter(workdir: Path) -> None:
    """Make workdir the working directory (after collection, which resolves test
    paths against the original one). The pipeline downloads inputs to paths
    relative to it, darts writes checkpoints to ./darts_logs and log_model copies
    MODEL_CODE_FILES from it, so every worker runs from its own directory holding
    links to them."""
    for name in MODEL_CODE_FILES:
        link = workdir / name
        if not link.exists():
            link.symlink_to(BACKEND_DIR / name)
    os.chdir(workdir)
