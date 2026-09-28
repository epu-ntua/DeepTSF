"""Local stand-ins for the services the pipeline talks to.

* ``S3Server``: moto's S3 server in a background thread, playing MinIO. The
  pipeline downloads its input CSVs from the ``dataset-storage`` bucket,
  reads/writes MLflow artifacts in ``mlflow-bucket`` with the minio client and
  the optuna search uploads its study to ``dagster-storage``.
* ``MlflowServer``: a real ``mlflow server`` subprocess that proxies artifacts to
  that bucket (``--serve-artifacts``). Artifact URIs are therefore
  ``mlflow-artifacts:/...``, exactly as with the deployed tracking servers, so
  the URI rewriting in assets.py / evaluate_forecasts.py is exercised as is.

Both only listen on 127.0.0.1 and keep their state below the given directory.
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# dagster-storage holds compute logs in production; the optuna search also uploads to it
BUCKETS = ["dataset-storage", "mlflow-bucket", "dagster-storage"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout: float, proc: subprocess.Popen = None, log: Path = None) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"{url} exited early:\n{log.read_text()[-3000:] if log else ''}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{url} did not come up within {timeout}s")


def _minio_like_paths(app):
    """MinIO ignores leading slashes in object names, and DeepTSF relies on it:
    download_online_file() asks for "/1/<run>/artifacts/..." (the part of the URI
    after the bucket name), i.e. the request path is /mlflow-bucket//1/.... moto
    would look up a key starting with "/", so collapse them like MinIO does."""
    def middleware(environ, start_response):
        for var in ("PATH_INFO", "RAW_URI", "REQUEST_URI"):   # moto reads the raw URI too
            parts = environ.get(var, "").split("/", 2)          # ['', bucket, key...]
            if len(parts) == 3 and parts[2].startswith("/"):
                environ[var] = f"/{parts[1]}/{parts[2].lstrip('/')}"
        return app(environ, start_response)
    return middleware


class S3Server:
    def __init__(self):
        self.port = free_port()
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self._server = None

    def start(self) -> "S3Server":
        import threading

        import boto3
        from moto.server import DomainDispatcherApplication, create_backend_app
        from werkzeug.serving import make_server

        app = _minio_like_paths(DomainDispatcherApplication(create_backend_app))
        self._server = make_server("127.0.0.1", self.port, app, threaded=True)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        s3 = boto3.client("s3", endpoint_url=self.endpoint, region_name="us-east-1",
                          aws_access_key_id="testing", aws_secret_access_key="testing")
        for bucket in BUCKETS:
            s3.create_bucket(Bucket=bucket)
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()


class MlflowServer:
    def __init__(self, state_dir: Path, s3_endpoint: str):
        self.port = free_port()
        self.uri = f"http://127.0.0.1:{self.port}"
        self.state_dir = state_dir
        self.s3_endpoint = s3_endpoint
        self.log = state_dir / "mlflow_server.log"
        self._proc = None

    def start(self) -> "MlflowServer":
        self.state_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, MLFLOW_S3_ENDPOINT_URL=self.s3_endpoint)
        cmd = [sys.executable, "-m", "mlflow", "server",
               "--host", "127.0.0.1", "--port", str(self.port), "--workers", "2",
               "--backend-store-uri", f"sqlite:///{self.state_dir / 'mlflow.db'}",
               "--artifacts-destination", "s3://mlflow-bucket",
               "--serve-artifacts"]
        # own process group: `mlflow server` starts gunicorn workers that outlive it otherwise
        self._proc = subprocess.Popen(cmd, env=env, cwd=self.state_dir, start_new_session=True,
                                      stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        _wait_http(f"{self.uri}/health", timeout=180, proc=self._proc, log=self.log)
        return self

    def stop(self) -> None:
        import signal

        if self._proc is None:
            return
        try:
            os.killpg(self._proc.pid, signal.SIGTERM)
            self._proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(self._proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def point_deeptsf_at(s3: S3Server, mlflow_server: MlflowServer) -> None:
    """Set the variables DeepTSF reads its service endpoints from."""
    os.environ.update({
        "MLFLOW_TRACKING_URI": mlflow_server.uri,
        "MLFLOW_S3_ENDPOINT_URL": s3.endpoint,
        "MINIO_CLIENT_URL": s3.endpoint.replace("http://", ""),
    })
