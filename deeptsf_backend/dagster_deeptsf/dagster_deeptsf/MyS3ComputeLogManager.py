import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional, Sequence
import boto3
from botocore.errorfactory import ClientError
from dagster_aws.s3.compute_log_manager import S3ComputeLogManager
from dagster._core.storage.local_compute_log_manager import LocalComputeLogSubscriptionManager
from dagster import (
    Field,
    Permissive,
    StringSource,
)
from dagster._config.config_type import Noneable
from dagster._core.storage.compute_log_manager import CapturedLogContext, ComputeIOType
from dagster._serdes import ConfigurableClassData
from typing_extensions import Self

import os
import shutil
import sys
from collections import defaultdict
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Final, Optional

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers.polling import PollingObserver

from dagster import (
    Field,
    Float,
    StringSource,
    _check as check,
)
from dagster._config.config_schema import UserConfigSchema
from dagster._core.storage.compute_log_manager import (
    CapturedLogContext,
    CapturedLogData,
    CapturedLogMetadata,
    CapturedLogSubscription,
    ComputeIOType,
    ComputeLogManager,
)
from dagster._serdes import ConfigurableClass, ConfigurableClassData
from dagster._seven import json
from dagster._utils import ensure_dir, ensure_file, touch_file
from dagster._utils.security import non_secure_md5_hash_str

import io
import os
import random
import string
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from contextlib import contextmanager

from dagster._core.execution.scripts import poll_compute_logs, watch_orphans
from dagster._serdes.ipc import interrupt_ipc_subprocess, open_ipc_subprocess
from dagster._seven import IS_WINDOWS
from dagster._utils import ensure_file

WIN_PY36_COMPUTE_LOG_DISABLED_MSG = """\u001b[33mWARNING: Compute log capture is disabled for the current environment. Set the environment variable `PYTHONLEGACYWINDOWSSTDIO` to enable.\n\u001b[0m"""


def create_compute_log_file_key():
    # Ensure that if user code has seeded the random module that it
    # doesn't cause the same file key for each step (but the random
    # seed is still restored afterwards)
    rng = random.Random(int.from_bytes(os.urandom(16), "big"))
    return "".join(rng.choice(string.ascii_lowercase) for x in range(8))


@contextmanager
def redirect_to_file(stream, filepath):
    with open(filepath, "a+", buffering=1, encoding="utf8") as file_stream:
        with redirect_stream(file_stream, stream):  # pyright: ignore[reportArgumentType]
            yield


@contextmanager
def mirror_stream_to_file(stream, filepath):
    ensure_file(filepath)
    with tail_to_stream(filepath, stream) as pids:
        print("SYS STREAM MIRROR", stream)
        with redirect_to_file(stream, filepath):
            yield pids


def should_disable_io_stream_redirect():
    # See https://stackoverflow.com/a/52377087
    # https://www.python.org/dev/peps/pep-0528/
    return os.name == "nt" and not os.environ.get("PYTHONLEGACYWINDOWSSTDIO")


def warn_if_compute_logs_disabled():
    if should_disable_io_stream_redirect():
        warnings.warn(WIN_PY36_COMPUTE_LOG_DISABLED_MSG)


@contextmanager
def redirect_stream(to_stream=os.devnull, from_stream=sys.stdout):
    # swap the file descriptors to capture system-level output in the process
    # From https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
    from_fd = _fileno(from_stream)
    to_fd = _fileno(to_stream)

    print("IN REDIRECT")
    print(from_fd, to_fd, should_disable_io_stream_redirect())
    if not from_fd or not to_fd or should_disable_io_stream_redirect():
        yield
        return

    with os.fdopen(os.dup(from_fd), "wb") as copied:
        from_stream.flush()
        try:
            os.dup2(_fileno(to_stream), from_fd)  # pyright: ignore[reportArgumentType]
        except ValueError:
            with open(to_stream, "wb") as to_file:
                os.dup2(to_file.fileno(), from_fd)
        try:
            yield from_stream
        finally:
            from_stream.flush()
            to_stream.flush()  # pyright: ignore[reportAttributeAccessIssue]
            os.dup2(copied.fileno(), from_fd)


@contextmanager
def tail_to_stream(path, stream):
    if IS_WINDOWS:
        with execute_windows_tail(path, stream) as pids:
            yield pids
    else:
        with execute_posix_tail(path, stream) as pids:
            yield pids


@contextmanager
def execute_windows_tail(path, stream):
    # Cannot use multiprocessing here because we already may be in a daemonized process
    # Instead, invoke a thin script to poll a file and dump output to stdout.  We pass the current
    # pid so that the poll process kills itself if it becomes orphaned
    poll_file = os.path.abspath(poll_compute_logs.__file__)
    stream = stream if _fileno(stream) else None

    with tempfile.TemporaryDirectory() as temp_dir:
        ipc_output_file = os.path.join(temp_dir, f"execute-windows-tail-{uuid.uuid4().hex}")

        try:
            tail_process = open_ipc_subprocess(
                [sys.executable, poll_file, path, str(os.getpid()), ipc_output_file], stdout=stream
            )
            yield (tail_process.pid, None)
        finally:
            if tail_process:
                start_time = time.time()
                while not os.path.isfile(ipc_output_file):
                    if time.time() - start_time > 15:
                        raise Exception("Timed out waiting for tail process to start")
                    time.sleep(1)

                # Now that we know the tail process has started, tell it to terminate once there is
                # nothing more to output
                interrupt_ipc_subprocess(tail_process)
                tail_process.communicate(timeout=30)


@contextmanager
def execute_posix_tail(path, stream):
    # open a subprocess to tail the file and print to stdout
    tail_cmd = ["tail", "-F", "-c", "+0", path]
    stream = stream if _fileno(stream) else None

    try:
        tail_process = None
        watcher_process = None
        tail_process = subprocess.Popen(tail_cmd, stdout=stream)

        # open a watcher process to check for the orphaning of the tail process (e.g. when the
        # current process is suddenly killed)
        watcher_file = os.path.abspath(watch_orphans.__file__)
        watcher_process = subprocess.Popen(
            [
                sys.executable,
                watcher_file,
                str(os.getpid()),
                str(tail_process.pid),
            ]
        )

        yield (tail_process.pid, watcher_process.pid)
    finally:
        # The posix tail process has default interval check 1s, which may lead to missing logs on stdout/stderr.
        # Allow users to add delay before killing tail process.
        # More here: https://github.com/dagster-io/dagster/issues/23336
        time.sleep(float(os.getenv("DAGSTER_COMPUTE_LOG_TAIL_WAIT_AFTER_FINISH", "0")))

        if tail_process:
            _clean_up_subprocess(tail_process)

        if watcher_process:
            _clean_up_subprocess(watcher_process)


def _clean_up_subprocess(subprocess_obj):
    try:
        if subprocess_obj:
            subprocess_obj.terminate()
            subprocess_obj.communicate(timeout=30)
    except OSError:
        pass


def _fileno(stream):
    try:
        print("in fileno", stream)
        fd = getattr(stream, "fileno", lambda: stream)()
        print("fd", fd)
    except io.UnsupportedOperation:
        # Test CLI runners will stub out stdout to a non-file stream, which will raise an
        # UnsupportedOperation if `fileno` is accessed.  We need to make sure we do not error out,
        # or tests will fail
        print("FAIL FAIL FAIL")
        return None

    if isinstance(fd, int):
        return fd

    return None

DEFAULT_WATCHDOG_POLLING_TIMEOUT: Final = 2.5

IO_TYPE_EXTENSION: Final[Mapping[ComputeIOType, str]] = {
    ComputeIOType.STDOUT: "out",
    ComputeIOType.STDERR: "err",
}

MAX_FILENAME_LENGTH: Final = 255



class MyLocalComputeLogManager(ComputeLogManager, ConfigurableClass):
    """Stores copies of stdout & stderr for each compute step locally on disk."""

    def __init__(
        self,
        base_dir: str,
        polling_timeout: Optional[float] = None,
        inst_data: Optional[ConfigurableClassData] = None,
    ):
        self._base_dir = base_dir
        self._polling_timeout = check.opt_float_param(
            polling_timeout, "polling_timeout", DEFAULT_WATCHDOG_POLLING_TIMEOUT
        )
        self._subscription_manager = LocalComputeLogSubscriptionManager(self)
        self._inst_data = check.opt_inst_param(inst_data, "inst_data", ConfigurableClassData)

    @property
    def inst_data(self) -> Optional[ConfigurableClassData]:
        return self._inst_data

    @property
    def polling_timeout(self) -> float:
        return self._polling_timeout

    @classmethod
    def config_type(cls) -> UserConfigSchema:
        return {
            "base_dir": StringSource,
            "polling_timeout": Field(Float, is_required=False),
        }

    @classmethod
    def from_config_value(
        cls, inst_data: Optional[ConfigurableClassData], config_value
    ) -> "MyLocalComputeLogManager":
        return MyLocalComputeLogManager(inst_data=inst_data, **config_value)

    @contextmanager
    def capture_logs(self, log_key: Sequence[str]) -> Generator[CapturedLogContext, None, None]:
        outpath = self.get_captured_local_path(log_key, IO_TYPE_EXTENSION[ComputeIOType.STDOUT])
        errpath = self.get_captured_local_path(log_key, IO_TYPE_EXTENSION[ComputeIOType.STDERR])
        print("capture_logs", outpath, log_key)
        print("SYS STREAM", sys.stdout)
        with mirror_stream_to_file(sys.stdout, outpath), mirror_stream_to_file(sys.stderr, errpath):
            yield CapturedLogContext(log_key)

        # leave artifact on filesystem so that we know the capture is completed
        touch_file(self.complete_artifact_path(log_key))

    @contextmanager
    def open_log_stream(
        self, log_key: Sequence[str], io_type: ComputeIOType
    ) -> Iterator[Optional[IO]]:
        path = self.get_captured_local_path(log_key, IO_TYPE_EXTENSION[io_type])
        ensure_file(path)
        with open(path, "+a", encoding="utf-8") as f:
            yield f

    def is_capture_complete(self, log_key: Sequence[str]) -> bool:
        return os.path.exists(self.complete_artifact_path(log_key))

    def get_log_data(
        self, log_key: Sequence[str], cursor: Optional[str] = None, max_bytes: Optional[int] = None
    ) -> CapturedLogData:
        stdout_cursor, stderr_cursor = self.parse_cursor(cursor)
        stdout, stdout_offset = self.get_log_data_for_type(
            log_key, ComputeIOType.STDOUT, offset=stdout_cursor, max_bytes=max_bytes
        )
        stderr, stderr_offset = self.get_log_data_for_type(
            log_key, ComputeIOType.STDERR, offset=stderr_cursor, max_bytes=max_bytes
        )
        return CapturedLogData(
            log_key=log_key,
            stdout=stdout,
            stderr=stderr,
            cursor=self.build_cursor(stdout_offset, stderr_offset),
        )

    def get_log_metadata(self, log_key: Sequence[str]) -> CapturedLogMetadata:
        return CapturedLogMetadata(
            stdout_location=self.get_captured_local_path(
                log_key, IO_TYPE_EXTENSION[ComputeIOType.STDOUT]
            ),
            stderr_location=self.get_captured_local_path(
                log_key, IO_TYPE_EXTENSION[ComputeIOType.STDERR]
            ),
            stdout_download_url=self.get_captured_log_download_url(log_key, ComputeIOType.STDOUT),
            stderr_download_url=self.get_captured_log_download_url(log_key, ComputeIOType.STDERR),
        )

    def delete_logs(
        self, log_key: Optional[Sequence[str]] = None, prefix: Optional[Sequence[str]] = None
    ):
        if log_key:
            paths = [
                self.get_captured_local_path(log_key, IO_TYPE_EXTENSION[ComputeIOType.STDOUT]),
                self.get_captured_local_path(log_key, IO_TYPE_EXTENSION[ComputeIOType.STDERR]),
                self.get_captured_local_path(
                    log_key, IO_TYPE_EXTENSION[ComputeIOType.STDOUT], partial=True
                ),
                self.get_captured_local_path(
                    log_key, IO_TYPE_EXTENSION[ComputeIOType.STDERR], partial=True
                ),
                self.get_captured_local_path(log_key, "complete"),
            ]
            for path in paths:
                if os.path.exists(path) and os.path.isfile(path):
                    os.remove(path)
        elif prefix:
            dir_to_delete = os.path.join(self._base_dir, *prefix)
            if os.path.exists(dir_to_delete) and os.path.isdir(dir_to_delete):
                # recursively delete all files in dir
                shutil.rmtree(dir_to_delete)
        else:
            check.failed("Must pass in either `log_key` or `prefix` argument to delete_logs")

    def get_log_data_for_type(
        self,
        log_key: Sequence[str],
        io_type: ComputeIOType,
        offset: Optional[int] = 0,
        max_bytes: Optional[int] = None,
    ):
        path = self.get_captured_local_path(log_key, IO_TYPE_EXTENSION[io_type])
        return self.read_path(path, offset or 0, max_bytes)

    def complete_artifact_path(self, log_key):
        return self.get_captured_local_path(log_key, "complete")

    def read_path(
        self,
        path: str,
        offset: int = 0,
        max_bytes: Optional[int] = None,
    ):
        if not os.path.exists(path) or not os.path.isfile(path):
            return None, offset

        with open(path, "rb") as f:
            f.seek(offset, os.SEEK_SET)
            if max_bytes is None:
                data = f.read()
            else:
                data = f.read(max_bytes)
            new_offset = f.tell()
        return data, new_offset

    def get_captured_log_download_url(self, log_key, io_type):
        check.inst_param(io_type, "io_type", ComputeIOType)
        url = "/logs"
        for part in log_key:
            url = f"{url}/{part}"

        return f"{url}/{IO_TYPE_EXTENSION[io_type]}"

    def get_captured_local_path(self, log_key: Sequence[str], extension: str, partial=False):
        [*namespace, filebase] = log_key
        filename = f"{filebase}.{extension}"
        if partial:
            filename = f"{filename}.partial"
        if len(filename) > MAX_FILENAME_LENGTH:
            filename = "{}.{}".format(non_secure_md5_hash_str(filebase.encode("utf-8")), extension)
        base_dir_path = Path(self._base_dir).resolve()
        log_path = base_dir_path.joinpath(*namespace, filename).resolve()
        if base_dir_path not in log_path.parents:
            raise ValueError("Invalid path")
        return str(log_path)

    def subscribe(
        self, log_key: Sequence[str], cursor: Optional[str] = None
    ) -> CapturedLogSubscription:
        subscription = CapturedLogSubscription(self, log_key, cursor)
        self._subscription_manager.add_subscription(subscription)
        return subscription

    def unsubscribe(self, subscription):
        self._subscription_manager.remove_subscription(subscription)

    def get_log_keys_for_log_key_prefix(
        self, log_key_prefix: Sequence[str], io_type: ComputeIOType
    ) -> Sequence[Sequence[str]]:
        """Returns the logs keys for a given log key prefix. This is determined by looking at the
        directory defined by the log key prefix and creating a log_key for each file in the directory.
        """
        base_dir_path = Path(self._base_dir).resolve()
        directory = base_dir_path.joinpath(*log_key_prefix)
        objects = directory.iterdir()
        results = []
        list_key_prefix = list(log_key_prefix)

        for obj in objects:
            if obj.is_file() and obj.suffix == "." + IO_TYPE_EXTENSION[io_type]:
                results.append(list_key_prefix + [obj.stem])

        return results

    def dispose(self) -> None:
        self._subscription_manager.dispose()


class MyS3ComputeLogManager(S3ComputeLogManager):
    """Custom class for S3ComputeLogManager with added support for accessKey and secretKey"""

    def __init__(
        self,
        bucket,
        local_dir=None,
        inst_data: Optional[ConfigurableClassData] = None,
        prefix="dagster",
        use_ssl=True,
        verify=True,
        verify_cert_path=None,
        endpoint_url=None,
        skip_empty_files=False,
        upload_interval=None,
        upload_extra_args=None,
        show_url_only=False,
        region=None,
        access_key=None, 
        secret_key=None,  
    ):
        super().__init__(
            bucket=bucket,
            local_dir=local_dir,
            inst_data=inst_data,
            prefix=prefix,
            use_ssl=use_ssl,
            verify=verify,
            verify_cert_path=verify_cert_path,
            endpoint_url=endpoint_url,
            skip_empty_files=skip_empty_files,
            upload_interval=upload_interval,
            upload_extra_args=upload_extra_args,
            show_url_only=show_url_only,
            region=region,
        )

        verify_param = verify_cert_path if verify_cert_path else verify

        self._local_manager = MyLocalComputeLogManager(local_dir)

        if access_key and secret_key:
            self._s3_session = boto3.resource(
                "s3",
                use_ssl=use_ssl,
                verify=verify_param,
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key, 
            ).meta.client
        else:
            self._s3_session = boto3.resource(
                "s3", use_ssl=use_ssl, verify=verify_param, endpoint_url=endpoint_url
            ).meta.client

    @classmethod
    def config_type(cls):
        return {
            "bucket": StringSource,
            "local_dir": Field(StringSource, is_required=False),
            "prefix": Field(StringSource, is_required=False, default_value="dagster"),
            "use_ssl": Field(bool, is_required=False, default_value=True),
            "verify": Field(bool, is_required=False, default_value=True),
            "verify_cert_path": Field(StringSource, is_required=False),
            "endpoint_url": Field(StringSource, is_required=False),
            "skip_empty_files": Field(bool, is_required=False, default_value=False),
            "upload_interval": Field(Noneable(int), is_required=False, default_value=None),
            "upload_extra_args": Field(
                Permissive(), is_required=False, description="Extra args for S3 file upload"
            ),
            "show_url_only": Field(bool, is_required=False, default_value=False),
            "region": Field(StringSource, is_required=False),
            "access_key": Field(StringSource, is_required=False),  
            "secret_key": Field(StringSource, is_required=False), 
        }
    
    @property
    def local_manager(self) -> MyLocalComputeLogManager:
        return self._local_manager


    @classmethod
    def from_config_value(
        cls, inst_data: ConfigurableClassData, config_value: Mapping[str, Any]
    ) -> Self:
        return cls(inst_data=inst_data, **config_value)