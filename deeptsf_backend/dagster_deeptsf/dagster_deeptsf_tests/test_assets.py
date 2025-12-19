import os
import time
import uuid
import runpy
from typing import Dict, List, Tuple, Any

import pytest
from dagster_graphql import DagsterGraphQLClient
from dagster_graphql.client import DagsterGraphQLClientError


# -------------------- Defaults (override via env) --------------------
DEFAULT_LAUNCH_FILE = os.getenv("DEEPTSF_LAUNCH_FILE", "launch_deeptsf.py")

DAGSTER_HOST = os.getenv("DAGSTER_HOST", "deeptsf-dagster.stage.aiodp.ai").strip()
DAGSTER_PORT = int(os.getenv("DAGSTER_PORT", "443"))  # if TLS terminates at ingress, 443 is typical
DAGSTER_JOB_NAME = os.getenv("DAGSTER_JOB_NAME", "deeptsf_dagster_job").strip()

RUN_TIMEOUT_SECONDS = int(os.getenv("DAGSTER_RUN_TIMEOUT_SECONDS", "3600"))
POLL_SECONDS = int(os.getenv("DAGSTER_POLL_SECONDS", "2"))

BEARER_TOKEN = os.getenv("DAGSTER_BEARER_TOKEN", "").strip()
PRESERVE_NAMES = os.getenv("DAGSTER_PRESERVE_NAMES", "0").strip().lower() in ("1", "true", "yes", "on")

HEADERS = {}
if BEARER_TOKEN:
    HEADERS["Authorization"] = f"Bearer {BEARER_TOKEN}"


# -------------------- Helpers --------------------
def _normalize_hostname(host: str) -> str:
    host = host.strip().replace("https://", "").replace("http://", "").rstrip("/")
    return host


def _load_launch_globals(launch_file: str) -> dict:
    if not os.path.exists(launch_file):
        raise FileNotFoundError(
            f"Could not find launch file at '{launch_file}'. "
            f"Set DEEPTSF_LAUNCH_FILE to the correct path."
        )
    return runpy.run_path(launch_file)


def _discover_all_top_level_dicts(globs: dict) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Collect every top-level dict defined in the launch file (excluding dunder/private names).
    This will pick up cfg1..cfgN and also run_config (and any other dict you define).
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name, val in globs.items():
        if not isinstance(val, dict):
            continue
        if name.startswith("__") or name.startswith("_"):
            continue
        # skip pytest internals or other accidental dicts if any
        if name in ("pytest",):
            continue
        out.append((name, val))

    # deterministic order
    out.sort(key=lambda x: x[0])
    return out


def _is_dagster_run_config(d: Dict[str, Any]) -> bool:
    # minimal check: run_config has "resources" at top level
    return isinstance(d, dict) and "resources" in d


def _wrap_as_run_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Your job expects config under resources.config.config
    return {"resources": {"config": {"config": cfg}}}


def _make_client(host: str, port: int) -> DagsterGraphQLClient:
    # Support both arg names across versions
    try:
        return DagsterGraphQLClient(
            hostname=host,
            port_number=port,
            use_ssl=True,
            headers=HEADERS or None,
        )
    except TypeError:
        return DagsterGraphQLClient(
            hostname=host,
            port_number=port,
            use_https=True,
            headers=HEADERS or None,
        )


def _get_run_status(client: DagsterGraphQLClient, run_id: str) -> str:
    try:
        st = client.get_run_status(run_id)
    except DagsterGraphQLClientError as e:
        raise RuntimeError(f"DagsterGraphQLClientError while fetching status for {run_id}: {e}") from e

    # depending on dagster version, this might be an Enum
    return getattr(st, "value", str(st))


TERMINAL_SUCCESS = {"SUCCESS"}
TERMINAL_FAIL = {"FAILURE", "CANCELED", "CANCELING"}


# -------------------- pytest plumbing --------------------
def pytest_generate_tests(metafunc):
    if "cfg_case" not in metafunc.fixturenames:
        return

    globs = _load_launch_globals(DEFAULT_LAUNCH_FILE)
    cases = _discover_all_top_level_dicts(globs)
    if not cases:
        raise RuntimeError(
            f"No top-level dicts found in {DEFAULT_LAUNCH_FILE}. "
            "Define your configs as top-level Python dict variables."
        )

    metafunc.parametrize("cfg_case", cases, ids=[name for name, _ in cases])


@pytest.fixture(scope="session")
def dagster_target():
    globs = _load_launch_globals(DEFAULT_LAUNCH_FILE)

    host = _normalize_hostname(str(globs.get("HOST", DAGSTER_HOST)).strip())
    port = int(str(globs.get("PORT", DAGSTER_PORT)).strip())
    job_name = str(globs.get("JOB_NAME", DAGSTER_JOB_NAME)).strip()

    if not host or not port or not job_name:
        raise RuntimeError(
            "Could not determine host/port/job. "
            "Set DAGSTER_HOST/DAGSTER_PORT/DAGSTER_JOB_NAME or define HOST/PORT/JOB_NAME in launch_deeptsf.py"
        )

    return {"host": host, "port": port, "job_name": job_name}


@pytest.mark.integration
def test_dagster_all_dicts(dagster_target, cfg_case):
    cfg_name, cfg = cfg_case

    host = dagster_target["host"]
    port = dagster_target["port"]
    job_name = dagster_target["job_name"]

    client = _make_client(host, port)

    # Build run_config depending on dict type
    if _is_dagster_run_config(cfg):
        run_config = cfg
    else:
        cfg_to_use = dict(cfg)

        # Make names unique to avoid collisions (optional)
        if not PRESERVE_NAMES:
            suffix = f"{cfg_name}-{uuid.uuid4().hex[:8]}"
            for k in ("experiment_name", "parent_run_name", "trial_name"):
                if k in cfg_to_use and isinstance(cfg_to_use[k], str) and cfg_to_use[k]:
                    cfg_to_use[k] = f"{cfg_to_use[k]}_{suffix}"

        run_config = _wrap_as_run_config(cfg_to_use)

    # 1) launch
    run_id = client.submit_job_execution(job_name, run_config=run_config)
    assert run_id, f"{cfg_name}: expected non-empty run_id"

    # 2) poll
    deadline = time.time() + RUN_TIMEOUT_SECONDS
    last_status = None

    while time.time() < deadline:
        last_status = _get_run_status(client, run_id)

        if last_status in TERMINAL_SUCCESS:
            return
        if last_status in TERMINAL_FAIL:
            pytest.fail(f"{cfg_name}: run {run_id} ended with status={last_status}")

        time.sleep(POLL_SECONDS)

    pytest.fail(
        f"{cfg_name}: timed out after {RUN_TIMEOUT_SECONDS}s waiting for run {run_id}. "
        f"Last status={last_status}"
    )
