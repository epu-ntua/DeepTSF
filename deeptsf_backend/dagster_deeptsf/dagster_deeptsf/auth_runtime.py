# dagster_deeptsf/auth_runtime.py
import os
import requests
import redis
from contextvars import ContextVar
import mlflow.utils.rest_utils as rest_utils
import mlflow.utils.request_utils as request_utils
from threading import RLock

REDIS_URL = os.environ["CELERY_BROKER_URL"]
# TOKEN_BROKER_URL = os.environ["TOKEN_BROKER_URL"]  # http://token-broker:8080/exchange/mlflow
TOKEN_BROKER_URL = "http://dagster_gateway:8090/exchange/mlflow"

_r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
_mlflow_patch_lock = RLock()
_mlflow_token_ctx: ContextVar[str | None] = ContextVar("mlflow_bearer_token", default=None)

def get_user_token_for_run(run_id: str) -> str:
    tok = _r.get(f"dagster:run:{run_id}:user_token")
    if not tok:
        raise RuntimeError(f"No user token found in Redis for run_id={run_id}")
    return tok

def exchange_for_mlflow(user_token: str) -> str:
    return user_token
    # r = requests.post(
    #     TOKEN_BROKER_URL,
    #     headers={"Authorization": f"Bearer {user_token}"},
    #     timeout=10,
    # )
    # r.raise_for_status()
    # return r.json()["access_token"]

# dagster_deeptsf/auth_runtime.py

def get_current_mlflow_token() -> str | None:
    return _mlflow_token_ctx.get()

def patch_mlflow_bearer(token: str) -> None:
    _mlflow_token_ctx.set(token)

    with _mlflow_patch_lock:
        if getattr(request_utils._get_http_response_with_retries, "_eg_patched", False):
            return

        orig = request_utils._get_http_response_with_retries

        def wrapped(*args, **kwargs):
            # Resolve token from execution-local context to support concurrent users.
            hdrs = dict(kwargs.get("headers") or {})
            token_in_context = _mlflow_token_ctx.get()
            if token_in_context:
                hdrs["Authorization"] = f"Bearer {token_in_context}"
            kwargs["headers"] = hdrs
            return orig(*args, **kwargs)

        wrapped._eg_patched = True
        request_utils._get_http_response_with_retries = wrapped
        # Keep both modules aligned in case MLflow resolves calls via rest_utils.
        rest_utils._get_http_response_with_retries = wrapped
