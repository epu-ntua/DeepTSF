# dagster_deeptsf/auth_runtime.py
#
# MLflow authentication for dagster / celery runs.
#
# The old design captured the launching user's Keycloak ACCESS token at launch
# and reused it for the whole run. Access tokens live ~5 minutes and die on
# logout, so any run outliving one failed with "Authentication required".
#
# Now:
#   1. The gateway enrolls each user once and stores their Keycloak OFFLINE
#      token, encrypted, in the shared eg_offline_tokens volume.
#   2. At launch the gateway records only `dagster:run:<run_id>:user` in redis.
#   3. Here, we resolve the run's user, read their offline token, and exchange
#      it for a fresh access token whenever the current one is near expiry.
#
# An offline token is not tied to the browser session, so the user can log out
# and the run keeps authenticating. Run length stops mattering entirely.
#
# Credentials are resolved PER RUN, never from a single process-wide global:
# several users' runs can share a worker process, and signing one user's MLflow
# calls with another's identity would silently cross-contaminate experiments.
# When the right user cannot be determined unambiguously we send nothing and
# let MLflow fail loudly rather than guess.
import os
import time
from contextvars import ContextVar
from threading import RLock

import redis
import requests

import mlflow.utils.rest_utils as rest_utils
import mlflow.utils.request_utils as request_utils

from dagster_deeptsf.offline_store import (
    delete_offline_token,
    read_offline_token,
    write_offline_token,
)

USE_AUTH = str(os.getenv("USE_AUTH", "")).strip().lower() not in {"", "false", "0", "none"}
REDIS_URL = os.getenv("CELERY_BROKER_URL")

# TOKEN_ISSUER_URL already points at .../protocol/openid-connect/token.
KEYCLOAK_TOKEN_URL = os.getenv("KEYCLOAK_TOKEN_URL") or os.getenv("TOKEN_ISSUER_URL")
KC_OFFLINE_ID = (os.getenv("KC_OFFLINE_ID") or "").strip()
KC_OFFLINE_SECRET = (os.getenv("KC_OFFLINE_SECRET") or "").strip()
# Renew this many seconds before the access token actually expires.
REFRESH_MARGIN_SECONDS = int(os.getenv("EG_REFRESH_MARGIN_SECONDS", "60"))
HTTP_TIMEOUT_S = 15

_r = redis.Redis.from_url(REDIS_URL, decode_responses=True) if USE_AUTH and REDIS_URL else None
_mlflow_patch_lock = RLock()

# Which user this execution context belongs to.
_username_ctx: ContextVar[str | None] = ContextVar("eg_username", default=None)

# run_id -> username. Threads spawned by darts/lightning start with an empty
# context, so the ContextVar alone is not visible there. Keyed by run so it can
# never collapse into "last writer wins" the way a single global would.
_run_users: dict[str, str] = {}
_run_users_lock = RLock()

# username -> (access_token, expires_at). Shared by all steps of that user in
# this process, so N concurrent steps cause one refresh rather than N.
_access_tokens: dict[str, tuple[str, float]] = {}
_access_lock = RLock()


class OfflineTokenError(RuntimeError):
    """The user's offline token is gone, expired, or revoked."""


def get_username_for_run(run_id: str) -> str | None:
    """Which user launched this dagster run."""
    if not USE_AUTH:
        return None
    if _r is None:
        raise RuntimeError("CELERY_BROKER_URL must be set when USE_AUTH is enabled")

    username = _r.get(f"dagster:run:{run_id}:user")
    if not username:
        raise RuntimeError(
            f"No user recorded in Redis for run_id={run_id}. Runs must be launched "
            "through the dagster gateway, which records who launched them; a run "
            "started directly against the dagster port bypasses it."
        )
    return username


def _refresh_access_token(username: str) -> tuple[str, float]:
    """Exchange the user's offline token for a fresh access token."""
    if not (KEYCLOAK_TOKEN_URL and KC_OFFLINE_ID and KC_OFFLINE_SECRET):
        raise RuntimeError(
            "KEYCLOAK_TOKEN_URL / KC_OFFLINE_ID / KC_OFFLINE_SECRET must be set "
            "for the worker to refresh access tokens"
        )

    offline_token = read_offline_token(username)
    if not offline_token:
        raise OfflineTokenError(
            f"No offline token stored for {username}. They need to open the "
            "dagster UI once to enroll."
        )

    resp = requests.post(
        KEYCLOAK_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": KC_OFFLINE_ID,
            "client_secret": KC_OFFLINE_SECRET,
            "refresh_token": offline_token,
        },
        timeout=HTTP_TIMEOUT_S,
    )

    if resp.status_code != 200:
        body = resp.text[:300]
        if "invalid_grant" in body:
            # Idle-expired (default 30 days unused) or revoked. Drop it so the
            # user's next visit to the UI re-enrolls them cleanly.
            delete_offline_token(username)
            raise OfflineTokenError(
                f"Keycloak rejected {username}'s offline token (expired or revoked). "
                f"They should open the dagster UI to re-enroll. Response: {body}"
            )
        raise RuntimeError(f"Token refresh for {username} failed ({resp.status_code}): {body}")

    payload = resp.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise RuntimeError(f"Token refresh for {username} returned no access_token")

    # Defensive: only happens if "Revoke Refresh Token" gets switched on for the
    # client. Persist the rotated token so the next refresh still works.
    rotated = payload.get("refresh_token")
    if rotated and rotated != offline_token:
        try:
            write_offline_token(username, rotated)
        except OSError as e:
            print(f"[auth_runtime] WARNING: could not persist rotated token for "
                  f"{username}: {e!r}")

    expires_at = time.time() + float(payload.get("expires_in", 300))
    return access_token, expires_at


def get_access_token(username: str, force_refresh: bool = False) -> str:
    """A currently-valid access token for *username*, refreshing as needed."""
    with _access_lock:
        cached = _access_tokens.get(username)
        if (not force_refresh and cached
                and time.time() < cached[1] - REFRESH_MARGIN_SECONDS):
            return cached[0]

        token, expires_at = _refresh_access_token(username)
        _access_tokens[username] = (token, expires_at)
        return token


def _resolve_username() -> str | None:
    username = _username_ctx.get()
    if username:
        return username
    # Off-context thread: only safe to infer when this process serves a single
    # run. Two concurrent runs means we cannot tell whose thread this is.
    with _run_users_lock:
        distinct = set(_run_users.values())
        if len(distinct) == 1:
            return next(iter(distinct))
    return None


def _install_mlflow_patch() -> None:
    with _mlflow_patch_lock:
        if getattr(request_utils._get_http_response_with_retries, "_eg_patched", False):
            return

        orig = request_utils._get_http_response_with_retries

        def wrapped(*args, **kwargs):
            username = _resolve_username()
            if not username:
                return orig(*args, **kwargs)

            headers = dict(kwargs.get("headers") or {})
            try:
                headers["Authorization"] = f"Bearer {get_access_token(username)}"
            except Exception as e:
                print(f"[auth_runtime] could not obtain access token for {username}: {e!r}")
                return orig(*args, **kwargs)
            kwargs["headers"] = headers

            resp = orig(*args, **kwargs)

            # The token was accepted a moment ago but rejected now (clock skew,
            # or keycloak rotated signing keys): mint a fresh one and retry once.
            if getattr(resp, "status_code", None) in (401, 403):
                try:
                    headers["Authorization"] = f"Bearer {get_access_token(username, force_refresh=True)}"
                except Exception as e:
                    print(f"[auth_runtime] forced refresh failed for {username}: {e!r}")
                    return resp
                kwargs["headers"] = headers
                resp = orig(*args, **kwargs)
            return resp

        wrapped._eg_patched = True
        request_utils._get_http_response_with_retries = wrapped
        # Keep both modules aligned in case MLflow resolves calls via rest_utils.
        rest_utils._get_http_response_with_retries = wrapped


def install_mlflow_auth_for_run(run_id: str) -> None:
    """Authenticate every MLflow call made while executing *run_id* as the user
    who launched it. Call once at the start of each asset."""
    if not USE_AUTH:
        return

    username = get_username_for_run(run_id)
    if not username:
        return

    _username_ctx.set(username)
    with _run_users_lock:
        _run_users[run_id] = username

    # Fail here, at the start of the step, rather than deep inside training.
    get_access_token(username)
    _install_mlflow_patch()


def get_current_mlflow_token() -> str | None:
    username = _resolve_username()
    return get_access_token(username) if username else None
