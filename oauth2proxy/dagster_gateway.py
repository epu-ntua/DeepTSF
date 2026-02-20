# dagster_gateway_rewritten.py
"""
Rewritten gateway with the SAME behavior as your original, plus one crucial fix:

- If oauth2-proxy does NOT forward bearer tokens to the gateway (which is what your logs show),
  we can still obtain the user access token by calling oauth2-proxy's /oauth2/auth endpoint
  using the browser session cookie that IS forwarded.

This is the "auth_request" pattern, but implemented inside the gateway.

You must set:
  OAUTH2_PROXY_INTERNAL_URL = "http://oauth2_proxy_dagster:4180"
  (or whatever service name/port oauth2-proxy listens on inside docker)

And your oauth2-proxy must be configured with:
  --set-xauthrequest=true
  --pass-access-token=true
so that /oauth2/auth returns X-Auth-Request-Access-Token.

Everything else remains as in your original: HTTP proxy + WebSocket proxy.
"""

import os
import json
import re
import asyncio
import httpx
import redis
import websockets
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from typing import Any

DAGSTER_UPSTREAM = os.environ["DAGSTER_UPSTREAM"].rstrip("/")  # e.g. http://dagster-webserver:8006
REDIS_URL = os.environ["REDIS_URL"]
RUN_TOKEN_TTL = int(os.environ.get("RUN_TOKEN_TTL_SECONDS", "21600"))  # 6h

# NEW: where the gateway can reach oauth2-proxy internally
OAUTH2_PROXY_INTERNAL_URL = os.environ.get("OAUTH2_PROXY_INTERNAL_URL", "").rstrip("/")
# optional: endpoint path, usually fixed
OAUTH2_AUTH_PATH = os.environ.get("OAUTH2_AUTH_PATH", "/oauth2/auth")

# Timeouts
HTTP_TIMEOUT = float(os.environ.get("GATEWAY_HTTP_TIMEOUT", "60"))
AUTH_TIMEOUT = float(os.environ.get("GATEWAY_AUTH_TIMEOUT", "5"))

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
app = FastAPI()


# -------------------------
# Token extraction helpers
# -------------------------
def _token_from_forwarded_headers(headers: dict) -> str | None:
    """
    Direct token forwarding (rare with oauth2-proxy in reverse-proxy mode).
    """
    # 1) oauth2-proxy xauthrequest header
    tok = headers.get("x-auth-request-access-token") or headers.get("X-Auth-Request-Access-Token")
    if tok:
        return tok.strip()

    # 2) custom/other
    tok = headers.get("x-forwarded-access-token") or headers.get("X-Forwarded-Access-Token")
    if tok:
        return tok.strip()

    # 3) Authorization bearer
    auth = headers.get("authorization") or headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    return None


async def _token_via_oauth2_auth(request: Request) -> str | None:
    """
    If the browser is authenticated via oauth2-proxy cookie (usual case),
    we can call oauth2-proxy /oauth2/auth with the Cookie header to retrieve
    X-Auth-Request-Access-Token.
    """
    if not OAUTH2_PROXY_INTERNAL_URL:
        return None

    cookie = request.headers.get("cookie")
    if not cookie:
        return None

    url = f"{OAUTH2_PROXY_INTERNAL_URL}{OAUTH2_AUTH_PATH}"
    headers = {
        "cookie": cookie,
        # these help oauth2-proxy build redirects correctly; not mandatory but harmless
        "x-forwarded-proto": request.headers.get("x-forwarded-proto", "https"),
        "x-forwarded-host": request.headers.get("host", ""),
        "x-original-uri": str(request.url.path),
    }

    try:
        async with httpx.AsyncClient(timeout=AUTH_TIMEOUT, follow_redirects=False) as client:
            resp = await client.get(url, headers=headers)

        # oauth2-proxy /oauth2/auth returns:
        # - 202 if authorized
        # - 401/403 if not
        if resp.status_code not in (200, 202):
            return None

        tok = resp.headers.get("X-Auth-Request-Access-Token")
        return tok.strip() if tok else None
    except Exception:
        return None


async def _get_user_token(request: Request) -> str | None:
    """
    Unified: try direct headers first, then fall back to oauth2-proxy /oauth2/auth using cookie.
    """
    direct = _token_from_forwarded_headers(dict(request.headers))
    if direct:
        return direct

    return await _token_via_oauth2_auth(request)


# -------------------------
# Dagster op detection
# -------------------------
LAUNCH_OPS = {
    # Dagster versions vary; include the common ones
    "LaunchPipelineExecution",
    "LaunchPipelineRun",
    "LaunchRun",
    "LaunchRunExecution",
    "LaunchRunReexecution",
    "LaunchPipelineReexecution",
}

def _is_launch(req_json: dict) -> bool:
    op = (req_json.get("operationName") or "").strip()
    if op in LAUNCH_OPS:
        return True
    q = (req_json.get("query") or "").lower()
    return ("mutation" in q and "launch" in q)

def _extract_run_ids(resp_json: dict) -> list[str]:
    out: list[str] = []

    def walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, dict):
            # Dagster often returns the run id as `run: { id: ... }`
            if "id" in x and isinstance(x["id"], str) and x.get("__typename") == "Run":
                out.append(x["id"])

            # Also keep these if present (optional)
            for k in ("runId", "rootRunId", "parentRunId"):
                v = x.get(k)
                if isinstance(v, str) and v:
                    out.append(v)

            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(resp_json.get("data"))

    # de-dup, preserve order
    seen = set()
    deduped = []
    for rid in out:
        if rid not in seen:
            seen.add(rid)
            deduped.append(rid)
    return deduped


def _to_ws_url(http_url: str) -> str:
    if http_url.startswith("https://"):
        return "wss://" + http_url[len("https://"):]
    if http_url.startswith("http://"):
        return "ws://" + http_url[len("http://"):]
    return "ws://" + http_url


# -------------------------
# WebSocket proxy
# -------------------------
@app.websocket("/graphql")
async def ws_graphql_proxy(client_ws: WebSocket):
    await client_ws.accept()

    qs = client_ws.scope.get("query_string", b"").decode("utf-8")
    upstream_base = _to_ws_url(DAGSTER_UPSTREAM)
    upstream_url = f"{upstream_base}/graphql" + (f"?{qs}" if qs else "")

    incoming_headers = {k.decode(): v.decode() for k, v in client_ws.scope.get("headers", [])}

    # Strip hop-by-hop
    for h in [
        "host", "connection", "upgrade",
        "sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions"
    ]:
        incoming_headers.pop(h, None)

    # IMPORTANT: In your chain oauth2-proxy is in front; the gateway sees Cookie.
    # We just forward Cookie upstream so Dagster webserver sees the same session behavior (it doesn’t validate it).
    # Token-capture for launches happens in the HTTP POST path; WS is mainly for subscriptions.
    try:
        async with websockets.connect(
            upstream_url,
            extra_headers=incoming_headers,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=None,
        ) as upstream_ws:

            async def client_to_upstream():
                while True:
                    msg = await client_ws.receive_text()
                    await upstream_ws.send(msg)

            async def upstream_to_client():
                while True:
                    msg = await upstream_ws.recv()
                    await client_ws.send_text(msg)

            await asyncio.gather(client_to_upstream(), upstream_to_client())

    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await client_ws.close(code=1011)
        except Exception:
            pass


# -------------------------
# HTTP proxy
# -------------------------
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_all(path: str, request: Request):
    # If something tries websocket upgrade outside the websocket route, fail cleanly
    if request.headers.get("upgrade", "").lower() == "websocket":
        return Response(status_code=426, content=b"WebSocket upgrade not supported on this route")

    url = f"{DAGSTER_UPSTREAM}/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    async with httpx.AsyncClient(follow_redirects=False, timeout=HTTP_TIMEOUT) as client:
        upstream = await client.request(
            request.method,
            url,
            content=body if body else None,
            params=dict(request.query_params),
            headers=headers,
        )

    # Capture token for launch mutations and store per runId
    if path == "graphql" and request.method == "POST":
        req_json = {}
        try:
            req_json = json.loads(body.decode("utf-8")) if body else {}
        except Exception as e:
            print("[gateway] could not parse request json:", repr(e))

        op = req_json.get("operationName")
        tok = await _get_user_token(request)
        print("[gateway] token_len:", len(tok) if tok else None)
        print(f"[gateway] graphql op={op} has_token={bool(tok)}")
        print("[gateway] authorization?", "authorization" in [k.lower() for k in request.headers.keys()])
        print("[gateway] cookie?", "cookie" in [k.lower() for k in request.headers.keys()])
        print("[gateway] oauth2_auth_fallback_enabled?", True)

        try:
            if _is_launch(req_json):
                user_token = tok
                print("[gateway] launch detected, upstream status:", upstream.status_code,
                      "content-type:", upstream.headers.get("content-type"))

                # IMPORTANT: don't assume JSON if upstream returned HTML error etc
                content_type = upstream.headers.get("content-type", "")
                raw = upstream.content[:2000]
                if "application/json" not in content_type.lower():
                    print("[gateway] upstream non-json body (first 2KB):", raw.decode("utf-8", "ignore"))
                    # Don't crash; just skip token storage
                    return Response(
                        content=upstream.content,
                        status_code=upstream.status_code,
                        headers={k: v for k, v in upstream.headers.items()
                                 if k.lower() not in ["content-encoding", "transfer-encoding", "connection"]},
                        media_type=content_type,
                    )

                resp_json = upstream.json()

                print("[gateway] launch response keys:", list(resp_json.keys()))
                print("[gateway] launch data:", json.dumps(resp_json.get("data"), indent=2)[:2000])
                print("[gateway] launch errors:", json.dumps(resp_json.get("errors"), indent=2)[:2000])

                run_ids = _extract_run_ids(resp_json)
                print("[gateway] extracted run_ids:", run_ids)

                if user_token and run_ids:
                    for run_id in run_ids:
                        r.setex(f"dagster:run:{run_id}:user_token", RUN_TOKEN_TTL, user_token)
                        print(f"[gateway] stored token for run_id={run_id}")
                else:
                    print("[gateway] NOT storing token: user_token?", bool(user_token), "run_ids?", run_ids)

        except Exception as e:
            print("[gateway] ERROR during launch handling:", repr(e))
    resp_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in ["content-encoding", "transfer-encoding", "connection", "content-length"]
    }

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )
