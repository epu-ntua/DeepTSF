"""One-time per-user enrollment that obtains a Keycloak OFFLINE token.

Why this exists: oauth2-proxy keeps its refresh token to itself (there is no
--pass-refresh-token), and Keycloak's token-exchange feature is disabled on this
realm. So the only way to get an offline token for a user is to run our own
authorization-code flow. Because the user already has a live Keycloak SSO
session by the time they reach the dagster UI, this costs one silent redirect
and no login prompt.

    GET /eg-auth/start     -> redirect to Keycloak (scope=...offline_access)
    GET /eg-auth/callback  <- Keycloak returns ?code=...; we swap it for the
                              offline token and store it encrypted.

An offline token, unlike a normal refresh token, survives the user logging out
and has no fixed expiry — it only dies after Keycloak's "Offline Session Idle"
window (default 30 days) with no use, or if revoked.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from offline_store import write_offline_token

KC_ISSUER_URL = os.environ.get("KC_ISSUER_URL", "").rstrip("/")
KC_OFFLINE_ID = os.environ.get("KC_OFFLINE_ID", "").strip()
KC_OFFLINE_SECRET = os.environ.get("KC_OFFLINE_SECRET", "").strip()
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
REDIRECT_PATH = "/eg-auth/callback"
STATE_TTL_SECONDS = 300
HTTP_TIMEOUT_S = 15.0

AUTH_ENDPOINT = f"{KC_ISSUER_URL}/protocol/openid-connect/auth"
TOKEN_ENDPOINT = f"{KC_ISSUER_URL}/protocol/openid-connect/token"

# 'email' is required: mlflow-oidc-auth identifies the user from that claim.
SCOPE = "openid email profile offline_access"


def configured() -> bool:
    return bool(KC_ISSUER_URL and KC_OFFLINE_ID and KC_OFFLINE_SECRET and PUBLIC_BASE_URL)


def _redirect_uri() -> str:
    return f"{PUBLIC_BASE_URL}{REDIRECT_PATH}"


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(os.urandom(64)).decode().rstrip("=")
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


def start(request: Request, redis_client, username: str) -> Response:
    """Kick off the authorization-code flow for *username*."""
    if not configured():
        return JSONResponse(
            status_code=500,
            content={"error": "offline enrollment is not configured on the gateway"},
        )

    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(32)
    next_url = request.query_params.get("next") or "/"
    # Only allow relative paths back, so ?next= can't be used as an open redirect.
    if not next_url.startswith("/") or next_url.startswith("//"):
        next_url = "/"

    redis_client.setex(
        f"eg:authstate:{state}",
        STATE_TTL_SECONDS,
        json.dumps({"verifier": verifier, "username": username, "next": next_url}),
    )

    params = {
        "client_id": KC_OFFLINE_ID,
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return RedirectResponse(f"{AUTH_ENDPOINT}?{urlencode(params)}", status_code=302)


def _jwt_payload(token: str) -> dict:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    seg = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(seg))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


async def callback(request: Request, redis_client) -> Response:
    """Handle Keycloak's redirect back: swap the code for an offline token."""
    error = request.query_params.get("error")
    if error:
        return JSONResponse(
            status_code=400,
            content={"error": error,
                     "description": request.query_params.get("error_description", "")},
        )

    code = request.query_params.get("code")
    state = request.query_params.get("state")
    if not code or not state:
        return JSONResponse(status_code=400, content={"error": "missing code or state"})

    raw_state = redis_client.get(f"eg:authstate:{state}")
    if not raw_state:
        # Expired or replayed. Send them back to the top so enrollment restarts.
        return RedirectResponse("/", status_code=302)
    redis_client.delete(f"eg:authstate:{state}")
    saved = json.loads(raw_state)

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            resp = await client.post(
                TOKEN_ENDPOINT,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": _redirect_uri(),
                    "client_id": KC_OFFLINE_ID,
                    "client_secret": KC_OFFLINE_SECRET,
                    "code_verifier": saved["verifier"],
                },
            )
    except httpx.HTTPError as e:
        print(f"[enroll] token request failed: {e!r}")
        return JSONResponse(status_code=502, content={"error": "keycloak unreachable"})

    if resp.status_code != 200:
        print(f"[enroll] token exchange failed {resp.status_code}: {resp.text[:300]}")
        return JSONResponse(status_code=502, content={"error": "token exchange failed"})

    payload = resp.json()
    offline_token = payload.get("refresh_token")
    if not offline_token:
        return JSONResponse(status_code=502, content={"error": "no refresh_token returned"})

    # Fail loudly now rather than silently three weeks from now: a plain refresh
    # token would die with the SSO session, defeating the whole point.
    typ = _jwt_payload(offline_token).get("typ", "")
    if typ != "Offline":
        print(f"[enroll] expected an Offline token, got typ={typ!r}. Check that the "
              f"user has the offline_access realm role and that the scope is allowed.")
        return JSONResponse(
            status_code=500,
            content={"error": "keycloak did not issue an offline token",
                     "typ": typ,
                     "hint": "grant the user the offline_access realm role"},
        )

    # Prefer the identity Keycloak just asserted over the one we guessed.
    claims = _jwt_payload(payload.get("access_token", "")) or {}
    username = (claims.get("email") or claims.get("preferred_username")
                or saved["username"]).strip().lower()

    write_offline_token(username, offline_token)
    print(f"[enroll] stored offline token for {username}")

    return RedirectResponse(saved.get("next") or "/", status_code=302)
