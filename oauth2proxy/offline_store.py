"""Shared on-disk store for per-user Keycloak offline tokens.

Lives in the eg_offline_tokens docker volume, mounted at /srv/eg-auth in both
the dagster gateway (which writes it during enrollment) and the dagster worker
(which reads it during every run):

    /srv/eg-auth/<sanitized-username>/offline_token.json

The token value is encrypted with Fernet using EG_TOKEN_ENC_KEY, which both
containers hold. That protects a volume backup or a stray filesystem copy — it
does not protect against someone who can already exec into either container,
since the key is in their environment.

NOTE: this file is duplicated in the worker image as
dagster_deeptsf/offline_store.py. The two copies must stay byte-compatible;
they are the read and write ends of the same format.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from cryptography.fernet import Fernet, InvalidToken

TOKEN_DIR = os.environ.get("EG_OFFLINE_TOKEN_DIR", "/srv/eg-auth")
TOKEN_FILENAME = "offline_token.json"
_ENC_KEY = os.environ.get("EG_TOKEN_ENC_KEY", "").strip()


def _fernet() -> Fernet:
    if not _ENC_KEY:
        raise RuntimeError("EG_TOKEN_ENC_KEY is not set; cannot read/write offline tokens")
    return Fernet(_ENC_KEY.encode())


def safe_dirname(username: str) -> str:
    """Filesystem-safe per-user directory. Unusual characters are escaped
    rather than dropped, so two different users can never collide."""
    return re.sub(r"[^A-Za-z0-9._@-]", lambda m: f"%{ord(m.group()):02x}", username)


def token_path(username: str) -> Path:
    return Path(TOKEN_DIR) / safe_dirname(username) / TOKEN_FILENAME


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Advisory lock so the gateway writing and a worker reading/rotating the
    same user's token cannot interleave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def read_offline_token(username: str) -> Optional[str]:
    """Return the user's decrypted offline token, or None if not enrolled."""
    path = token_path(username)
    if not path.exists():
        return None
    try:
        with file_lock(path):
            record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    blob = record.get("token")
    if not blob:
        return None
    try:
        return _fernet().decrypt(blob.encode()).decode()
    except (InvalidToken, ValueError):
        # Wrong/rotated EG_TOKEN_ENC_KEY, or a corrupt file. Treat as
        # un-enrolled so the user is simply asked to enroll again.
        return None


def write_offline_token(username: str, offline_token: str) -> None:
    """Encrypt and persist the user's offline token (atomically, 0600)."""
    path = token_path(username)
    record = {
        "username": username,
        "token": _fernet().encrypt(offline_token.encode()).decode(),
        "stored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
    }
    with file_lock(path):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2))
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)


def delete_offline_token(username: str) -> None:
    """Forget a user's token, e.g. after Keycloak rejects it as expired or
    revoked, so their next visit re-enrolls them."""
    with contextlib.suppress(OSError):
        token_path(username).unlink()


def is_enrolled(username: str) -> bool:
    return read_offline_token(username) is not None
