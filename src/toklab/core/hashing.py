"""Deterministic hashing helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    """Return a SHA-256 hex digest of ``config`` as canonical sorted JSON."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)
