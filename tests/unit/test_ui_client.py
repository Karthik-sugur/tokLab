"""UI client fails gracefully when the API is unreachable or returns errors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

# ui/ is not a package under src/; put it on path for tests.
_UI = Path(__file__).resolve().parents[2] / "ui"
if str(_UI) not in sys.path:
    sys.path.insert(0, str(_UI))

from api_client import (  # noqa: E402
    ApiUnavailable,
    TokLabClient,
    api_unavailable_message,
    readable_http_error,
)


def test_unavailable_message_text() -> None:
    msg = api_unavailable_message("http://127.0.0.1:9")
    assert msg == (
        "Cannot reach the API at http://127.0.0.1:9. "
        "Start it with: toklab serve-api"
    )


def test_api_unavailable_stores_base_url() -> None:
    exc = ApiUnavailable("http://127.0.0.1:9/")
    assert exc.base_url == "http://127.0.0.1:9"
    assert "toklab serve-api" in str(exc)


def test_client_closed_port_raises_api_unavailable() -> None:
    """Closed port → ApiUnavailable (never a raw ConnectError to the UI)."""
    cli = TokLabClient("http://127.0.0.1:9")
    with pytest.raises(ApiUnavailable) as ei:
        cli.datasets()
    assert ei.value.base_url == "http://127.0.0.1:9"
    assert "Cannot reach the API at http://127.0.0.1:9" in str(ei.value)
    assert "toklab serve-api" in str(ei.value)
    assert ei.type is ApiUnavailable


def test_client_preview_closed_port_raises_api_unavailable() -> None:
    cli = TokLabClient("http://127.0.0.1:9")
    with pytest.raises(ApiUnavailable) as ei:
        cli.preview("x = 1\n", ["C6"])
    assert "Cannot reach the API at http://127.0.0.1:9" in str(ei.value)


def _status_exc(status: int, body: dict | str) -> httpx.HTTPStatusError:
    if isinstance(body, dict):
        content = json.dumps(body).encode("utf-8")
        headers = {"content-type": "application/json"}
    else:
        content = str(body).encode("utf-8")
        headers = {"content-type": "text/plain"}
    req = httpx.Request("GET", "http://127.0.0.1:8000/x")
    resp = httpx.Response(status, content=content, headers=headers, request=req)
    return httpx.HTTPStatusError("err", request=req, response=resp)


def test_readable_http_error_404() -> None:
    body = {"detail": {"detail": "experiment missing", "code": "missing"}}
    msg = readable_http_error(_status_exc(404, body))
    assert msg.startswith("API error 404:")
    assert "experiment missing" in msg


def test_readable_http_error_409() -> None:
    msg = readable_http_error(
        _status_exc(409, {"detail": "results not ready"})
    )
    assert msg == "API error 409: results not ready"


def test_readable_http_error_422() -> None:
    msg = readable_http_error(
        _status_exc(422, {"detail": "conditions must be non-empty"})
    )
    assert msg == "API error 422: conditions must be non-empty"
