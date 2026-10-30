"""UI client fails gracefully when the API is unreachable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ui/ is not a package under src/; put it on path for tests.
_UI = Path(__file__).resolve().parents[2] / "ui"
if str(_UI) not in sys.path:
    sys.path.insert(0, str(_UI))

from api_client import (  # noqa: E402
    ApiUnreachableError,
    TokLabClient,
    api_unreachable_message,
)


def test_unreachable_message_text() -> None:
    msg = api_unreachable_message("http://127.0.0.1:9")
    assert msg == (
        "Cannot reach the API at http://127.0.0.1:9. "
        "Start it with: toklab serve-api"
    )


def test_client_closed_port_friendly_error_no_traceback() -> None:
    """Point at a closed port: friendly ApiUnreachableError, not ConnectError."""
    cli = TokLabClient("http://127.0.0.1:9")
    with pytest.raises(ApiUnreachableError) as ei:
        cli.datasets()
    text = str(ei.value)
    assert "Cannot reach the API at http://127.0.0.1:9" in text
    assert "toklab serve-api" in text
    # Cause is chained for debugging, but the raised type is friendly.
    assert ei.type is ApiUnreachableError
    assert not isinstance(ei.value, ConnectionError)


def test_client_post_closed_port_friendly_error() -> None:
    cli = TokLabClient("http://127.0.0.1:9")
    with pytest.raises(ApiUnreachableError) as ei:
        cli.preview("x = 1\n", ["C6"])
    assert "Cannot reach the API at http://127.0.0.1:9" in str(ei.value)
