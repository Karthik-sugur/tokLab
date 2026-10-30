"""Preview token HTML escapes user text and keeps one span per token."""

from __future__ import annotations

import sys
from pathlib import Path

_UI = Path(__file__).resolve().parents[2] / "ui"
if str(_UI) not in sys.path:
    sys.path.insert(0, str(_UI))

from token_highlight import (  # noqa: E402
    render_token_highlight_html,
    visible_token_inner_html,
)


def test_visible_inner_marks_leading_space_and_newline() -> None:
    inner = visible_token_inner_html("  a\nb")
    assert "·" in inner
    assert "↵" in inner
    assert "a" in inner
    assert "\n" in inner


def test_render_escapes_specials_and_span_count() -> None:
    text = "Make `toklab serve-ui` print *this*_ & <that>.\n# hi"
    # Explicit spans covering backticks, *, _, &, <, >, #, and newline.
    pieces = [
        "Make ",
        "`toklab serve-ui`",
        " print ",
        "*this*",
        "_",
        " & ",
        "<that>",
        ".",
        "\n",
        "# hi",
    ]
    assert "".join(pieces) == text
    spans = []
    pos = 0
    for p in pieces:
        spans.append({"start": pos, "end": pos + len(p)})
        pos += len(p)
    ids = list(range(len(spans)))
    out = render_token_highlight_html(text, spans, ids, "#1f77b4")

    assert out.count('class="tok"') == len(spans)

    # No unescaped user HTML; markdown-sensitive chars stay as text.
    assert "<that>" not in out
    assert "&lt;that&gt;" in out
    assert "&amp;" in out
    assert "`toklab serve-ui`" in out
    assert "*this*" in out
    assert "_" in out
    assert "#" in out and "hi" in out
    assert "white-space:pre-wrap" in out
    assert "monospace" in out
    assert "↵" in out
    assert "& " not in out
