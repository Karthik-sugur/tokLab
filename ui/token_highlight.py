"""Safe HTML for Preview token spans (no Streamlit markdown)."""

from __future__ import annotations

import html
from typing import Any


def visible_token_inner_html(piece: str) -> str:
    """Escape token text; mark leading spaces and newlines for readability."""
    if piece == "":
        return "∅"
    parts: list[str] = []
    at_line_start = True
    faint = 'style="opacity:0.4"'
    for ch in piece:
        if ch == "\n":
            parts.append(f'<span aria-hidden="true" {faint}>↵</span>\n')
            at_line_start = True
        elif ch == " " and at_line_start:
            parts.append(f'<span aria-hidden="true" {faint}>·</span>')
        else:
            at_line_start = False
            parts.append(html.escape(ch))
    return "".join(parts)


def render_token_highlight_html(
    text: str,
    spans: list[dict[str, Any]],
    ids: list[Any],
    color: str,
) -> str:
    """Build monospace pre-wrap HTML for one condition's token spans."""
    parts: list[str] = []
    for i, sp in enumerate(spans):
        piece = text[int(sp["start"]) : int(sp["end"])]
        style = (
            f"background:{color}22;border-bottom:2px solid {color};"
            "padding:1px 2px;margin:1px"
        )
        tid = ids[i] if i < len(ids) else "?"
        inner = visible_token_inner_html(piece)
        parts.append(
            f'<span class="tok" style="{style}" title="id={html.escape(str(tid))}">'
            f"{inner}</span>"
        )
    wrapper = (
        "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
        "white-space:pre-wrap;line-height:1.6;word-break:break-word"
    )
    return f'<div style="{wrapper}">{"".join(parts)}</div>'
