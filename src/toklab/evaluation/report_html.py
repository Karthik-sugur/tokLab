"""Self-contained HTML rendering of report.md (no server, no extra deps).

Handles the Markdown subset reports/templates/report.md.j2 emits: ATX
headings, pipe tables, ``-`` bullet lists, paragraphs, **bold**, `code` and
backslash escapes. The text is converted as-is; nothing is computed.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#222}
table{border-collapse:collapse;margin:1rem 0;font-size:.9rem}
th,td{border:1px solid #ccc;padding:.25rem .5rem;text-align:left}
th{background:#f3f3f3}
code{background:#f3f3f3;padding:0 .2rem;border-radius:3px}
h1{border-bottom:2px solid #333}h2{border-bottom:1px solid #ccc}
"""

_ESCAPED = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|])")
_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _inline(text: str) -> str:
    """Escape HTML, then apply code / bold / backslash escapes."""
    slots: list[str] = []

    def stash(fragment: str) -> str:
        slots.append(fragment)
        return f"\x00{len(slots) - 1}\x00"

    text = _CODE.sub(lambda m: stash(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = _ESCAPED.sub(lambda m: stash(html.escape(m.group(1))), text)
    text = html.escape(text, quote=False)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    return re.sub("\x00(\\d+)\x00", lambda m: slots[int(m.group(1))], text)


def _split_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    cells, cur, i = [], [], 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            cur.append(body[i : i + 2])
            i += 2
            continue
        if body[i] == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(body[i])
        i += 1
    cells.append("".join(cur).strip())
    return cells


def markdown_to_html(md: str, title: str = "TokLab report") -> str:
    out: list[str] = []
    para: list[str] = []
    lines = md.splitlines()

    def flush_para() -> None:
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_para()
            i += 1
            continue
        heading = _HEADING.match(stripped)
        if heading:
            flush_para()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            i += 1
            continue
        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and _TABLE_RULE.match(lines[i + 1].strip())
        ):
            flush_para()
            head = "".join(f"<th>{_inline(c)}</th>" for c in _split_row(stripped))
            body: list[str] = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = _split_row(lines[i])
                body.append(
                    "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>"
                )
                i += 1
            out.append(
                f"<table><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table>"
            )
            continue
        if stripped.startswith("- "):
            flush_para()
            items: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                item = [lines[i].strip()[2:]]
                i += 1
                while (
                    i < len(lines)
                    and lines[i].startswith("  ")
                    and lines[i].strip()
                    and not lines[i].strip().startswith("- ")
                ):
                    item.append(lines[i].strip())
                    i += 1
                items.append(f"<li>{_inline(' '.join(item))}</li>")
            out.append(f"<ul>{''.join(items)}</ul>")
            continue
        para.append(stripped.rstrip("\\").rstrip())
        i += 1
    flush_para()
    return (
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body>\n{chr(10).join(out)}\n</body></html>\n"
    )


def write_report_html(report_md: Path, out: Path) -> Path:
    md = report_md.read_text(encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown_to_html(md), encoding="utf-8")
    return out
