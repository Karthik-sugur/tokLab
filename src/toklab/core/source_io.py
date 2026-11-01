"""Byte-exact source file I/O.

Source files are always read as bytes and decoded as strict UTF-8, so CRLF,
lone CR and a leading BOM (kept as U+FEFF) survive unchanged. Never use
``Path.read_text`` / text-mode ``open`` for corpus files: universal-newline
translation silently rewrites ``\\r\\n`` to ``\\n``.
"""

from __future__ import annotations

from pathlib import Path


def read_source_text(path: Path | str) -> str:
    """Return ``path``'s bytes decoded as strict UTF-8 (no newline translation)."""
    return Path(path).read_bytes().decode("utf-8")


def write_source_text(path: Path | str, text: str) -> None:
    """Write ``text`` as UTF-8 bytes (no newline translation); create parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
