"""Frozen shared data types used across TokLab."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """A contiguous slice of source text with a kind and role.

    kinds: ident, keyword, op, delim, string, comment, number, gap
    """

    start: int
    end: int
    kind: str
    role: str


@dataclass(frozen=True)
class SourceFile:
    """One source file in a dataset split."""

    file_id: str
    repo: str
    path: str
    sha256: str
    language: str
    n_bytes: int
    split: str


@dataclass(frozen=True)
class EncodeResult:
    """Tokenizer output: token ids and character spans in the source.

    spans[i] = (start_char, end_char) covered by token i.
    """

    ids: list[int]
    spans: list[tuple[int, int]]
