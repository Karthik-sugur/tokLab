"""Byte-level tokenizer (condition C6): one token per UTF-8 byte."""

from __future__ import annotations

from toklab.core.types import EncodeResult


class ByteLevelTokenizer:
    """Ids are raw UTF-8 bytes in ``0..255``; vocab size is 256."""

    name = "C6"

    def encode(self, text: str) -> EncodeResult:
        raw = text.encode("utf-8")
        ids = list(raw)
        spans: list[tuple[int, int]] = []
        for char_i, ch in enumerate(text):
            n = len(ch.encode("utf-8"))
            for _ in range(n):
                spans.append((char_i, char_i + 1))
        if len(ids) != len(spans):
            raise RuntimeError("byte/span length mismatch")
        return EncodeResult(ids=ids, spans=spans)

    def decode(self, ids: list[int]) -> str:
        if any(i < 0 or i > 255 for i in ids):
            raise ValueError("byte-level ids must be in 0..255")
        return bytes(ids).decode("utf-8")

    @property
    def vocab_size(self) -> int:
        return 256
