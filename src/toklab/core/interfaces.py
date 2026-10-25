"""Protocols and shared helpers for segmenters and tokenizers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from toklab.core.types import EncodeResult, Segment


def assert_lossless_segments(text: str, segments: list[Segment]) -> None:
    """Assert segments are contiguous, cover ``text``, and reconstruct it.

    Raises:
        AssertionError: with a clear message on gap, overlap, or uncovered text.
    """
    if not segments:
        if text:
            raise AssertionError(
                f"segments are empty but text has length {len(text)}"
            )
        return

    if segments[0].start != 0:
        raise AssertionError(
            f"first segment starts at {segments[0].start}, expected 0 "
            f"(uncovered prefix)"
        )

    for i, seg in enumerate(segments):
        if seg.start > seg.end:
            raise AssertionError(
                f"segment {i} has start {seg.start} > end {seg.end}"
            )
        if i > 0:
            prev = segments[i - 1]
            if seg.start < prev.end:
                raise AssertionError(
                    f"overlap: segment {i - 1} ends at {prev.end}, "
                    f"segment {i} starts at {seg.start}"
                )
            if seg.start > prev.end:
                raise AssertionError(
                    f"gap: segment {i - 1} ends at {prev.end}, "
                    f"segment {i} starts at {seg.start} "
                    f"(uncovered [{prev.end}:{seg.start}])"
                )

    if segments[-1].end != len(text):
        raise AssertionError(
            f"last segment ends at {segments[-1].end}, "
            f"expected {len(text)} (uncovered suffix)"
        )

    rebuilt = "".join(text[s.start : s.end] for s in segments)
    if rebuilt != text:
        raise AssertionError(
            "segment slices do not reconstruct text "
            f"(rebuilt length {len(rebuilt)}, text length {len(text)})"
        )


@runtime_checkable
class Segmenter(Protocol):
    """Split source text into contiguous, lossless segments."""

    def segment(self, text: str) -> list[Segment]:
        """Return segments that cover ``text`` without gaps or overlaps."""
        ...


@runtime_checkable
class Tokenizer(Protocol):
    """Encode and decode text with a fixed vocabulary."""

    name: str

    def encode(self, text: str) -> EncodeResult:
        """Encode ``text`` into token ids and character spans."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text (must be lossless with encode)."""
        ...

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        ...
