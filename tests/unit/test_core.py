"""Unit tests for core types, hashing, and lossless segment checks."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from toklab.core.hashing import config_hash, sha256_text
from toklab.core.interfaces import assert_lossless_segments
from toklab.core.types import Segment


class _CoveringSegmenter:
    """Dummy Segmenter that returns one gap segment covering the whole text."""

    def segment(self, text: str) -> list[Segment]:
        return [Segment(start=0, end=len(text), kind="gap", role="all")]


def test_config_hash_ignores_key_order() -> None:
    a = {"seed": 7, "name": "toklab", "nested": {"b": 2, "a": 1}}
    b = {"nested": {"a": 1, "b": 2}, "name": "toklab", "seed": 7}
    assert config_hash(a) == config_hash(b)


def test_sha256_text_stable() -> None:
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("world")


def test_segment_is_immutable() -> None:
    seg = Segment(start=0, end=1, kind="ident", role="name")
    with pytest.raises(FrozenInstanceError):
        seg.start = 2  # type: ignore[misc]


def test_dummy_segmenter_is_lossless() -> None:
    text = "def foo():\n    return 1\n"
    segments = _CoveringSegmenter().segment(text)
    assert_lossless_segments(text, segments)


def test_assert_lossless_segments_detects_gap() -> None:
    text = "abcdef"
    segments = [
        Segment(0, 2, "ident", "a"),
        Segment(4, 6, "ident", "b"),
    ]
    with pytest.raises(AssertionError, match="gap"):
        assert_lossless_segments(text, segments)


def test_assert_lossless_segments_detects_overlap() -> None:
    text = "abcdef"
    segments = [
        Segment(0, 3, "ident", "a"),
        Segment(2, 6, "ident", "b"),
    ]
    with pytest.raises(AssertionError, match="overlap"):
        assert_lossless_segments(text, segments)


def test_assert_lossless_segments_detects_uncovered_prefix() -> None:
    text = "abcdef"
    segments = [Segment(1, 6, "gap", "all")]
    with pytest.raises(AssertionError, match="uncovered prefix"):
        assert_lossless_segments(text, segments)
