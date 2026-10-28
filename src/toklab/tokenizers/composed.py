"""Structure-aware composed tokenizer (C4 / later C5)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from toklab.core.types import EncodeResult, Segment
from toklab.parsing.roles import assign_ident_roles
from toklab.parsing.segmenters import TreeSitterSegmenter
from toklab.tokenizers.native_hf import NativeHFTokenizer


def merge_leading_space_gaps(segments: list[Segment], text: str) -> list[Segment]:
    """Merge a single-space ``gap`` into the following segment.

    Matches pretrained BPE units like ``Ġfoo`` (see docs/decisions/002).
    """
    if not segments:
        return segments
    out: list[Segment] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if (
            i + 1 < len(segments)
            and seg.kind == "gap"
            and text[seg.start : seg.end] == " "
        ):
            nxt = segments[i + 1]
            out.append(
                replace(
                    nxt,
                    start=seg.start,
                    end=nxt.end,
                )
            )
            i += 2
            continue
        out.append(seg)
        i += 1
    return out


class ComposedTokenizer:
    """Encode via segmenter → per-segment base BPE; concatenate ids/spans.

    ``attach_leading_space=True`` (default per spike 2 / decision 002) merges
    a single leading space gap into the next segment before BPE.

    ``indent_tokens`` is an ablation flag (default OFF); when True it is
    accepted but does not change encoding yet (reserved for later ablations).
    """

    def __init__(
        self,
        segmenter: Any,
        base_hf_tokenizer: NativeHFTokenizer,
        adapted_vocab: dict[str, int] | None = None,
        *,
        attach_leading_space: bool = True,
        indent_tokens: bool = False,
        name: str = "C4",
    ) -> None:
        self.segmenter = segmenter
        self.base = base_hf_tokenizer
        self.adapted_vocab = adapted_vocab
        self.attach_leading_space = attach_leading_space
        self.indent_tokens = indent_tokens
        self.name = name

    def _segments(self, text: str) -> list[Segment]:
        segs = self.segmenter.segment(text)
        if isinstance(self.segmenter, TreeSitterSegmenter):
            segs = assign_ident_roles(text, segs)
        if self.attach_leading_space:
            segs = merge_leading_space_gaps(segs, text)
        # indent_tokens: reserved ablation; no-op while False or True for now.
        _ = self.indent_tokens
        return segs

    def _encode_piece(self, piece: str) -> EncodeResult:
        """Encode one segment text; optional adapted-vocab longest match later."""
        if self.adapted_vocab and piece in self.adapted_vocab:
            # Exact whole-segment match only (boundary-aware; not HF add_tokens).
            tid = self.adapted_vocab[piece]
            return EncodeResult(ids=[tid], spans=[(0, len(piece))])
        return self.base.encode(piece)

    def encode(self, text: str) -> EncodeResult:
        if text == "":
            return EncodeResult(ids=[], spans=[])
        ids: list[int] = []
        spans: list[tuple[int, int]] = []
        for seg in self._segments(text):
            piece = text[seg.start : seg.end]
            if piece == "":
                continue
            sub = self._encode_piece(piece)
            for tid, (s, e) in zip(sub.ids, sub.spans, strict=True):
                ids.append(tid)
                spans.append((seg.start + s, seg.start + e))
        return EncodeResult(ids=ids, spans=spans)

    def decode(self, ids: list[int]) -> str:
        return self.base.decode(ids)

    @property
    def vocab_size(self) -> int:
        if self.adapted_vocab:
            return max(self.base.vocab_size, max(self.adapted_vocab.values()) + 1)
        return self.base.vocab_size
