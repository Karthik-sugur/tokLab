"""Structure-aware / domain-adapted composed tokenizers (C3–C5)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from toklab.core.types import EncodeResult, Segment
from toklab.parsing.roles import assign_ident_roles
from toklab.parsing.segmenters import TreeSitterSegmenter
from toklab.tokenizers.native_hf import NativeHFTokenizer
from toklab.vocab.adapted_vocab import AdaptedVocab

MatchMode = Literal["none", "native_longest", "structural_ident"]


def merge_leading_space_gaps(segments: list[Segment], text: str) -> list[Segment]:
    """Merge a single-space ``gap`` into the following segment."""
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
            out.append(replace(nxt, start=seg.start, end=nxt.end))
            i += 2
            continue
        out.append(seg)
        i += 1
    return out


def _is_ident_boundary(text: str, start: int, end: int) -> bool:
    """True if text[start:end] is a full identifier at word boundaries."""
    if start < 0 or end > len(text) or start >= end:
        return False
    piece = text[start:end]
    if not piece.isidentifier():
        return False
    if start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
        return False
    if end < len(text) and (text[end].isalnum() or text[end] == "_"):
        return False
    return True


def _sorted_entries(adapted: dict[str, int]) -> list[str]:
    return sorted(adapted.keys(), key=lambda s: (-len(s), s))


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def entries_are_plain_identifiers(adapted: dict[str, int]) -> bool:
    """True if every entry consists only of ``isalnum()`` / ``_`` characters."""
    return all(all(_is_ident_char(c) for c in ent) for ent in adapted)


def encode_native_longest_match(
    text: str,
    base: NativeHFTokenizer,
    adapted: dict[str, int],
    *,
    plain_identifiers: bool | None = None,
) -> EncodeResult:
    """C3 encode: longest-match adapted ids on identifier boundaries; else base BPE.

    With an empty adapted map this is exactly ``base.encode(text)`` (≡ C2).

    When every entry is made of identifier characters only, a match must
    start and end on identifier boundaries, so the only entry that can match
    at a boundary is the whole identifier run there: one dict lookup per run
    gives the same result as scanning all entries.
    """
    if not adapted:
        return base.encode(text)
    if plain_identifiers is None:
        plain_identifiers = entries_are_plain_identifiers(adapted)
    if plain_identifiers:
        return _encode_runs(text, base, adapted)
    entries = _sorted_entries(adapted)
    ids: list[int] = []
    spans: list[tuple[int, int]] = []
    i = 0
    unmatched_start: int | None = None

    def flush_unmatched(up_to: int) -> None:
        nonlocal unmatched_start
        if unmatched_start is None or unmatched_start >= up_to:
            unmatched_start = None
            return
        piece = text[unmatched_start:up_to]
        sub = base.encode(piece)
        for tid, (s, e) in zip(sub.ids, sub.spans, strict=True):
            ids.append(tid)
            spans.append((unmatched_start + s, unmatched_start + e))
        unmatched_start = None

    while i < len(text):
        matched = False
        for ent in entries:
            end = i + len(ent)
            if end <= len(text) and text.startswith(ent, i) and _is_ident_boundary(
                text, i, end
            ):
                flush_unmatched(i)
                ids.append(adapted[ent])
                spans.append((i, end))
                i = end
                matched = True
                break
        if matched:
            continue
        if unmatched_start is None:
            unmatched_start = i
        i += 1
    flush_unmatched(len(text))
    return EncodeResult(ids=ids, spans=spans)


def _encode_runs(
    text: str, base: NativeHFTokenizer, adapted: dict[str, int]
) -> EncodeResult:
    ids: list[int] = []
    spans: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    unmatched_start: int | None = None

    def flush_unmatched(up_to: int) -> None:
        nonlocal unmatched_start
        if unmatched_start is not None and unmatched_start < up_to:
            sub = base.encode(text[unmatched_start:up_to])
            for tid, (s, e) in zip(sub.ids, sub.spans, strict=True):
                ids.append(tid)
                spans.append((unmatched_start + s, unmatched_start + e))
        unmatched_start = None

    while i < n:
        if _is_ident_char(text[i]) and (i == 0 or not _is_ident_char(text[i - 1])):
            j = i + 1
            while j < n and _is_ident_char(text[j]):
                j += 1
            piece = text[i:j]
            tid = adapted.get(piece)
            if tid is not None and piece.isidentifier():
                flush_unmatched(i)
                ids.append(tid)
                spans.append((i, j))
            elif unmatched_start is None:
                unmatched_start = i
            i = j
            continue
        if unmatched_start is None:
            unmatched_start = i
        i += 1
    flush_unmatched(n)
    return EncodeResult(ids=ids, spans=spans)


def _lookup_adapted_piece(
    piece: str, adapted: dict[str, int]
) -> EncodeResult | None:
    """Exact / leading-space-stripped match for a structural segment piece."""
    if piece in adapted:
        return EncodeResult(ids=[adapted[piece]], spans=[(0, len(piece))])
    if piece.startswith(" ") and piece[1:] in adapted:
        # Leading space + adapted bare ident (attach_leading_space merge).
        # Emit space via a 1-char base encode + adapted id.
        return None  # handled by caller with split
    return None


class ComposedTokenizer:
    """C3/C4/C5 tokenizer: optional structure + optional adapted vocab.

    ``match_mode``:
      - ``none``: per-segment base BPE only (C4 when no adapted vocab)
      - ``native_longest``: C3 full-text longest-match (K=0 ⇒ C2)
      - ``structural_ident``: C5 match on tree-sitter ident segments
    """

    def __init__(
        self,
        segmenter: Any,
        base_hf_tokenizer: NativeHFTokenizer,
        adapted_vocab: AdaptedVocab | dict[str, int] | None = None,
        *,
        attach_leading_space: bool = True,
        indent_tokens: bool = False,
        match_mode: MatchMode = "none",
        name: str = "C4",
    ) -> None:
        self.segmenter = segmenter
        self.base = base_hf_tokenizer
        self.attach_leading_space = attach_leading_space
        self.indent_tokens = indent_tokens
        self.match_mode = match_mode
        self.name = name
        self._adapted_obj: AdaptedVocab | None
        self.adapted_vocab: dict[str, int]
        if adapted_vocab is None:
            self._adapted_obj = None
            self.adapted_vocab = {}
        elif isinstance(adapted_vocab, AdaptedVocab):
            self._adapted_obj = adapted_vocab
            self.adapted_vocab = adapted_vocab.as_dict()
        else:
            self._adapted_obj = None
            self.adapted_vocab = dict(adapted_vocab)
        self._rev = {i: t for t, i in self.adapted_vocab.items()}
        self._plain_identifiers = entries_are_plain_identifiers(self.adapted_vocab)

    def _segments(self, text: str) -> list[Segment]:
        segs = self.segmenter.segment(text)
        if isinstance(self.segmenter, TreeSitterSegmenter):
            segs = assign_ident_roles(text, segs)
        if self.attach_leading_space:
            segs = merge_leading_space_gaps(segs, text)
        _ = self.indent_tokens
        return segs

    def _encode_structural_piece(self, piece: str) -> EncodeResult:
        if not self.adapted_vocab:
            return self.base.encode(piece)
        if piece in self.adapted_vocab:
            tid = self.adapted_vocab[piece]
            return EncodeResult(ids=[tid], spans=[(0, len(piece))])
        if piece.startswith(" ") and piece[1:] in self.adapted_vocab:
            space_enc = self.base.encode(" ")
            tid = self.adapted_vocab[piece[1:]]
            ids = list(space_enc.ids) + [tid]
            spans = list(space_enc.spans) + [(1, len(piece))]
            return EncodeResult(ids=ids, spans=spans)
        return self.base.encode(piece)

    def encode(self, text: str) -> EncodeResult:
        if text == "":
            return EncodeResult(ids=[], spans=[])

        if self.match_mode == "native_longest":
            return encode_native_longest_match(
                text,
                self.base,
                self.adapted_vocab,
                plain_identifiers=self._plain_identifiers,
            )

        # C4 / C5: segment then encode.
        ids: list[int] = []
        spans: list[tuple[int, int]] = []
        for seg in self._segments(text):
            piece = text[seg.start : seg.end]
            if piece == "":
                continue
            if (
                self.match_mode == "structural_ident"
                and self.adapted_vocab
                and seg.kind == "ident"
            ):
                sub = self._encode_structural_piece(piece)
            elif (
                self.match_mode == "structural_ident"
                and self.adapted_vocab
                and piece.startswith(" ")
                and piece[1:].isidentifier()
                and piece[1:] in self.adapted_vocab
            ):
                # Merged leading-space + ident.
                sub = self._encode_structural_piece(piece)
            else:
                sub = self.base.encode(piece)
            for tid, (s, e) in zip(sub.ids, sub.spans, strict=True):
                ids.append(tid)
                spans.append((seg.start + s, seg.start + e))
        return EncodeResult(ids=ids, spans=spans)

    def decode(self, ids: list[int]) -> str:
        if not self._rev:
            return self.base.decode(ids)
        parts: list[str] = []
        buf: list[int] = []

        def flush() -> None:
            if buf:
                parts.append(self.base.decode(buf))
                buf.clear()

        for tid in ids:
            if tid in self._rev:
                flush()
                parts.append(self._rev[tid])
            else:
                buf.append(tid)
        flush()
        return "".join(parts)

    @property
    def vocab_size(self) -> int:
        if self._adapted_obj is not None and len(self._adapted_obj) > 0:
            return max(e.id for e in self._adapted_obj.entries) + 1
        if self.adapted_vocab:
            return max(self.base.vocab_size, max(self.adapted_vocab.values()) + 1)
        return self.base.vocab_size
