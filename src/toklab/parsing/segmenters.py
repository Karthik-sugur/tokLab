"""Lossless segmenters: HF pre-tokenizer spans and tree-sitter leaves."""

from __future__ import annotations

from typing import Any

from toklab.core.types import Segment

# tree-sitter-python node types → Segment.kind
_TS_KIND: dict[str, str] = {
    "identifier": "ident",
    "type_identifier": "ident",
    "comment": "comment",
    "string": "string",
    "string_start": "string",
    "string_content": "string",
    "string_end": "string",
    "escape_sequence": "string",
    "interpolation": "string",
    "integer": "number",
    "float": "number",
    "true": "keyword",
    "false": "keyword",
    "none": "keyword",
}

_TS_KEYWORDS = frozenset(
    {
        "def",
        "class",
        "return",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "import",
        "from",
        "as",
        "with",
        "pass",
        "break",
        "continue",
        "try",
        "except",
        "finally",
        "raise",
        "yield",
        "async",
        "await",
        "lambda",
        "global",
        "nonlocal",
        "assert",
        "del",
        "in",
        "is",
        "not",
        "and",
        "or",
        "True",
        "False",
        "None",
    }
)

_OP_CHARS = frozenset("=+-*/%<>!&|^~@")


def _byte_to_char_index(text: str, byte_len: int) -> list[int]:
    """Map each byte offset 0..nbytes to a char index (len = nbytes+1)."""
    raw = text.encode("utf-8")
    index = [0] * (len(raw) + 1)
    char_i = 0
    byte_i = 0
    for ch in text:
        n = len(ch.encode("utf-8"))
        for _ in range(n):
            index[byte_i] = char_i
            byte_i += 1
        char_i += 1
    index[len(raw)] = char_i
    return index


def _kind_for_leaf(node_type: str, piece: str) -> str:
    if node_type in _TS_KIND:
        return _TS_KIND[node_type]
    if node_type in {"ERROR", "MISSING"} or node_type.startswith("ERROR"):
        # Keep error text; classify lightly.
        if piece.isidentifier():
            return "ident"
        return "gap" if piece.isspace() else "op"
    if piece in _TS_KEYWORDS or node_type in _TS_KEYWORDS:
        return "keyword"
    if piece.isidentifier():
        return "ident"
    if piece and all(c in _OP_CHARS for c in piece):
        return "op"
    if piece and all(c in "()[]{}.,:;\\" or c.isspace() for c in piece):
        # punctuation / delimiters (allow mixed with spaces only if pure delim)
        if piece.isspace():
            return "gap"
        return "delim"
    if piece.isspace():
        return "gap"
    return "op"


def _guess_kind_from_text(piece: str) -> str:
    if not piece:
        return "gap"
    if piece.isspace():
        return "gap"
    if piece in _TS_KEYWORDS:
        return "keyword"
    if piece.isidentifier():
        return "ident"
    if piece[0] in "\"'":
        return "string"
    if piece[0].isdigit() or (
        piece[0] == "." and len(piece) > 1 and piece[1].isdigit()
    ):
        return "number"
    if piece.startswith("#"):
        return "comment"
    if all(c in _OP_CHARS for c in piece):
        return "op"
    if all(c in "()[]{}.,:;\\" for c in piece):
        return "delim"
    return "op"


class NativeSegmenter:
    """Segment using an HF tokenizer's ``return_offsets_mapping`` spans."""

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tok = hf_tokenizer

    def segment(self, text: str) -> list[Segment]:
        if text == "":
            return []
        encoded = self._tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        offsets = [(int(s), int(e)) for s, e in encoded["offset_mapping"]]
        segments: list[Segment] = []
        cursor = 0
        for start, end in offsets:
            if start < cursor:
                # Overlapping offset — advance without duplicating.
                start = cursor
            if start > cursor:
                gap = text[cursor:start]
                segments.append(
                    Segment(cursor, start, "gap", "other")
                    if gap.isspace() or gap == ""
                    else Segment(cursor, start, _guess_kind_from_text(gap), "other")
                )
            if end > start:
                piece = text[start:end]
                segments.append(
                    Segment(start, end, _guess_kind_from_text(piece), "other")
                )
            cursor = max(cursor, end)
        if cursor < len(text):
            piece = text[cursor:]
            kind = "gap" if piece.isspace() else _guess_kind_from_text(piece)
            segments.append(Segment(cursor, len(text), kind, "other"))
        return segments


class TreeSitterSegmenter:
    """Walk tree-sitter leaf nodes; insert ``gap`` segments for holes.

    Parse errors do not drop text: error/missing leaves and any uncovered
    ranges become ordinary segments so coverage stays lossless.
    """

    def __init__(self) -> None:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        self._language = Language(tspython.language())
        self._parser = Parser(self._language)

    def segment(self, text: str) -> list[Segment]:
        if text == "":
            return []
        raw = text.encode("utf-8")
        tree = self._parser.parse(raw)
        b2c = _byte_to_char_index(text, len(raw))
        leaves: list[tuple[int, int, str]] = []

        def walk(node: Any) -> None:
            if node.child_count == 0:
                start_b, end_b = node.start_byte, node.end_byte
                if end_b > start_b:
                    leaves.append((start_b, end_b, node.type))
                return
            for i in range(node.child_count):
                walk(node.child(i))

        walk(tree.root_node)
        leaves.sort(key=lambda t: (t[0], t[1]))

        segments: list[Segment] = []
        cursor_b = 0
        for start_b, end_b, ntype in leaves:
            if start_b < cursor_b:
                start_b = cursor_b
            if start_b > cursor_b:
                cs, ce = b2c[cursor_b], b2c[start_b]
                if ce > cs:
                    piece = text[cs:ce]
                    kind = "gap" if piece.isspace() else _guess_kind_from_text(piece)
                    segments.append(Segment(cs, ce, kind, "other"))
            if end_b > start_b:
                cs, ce = b2c[start_b], b2c[end_b]
                if ce > cs:
                    piece = text[cs:ce]
                    kind = _kind_for_leaf(ntype, piece)
                    segments.append(Segment(cs, ce, kind, "other"))
            cursor_b = max(cursor_b, end_b)

        if cursor_b < len(raw):
            cs, ce = b2c[cursor_b], b2c[len(raw)]
            if ce > cs:
                piece = text[cs:ce]
                kind = "gap" if piece.isspace() else _guess_kind_from_text(piece)
                segments.append(Segment(cs, ce, kind, "other"))

        # Collapse adjacent empties / ensure contiguity via merge of zero-width.
        return [s for s in segments if s.end > s.start]
