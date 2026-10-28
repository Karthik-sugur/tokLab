"""Assign semantic roles to identifier segments using tree-sitter context."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from toklab.core.types import Segment


def _byte_to_char_index(text: str) -> list[int]:
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


def _role_for_ident_node(node: Any) -> str:
    """Map an identifier leaf's parent chain to a role label."""
    parent = node.parent
    if parent is None:
        return "other"
    ptype = parent.type

    # def / async def name
    if ptype == "function_definition":
        # name field is the first identifier child typically
        name_node = parent.child_by_field_name("name")
        if name_node is not None and name_node.id == node.id:
            return "def_name"
    if ptype == "class_definition":
        name_node = parent.child_by_field_name("name")
        if name_node is not None and name_node.id == node.id:
            return "class_name"

    # parameters
    param_types = {
        "parameters",
        "lambda_parameters",
        "typed_parameter",
        "default_parameter",
    }
    if ptype in param_types:
        return "parameter"

    # attribute: obj.attr — the attr field
    if ptype == "attribute":
        attr = parent.child_by_field_name("attribute")
        if attr is not None and attr.id == node.id:
            return "attribute"

    # call: function name
    if ptype == "call":
        func = parent.child_by_field_name("function")
        if func is not None and func.id == node.id:
            return "call"
    grand = parent.parent
    if ptype == "attribute" and grand is not None and grand.type == "call":
        # foo.bar() — method name treated as call
        func = grand.child_by_field_name("function")
        if func is not None and func.id == parent.id:
            attr = parent.child_by_field_name("attribute")
            if attr is not None and attr.id == node.id:
                return "call"

    return "other"


def assign_ident_roles(
    text: str,
    segments: list[Segment],
    tree: Any | None = None,
) -> list[Segment]:
    """Return segments with ``role`` filled for ``kind=="ident"``.

    If ``tree`` is None, parse ``text`` with tree-sitter Python.
    Non-ident segments keep their existing role (usually ``other``).
    """
    if not segments:
        return segments

    if tree is None:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        parser = Parser(Language(tspython.language()))
        tree = parser.parse(text.encode("utf-8"))

    b2c = _byte_to_char_index(text)
    # Map (start_char, end_char) → role for identifier leaves.
    roles: dict[tuple[int, int], str] = {}

    def walk(node: Any) -> None:
        if node.child_count == 0 and node.type in {"identifier", "type_identifier"}:
            cs, ce = b2c[node.start_byte], b2c[node.end_byte]
            roles[(cs, ce)] = _role_for_ident_node(node)
            return
        for i in range(node.child_count):
            walk(node.child(i))

    walk(tree.root_node)

    out: list[Segment] = []
    for seg in segments:
        if seg.kind != "ident":
            out.append(seg)
            continue
        role = roles.get((seg.start, seg.end), "other")
        out.append(replace(seg, role=role))
    return out
