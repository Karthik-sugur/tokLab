"""AST alpha-equivalence check for renamed files.

Identifiers at ``Name``, ``FunctionDef``/``AsyncFunctionDef``/``ClassDef``
names, ``arg`` and ``Attribute.attr`` are mapped to ``ID<n>`` in first-seen
order; every other field (keyword names, import aliases, strings, ...) must
match verbatim.
"""

from __future__ import annotations

import ast


class _Normalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.id_map: dict[str, str] = {}

    def _map(self, name: str) -> str:
        if name not in self.id_map:
            self.id_map[name] = f"ID{len(self.id_map)}"
        return self.id_map[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        self.generic_visit(node)
        node.id = self._map(node.id)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.name = self._map(node.name)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.name = self._map(node.name)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.name = self._map(node.name)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        self.generic_visit(node)
        node.arg = self._map(node.arg)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        self.generic_visit(node)
        node.attr = self._map(node.attr)
        return node


def normalize_identifiers(tree: ast.AST) -> ast.AST:
    """Rewrite identifiers in-place to ``ID<n>`` (first-seen order)."""
    return _Normalizer().visit(tree)


def alpha_equivalent(original: str, renamed: str) -> bool:
    """True iff both sources parse and their normalized ASTs are identical.

    Raises ``SyntaxError`` if either source does not parse with ``ast``.
    """
    a = normalize_identifiers(ast.parse(original))
    b = normalize_identifiers(ast.parse(renamed))
    return ast.dump(a) == ast.dump(b)
