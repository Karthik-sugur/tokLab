"""Smoke tests for package install and tree-sitter Python parsing."""

import tree_sitter_python as tspython
from tree_sitter import Language, Parser


def test_import_toklab() -> None:
    import toklab

    assert toklab.__version__


def test_tree_sitter_parses_assignment() -> None:
    language = Language(tspython.language())
    parser = Parser(language)
    tree = parser.parse(b"x = 1")
    assert not tree.root_node.has_error
