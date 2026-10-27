"""Tests for synthetic enterprise lexicon, rename, and drift."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from toklab.data.synth.drift import apply_drift
from toklab.data.synth.lexicon import EnterpriseLexicon
from toklab.data.synth.rename import (
    collect_renameable_names,
    rename_file_with_lexicon,
    rename_source,
)


def _normalize_identifiers(tree: ast.AST) -> ast.AST:
    counter = {"n": 0}
    id_map: dict[str, str] = {}

    class Norm(ast.NodeTransformer):
        def _map(self, name: str) -> str:
            if name not in id_map:
                id_map[name] = f"ID{counter['n']}"
                counter["n"] += 1
            return id_map[name]

        def visit_Name(self, node: ast.Name) -> ast.Name:
            self.generic_visit(node)
            return ast.Name(id=self._map(node.id), ctx=node.ctx)

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

    return Norm().visit(tree)


def _alpha_eq(a: str, b: str) -> bool:
    return ast.dump(_normalize_identifiers(ast.parse(a))) == ast.dump(
        _normalize_identifiers(ast.parse(b))
    )


FIXTURE_SRC = '''\
"""Sample project module for synth tests."""
from math import sqrt

__all__ = ["Widget", "MAX_BATCH", "build_widget"]

MAX_BATCH = 10
_PRIVATE = 1

class Widget:
    def process(self, x):
        return sqrt(x)

    def __repr__(self):
        return "Widget"

def build_widget():
    w = Widget()
    return w.process(4)

def helper():
    name = "process"
    return getattr(Widget(), name)
'''


def test_adapt_drift_lexicons_share_no_terms() -> None:
    lex = EnterpriseLexicon.from_yaml(seed=0)
    overlap = lex.lexicon_adapt.all_terms() & lex.lexicon_drift.all_terms()
    assert not overlap


def test_zipf_names_are_enterprise_shaped() -> None:
    lex = EnterpriseLexicon.from_yaml(seed=1)
    name = lex.next_name("adapt")
    assert name[:1].isupper()
    assert name.isidentifier()
    # Must be composed from adapt terms only.
    adapt = lex.lexicon_adapt
    assert any(name.startswith(o) for o in adapt.orgs)


def test_dunder_and_builtin_never_renamed() -> None:
    lex = EnterpriseLexicon.from_yaml(seed=0)
    result = rename_file_with_lexicon(FIXTURE_SRC, lex, which="adapt")
    tree = ast.parse(result.source)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "__repr__" in names
    # builtins / imports not rewritten into enterprise names as defs
    assert "sqrt" not in result.rename_map
    assert "__repr__" not in result.rename_map
    assert "__all__" not in result.rename_map


def test_alpha_equivalence_on_fixture() -> None:
    lex = EnterpriseLexicon.from_yaml(seed=0)
    result = rename_file_with_lexicon(FIXTURE_SRC, lex, which="adapt")
    assert result.rename_map
    assert _alpha_eq(FIXTURE_SRC, result.source)


def test_getattr_string_target_not_renamed_as_definition() -> None:
    # `process` appears in a string for getattr → frozen, must not be renamed.
    names = collect_renameable_names(FIXTURE_SRC)
    assert "process" not in names


def test_same_seed_identical_output(tmp_path: Path) -> None:
    lex_a = EnterpriseLexicon.from_yaml(seed=42)
    lex_b = EnterpriseLexicon.from_yaml(seed=42)
    a = rename_file_with_lexicon(FIXTURE_SRC, lex_a, which="adapt")
    b = rename_file_with_lexicon(FIXTURE_SRC, lex_b, which="adapt")
    assert a.source == b.source
    assert a.rename_map == b.rename_map


def test_drift_names_never_in_adapt_output() -> None:
    lex = EnterpriseLexicon.from_yaml(seed=0)
    adapt = rename_file_with_lexicon(FIXTURE_SRC, lex, which="adapt")
    # Fresh lexicon with same seed so pools match; continue RNG via new apply.
    lex2 = EnterpriseLexicon.from_yaml(seed=0)
    # Burn the same number of adapt draws so drift stream differs but pools match.
    _ = rename_file_with_lexicon(FIXTURE_SRC, lex2, which="adapt")
    drift = apply_drift(FIXTURE_SRC, lex2, proportion=50, seed=0)
    adapt_names = set(adapt.rename_map.values())
    drift_names = set(drift.rename_map.values())
    assert adapt_names.isdisjoint(drift_names)
    # Drift terms inventory must stay disjoint from adapt terms.
    assert not (
        lex.lexicon_adapt.all_terms() & lex.lexicon_drift.all_terms()
    )


def test_drift_proportion_zero_is_identity() -> None:
    lex = EnterpriseLexicon.from_yaml(seed=0)
    out = apply_drift(FIXTURE_SRC, lex, proportion=0, seed=0)
    assert out.source == FIXTURE_SRC
    assert out.rename_map == {}


def test_pipeline_deterministic(tmp_path: Path) -> None:
    from toklab.data.splits import build_dataset
    from toklab.data.synth.pipeline import run_synth

    repos = tmp_path / "repos"
    for i, name in enumerate("abcde"):
        p = repos / f"repo_{name}" / "mod.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            f"class Thing{i}:\n    def run(self):\n        return {i}\n",
            encoding="utf-8",
        )
    manifest_dir = tmp_path / "manifest"
    build_dataset(repos, manifest_dir, seed=0)
    out1 = tmp_path / "s1"
    out2 = tmp_path / "s2"
    run_synth(
        manifest=manifest_dir / "manifest.parquet",
        repos_dir=repos,
        out_dir=out1,
        seed=0,
    )
    run_synth(
        manifest=manifest_dir / "manifest.parquet",
        repos_dir=repos,
        out_dir=out2,
        seed=0,
    )
    m1 = json.loads((out1 / "rename_map.json").read_text(encoding="utf-8"))
    m2 = json.loads((out2 / "rename_map.json").read_text(encoding="utf-8"))
    assert m1 == m2


def test_manual_map_preserves_alpha_eq() -> None:
    src = "def foo():\n    return foo\n"
    out = rename_source(src, {"foo": "AcmeLogisticsShipmentRouter"})
    assert "AcmeLogisticsShipmentRouter" in out.source
    assert _alpha_eq(src, out.source)


def test_builtin_name_in_map_ignored() -> None:
    src = "def foo():\n    return len([1])\n"
    out = rename_source(src, {"len": "AcmeNope", "foo": "AcmeYes"})
    assert "len(" in out.source
    assert "AcmeYes" in out.source
    assert "AcmeNope" not in out.source
