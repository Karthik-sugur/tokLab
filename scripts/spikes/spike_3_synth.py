#!/usr/bin/env python3
"""Spike 3: libcst rename of top-level names + token-count impact.

Prints observed facts only. Not library code.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PREFERRED = "Qwen/Qwen2.5-Coder-0.5B"
FALLBACKS = [
    "Salesforce/codegen-350M-mono",
    "microsoft/CodeGPT-small-py",
]

# Long made-up names for top-level functions / classes
FAKE_FUNC = "AcmePaymentSettlementBatchProcessor"
FAKE_CLASS = "AcmeCustomerLedgerReconciliationService"
FAKE_FUNC_2 = "AcmeInvoiceClearingWorkflowRunner"


def _load_code_tokenizer():
    from transformers import AutoTokenizer

    for name in [PREFERRED, *FALLBACKS]:
        try:
            print(f"Trying tokenizer: {name}")
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            print(f"Loaded tokenizer: {name}")
            return name, tok
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load {name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load any tokenizer")


def _pick_stdlib_modules() -> list[Path]:
    root = Path(sys.base_prefix) / "lib" / (
        f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    if not root.is_dir():
        candidates = list(Path(sys.base_prefix).rglob("os.py"))
        if not candidates:
            raise RuntimeError(f"Could not find stdlib under {sys.base_prefix}")
        root = candidates[0].parent

    # Prefer small, pure-Python modules with top-level defs/classes
    preferred = ["quopri.py", "uu.py", "imaplib.py", "colorsys.py", "nturl2path.py"]
    found: list[Path] = []
    for name in preferred:
        p = root / name
        if p.is_file():
            found.append(p)
        if len(found) >= 3:
            break
    if len(found) < 3:
        for p in sorted(root.glob("*.py")):
            if p in found:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text)
            tops = [
                n
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            if tops:
                found.append(p)
            if len(found) >= 3:
                break
    if len(found) < 3:
        raise RuntimeError(f"Could not find 3 suitable stdlib modules under {root}")
    print(f"stdlib_root: {root}")
    return found[:3]


class _RenameTopLevel:
    """libcst transformer: rename top-level function/class definitions + refs."""

    def __init__(self, mapping: dict[str, str]):
        import libcst as cst

        self._cst = cst
        self.mapping = mapping

    def transform(self, module):
        cst = self._cst

        class Visitor(cst.CSTTransformer):
            def __init__(self, mapping: dict[str, str]):
                super().__init__()
                self.mapping = mapping
                self._stack: list[str] = ["module"]

            def visit_FunctionDef(self, node):  # noqa: N802
                self._stack.append("function")

            def leave_FunctionDef(self, original_node, updated_node):  # noqa: N802
                self._stack.pop()
                name = original_node.name.value
                if self._stack == ["module"] and name in self.mapping:
                    return updated_node.with_changes(
                        name=cst.Name(self.mapping[name])
                    )
                return updated_node

            def visit_ClassDef(self, node):  # noqa: N802
                self._stack.append("class")

            def leave_ClassDef(self, original_node, updated_node):  # noqa: N802
                self._stack.pop()
                name = original_node.name.value
                if self._stack == ["module"] and name in self.mapping:
                    return updated_node.with_changes(
                        name=cst.Name(self.mapping[name])
                    )
                return updated_node

            def leave_Name(self, original_node, updated_node):  # noqa: N802
                if original_node.value in self.mapping:
                    return updated_node.with_changes(
                        value=self.mapping[original_node.value]
                    )
                return updated_node

        return module.visit(Visitor(self.mapping))


def _normalize_identifiers(tree: ast.AST) -> ast.AST:
    """Replace all identifier names with sequential placeholders."""
    counter = {"n": 0}
    id_map: dict[str, str] = {}

    class Norm(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            self.generic_visit(node)
            if node.id not in id_map:
                id_map[node.id] = f"ID{counter['n']}"
                counter["n"] += 1
            return ast.Name(id=id_map[node.id], ctx=node.ctx)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            self.generic_visit(node)
            if node.name not in id_map:
                id_map[node.name] = f"ID{counter['n']}"
                counter["n"] += 1
            node.name = id_map[node.name]
            return node

        def visit_AsyncFunctionDef(
            self, node: ast.AsyncFunctionDef
        ) -> ast.AsyncFunctionDef:
            self.generic_visit(node)
            if node.name not in id_map:
                id_map[node.name] = f"ID{counter['n']}"
                counter["n"] += 1
            node.name = id_map[node.name]
            return node

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
            self.generic_visit(node)
            if node.name not in id_map:
                id_map[node.name] = f"ID{counter['n']}"
                counter["n"] += 1
            node.name = id_map[node.name]
            return node

        def visit_arg(self, node: ast.arg) -> ast.arg:
            self.generic_visit(node)
            if node.arg not in id_map:
                id_map[node.arg] = f"ID{counter['n']}"
                counter["n"] += 1
            node.arg = id_map[node.arg]
            return node

        def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
            self.generic_visit(node)
            if node.attr not in id_map:
                id_map[node.attr] = f"ID{counter['n']}"
                counter["n"] += 1
            node.attr = id_map[node.attr]
            return node

    return Norm().visit(tree)


def _top_level_mapping(tree: ast.AST) -> dict[str, str]:
    mapping: dict[str, str] = {}
    fake_cycle = [FAKE_FUNC, FAKE_CLASS, FAKE_FUNC_2]
    i = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mapping[node.name] = fake_cycle[i % len(fake_cycle)] + (
                f"_{i}" if i >= len(fake_cycle) else ""
            )
            # Prefer distinct long names without suffix when possible
            if i < len(fake_cycle):
                mapping[node.name] = fake_cycle[i]
            else:
                mapping[node.name] = f"{fake_cycle[i % len(fake_cycle)]}V{i}"
            i += 1
    return mapping


def _token_count(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def main() -> int:
    import libcst as cst

    code_name, tok = _load_code_tokenizer()
    modules = _pick_stdlib_modules()
    print(f"modules: {[m.name for m in modules]}")

    for path in modules:
        print()
        print(f"=== {path.name} ===")
        original = path.read_text(encoding="utf-8")
        tree = ast.parse(original)
        mapping = _top_level_mapping(tree)
        print(f"rename_mapping: {mapping}")
        if not mapping:
            print("SKIP: no top-level functions/classes")
            continue

        wrapper = cst.parse_module(original)
        renamed_mod = _RenameTopLevel(mapping).transform(wrapper)
        renamed = renamed_mod.code

        orig_norm = ast.dump(_normalize_identifiers(ast.parse(original)))
        ren_norm = ast.dump(_normalize_identifiers(ast.parse(renamed)))
        equal = orig_norm == ren_norm
        print(f"ast_dump_equal_after_normalising_identifiers: {equal}")
        if not equal:
            print(f"  orig_norm_len={len(orig_norm)} ren_norm_len={len(ren_norm)}")

        n_orig = _token_count(tok, original)
        n_ren = _token_count(tok, renamed)
        print(f"token_count_original ({code_name}): {n_orig}")
        print(f"token_count_renamed  ({code_name}): {n_ren}")
        print(f"token_count_delta: {n_ren - n_orig}")

    print()
    print("FINDINGS:")
    print(
        f"- Renamed top-level function/class names in {len(modules)} stdlib "
        "modules via libcst; see per-module mapping and ast equality above."
    )
    print(
        f"- Token counts use tokenizer {code_name}; deltas printed per module."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
