"""Scope-aware libcst renamer for project-defined identifiers."""

from __future__ import annotations

import builtins
import json
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
from libcst.metadata import (
    Assignment,
    ImportAssignment,
    MetadataWrapper,
    ScopeProvider,
)

_BUILTIN_NAMES = frozenset(dir(builtins))


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__") and len(name) >= 4


def _is_constant_name(name: str) -> bool:
    """Module-level CONSTANT style: uppercase letters/digits/underscores."""
    if not name or _is_dunder(name):
        return False
    return (
        name.isidentifier()
        and name.upper() == name
        and any(c.isalpha() for c in name)
    )


def _getattr_like_func(func: cst.BaseExpression) -> bool:
    if isinstance(func, cst.Name):
        return func.value in {"getattr", "hasattr", "setattr", "delattr"}
    return False


class _StringNameCollector(cst.CSTVisitor):
    """Collect identifier-like strings used with getattr* or any string lit."""

    def __init__(self) -> None:
        self.frozen: set[str] = set()

    def visit_Call(self, node: cst.Call) -> bool:
        if not _getattr_like_func(node.func):
            return True
        if len(node.args) >= 2:
            arg = node.args[1].value
            if isinstance(arg, cst.SimpleString):
                raw = arg.raw_value
                if isinstance(raw, str) and raw.isidentifier():
                    self.frozen.add(raw)
        return True

    def visit_SimpleString(self, node: cst.SimpleString) -> bool:
        raw = node.raw_value
        if isinstance(raw, str) and raw.isidentifier():
            self.frozen.add(raw)
        return False


class _DefinitionCollector(cst.CSTVisitor):
    """Find project-defined classes, functions/methods, module constants."""

    METADATA_DEPENDENCIES = (ScopeProvider,)

    def __init__(self, frozen: set[str]) -> None:
        self.frozen = frozen
        self._stack: list[str] = ["module"]
        # Assignment object → original name
        self.assignments: dict[Assignment, str] = {}
        self.names_in_all: set[str] = set()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        # Includes async def (libcst uses FunctionDef + asynchronous field).
        self._record_def_name(node.name)
        self._stack.append("function")
        return True

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._stack.pop()

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._record_def_name(node.name)
        self._stack.append("class")
        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._stack.pop()

    def visit_Assign(self, node: cst.Assign) -> bool:
        self._maybe_module_constants(node.targets)
        self._capture_all(node)
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        if self._stack == ["module"] and isinstance(node.target, cst.Name):
            self._maybe_one_constant(node.target)
        return True

    def _capture_all(self, node: cst.Assign) -> None:
        if len(node.targets) != 1:
            return
        target = node.targets[0].target
        if not isinstance(target, cst.Name) or target.value != "__all__":
            return
        val = node.value
        if not isinstance(val, (cst.List, cst.Tuple)):
            return
        for elt in val.elements:
            if isinstance(elt.value, cst.SimpleString):
                raw = elt.value.raw_value
                if isinstance(raw, str):
                    self.names_in_all.add(raw)

    def _maybe_module_constants(self, targets: list[cst.AssignTarget]) -> None:
        if self._stack != ["module"]:
            return
        for t in targets:
            if isinstance(t.target, cst.Name):
                self._maybe_one_constant(t.target)

    def _maybe_one_constant(self, name_node: cst.Name) -> None:
        name = name_node.value
        if not _is_constant_name(name):
            return
        if not self._eligible(name):
            return
        self._bind(name_node, name)

    def _record_def_name(self, name_node: cst.Name) -> None:
        name = name_node.value
        if not self._eligible(name):
            return
        self._bind(name_node, name)

    def _eligible(self, name: str) -> bool:
        if _is_dunder(name):
            return False
        if name in _BUILTIN_NAMES:
            return False
        if name in self.frozen:
            return False
        return True

    def _bind(self, name_node: cst.Name, name: str) -> None:
        scope = self.get_metadata(ScopeProvider, name_node, None)
        if scope is None:
            return
        # libcst: FunctionDef/ClassDef assignments use the def node;
        # Assign targets use the Name node.
        for assignment in scope[name]:
            if isinstance(assignment, ImportAssignment):
                continue
            if not isinstance(assignment, Assignment):
                continue
            if assignment.name != name:
                continue
            node = assignment.node
            if node is name_node:
                self.assignments[assignment] = name
                return
            if (
                isinstance(node, (cst.FunctionDef, cst.ClassDef))
                and node.name is name_node
            ):
                self.assignments[assignment] = name
                return
            if isinstance(node, cst.Name) and node.value == name:
                self.assignments[assignment] = name
                return


class _RenameTransformer(cst.CSTTransformer):
    """Rename selected Assignments and update ``__all__`` string entries."""

    METADATA_DEPENDENCIES = (ScopeProvider,)

    def __init__(self, assignment_map: dict[Assignment, str], name_map: dict[str, str]):
        super().__init__()
        self.assignment_map = assignment_map
        self.name_map = name_map  # old → new for __all__ / Attribute.attr

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.Name:
        scope = self.get_metadata(ScopeProvider, original_node, None)
        if scope is None:
            return updated_node
        name = original_node.value
        matched: set[str] = set()
        for assignment in scope[name]:
            if assignment in self.assignment_map:
                matched.add(self.assignment_map[assignment])
        if len(matched) == 1:
            return updated_node.with_changes(value=next(iter(matched)))
        return updated_node

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.Attribute:
        # Rename method / nested access when attr is in the name map and the
        # corresponding definition was selected (string-keyed, per-file).
        attr = original_node.attr
        if isinstance(attr, cst.Name) and attr.value in self.name_map:
            new = self.name_map[attr.value]
            return updated_node.with_changes(attr=cst.Name(new))
        return updated_node

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.Assign:
        if len(updated_node.targets) != 1:
            return updated_node
        target = updated_node.targets[0].target
        if not isinstance(target, cst.Name) or target.value != "__all__":
            return updated_node
        val = updated_node.value
        if not isinstance(val, (cst.List, cst.Tuple)):
            return updated_node

        def rewrite_elt(elt: cst.BaseElement) -> cst.BaseElement:
            if not isinstance(elt.value, cst.SimpleString):
                return elt
            raw = elt.value.raw_value
            if not isinstance(raw, str) or raw not in self.name_map:
                return elt
            # Preserve quote style from the original literal.
            quote = '"' if elt.value.value.startswith('"') else "'"
            new_lit = cst.SimpleString(value=f"{quote}{self.name_map[raw]}{quote}")
            return elt.with_changes(value=new_lit)

        new_elements = [rewrite_elt(e) for e in val.elements]
        return updated_node.with_changes(
            value=val.with_changes(elements=new_elements)
        )


@dataclass(frozen=True)
class RenameResult:
    source: str
    rename_map: dict[str, str]


def collect_renameable_names(source: str) -> list[str]:
    """Return ordered unique project-defined names eligible for renaming."""
    module = cst.parse_module(source)
    frozen_visitor = _StringNameCollector()
    module.visit(frozen_visitor)
    wrapper = MetadataWrapper(module)
    collector = _DefinitionCollector(frozen_visitor.frozen)
    wrapper.visit(collector)
    # Preserve first-seen order of original names.
    ordered: list[str] = []
    seen: set[str] = set()
    for name in collector.assignments.values():
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def rename_source(source: str, mapping: dict[str, str]) -> RenameResult:
    """Apply ``mapping`` (old→new) to project definitions and their uses."""
    if not mapping:
        return RenameResult(source=source, rename_map={})

    module = cst.parse_module(source)
    frozen_visitor = _StringNameCollector()
    module.visit(frozen_visitor)
    # Never rename frozen / dunder / builtin even if mapping asks.
    clean = {
        o: n
        for o, n in mapping.items()
        if o not in frozen_visitor.frozen
        and not _is_dunder(o)
        and o not in _BUILTIN_NAMES
    }
    if not clean:
        return RenameResult(source=source, rename_map={})

    # Collect and transform on the same wrapper so Assignment identity matches.
    wrapper = MetadataWrapper(cst.parse_module(source))
    collector = _DefinitionCollector(frozen_visitor.frozen)
    wrapper.visit(collector)

    assignment_map: dict[Assignment, str] = {}
    applied: dict[str, str] = {}
    for assignment, old_name in collector.assignments.items():
        if old_name in clean:
            assignment_map[assignment] = clean[old_name]
            applied[old_name] = clean[old_name]

    if not assignment_map:
        return RenameResult(source=source, rename_map={})

    transformed = wrapper.visit(_RenameTransformer(assignment_map, applied))
    assert isinstance(transformed, cst.Module)
    return RenameResult(
        source=transformed.code,
        rename_map=dict(sorted(applied.items())),
    )


def rename_file_with_lexicon(
    source: str,
    name_allocator,
    which: str = "adapt",
) -> RenameResult:
    """Allocate lexicon names for all renameable defs and apply them."""
    olds = collect_renameable_names(source)
    news = name_allocator.allocate_names(len(olds), which=which)
    mapping = dict(zip(olds, news, strict=True))
    return rename_source(source, mapping)


def write_rename_map(path: Path | str, rename_maps: dict[str, dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rename_maps, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
