"""Per-file, string-consistent libcst renamer for project-defined identifiers.

A candidate name (class, function/method, module-level CONSTANT) is renamed
at *every* identifier occurrence in the file (``Name``, def/class name,
parameter, ``.attr``), so the rename is a bijection on identifier strings.

A candidate is skipped when the file uses the same string somewhere a rename
cannot be proven consistent:

* import statements (module path, imported name, ``as`` alias)
* keyword-argument names, ``global``/``nonlocal``, ``except ... as``,
  match-pattern captures and keyword attributes
* any identifier-like string literal (``getattr(x, "name")``, ``__all__``)
* attribute access whose base is not ``self``, ``cls``, ``super()`` or a
  class defined in this file (``os.path``, ``obj.json`` may be external)

Every output is checked with :func:`alpha_equivalent`; on failure the
original source is returned with ``renamed=False`` and a reason.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import libcst as cst

from toklab.data.synth.validate import alpha_equivalent

_BUILTIN_NAMES = frozenset(dir(builtins))
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SELF_NAMES = frozenset({"self", "cls"})

REASON_OK = ""
REASON_LIBCST_PARSE = "libcst_parse_error"
REASON_AST_PARSE = "ast_parse_error"
REASON_NO_CANDIDATES = "no_eligible_identifiers"
REASON_ALPHA_FAIL = "alpha_equivalence_failed"


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


@dataclass(frozen=True)
class _FileFacts:
    candidates: tuple[str, ...]
    blocked: frozenset[str]


def _module_constant_targets(tree: ast.Module) -> list[ast.Name]:
    out: list[ast.Name] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            out.extend(t for t in stmt.targets if isinstance(t, ast.Name))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            out.append(stmt.target)
    return [t for t in out if _is_constant_name(t.id)]


def _allowed_attr_base(base: ast.expr, local_classes: set[str]) -> bool:
    if isinstance(base, ast.Name):
        return base.id in _SELF_NAMES or base.id in local_classes
    return (
        isinstance(base, ast.Call)
        and isinstance(base.func, ast.Name)
        and base.func.id == "super"
    )


def _file_facts(tree: ast.Module) -> _FileFacts:
    local_classes = {
        n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
    }
    blocked: set[str] = set()
    defs: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.append((node.lineno, node.col_offset, node.name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                blocked.update(node.module.split("."))
            for alias in node.names:
                blocked.update(alias.name.split("."))
                if alias.asname:
                    blocked.add(alias.asname)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            blocked.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            blocked.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            blocked.add(node.arg)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            blocked.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            blocked.add(node.rest)
        elif isinstance(node, ast.MatchClass):
            blocked.update(node.kwd_attrs)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.isidentifier():
                blocked.add(node.value)
        elif isinstance(node, ast.Attribute):
            if not _allowed_attr_base(node.value, local_classes):
                blocked.add(node.attr)
    for target in _module_constant_targets(tree):
        defs.append((target.lineno, target.col_offset, target.id))

    ordered: list[str] = []
    seen: set[str] = set()
    for _, _, name in sorted(defs):
        if name in seen:
            continue
        seen.add(name)
        if _eligible(name, blocked):
            ordered.append(name)
    return _FileFacts(candidates=tuple(ordered), blocked=frozenset(blocked))


def _eligible(name: str, blocked: set[str] | frozenset[str]) -> bool:
    return not (_is_dunder(name) or name in _BUILTIN_NAMES or name in blocked)


class _RenameTransformer(cst.CSTTransformer):
    """Rename every ``Name`` whose value is in ``name_map``."""

    def __init__(self, name_map: dict[str, str]) -> None:
        super().__init__()
        self.name_map = name_map

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.Name:
        new = self.name_map.get(original_node.value)
        if new is None:
            return updated_node
        return updated_node.with_changes(value=new)


@dataclass(frozen=True)
class RenameResult:
    source: str
    rename_map: dict[str, str]
    renamed: bool = False
    reason: str = field(default=REASON_NO_CANDIDATES)


def collect_renameable_names(source: str) -> list[str]:
    """Ordered unique project-defined names that can be renamed consistently.

    Returns ``[]`` if ``ast`` cannot parse the source.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return list(_file_facts(tree).candidates)


def rename_source(source: str, mapping: dict[str, str]) -> RenameResult:
    """Apply ``mapping`` (old→new) to every identifier occurrence of each old.

    Entries for ineligible names, or whose new name already occurs in the
    file or is used twice, are dropped. The output is alpha-equivalence
    checked; on failure the original is returned with ``renamed=False``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return RenameResult(source, {}, False, REASON_AST_PARSE)
    facts = _file_facts(tree)
    existing = set(_IDENT_RE.findall(source))
    clean: dict[str, str] = {}
    used: set[str] = set()
    for old, new in mapping.items():
        if not _eligible(old, facts.blocked) or old not in existing:
            continue
        if not new.isidentifier() or new in existing or new in used:
            continue
        clean[old] = new
        used.add(new)
    if not clean:
        return RenameResult(source, {}, False, REASON_NO_CANDIDATES)

    try:
        module = cst.parse_module(source)
    except (cst.ParserSyntaxError, cst.CSTLogicError):
        return RenameResult(source, {}, False, REASON_LIBCST_PARSE)
    transformed = module.visit(_RenameTransformer(clean))
    assert isinstance(transformed, cst.Module)
    out = transformed.code
    try:
        ok = alpha_equivalent(source, out)
    except SyntaxError:
        ok = False
    if not ok:
        return RenameResult(source, {}, False, REASON_ALPHA_FAIL)
    return RenameResult(out, dict(sorted(clean.items())), True, REASON_OK)


def rename_file_with_lexicon(
    source: str,
    name_allocator,
    which: str = "adapt",
) -> RenameResult:
    """Allocate lexicon names for all renameable defs and apply them.

    Files that ``ast`` or libcst cannot parse are returned unchanged with
    ``renamed=False`` and the reason.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return RenameResult(source, {}, False, REASON_AST_PARSE)
    olds = _file_facts(tree).candidates
    if not olds:
        return RenameResult(source, {}, False, REASON_NO_CANDIDATES)
    news = name_allocator.allocate_names(len(olds), which=which)
    return rename_source(source, dict(zip(olds, news, strict=True)))


def write_rename_map(path: Path | str, rename_maps: dict[str, dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rename_maps, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
