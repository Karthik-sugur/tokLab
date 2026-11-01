"""Diagnose alpha-equivalence failures in the synthetic adapt layer.

Usage:
    python scripts/diagnose_synth.py \
        --repos data/repos --synthetic data/synthetic \
        --manifest data/processed/manifest.parquet \
        --log logs/step2_synth_validate.log

The validate log lists only the first 50 failures, so the failing set is
recomputed by comparing each adapt original with its synthetic copy. Only
those failing files are diagnosed: for each, the first AST node where the
normalized original and renamed trees differ is located and classified.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import libcst as cst

from toklab.data.splits import ManifestView
from toklab.data.synth.validate import alpha_equivalent, normalize_identifiers

CATEGORIES = (
    "attribute access not renamed (self.x / obj.x)",
    "keyword argument name",
    "string literal or __all__ entry",
    "decorator",
    "class/dataclass field",
    "import statement",
    "name collision/shadowing",
    "other",
)

_IDENT_FIELDS = {
    ast.Name: "id",
    ast.FunctionDef: "name",
    ast.AsyncFunctionDef: "name",
    ast.ClassDef: "name",
    ast.arg: "arg",
    ast.Attribute: "attr",
}


def _annotate_raw(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        field = _IDENT_FIELDS.get(type(node))
        if field is not None:
            node._raw = getattr(node, field)  # type: ignore[attr-defined]


Ancestors = list[tuple[ast.AST, str]]
Diff = tuple[Any, Any, str | None, Ancestors]


def _first_diff(a: Any, b: Any, ancestors: Ancestors) -> Diff | None:
    if type(a) is not type(b):
        return (a, b, None, ancestors)
    if isinstance(a, ast.AST):
        # Same order as the normalizer: children first, own identifier last,
        # so the first mismatch is where the ID numbering first diverges.
        ident = _IDENT_FIELDS.get(type(a))
        fields = [f for f in a._fields if f != ident]
        if ident is not None:
            fields.append(ident)
        for field in fields:
            va = getattr(a, field, None)
            vb = getattr(b, field, None)
            path = [*ancestors, (a, field)]
            if isinstance(va, ast.AST) or isinstance(vb, ast.AST):
                found = _first_diff(va, vb, path)
            elif isinstance(va, list) and isinstance(vb, list):
                found = None
                for ea, eb in zip(va, vb, strict=False):
                    found = _first_diff(ea, eb, path)
                    if found:
                        break
                if found is None and len(va) != len(vb):
                    found = (a, b, field, ancestors)
            elif va != vb:
                found = (a, b, field, ancestors)
            else:
                found = None
            if found:
                return found
    elif a != b:
        return (a, b, None, ancestors)
    return None


def _lineno(node: Any, ancestors: Ancestors) -> int | None:
    if isinstance(node, ast.AST) and getattr(node, "lineno", None):
        return node.lineno
    for anc, _ in reversed(ancestors):
        if getattr(anc, "lineno", None):
            return anc.lineno
    return None


def _in_class_field(ancestors: Ancestors) -> bool:
    for i in range(len(ancestors) - 1, -1, -1):
        node, field = ancestors[i]
        if isinstance(node, ast.ClassDef) and field == "body":
            if i + 1 < len(ancestors):
                return isinstance(ancestors[i + 1][0], (ast.Assign, ast.AnnAssign))
            return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return False
    return False


def classify(diff: Diff) -> tuple[str, str]:
    """Return ``(category, detail)`` for the first differing node."""
    na, nb, field, ancestors = diff
    chain = [n for n, _ in ancestors] + [na]
    raw_a = getattr(na, "_raw", None)
    raw_b = getattr(nb, "_raw", None)
    detail = f"{type(na).__name__}.{field}"
    if raw_a is not None or raw_b is not None:
        detail += f" {raw_a!r}->{raw_b!r}"
    elif field is not None and isinstance(na, ast.AST):
        detail += f" {getattr(na, field, None)!r}->{getattr(nb, field, None)!r}"
    if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in chain):
        return CATEGORIES[5], detail
    if isinstance(na, ast.keyword) and field == "arg":
        return CATEGORIES[1], detail
    if any(f == "decorator_list" for _, f in ancestors):
        return CATEGORIES[3], detail
    if isinstance(na, ast.Constant):
        return CATEGORIES[2], detail
    if field is not None and _IDENT_FIELDS.get(type(na)) == field:
        if _in_class_field(ancestors):
            return CATEGORIES[4], detail
        if isinstance(na, ast.Attribute):
            return CATEGORIES[0], detail
        return CATEGORIES[6], detail
    return CATEGORIES[7], detail


def diagnose_pair(original: str, renamed: str) -> dict[str, Any] | None:
    try:
        ta = ast.parse(original)
        tb = ast.parse(renamed)
    except SyntaxError as exc:
        return {
            "category": CATEGORIES[7],
            "detail": f"ast.SyntaxError line {exc.lineno}: {exc.msg}",
            "line_a": exc.lineno,
            "line_b": exc.lineno,
            "snippet_a": (exc.text or "").strip(),
            "snippet_b": "",
        }
    _annotate_raw(ta)
    _annotate_raw(tb)
    normalize_identifiers(ta)
    normalize_identifiers(tb)
    diff = _first_diff(ta, tb, [])
    if diff is None:
        return None
    category, detail = classify(diff)
    la = _lineno(diff[0], diff[3])
    lb = _lineno(diff[1], diff[3])
    a_lines = original.splitlines()
    b_lines = renamed.splitlines()
    return {
        "category": category,
        "detail": detail,
        "line_a": la,
        "line_b": lb,
        "snippet_a": a_lines[la - 1].strip() if la and la <= len(a_lines) else "",
        "snippet_b": b_lines[lb - 1].strip() if lb and lb <= len(b_lines) else "",
    }


def _logged_counts(log: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not log.is_file():
        return out
    for line in log.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\w+)=(\d+)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", type=Path, default=Path("data/repos"))
    ap.add_argument("--synthetic", type=Path, default=Path("data/synthetic"))
    ap.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest.parquet")
    )
    ap.add_argument("--log", type=Path, default=Path("logs/step2_synth_validate.log"))
    ap.add_argument(
        "--write-failing",
        type=Path,
        default=None,
        help="optional path to write the failing repo/path list",
    )
    args = ap.parse_args()

    logged = _logged_counts(args.log)
    print(f"log {args.log}: {logged}")

    view = ManifestView.from_parquet(args.manifest, "adapt")
    failing: list[tuple[str, str, str]] = []
    unchanged: list[tuple[str, str]] = []
    for sf in view.files():
        rel = f"{sf.repo}/{sf.path}"
        orig = (args.repos / sf.repo / sf.path).read_text(encoding="utf-8")
        new = (args.synthetic / "adapt" / sf.repo / sf.path).read_text(
            encoding="utf-8"
        )
        if new == orig:
            unchanged.append((rel, orig))
            continue
        try:
            ok = alpha_equivalent(orig, new)
        except SyntaxError:
            ok = False
        if not ok:
            failing.append((rel, orig, new))
    print(f"recomputed adapt_files={len(view)} alpha_equivalence_fail={len(failing)}")
    if args.write_failing is not None:
        args.write_failing.parent.mkdir(parents=True, exist_ok=True)
        args.write_failing.write_text(
            "".join(f"{rel}\n" for rel, _, _ in failing), encoding="utf-8"
        )
        print(f"wrote failing list: {args.write_failing}")

    counts: Counter[str] = Counter()
    repos: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    details: dict[str, Counter[str]] = defaultdict(Counter)
    for rel, orig, new in failing:
        info = diagnose_pair(orig, new)
        if info is None:
            continue
        cat = info["category"]
        counts[cat] += 1
        repos[cat][rel.split("/", 1)[0]] += 1
        details[cat][info["detail"].split(" ", 1)[0].rstrip(":")] += 1
        if len(examples[cat]) < 2:
            examples[cat].append((rel, info))

    print("\n=== failures by category (first differing node) ===")
    for cat in CATEGORIES:
        print(f"{counts.get(cat, 0):5d}  {cat}")
    for cat in CATEGORIES:
        if not counts.get(cat):
            continue
        print(f"\n--- {cat} ({counts[cat]}) ---")
        top = ", ".join(f"{r}={n}" for r, n in repos[cat].most_common(3))
        print(f"top repos: {top}")
        nodes = ", ".join(f"{d}={n}" for d, n in details[cat].most_common(5))
        print(f"node.field: {nodes}")
        for rel, info in examples[cat]:
            print(f"  file: {rel}  ({info['detail']})")
            print(f"    original L{info['line_a']}: {info['snippet_a']}")
            print(f"    renamed  L{info['line_b']}: {info['snippet_b']}")

    print("\n=== libcst-unparseable adapt files (copied unchanged) ===")
    n_bad = 0
    for rel, orig in unchanged:
        try:
            cst.parse_module(orig)
        except (cst.ParserSyntaxError, cst.CSTLogicError) as exc:
            n_bad += 1
            msg = str(exc).strip().splitlines()
            first = msg[0] if msg else type(exc).__name__
            print(f"  {rel}: {type(exc).__name__}: {first}")
    print(f"libcst_unparseable={n_bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
