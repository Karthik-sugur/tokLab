"""Row filters for stored result tables (display only; no metrics computed)."""

from __future__ import annotations

from typing import Any

Row = dict[str, Any]


def eval_sets(rows: list[Row]) -> list[str]:
    """Eval sets in first-seen order (the runner writes the primary set first)."""
    seen: list[str] = []
    for r in rows:
        s = r.get("eval_set")
        if s is not None and str(s) not in seen:
            seen.append(str(s))
    return seen


def k_values(rows: list[Row]) -> list[int]:
    return sorted({int(r["k"]) for r in rows if r.get("k") is not None})


def k_star(factorial_rows: list[Row]) -> int | None:
    """K* as flagged by the runner in factorial_effects (None if not flagged)."""
    ks = {int(r["k"]) for r in factorial_rows if r.get("k_star") is True}
    return ks.pop() if len(ks) == 1 else None


def filter_rows(
    rows: list[Row], eval_set: str | None = None, k: int | None = None
) -> list[Row]:
    """Rows of one eval set and one K; tables without the column pass through."""
    out = []
    for r in rows:
        if eval_set is not None and "eval_set" in r and str(r["eval_set"]) != eval_set:
            continue
        if k is not None and r.get("k") is not None and int(r["k"]) != k:
            continue
        out.append(r)
    return out


def k_label(k: int, kstar: int | None) -> str:
    return f"{k} (K*)" if kstar is not None and k == kstar else str(k)
