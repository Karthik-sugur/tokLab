"""Intrinsic tokenizer metrics (compression, tokens per KB)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from toklab.core.interfaces import Tokenizer
from toklab.core.types import SourceFile


def original_n_bytes(text: str) -> int:
    """UTF-8 byte length of the original source (golden rule 4)."""
    return len(text.encode("utf-8"))


def compression_ratio(n_bytes: int, n_tokens: int) -> float:
    """``n_bytes / n_tokens``; higher is better. Undefined if n_tokens==0."""
    if n_tokens == 0:
        if n_bytes == 0:
            return float("nan")
        raise ZeroDivisionError("n_tokens is 0 for non-empty source")
    return n_bytes / n_tokens


def tokens_per_kb(n_tokens: int, n_bytes: int) -> float:
    """Tokens per kilobyte of original UTF-8 bytes."""
    if n_bytes == 0:
        return float("nan")
    return n_tokens * 1024.0 / n_bytes


def measure_file(
    tokenizer: Tokenizer,
    file_id: str,
    text: str,
    *,
    condition: str | None = None,
) -> list[dict[str, object]]:
    """Return metric rows for one file: n_tokens, n_bytes, CR, tokens_per_kb."""
    cond = condition if condition is not None else tokenizer.name
    n_bytes = original_n_bytes(text)
    n_tokens = len(tokenizer.encode(text).ids)
    cr = compression_ratio(n_bytes, n_tokens) if n_tokens or n_bytes == 0 else float(
        "nan"
    )
    tpk = tokens_per_kb(n_tokens, n_bytes)
    return [
        {
            "condition": cond,
            "file_id": file_id,
            "metric": "n_tokens",
            "value": n_tokens,
        },
        {
            "condition": cond,
            "file_id": file_id,
            "metric": "n_bytes",
            "value": n_bytes,
        },
        {
            "condition": cond,
            "file_id": file_id,
            "metric": "compression_ratio",
            "value": cr,
        },
        {
            "condition": cond,
            "file_id": file_id,
            "metric": "tokens_per_kb",
            "value": tpk,
        },
    ]


def measure_files(
    tokenizer: Tokenizer,
    files: Sequence[SourceFile],
    *,
    texts: dict[str, str],
    condition: str | None = None,
) -> pd.DataFrame:
    """Build a long-form metrics table for many files."""
    rows: list[dict[str, object]] = []
    for sf in files:
        rows.extend(
            measure_file(
                tokenizer,
                sf.file_id,
                texts[sf.file_id],
                condition=condition,
            )
        )
    return pd.DataFrame(rows, columns=["condition", "file_id", "metric", "value"])


def write_metrics_parquet(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def context_utilisation(
    files: Sequence[tuple[str, int, int]],
    budgets: Sequence[int],
) -> pd.DataFrame:
    """Greedy first-fit packing of files into token budgets.

    ``files`` is an ordered sequence of ``(file_id, n_tokens, n_bytes)``.
    For each budget ``C``, walk files in that fixed order and include a file
    when ``used + n_tokens <= C`` (skip if it does not fit; continue).

    Returns rows: budget, files_fit, bytes_fit, tokens_used.
    """
    rows: list[dict[str, object]] = []
    ordered = list(files)
    for c in budgets:
        used = 0
        n_files = 0
        n_bytes = 0
        for _fid, n_tok, nbytes in ordered:
            if n_tok < 0:
                raise ValueError("n_tokens must be non-negative")
            if used + n_tok <= c:
                used += n_tok
                n_files += 1
                n_bytes += nbytes
        rows.append(
            {
                "budget_tokens": int(c),
                "files_fit": n_files,
                "bytes_fit": n_bytes,
                "tokens_used": used,
            }
        )
    return pd.DataFrame(rows)


def context_utilisation_for_tokenizer(
    tokenizer: Tokenizer,
    source_files: Sequence[SourceFile],
    texts: dict[str, str],
    budgets: Sequence[int],
    *,
    condition: str | None = None,
) -> pd.DataFrame:
    """Compute context utilisation using a fixed file order (sorted file_id)."""
    ordered_sf = sorted(source_files, key=lambda f: f.file_id)
    packed: list[tuple[str, int, int]] = []
    for sf in ordered_sf:
        text = texts[sf.file_id]
        n_tok = len(tokenizer.encode(text).ids)
        n_bytes = original_n_bytes(text)
        packed.append((sf.file_id, n_tok, n_bytes))
    df = context_utilisation(packed, budgets)
    cond = condition if condition is not None else tokenizer.name
    df.insert(0, "condition", cond)
    return df
