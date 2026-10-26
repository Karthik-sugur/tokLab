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
