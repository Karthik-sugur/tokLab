"""Compare token counts across conditions on a manifest split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toklab.core.types import SourceFile
from toklab.data.splits import ManifestView
from toklab.tokenizers.registry import get_tokenizer


def _load_texts(repos_dir: Path, files: tuple[SourceFile, ...]) -> dict[str, str]:
    return {
        sf.file_id: (repos_dir / sf.repo / sf.path).read_text(encoding="utf-8")
        for sf in files
    }


def compare_conditions(
    *,
    manifest: Path | str,
    repos_dir: Path | str,
    conditions: list[str],
    split: str = "test",
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return a plain-text table of total / mean tokens per condition."""
    repos_dir = Path(repos_dir)
    view = ManifestView.from_parquet(manifest, split)
    files = view.files()
    if not files:
        raise ValueError(f"no files in split {split!r}")
    texts = _load_texts(repos_dir, files)

    lines = [
        f"{'condition':<10} {'files':>6} {'total_tokens':>14} {'mean_tokens':>12}",
        "-" * 46,
    ]
    for cond in conditions:
        tok = get_tokenizer(cond, cfg)
        totals: list[int] = []
        for sf in files:
            n = len(tok.encode(texts[sf.file_id]).ids)
            totals.append(n)
        total = sum(totals)
        mean = total / len(totals)
        lines.append(f"{cond:<10} {len(totals):6d} {total:14d} {mean:12.2f}")
    return "\n".join(lines)
