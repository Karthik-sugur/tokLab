"""Lossless round-trip and parse-fidelity checks."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from toklab.core.interfaces import Tokenizer
from toklab.core.source_io import read_source_text
from toklab.core.types import SourceFile


@dataclass
class FidelityReport:
    """Summary of round-trip and re-parse checks for one tokenizer."""

    tokenizer_name: str
    n_files: int = 0
    n_roundtrip_ok: int = 0
    n_roundtrip_fail: int = 0
    n_orig_parsed: int = 0
    n_reparse_ok: int = 0
    n_reparse_fail: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def all_roundtrips_ok(self) -> bool:
        return self.n_roundtrip_fail == 0 and self.n_files == self.n_roundtrip_ok


def _load_text(repos_dir: Path, sf: SourceFile) -> str:
    return read_source_text(repos_dir / sf.repo / sf.path)


def check_roundtrip(
    tokenizer: Tokenizer,
    files: Sequence[SourceFile],
    *,
    texts: dict[str, str] | None = None,
    repos_dir: Path | str | None = None,
) -> FidelityReport:
    """Check ``decode(encode(x)) == x`` and re-``ast.parse`` when original parses.

    Provide either ``texts`` (file_id → source) or ``repos_dir`` to load files.
    """
    report = FidelityReport(tokenizer_name=tokenizer.name)
    root = Path(repos_dir) if repos_dir is not None else None

    for sf in files:
        report.n_files += 1
        if texts is not None:
            text = texts[sf.file_id]
        elif root is not None:
            text = _load_text(root, sf)
        else:
            raise ValueError("provide texts= or repos_dir= to load source")

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        if decoded == text:
            report.n_roundtrip_ok += 1
        else:
            report.n_roundtrip_fail += 1
            report.failures.append(
                f"roundtrip:{sf.file_id}:{sf.repo}/{sf.path} "
                f"len_in={len(text)} len_out={len(decoded)}"
            )
            continue

        # Re-parse only when the original parses as Python.
        try:
            ast.parse(text)
        except SyntaxError:
            continue
        report.n_orig_parsed += 1
        try:
            ast.parse(decoded)
            report.n_reparse_ok += 1
        except SyntaxError as exc:
            report.n_reparse_fail += 1
            report.failures.append(
                f"reparse:{sf.file_id}:{sf.repo}/{sf.path}: {exc}"
            )

    return report


def roundtrip_with_counts(
    tokenizer: Tokenizer,
    files: Sequence[SourceFile],
    texts: dict[str, str],
    *,
    orig_parses: dict[str, bool] | None = None,
) -> tuple[FidelityReport, dict[str, int]]:
    """``check_roundtrip`` plus token count per file, encoding each file once.

    Same check as :func:`check_roundtrip` (``decode(encode(x)) == x``). When
    the round trip is exact the decoded text *is* the original, so it re-parses
    exactly when the original parses; ``orig_parses`` (file_id → parses) lets
    callers compute that once per file instead of once per tokenizer.
    """
    report = FidelityReport(tokenizer_name=tokenizer.name)
    counts: dict[str, int] = {}
    for sf in files:
        report.n_files += 1
        text = texts[sf.file_id]
        encoded = tokenizer.encode(text)
        counts[sf.file_id] = len(encoded.ids)
        decoded = tokenizer.decode(encoded.ids)
        if decoded != text:
            report.n_roundtrip_fail += 1
            report.failures.append(
                f"roundtrip:{sf.file_id}:{sf.repo}/{sf.path} "
                f"len_in={len(text)} len_out={len(decoded)}"
            )
            continue
        report.n_roundtrip_ok += 1
        parses = (
            orig_parses[sf.file_id]
            if orig_parses is not None and sf.file_id in orig_parses
            else source_parses(text)
        )
        if parses:
            report.n_orig_parsed += 1
            report.n_reparse_ok += 1
    return report, counts


def source_parses(text: str) -> bool:
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def check_roundtrip_text(tokenizer: Tokenizer, texts: Iterable[str]) -> list[str]:
    """Return failure messages for any text that does not round-trip."""
    failures: list[str] = []
    for i, text in enumerate(texts):
        decoded = tokenizer.decode(tokenizer.encode(text).ids)
        if decoded != text:
            failures.append(f"index={i} len_in={len(text)} len_out={len(decoded)}")
    return failures
