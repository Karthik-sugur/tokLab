"""Mine identifier candidates from the ADAPT split only."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from toklab.core.types import SourceFile
from toklab.data.splits import ManifestView
from toklab.parsing.segmenters import NativeSegmenter, TreeSitterSegmenter

MineMode = Literal["native", "structural"]


@dataclass(frozen=True)
class CandidateStats:
    """Frequency stats for one mined identifier string."""

    token: str
    freq: int
    n_repos: int


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _join_native_identifiers(text: str, spans: list[tuple[int, int]]) -> list[str]:
    """Join adjacent native pre-token spans that form a valid identifier."""
    out: list[str] = []
    i = 0
    n = len(spans)
    while i < n:
        start, end = spans[i]
        if end <= start:
            i += 1
            continue
        first = text[start:end]
        if not first or not (first[0].isalpha() or first[0] == "_"):
            i += 1
            continue
        j = i
        cur_end = end
        while j + 1 < n and spans[j + 1][0] == cur_end:
            nxt_end = spans[j + 1][1]
            trial = text[start:nxt_end]
            if trial and all(_is_ident_char(c) for c in trial):
                j += 1
                cur_end = nxt_end
            else:
                break
        candidate = text[start:cur_end]
        if candidate.isidentifier():
            out.append(candidate)
        i = j + 1
    return out


def _native_candidates(text: str, hf_tokenizer: Any) -> list[str]:
    segments = NativeSegmenter(hf_tokenizer).segment(text)
    spans = [(s.start, s.end) for s in segments if s.end > s.start]
    return _join_native_identifiers(text, spans)


def _structural_candidates(text: str, ts: TreeSitterSegmenter) -> list[str]:
    out: list[str] = []
    for s in ts.segment(text):
        if s.kind != "ident":
            continue
        piece = text[s.start : s.end]
        if piece.isidentifier():
            out.append(piece)
    return out


def mine_candidates(
    view: ManifestView,
    *,
    repos_dir: Path | str,
    mode: MineMode,
    base_tokenizer: Any,
    tree_segmenter: TreeSitterSegmenter | None = None,
) -> list[CandidateStats]:
    """Count identifier candidates from an ADAPT-only ``ManifestView``.

    Raises:
        PermissionError: if ``view.split`` is not ``"adapt"``.
    """
    if view.split != "adapt":
        raise PermissionError(
            f"vocabulary mining requires ManifestView(adapt); "
            f"got split={view.split!r}"
        )
    repos_dir = Path(repos_dir)
    hf = getattr(base_tokenizer, "_tok", base_tokenizer)
    ts = tree_segmenter or TreeSitterSegmenter()

    freq: dict[str, int] = defaultdict(int)
    repos_for: dict[str, set[str]] = defaultdict(set)

    for sf in view.files():
        text = (repos_dir / sf.repo / sf.path).read_text(encoding="utf-8")
        if mode == "native":
            cands = _native_candidates(text, hf)
        elif mode == "structural":
            cands = _structural_candidates(text, ts)
        else:
            raise ValueError(f"unknown mine mode {mode!r}")
        for c in cands:
            freq[c] += 1
            repos_for[c].add(sf.repo)

    return [
        CandidateStats(token=t, freq=freq[t], n_repos=len(repos_for[t]))
        for t in sorted(freq.keys())
    ]


def mine_from_files(
    files: list[SourceFile],
    texts: dict[str, str],
    *,
    mode: MineMode,
    base_tokenizer: Any,
    required_split: str = "adapt",
) -> list[CandidateStats]:
    """Mine from files; raises if any file is not ``required_split``."""
    bad = [f for f in files if f.split != required_split]
    if bad:
        raise PermissionError(
            f"refusing to mine from non-{required_split} files: "
            f"{bad[0].repo}/{bad[0].path} split={bad[0].split!r}"
        )
    hf = getattr(base_tokenizer, "_tok", base_tokenizer)
    ts = TreeSitterSegmenter()
    freq: dict[str, int] = defaultdict(int)
    repos_for: dict[str, set[str]] = defaultdict(set)
    for sf in files:
        text = texts[sf.file_id]
        cands = (
            _native_candidates(text, hf)
            if mode == "native"
            else _structural_candidates(text, ts)
        )
        for c in cands:
            freq[c] += 1
            repos_for[c].add(sf.repo)
    return [
        CandidateStats(token=t, freq=freq[t], n_repos=len(repos_for[t]))
        for t in sorted(freq.keys())
    ]
