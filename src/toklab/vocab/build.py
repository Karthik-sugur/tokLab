"""Build AdaptedVocab from an adapt-only ManifestView."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toklab.data.splits import ManifestView
from toklab.tokenizers.native_hf import NativeHFTokenizer
from toklab.vocab.adapted_vocab import AdaptedVocab, empty_adapted_vocab
from toklab.vocab.mining import MineMode, mine_candidates
from toklab.vocab.selection import ScoredCandidate, score_candidates, select_vocab


def _require_adapt(view: ManifestView, what: str) -> None:
    if view.split != "adapt":
        raise PermissionError(f"{what} requires adapt split, got {view.split!r}")


def build_adapted_vocab(
    view: ManifestView,
    *,
    repos_dir: Path | str,
    base: NativeHFTokenizer,
    mode: MineMode,
    k: int,
    min_freq: int = 1,
    min_repos: int = 1,
    base_model: str = "",
) -> AdaptedVocab:
    """Mine + select + assign append-only ids (offset = base.vocab_size).

    ``repos_dir`` holds ``<repo>/<path>`` for the adapt files (for a synthetic
    corpus that is ``<synthetic>/adapt``).
    """
    _require_adapt(view, "build_adapted_vocab")
    id_offset = base.vocab_size
    if k == 0:
        return empty_adapted_vocab(id_offset, mode=mode, base_model=base_model)
    candidates = mine_candidates(
        view,
        repos_dir=repos_dir,
        mode=mode,
        base_tokenizer=base,
    )
    selected = select_vocab(
        candidates,
        base,
        k=k,
        min_freq=min_freq,
        min_repos=min_repos,
    )
    return AdaptedVocab.from_selected(
        selected,
        id_offset=id_offset,
        mode=mode,
        base_model=base_model,
    )


def scored_adapt_candidates(
    view: ManifestView,
    *,
    repos_dir: Path | str,
    base: NativeHFTokenizer,
    mode: MineMode,
    min_freq: int = 1,
    min_repos: int = 1,
) -> list[ScoredCandidate]:
    """All positive-gain adapt candidates, sorted as ``select_vocab`` ranks them.

    ``vocab_from_scored(scored, k)`` equals ``build_adapted_vocab(..., k=k)``
    for every ``k``, so a K sweep mines the adapt split once per mode.
    """
    _require_adapt(view, "scored_adapt_candidates")
    candidates = mine_candidates(
        view, repos_dir=repos_dir, mode=mode, base_tokenizer=base
    )
    return score_candidates(
        candidates, base, min_freq=min_freq, min_repos=min_repos
    )


def vocab_from_scored(
    scored: list[ScoredCandidate],
    *,
    k: int,
    id_offset: int,
    mode: MineMode,
    base_model: str = "",
) -> AdaptedVocab:
    """Top-``k`` of pre-scored candidates as an append-only vocabulary."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k == 0:
        return empty_adapted_vocab(id_offset, mode=mode, base_model=base_model)
    return AdaptedVocab.from_selected(
        scored[:k], id_offset=id_offset, mode=mode, base_model=base_model
    )


def shared_base_from_cfg(cfg: dict[str, Any]) -> NativeHFTokenizer:
    """Load the shared C2 base tokenizer from config (same instance config)."""
    model_id = cfg.get("tokenizer_name") or cfg["model_name"]
    return NativeHFTokenizer.from_pretrained(str(model_id), name="C2")
