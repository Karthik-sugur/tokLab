"""Build AdaptedVocab from an adapt-only ManifestView."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toklab.data.splits import ManifestView
from toklab.tokenizers.native_hf import NativeHFTokenizer
from toklab.vocab.adapted_vocab import AdaptedVocab, empty_adapted_vocab
from toklab.vocab.mining import MineMode, mine_candidates
from toklab.vocab.selection import select_vocab


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
    """Mine + select + assign append-only ids (offset = base.vocab_size)."""
    if view.split != "adapt":
        raise PermissionError(
            f"build_adapted_vocab requires adapt split, got {view.split!r}"
        )
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


def shared_base_from_cfg(cfg: dict[str, Any]) -> NativeHFTokenizer:
    """Load the shared C2 base tokenizer from config (same instance config)."""
    model_id = cfg.get("tokenizer_name") or cfg["model_name"]
    return NativeHFTokenizer.from_pretrained(str(model_id), name="C2")
