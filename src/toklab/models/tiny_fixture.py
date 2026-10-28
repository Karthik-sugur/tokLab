"""Tiny random-init causal LM fixture for CPU tests and dry-runs.

Does not download Hugging Face weights. Not used for published research
claims — fixture-only. Round-trip is lossless for ASCII source text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from toklab.core.types import EncodeResult
from toklab.vocab.adapted_vocab import AdaptedEntry, AdaptedVocab

# ASCII codepoints 0..127 are base ids; adapted ids append at >= 128.
ASCII_BASE = 128


@dataclass
class TinyBundle:
    model: Any
    hf_tokenizer: Any
    tokenizer: ByteFallbackTokenizer
    vocab_size: int
    tie_word_embeddings: bool
    id_offset: int


class ByteFallbackTokenizer:
    """ASCII-lossless fixture tokenizer with optional adapted whole-token ids."""

    def __init__(
        self,
        *,
        vocab_size: int,
        name: str = "tiny",
        adapted: dict[str, int] | None = None,
        id_offset: int = ASCII_BASE,
    ) -> None:
        if vocab_size < ASCII_BASE:
            raise ValueError("vocab_size must be >= 128 for ASCII base ids")
        self.vocab_size = vocab_size
        self.name = name
        self._adapted = dict(adapted or {})
        self.id_offset = id_offset

    def encode(self, text: str) -> EncodeResult:
        ids: list[int] = []
        spans: list[tuple[int, int]] = []
        entries = sorted(self._adapted.keys(), key=lambda s: (-len(s), s))
        i = 0
        while i < len(text):
            matched = False
            for tok in entries:
                if text.startswith(tok, i):
                    ids.append(self._adapted[tok])
                    spans.append((i, i + len(tok)))
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            code = ord(ch)
            if code >= ASCII_BASE:
                raise ValueError(
                    "tiny fixture tokenizer only round-trips ASCII; "
                    f"got codepoint {code}"
                )
            ids.append(code)
            spans.append((i, i + 1))
            i += 1
        return EncodeResult(ids=ids, spans=spans)

    def decode(self, ids: list[int]) -> str:
        rev = {v: k for k, v in self._adapted.items()}
        out: list[str] = []
        for tid in ids:
            if tid in rev:
                out.append(rev[tid])
            elif 0 <= tid < ASCII_BASE:
                out.append(chr(tid))
            else:
                raise ValueError(f"unknown id {tid} in tiny decode")
        return "".join(out)

    def with_adapted(self, adapted: AdaptedVocab) -> ByteFallbackTokenizer:
        max_id = max((e.id for e in adapted.entries), default=self.vocab_size - 1)
        return ByteFallbackTokenizer(
            vocab_size=max(self.vocab_size, max_id + 1),
            name=self.name,
            adapted=adapted.as_dict(),
            id_offset=adapted.id_offset,
        )


class _HFStubTokenizer:
    """Minimal encode API for adapt_embeddings BPE decomposition."""

    def __init__(self, id_offset: int = ASCII_BASE) -> None:
        self.id_offset = id_offset

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(c) for c in text if ord(c) < self.id_offset]

    def __len__(self) -> int:
        return self.id_offset


def build_tiny_bundle(
    *,
    seed: int = 0,
    vocab_size: int = ASCII_BASE,
    n_embd: int = 32,
    n_layer: int = 2,
    n_head: int = 2,
    n_positions: int = 64,
    tie_word_embeddings: bool = True,
) -> TinyBundle:
    """Construct a tiny GPT-2-config causal LM with random weights."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_embd * 4,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    tok = ByteFallbackTokenizer(
        vocab_size=vocab_size, name="tiny", id_offset=ASCII_BASE
    )
    return TinyBundle(
        model=model,
        hf_tokenizer=_HFStubTokenizer(ASCII_BASE),
        tokenizer=tok,
        vocab_size=vocab_size,
        tie_word_embeddings=tie_word_embeddings,
        id_offset=ASCII_BASE,
    )


def tiny_adapted_vocab(
    bundle: TinyBundle, tokens: list[str] | None = None
) -> AdaptedVocab:
    """Append-only adapted vocab for the tiny fixture (ids >= vocab_size)."""
    tokens = tokens or ["AcmeWidgetFactory", "HelperTool"]
    entries = [
        AdaptedEntry(
            token=t,
            id=bundle.vocab_size + i,
            gain=1.0,
            freq=1,
            n_repos=1,
        )
        for i, t in enumerate(tokens)
    ]
    return AdaptedVocab(
        id_offset=bundle.vocab_size,
        entries=entries,
        mode="native",
        base_model="tiny_fixture",
    )
