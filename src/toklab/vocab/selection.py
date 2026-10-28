"""Select top-K adapted vocabulary entries by compression gain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from toklab.vocab.mining import CandidateStats


@dataclass(frozen=True)
class ScoredCandidate:
    token: str
    freq: int
    n_repos: int
    n_base_tokens: int
    gain: float


def base_token_count(token: str, base_tokenizer: Any) -> int:
    """Number of base-BPE tokens for ``token`` alone (no special tokens)."""
    encode = getattr(base_tokenizer, "encode", None)
    if encode is not None:
        result = encode(token)
        ids = result.ids if hasattr(result, "ids") else result
        return len(ids)
    # Raw HF tokenizer
    return len(base_tokenizer.encode(token, add_special_tokens=False))


def score_candidates(
    candidates: list[CandidateStats],
    base_tokenizer: Any,
    *,
    min_freq: int = 1,
    min_repos: int = 1,
) -> list[ScoredCandidate]:
    """Filter and score candidates: ``gain = freq * (n_base_tokens - 1)``."""
    scored: list[ScoredCandidate] = []
    for c in candidates:
        if c.freq < min_freq or c.n_repos < min_repos:
            continue
        n_tok = base_token_count(c.token, base_tokenizer)
        gain = float(c.freq * (n_tok - 1))
        if gain <= 0:
            continue
        scored.append(
            ScoredCandidate(
                token=c.token,
                freq=c.freq,
                n_repos=c.n_repos,
                n_base_tokens=n_tok,
                gain=gain,
            )
        )
    # Deterministic: gain desc, then alphabetical token.
    scored.sort(key=lambda s: (-s.gain, s.token))
    return scored


def select_vocab(
    candidates: list[CandidateStats],
    base_tokenizer: Any,
    *,
    k: int,
    min_freq: int = 1,
    min_repos: int = 1,
) -> list[ScoredCandidate]:
    """Pick top ``k`` candidates by gain (alphabetical tie-break)."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    scored = score_candidates(
        candidates,
        base_tokenizer,
        min_freq=min_freq,
        min_repos=min_repos,
    )
    return scored[:k]
