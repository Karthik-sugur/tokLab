"""Vocabulary / KV-cache / FLOPs cost estimators (pure functions).

All FLOPs / attention figures are labelled ESTIMATE: they use simplified
transformer accounting, not measured hardware counters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VocabCostEstimate:
    """Parameter and memory overhead from appending ``delta_v`` vocab rows."""

    delta_v: int
    d_model: int
    tie_word_embeddings: bool
    delta_params: int
    bytes_per_param: int
    delta_bytes: int
    label: str = "ESTIMATE"


@dataclass(frozen=True)
class KvCacheEstimate:
    """KV-cache bytes for a single sequence of length ``T`` (ESTIMATE)."""

    layers: int
    kv_heads: int
    head_dim: int
    seq_len: int
    bytes_per_elem: int
    kv_cache_bytes: int
    label: str = "ESTIMATE"


@dataclass(frozen=True)
class FlopsEstimate:
    """Per-token FLOPs ESTIMATE for MLP + attention (see docstring)."""

    d_model: int
    layers: int
    seq_len: int
    flops_mlp_per_token: float
    flops_attn_per_token: float
    flops_per_token_estimate: float
    label: str = "ESTIMATE"


def delta_params(
    delta_v: int,
    d_model: int,
    *,
    tie_word_embeddings: bool,
) -> int:
    """``delta_V * d_model * (2 if untied else 1)``."""
    if delta_v < 0 or d_model <= 0:
        raise ValueError("delta_v >= 0 and d_model > 0 required")
    factor = 1 if tie_word_embeddings else 2
    return int(delta_v * d_model * factor)


def delta_bytes(n_params: int, *, bytes_per_param: int = 2) -> int:
    """Storage bytes for ``n_params`` (default 2 = fp16)."""
    if n_params < 0 or bytes_per_param <= 0:
        raise ValueError("n_params >= 0 and bytes_per_param > 0")
    return int(n_params * bytes_per_param)


def kv_cache_bytes(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    bytes_per_elem: int = 2,
) -> int:
    """``2 * layers * kv_heads * head_dim * T * bytes_per_elem``.

    The leading 2 accounts for separate K and V caches.
    """
    if min(layers, kv_heads, head_dim, seq_len, bytes_per_elem) <= 0:
        raise ValueError("all kv_cache dimensions must be positive")
    return int(2 * layers * kv_heads * head_dim * seq_len * bytes_per_elem)


def flops_per_token_estimate(
    *,
    d_model: int,
    layers: int,
    seq_len: int,
    ffn_mult: float = 4.0,
) -> FlopsEstimate:
    """ESTIMATE FLOPs per generated token (forward, decode step).

    Formulas (documented, not measured):

    * MLP per layer per token: ``2 * d_model * (ffn_mult * d_model) * 2``
      ≈ matmul up + matmul down (multiply-add counted as 2 FLOPs each way
      simplified to ``2 * d * d_ff * 2`` with ``d_ff = ffn_mult * d``).
      We use ``flops_mlp = layers * 2 * d_model * (ffn_mult * d_model)``.

    * Attention per layer per token (decode, attending to ``T`` keys):
      ``flops_attn = layers * (4 * d_model * d_model + 2 * seq_len * d_model)``
      — QKV/out projections ≈ ``4 * d^2``, score+weighted sum ≈ ``2 * T * d``.

    Total: ``flops_mlp + flops_attn`` (ESTIMATE).
    """
    if d_model <= 0 or layers <= 0 or seq_len <= 0:
        raise ValueError("d_model, layers, seq_len must be positive")
    d_ff = ffn_mult * d_model
    flops_mlp = float(layers * 2 * d_model * d_ff)
    flops_attn = float(
        layers * (4 * d_model * d_model + 2 * seq_len * d_model)
    )
    return FlopsEstimate(
        d_model=d_model,
        layers=layers,
        seq_len=seq_len,
        flops_mlp_per_token=flops_mlp,
        flops_attn_per_token=flops_attn,
        flops_per_token_estimate=flops_mlp + flops_attn,
        label="ESTIMATE",
    )


def estimate_vocab_cost(
    *,
    delta_v: int,
    d_model: int,
    tie_word_embeddings: bool,
    bytes_per_param: int = 2,
) -> VocabCostEstimate:
    """Parameter + byte overhead for ``delta_v`` new embedding rows."""
    dp = delta_params(
        delta_v, d_model, tie_word_embeddings=tie_word_embeddings
    )
    return VocabCostEstimate(
        delta_v=delta_v,
        d_model=d_model,
        tie_word_embeddings=tie_word_embeddings,
        delta_params=dp,
        bytes_per_param=bytes_per_param,
        delta_bytes=delta_bytes(dp, bytes_per_param=bytes_per_param),
        label="ESTIMATE",
    )


def estimate_kv_cache(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    bytes_per_elem: int = 2,
) -> KvCacheEstimate:
    return KvCacheEstimate(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        bytes_per_elem=bytes_per_elem,
        kv_cache_bytes=kv_cache_bytes(
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            bytes_per_elem=bytes_per_elem,
        ),
        label="ESTIMATE",
    )


def as_plain_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
