"""Quality metrics: bits-per-byte (BPB) and identifier-completion EM.

Quality rows are NEVER produced for conditions whose gate verdict is
``UNTRAINED_NEW_ROWS`` or ``INCOMPATIBLE`` (golden rule 8).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from toklab.models.gate import GateResult, GateVerdict

# Identifiers: Python-like names for completion probes.
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b")


class QualitySuppressedError(RuntimeError):
    """Raised when quality is requested for a gated-out condition."""


def bits_per_byte(*, nll_nats_sum: float, n_bytes: int) -> float:
    """BPB = (sum of NLL in nats / ln 2) / original UTF-8 bytes.

    ``nll_nats_sum`` must be the sum of per-token negative log-likelihoods
    in nats over **all** predicted tokens in the evaluation texts.
    ``n_bytes`` is the sum of original UTF-8 byte lengths (golden rule 4/7).
    """
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")
    if not math.isfinite(nll_nats_sum):
        raise ValueError("nll_nats_sum must be finite")
    return (nll_nats_sum / math.log(2.0)) / float(n_bytes)


def assert_quality_allowed(gate_result: GateResult | GateVerdict) -> None:
    """Refuse quality for UNTRAINED_NEW_ROWS / INCOMPATIBLE."""
    verdict = (
        gate_result.verdict
        if isinstance(gate_result, GateResult)
        else gate_result
    )
    if verdict in {
        GateVerdict.UNTRAINED_NEW_ROWS,
        GateVerdict.INCOMPATIBLE,
    }:
        raise QualitySuppressedError(
            f"quality suppressed for gate verdict {verdict.value} "
            "(golden rule 8: no untrained / incompatible quality rows)"
        )


def causal_nll_nats_sum(
    model: Any,
    input_ids: Any,
    *,
    ignore_index: int = -100,
) -> float:
    """Sum of per-token NLL in nats for a single causal sequence (batch=1).

    Uses teacher forcing: predicts token t from tokens < t. The first token
    has no prediction and contributes 0. Counts every remaining token.
    """
    import torch
    import torch.nn.functional as F

    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("input_ids must have shape (1, T)")
    if input_ids.size(1) < 2:
        return 0.0

    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits  # (1, T, V)
        # Predict token i+1 from position i.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        nll = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
    return float(nll.item())


def bpb_for_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: str | None = None,
) -> tuple[float, float, int]:
    """Return ``(bpb, nll_nats_sum, n_bytes)`` over all texts / all tokens."""
    import torch

    if device is None:
        device = str(next(model.parameters()).device)
    nll_sum = 0.0
    n_bytes = 0
    for text in texts:
        raw = text.encode("utf-8")
        n_bytes += len(raw)
        ids = tokenizer.encode(text).ids
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        nll_sum += causal_nll_nats_sum(model, input_ids)
    if n_bytes <= 0:
        raise ValueError("no bytes in texts")
    return bits_per_byte(nll_nats_sum=nll_sum, n_bytes=n_bytes), nll_sum, n_bytes


@dataclass(frozen=True)
class CompletionProbe:
    """One identifier-completion example."""

    context: str
    target: str
    file_id: str = ""


@dataclass
class QualityReport:
    condition: str
    gate_verdict: str
    bpb: float
    nll_nats_sum: float
    n_bytes: int
    completion_exact_match: float | None
    n_completion: int
    notes: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_identifier_probes(
    texts: Sequence[tuple[str, str]],
    *,
    min_len: int = 4,
) -> list[CompletionProbe]:
    """Build completion probes from ``(file_id, source)`` synthetic files.

    For each long identifier occurrence, the probe context is the source with
    that occurrence removed (prefix kept); target is the identifier string.
    """
    probes: list[CompletionProbe] = []
    for file_id, text in texts:
        for m in _IDENT_RE.finditer(text):
            name = m.group(1)
            if len(name) < min_len:
                continue
            # Skip all-lowercase shortish builtins-ish.
            if name.islower() and len(name) < 8:
                continue
            start = m.start(1)
            probes.append(
                CompletionProbe(
                    context=text[:start],
                    target=name,
                    file_id=file_id,
                )
            )
    return probes


def identifier_completion_exact_match(
    model: Any,
    tokenizer: Any,
    probes: Sequence[CompletionProbe],
    *,
    device: str | None = None,
    max_new_tokens: int | None = None,
) -> tuple[float, int]:
    """Greedy generate ``len(target)``-ish tokens; exact-match on decoded text.

    Uses character-length of the target as a stop heuristic via encoding the
    target with the same tokenizer to know how many ids to generate.
    """
    import torch

    if not probes:
        return float("nan"), 0
    if device is None:
        device = str(next(model.parameters()).device)
    model.eval()
    n_ok = 0
    for probe in probes:
        target_ids = tokenizer.encode(probe.target).ids
        if not target_ids:
            continue
        n_gen = max_new_tokens or len(target_ids)
        ctx_ids = tokenizer.encode(probe.context).ids
        if not ctx_ids:
            continue
        input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        generated: list[int] = []
        with torch.no_grad():
            cur = input_ids
            for _ in range(n_gen):
                out = model(input_ids=cur)
                next_id = int(out.logits[0, -1, :].argmax().item())
                generated.append(next_id)
                cur = torch.tensor(
                    [ctx_ids + generated], dtype=torch.long, device=device
                )
        pred = tokenizer.decode(generated)
        if pred == probe.target:
            n_ok += 1
    return n_ok / len(probes), len(probes)


def evaluate_quality(
    *,
    condition: str,
    model: Any,
    tokenizer: Any,
    gate_result: GateResult,
    texts: Sequence[str],
    completion_sources: Sequence[tuple[str, str]] | None = None,
    device: str | None = None,
) -> QualityReport:
    """Compute BPB (+ optional completion EM). Refuses gated-out conditions."""
    assert_quality_allowed(gate_result)
    bpb, nll_sum, n_bytes = bpb_for_texts(
        model, tokenizer, texts, device=device
    )
    em: float | None = None
    n_comp = 0
    if completion_sources:
        probes = build_identifier_probes(list(completion_sources))
        if probes:
            em, n_comp = identifier_completion_exact_match(
                model, tokenizer, probes, device=device
            )
    return QualityReport(
        condition=condition,
        gate_verdict=gate_result.verdict.value,
        bpb=bpb,
        nll_nats_sum=nll_sum,
        n_bytes=n_bytes,
        completion_exact_match=em,
        n_completion=n_comp,
        notes="BPB over all tokens / original UTF-8 bytes",
    )
