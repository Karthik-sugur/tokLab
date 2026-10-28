"""Model × tokenizer compatibility gate (honest quality eligibility)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from toklab.models.adapt_embeddings import EmbeddingInitRecord


class GateVerdict(StrEnum):
    """Single gate outcome for a (tokenizer, model_variant) pair."""

    OK_NATIVE = "OK_NATIVE"
    OK_ADAPTED_TRAINED = "OK_ADAPTED_TRAINED"
    UNTRAINED_NEW_ROWS = "UNTRAINED_NEW_ROWS"
    BOUNDARY_SHIFT_UNVALIDATED = "BOUNDARY_SHIFT_UNVALIDATED"
    INCOMPATIBLE = "INCOMPATIBLE"


@dataclass
class ModelVariant:
    """Model state presented to the gate.

    ``uses_boundary_shift``: True when tokenization boundaries differ from the
    model's native BPE (e.g. structure-aware C4/C5) even if vocab ids fit.
    """

    model: Any
    n_embedding_rows: int | None = None
    tie_word_embeddings: bool | None = None
    init_record: EmbeddingInitRecord | None = None
    trained_new_rows: bool | None = None
    has_new_vocab_rows: bool = False
    uses_boundary_shift: bool = False
    name: str = "variant"
    meta: dict[str, Any] = field(default_factory=dict)

    def resolve_rows(self) -> int:
        if self.n_embedding_rows is not None:
            return int(self.n_embedding_rows)
        return int(self.model.get_input_embeddings().weight.shape[0])

    def resolve_tie(self) -> bool:
        if self.tie_word_embeddings is not None:
            return bool(self.tie_word_embeddings)
        return bool(
            getattr(self.model.config, "tie_word_embeddings", False)
        )

    def resolve_trained(self) -> bool:
        if self.trained_new_rows is not None:
            return bool(self.trained_new_rows)
        if self.init_record is not None:
            return bool(self.init_record.trained_new_rows)
        return not self.has_new_vocab_rows


@dataclass(frozen=True)
class GateResult:
    verdict: GateVerdict
    reasons: tuple[str, ...]
    max_token_id: int | None
    embedding_rows: int
    roundtrip_ok: bool | None
    shapes_ok: bool
    init_recorded: bool
    zero_shot_bpb: float | None
    c2_bpb: float | None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    @property
    def quality_allowed(self) -> bool:
        return self.verdict not in {
            GateVerdict.UNTRAINED_NEW_ROWS,
            GateVerdict.INCOMPATIBLE,
        }


def _embeddings_tied(model: Any) -> bool:
    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    if out_emb is None:
        return bool(getattr(model.config, "tie_word_embeddings", False))
    return in_emb.weight.data_ptr() == out_emb.weight.data_ptr()


def _check_shapes(model: Any, tie_expected: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    in_rows, in_dim = int(in_emb.weight.shape[0]), int(in_emb.weight.shape[1])
    if out_emb is None:
        if tie_expected:
            reasons.append("tie_word_embeddings=True but output embeddings are None")
            return False, reasons
        return True, reasons
    out_rows, out_dim = int(out_emb.weight.shape[0]), int(out_emb.weight.shape[1])
    if in_rows != out_rows or in_dim != out_dim:
        reasons.append(
            f"embedding/head shape mismatch: in={(in_rows, in_dim)} "
            f"out={(out_rows, out_dim)}"
        )
        return False, reasons
    tied = _embeddings_tied(model)
    if tie_expected and not tied:
        reasons.append("tie_word_embeddings=True but input/output weights diverge")
        return False, reasons
    if (not tie_expected) and tied:
        # Not fatal: some configs leave ptr equal; note only.
        reasons.append("note: untied config but input/output share storage")
    return True, reasons


def _max_token_id(tokenizer: Any, texts: Sequence[str]) -> int:
    mx = -1
    for text in texts:
        ids = tokenizer.encode(text).ids
        if ids:
            mx = max(mx, max(ids))
    return mx


def _roundtrip_ok(tokenizer: Any, texts: Sequence[str]) -> bool:
    for text in texts:
        decoded = tokenizer.decode(tokenizer.encode(text).ids)
        if decoded != text:
            return False
    return True


def gate(
    tokenizer: Any,
    model_variant: ModelVariant,
    *,
    sample_texts: Sequence[str] | None = None,
    c2_bpb: float | None = None,
    zero_shot_bpb: float | None = None,
) -> GateResult:
    """Return one compatibility verdict for ``tokenizer`` × ``model_variant``.

    Checks (in order of hard failure):
    1. ``max(token id) < embedding rows``
    2. head/embedding shapes vs tying
    3. lossless round-trip on ``sample_texts``
    4. init recorded when new vocab rows exist
    5. training status of new rows
    6. boundary-shift flag (structure-aware native vocab)
    7. optional zero-shot BPB vs C2 (recorded; does not invent a pass)

    Quality must not be published for ``UNTRAINED_NEW_ROWS`` or ``INCOMPATIBLE``.
    """
    texts = list(sample_texts or ["def f(x):\n    return x\n"])
    rows = model_variant.resolve_rows()
    tie = model_variant.resolve_tie()
    reasons: list[str] = []

    max_id = _max_token_id(tokenizer, texts)
    if max_id >= rows:
        return GateResult(
            verdict=GateVerdict.INCOMPATIBLE,
            reasons=(
                f"max token id {max_id} >= embedding rows {rows}",
            ),
            max_token_id=max_id,
            embedding_rows=rows,
            roundtrip_ok=None,
            shapes_ok=False,
            init_recorded=model_variant.init_record is not None,
            zero_shot_bpb=zero_shot_bpb,
            c2_bpb=c2_bpb,
        )

    shapes_ok, shape_notes = _check_shapes(model_variant.model, tie)
    reasons.extend(shape_notes)
    if not shapes_ok:
        return GateResult(
            verdict=GateVerdict.INCOMPATIBLE,
            reasons=tuple(reasons) or ("embedding/head shape check failed",),
            max_token_id=max_id,
            embedding_rows=rows,
            roundtrip_ok=None,
            shapes_ok=False,
            init_recorded=model_variant.init_record is not None,
            zero_shot_bpb=zero_shot_bpb,
            c2_bpb=c2_bpb,
        )

    rt_ok = _roundtrip_ok(tokenizer, texts)
    if not rt_ok:
        return GateResult(
            verdict=GateVerdict.INCOMPATIBLE,
            reasons=("round-trip decode(encode(x)) != x on sample_texts",),
            max_token_id=max_id,
            embedding_rows=rows,
            roundtrip_ok=False,
            shapes_ok=True,
            init_recorded=model_variant.init_record is not None,
            zero_shot_bpb=zero_shot_bpb,
            c2_bpb=c2_bpb,
        )

    has_new = bool(model_variant.has_new_vocab_rows) or (
        model_variant.init_record is not None
        and model_variant.init_record.n_new_rows > 0
    )
    init_recorded = model_variant.init_record is not None
    if has_new and not init_recorded:
        return GateResult(
            verdict=GateVerdict.INCOMPATIBLE,
            reasons=(
                "new vocab rows present but embedding init was not recorded",
            ),
            max_token_id=max_id,
            embedding_rows=rows,
            roundtrip_ok=True,
            shapes_ok=True,
            init_recorded=False,
            zero_shot_bpb=zero_shot_bpb,
            c2_bpb=c2_bpb,
        )

    details: dict[str, Any] = {}
    if zero_shot_bpb is not None and c2_bpb is not None and math.isfinite(c2_bpb):
        details["bpb_delta_vs_c2"] = float(zero_shot_bpb) - float(c2_bpb)
        details["zero_shot_bpb"] = float(zero_shot_bpb)
        details["c2_bpb"] = float(c2_bpb)

    if has_new:
        trained = model_variant.resolve_trained()
        if not trained:
            reasons.append(
                "new embedding rows initialised but not trained "
                "(quality must be suppressed)"
            )
            return GateResult(
                verdict=GateVerdict.UNTRAINED_NEW_ROWS,
                reasons=tuple(reasons),
                max_token_id=max_id,
                embedding_rows=rows,
                roundtrip_ok=True,
                shapes_ok=True,
                init_recorded=True,
                zero_shot_bpb=zero_shot_bpb,
                c2_bpb=c2_bpb,
                details=details,
            )
        reasons.append("adapted vocab rows present and marked trained")
        return GateResult(
            verdict=GateVerdict.OK_ADAPTED_TRAINED,
            reasons=tuple(reasons),
            max_token_id=max_id,
            embedding_rows=rows,
            roundtrip_ok=True,
            shapes_ok=True,
            init_recorded=init_recorded,
            zero_shot_bpb=zero_shot_bpb,
            c2_bpb=c2_bpb,
            details=details,
        )

    if model_variant.uses_boundary_shift:
        reasons.append(
            "token boundaries differ from native BPE "
            "(structure-aware); embedding rows exist but shift unvalidated"
        )
        return GateResult(
            verdict=GateVerdict.BOUNDARY_SHIFT_UNVALIDATED,
            reasons=tuple(reasons),
            max_token_id=max_id,
            embedding_rows=rows,
            roundtrip_ok=True,
            shapes_ok=True,
            init_recorded=init_recorded,
            zero_shot_bpb=zero_shot_bpb,
            c2_bpb=c2_bpb,
            details=details,
        )

    reasons.append("native vocab; shapes and round-trip OK")
    return GateResult(
        verdict=GateVerdict.OK_NATIVE,
        reasons=tuple(reasons),
        max_token_id=max_id,
        embedding_rows=rows,
        roundtrip_ok=True,
        shapes_ok=True,
        init_recorded=init_recorded,
        zero_shot_bpb=zero_shot_bpb,
        c2_bpb=c2_bpb,
        details=details,
    )
