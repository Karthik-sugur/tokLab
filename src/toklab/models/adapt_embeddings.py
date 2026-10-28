"""Resize embeddings for AdaptedVocab; init new rows from BPE mean."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from toklab.vocab.adapted_vocab import AdaptedVocab

INIT_METHOD = "mean_of_base_bpe_decomposition"


@dataclass(frozen=True)
class EmbeddingInitRecord:
    """Provenance for how new embedding rows were initialised."""

    method: str
    n_new_rows: int
    id_offset: int
    target_rows: int
    tie_word_embeddings: bool
    still_tied_after: bool
    per_token: tuple[dict[str, Any], ...] = ()
    trained_new_rows: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_token"] = list(self.per_token)
        return d


@dataclass
class AdaptedModel:
    """Model after vocab resize + init metadata."""

    model: Any
    init_record: EmbeddingInitRecord
    adapted_vocab: AdaptedVocab
    meta: dict[str, Any] = field(default_factory=dict)


def _bpe_ids(hf_tokenizer: Any, token: str) -> list[int]:
    """Original base BPE decomposition of ``token`` (no special tokens)."""
    ids = hf_tokenizer.encode(token, add_special_tokens=False)
    return [int(i) for i in ids]


def _embeddings_tied(model: Any) -> bool:
    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    if out_emb is None:
        return bool(getattr(model.config, "tie_word_embeddings", False))
    return in_emb.weight.data_ptr() == out_emb.weight.data_ptr()


def adapt_embeddings(
    model: Any,
    adapted_vocab: AdaptedVocab,
    hf_tokenizer: Any,
    *,
    mark_trained: bool = False,
) -> AdaptedModel:
    """Resize embeddings for ``adapted_vocab`` and mean-init new rows.

    For each adapted entry, the new **input** embedding row is set to the
    mean of the embedding vectors of that entry's original base BPE
    subword ids. If embeddings are untied, the output head row is set the
    same way; if tied, the single shared matrix is updated once.

    Returns the model plus an ``EmbeddingInitRecord`` (init method label).
    New rows are **not** considered trained unless ``mark_trained=True``.
    """
    import torch

    if len(adapted_vocab) == 0:
        in_emb = model.get_input_embeddings()
        record = EmbeddingInitRecord(
            method=INIT_METHOD,
            n_new_rows=0,
            id_offset=adapted_vocab.id_offset,
            target_rows=int(in_emb.weight.shape[0]),
            tie_word_embeddings=bool(
                getattr(model.config, "tie_word_embeddings", False)
            ),
            still_tied_after=_embeddings_tied(model),
            per_token=(),
            trained_new_rows=True,  # nothing new to train
            notes="empty AdaptedVocab; no resize",
        )
        return AdaptedModel(
            model=model, init_record=record, adapted_vocab=adapted_vocab
        )

    max_id = max(e.id for e in adapted_vocab.entries)
    target_rows = max_id + 1
    old_rows = int(model.get_input_embeddings().weight.shape[0])
    if target_rows > old_rows:
        model.resize_token_embeddings(target_rows)

    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    tied_cfg = bool(getattr(model.config, "tie_word_embeddings", False))
    tied_now = _embeddings_tied(model)

    per_token: list[dict[str, Any]] = []
    with torch.no_grad():
        weight = in_emb.weight
        for entry in adapted_vocab.entries:
            sub_ids = _bpe_ids(hf_tokenizer, entry.token)
            # Fall back to mean of all base rows if decomposition empty.
            if not sub_ids:
                vec = weight[: adapted_vocab.id_offset].mean(dim=0)
                used_ids: list[int] = []
            else:
                # Clamp to rows that existed before new adapted ids.
                valid = [i for i in sub_ids if 0 <= i < adapted_vocab.id_offset]
                if not valid:
                    valid = [i for i in sub_ids if 0 <= i < old_rows]
                if not valid:
                    vec = weight[: min(old_rows, weight.shape[0])].mean(dim=0)
                    used_ids = []
                else:
                    idx = torch.tensor(valid, dtype=torch.long, device=weight.device)
                    vec = weight.index_select(0, idx).mean(dim=0)
                    used_ids = valid
            weight[entry.id].copy_(vec)
            if out_emb is not None and not tied_now:
                out_emb.weight[entry.id].copy_(vec)
            per_token.append(
                {
                    "token": entry.token,
                    "id": entry.id,
                    "bpe_ids": used_ids,
                    "method": INIT_METHOD,
                }
            )

    still_tied = _embeddings_tied(model)
    record = EmbeddingInitRecord(
        method=INIT_METHOD,
        n_new_rows=len(adapted_vocab),
        id_offset=adapted_vocab.id_offset,
        target_rows=int(model.get_input_embeddings().weight.shape[0]),
        tie_word_embeddings=tied_cfg,
        still_tied_after=still_tied,
        per_token=tuple(per_token),
        trained_new_rows=bool(mark_trained),
        notes=(
            "new rows mean-initialised from base BPE; "
            "not trained unless mark_trained=True"
        ),
    )
    return AdaptedModel(
        model=model, init_record=record, adapted_vocab=adapted_vocab
    )
