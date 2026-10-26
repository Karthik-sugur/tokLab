"""Hugging Face tokenizer wrapped as a TokLab Tokenizer."""

from __future__ import annotations

from typing import Any

from toklab.core.types import EncodeResult


class NativeHFTokenizer:
    """Lossless HF wrapper using ``return_offsets_mapping`` for char spans."""

    def __init__(self, tokenizer: Any, name: str) -> None:
        self._tok = tokenizer
        self.name = name

    @classmethod
    def from_pretrained(cls, model_id: str, *, name: str) -> NativeHFTokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        if not getattr(tok, "is_fast", False):
            raise RuntimeError(
                f"tokenizer for {model_id!r} is not a fast tokenizer; "
                "char spans via return_offsets_mapping are required"
            )
        return cls(tok, name=name)

    def encode(self, text: str) -> EncodeResult:
        encoded = self._tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = list(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        spans = [(int(s), int(e)) for s, e in offsets]
        if len(ids) != len(spans):
            raise RuntimeError(
                f"{self.name}: id/span length mismatch "
                f"({len(ids)} ids vs {len(spans)} spans)"
            )
        return EncodeResult(ids=ids, spans=spans)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    @property
    def vocab_size(self) -> int:
        return int(len(self._tok))
