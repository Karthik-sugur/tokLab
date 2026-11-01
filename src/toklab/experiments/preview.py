"""Snippet tokenize preview (API-facing; no science metrics)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from toklab.tokenizers.registry import get_tokenizer, load_base_model_config
from toklab.vocab.adapted_vocab import AdaptedVocab

MAX_PREVIEW_CHARS = 50_000
NO_VOCAB_NOTE = "no adapted vocabulary loaded, so C3/C5 equal C2/C4"


def preview_tokenize(
    text: str,
    conditions: list[str],
    *,
    cfg: dict[str, Any] | None = None,
    adapted: Mapping[str, AdaptedVocab] | None = None,
    experiment_id: str | None = None,
    k_star: int | None = None,
) -> dict[str, Any]:
    """Encode ``text`` under each condition; return ids + char spans.

    ``adapted`` (``{"native": ..., "structural": ...}``) is an experiment's
    K* vocabulary: C3 uses ``native``, C5 ``structural``. Without it C3/C5
    have no adapted entries and tokenize exactly like C2/C4.

    Raises:
        ValueError: empty conditions, text exceeds ``MAX_PREVIEW_CHARS``, or
            the vocabulary was built for a different base vocabulary size.
    """
    if not conditions:
        raise ValueError("conditions must be non-empty")
    if len(text) > MAX_PREVIEW_CHARS:
        raise ValueError(
            f"snippet too large ({len(text)} chars; max {MAX_PREVIEW_CHARS})"
        )

    cfg = dict(cfg or {})
    try:
        cfg = {**load_base_model_config(cfg.get("base_model_yaml")), **cfg}
    except (FileNotFoundError, ValueError, OSError):
        if "model_name" not in cfg:
            cfg["model_name"] = "Qwen/Qwen2.5-Coder-0.5B"
    if "attach_leading_space" not in cfg:
        cfg["attach_leading_space"] = True

    if adapted:
        base = get_tokenizer("C2", cfg)
        for mode, v in adapted.items():
            if v.id_offset != base.vocab_size:
                raise ValueError(
                    f"{mode} vocabulary id_offset {v.id_offset} does not match "
                    f"base vocab size {base.vocab_size}"
                )
        cfg["base_tokenizer"] = base

    results: list[dict[str, Any]] = []
    for cond in conditions:
        cond_cfg = dict(cfg)
        if adapted and cond == "C3":
            cond_cfg["adapted_vocab"] = adapted["native"]
        elif adapted and cond == "C5":
            cond_cfg["adapted_vocab"] = adapted["structural"]
        tok = get_tokenizer(cond, cond_cfg)
        enc = tok.encode(text)
        results.append(
            {
                "condition": cond,
                "n_tokens": len(enc.ids),
                "ids": list(enc.ids),
                "spans": [{"start": s, "end": e} for s, e in enc.spans],
            }
        )
    if adapted:
        sizes = {m: len(v) for m, v in adapted.items()}
        note = (
            f"adapted vocabulary of experiment {experiment_id} at K*={k_star} "
            f"(C3 native={sizes.get('native', 0)}, "
            f"C5 structural={sizes.get('structural', 0)} entries)"
        )
    else:
        sizes = {}
        note = NO_VOCAB_NOTE
    return {
        "text": text,
        "n_chars": len(text),
        "n_bytes": len(text.encode("utf-8")),
        "conditions": results,
        "experiment_id": experiment_id if adapted else None,
        "k_star": k_star if adapted else None,
        "vocab_sizes": sizes,
        "note": note,
    }
