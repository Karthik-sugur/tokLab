"""Snippet tokenize preview (API-facing; no science metrics)."""

from __future__ import annotations

from typing import Any

from toklab.tokenizers.registry import get_tokenizer, load_base_model_config

MAX_PREVIEW_CHARS = 50_000


def preview_tokenize(
    text: str,
    conditions: list[str],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode ``text`` under each condition; return ids + char spans.

    Raises:
        ValueError: empty conditions or text exceeds ``MAX_PREVIEW_CHARS``.
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

    results: list[dict[str, Any]] = []
    for cond in conditions:
        tok = get_tokenizer(cond, cfg)
        enc = tok.encode(text)
        results.append(
            {
                "condition": cond,
                "n_tokens": len(enc.ids),
                "ids": list(enc.ids),
                "spans": [{"start": s, "end": e} for s, e in enc.spans],
            }
        )
    return {
        "text": text,
        "n_chars": len(text),
        "n_bytes": len(text.encode("utf-8")),
        "conditions": results,
    }
