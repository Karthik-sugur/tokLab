"""Condition → Tokenizer factory (C1–C6 grid)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toklab.tokenizers.byte_level import ByteLevelTokenizer
from toklab.tokenizers.composed import ComposedTokenizer
from toklab.tokenizers.native_hf import NativeHFTokenizer
from toklab.vocab.adapted_vocab import AdaptedVocab

BASELINE_CONDITIONS = ("C1", "C2", "C6")
GRID_CONDITIONS = ("C2", "C3", "C4", "C5")
KNOWN_CONDITIONS = ("C1", "C2", "C3", "C4", "C5", "C6")


def _default_base_model_yaml() -> Path:
    candidates = [
        Path.cwd() / "configs" / "models" / "base_model.yaml",
        Path(__file__).resolve().parents[3]
        / "configs"
        / "models"
        / "base_model.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def load_base_model_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load ``configs/models/base_model.yaml`` (or an override path)."""
    import yaml

    cfg_path = Path(path) if path is not None else _default_base_model_yaml()
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "model_name" not in data:
        raise ValueError(f"{cfg_path} must define model_name")
    return data


def _resolve_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(cfg or {})
    if "model_name" not in cfg:
        loaded = load_base_model_config(cfg.get("base_model_yaml"))
        cfg = {**loaded, **cfg}
    return cfg


def _base_tok(cfg: dict[str, Any]) -> NativeHFTokenizer:
    if "base_tokenizer" in cfg and cfg["base_tokenizer"] is not None:
        return cfg["base_tokenizer"]
    model_id = cfg.get("tokenizer_name") or cfg["model_name"]
    return NativeHFTokenizer.from_pretrained(str(model_id), name="C2")


def _adapted(cfg: dict[str, Any]) -> AdaptedVocab | dict[str, int] | None:
    return cfg.get("adapted_vocab")


def get_tokenizer(condition: str, cfg: dict[str, Any] | None = None) -> Any:
    """Return a Tokenizer for C1–C6.

    C2..C5 share the same base model config; they differ only in structure
    (off/on) and domain vocab (off/on). Pass ``adapted_vocab`` / ``base_tokenizer``
    in ``cfg`` for C3/C5 (and optionally C4) so the base instance is shared.
    """
    cfg = _resolve_cfg(cfg)
    attach = bool(cfg.get("attach_leading_space", True))
    indent = bool(cfg.get("indent_tokens", False))

    if condition == "C1":
        return NativeHFTokenizer.from_pretrained("gpt2", name="C1")
    if condition == "C2":
        base = _base_tok(cfg)
        base.name = "C2"
        return base
    if condition == "C6":
        return ByteLevelTokenizer()
    if condition == "C3":
        base = _base_tok(cfg)
        return ComposedTokenizer(
            segmenter=None,
            base_hf_tokenizer=base,
            adapted_vocab=_adapted(cfg),
            attach_leading_space=attach,
            indent_tokens=indent,
            match_mode="native_longest",
            name="C3",
        )
    if condition in {"C4", "C5"}:
        from toklab.parsing.segmenters import TreeSitterSegmenter

        base = _base_tok(cfg)
        adapted = _adapted(cfg) if condition == "C5" else None
        mode = "structural_ident" if condition == "C5" else "none"
        return ComposedTokenizer(
            TreeSitterSegmenter(),
            base,
            adapted_vocab=adapted,
            attach_leading_space=attach,
            indent_tokens=indent,
            match_mode=mode,
            name=condition,
        )
    raise ValueError(
        f"unknown condition {condition!r}; known: {KNOWN_CONDITIONS}"
    )


get = get_tokenizer
