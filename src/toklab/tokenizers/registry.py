"""Condition → Tokenizer factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toklab.tokenizers.byte_level import ByteLevelTokenizer
from toklab.tokenizers.composed import ComposedTokenizer
from toklab.tokenizers.native_hf import NativeHFTokenizer

BASELINE_CONDITIONS = ("C1", "C2", "C6")
STRUCTURE_CONDITIONS = ("C4",)
KNOWN_CONDITIONS = BASELINE_CONDITIONS + STRUCTURE_CONDITIONS


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


def get_tokenizer(condition: str, cfg: dict[str, Any] | None = None) -> Any:
    """Return a Tokenizer for ``condition`` (``C1`` / ``C2`` / ``C4`` / ``C6``).

    ``cfg`` may include ``model_name`` / ``tokenizer_name``,
    ``attach_leading_space`` (C4, default True per decision 002),
    ``indent_tokens`` (C4 ablation, default False).
    """
    cfg = dict(cfg or {})
    if "model_name" not in cfg:
        yaml_path = cfg.get("base_model_yaml")
        loaded = load_base_model_config(yaml_path)
        cfg = {**loaded, **cfg}

    if condition == "C1":
        return NativeHFTokenizer.from_pretrained("gpt2", name="C1")
    if condition == "C2":
        model_id = cfg.get("tokenizer_name") or cfg["model_name"]
        return NativeHFTokenizer.from_pretrained(str(model_id), name="C2")
    if condition == "C6":
        return ByteLevelTokenizer()
    if condition == "C4":
        from toklab.parsing.segmenters import TreeSitterSegmenter

        model_id = cfg.get("tokenizer_name") or cfg["model_name"]
        base = NativeHFTokenizer.from_pretrained(str(model_id), name="C2")
        attach = cfg.get("attach_leading_space", True)
        indent = bool(cfg.get("indent_tokens", False))
        return ComposedTokenizer(
            TreeSitterSegmenter(),
            base,
            adapted_vocab=None,
            attach_leading_space=bool(attach),
            indent_tokens=indent,
            name="C4",
        )
    raise ValueError(
        f"unknown condition {condition!r}; known: {KNOWN_CONDITIONS}"
    )


# Alias matching the task wording ``get(...)``.
get = get_tokenizer
