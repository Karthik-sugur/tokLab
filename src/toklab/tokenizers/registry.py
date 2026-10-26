"""Condition → Tokenizer factory for baseline (and later) conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toklab.tokenizers.byte_level import ByteLevelTokenizer
from toklab.tokenizers.native_hf import NativeHFTokenizer

BASELINE_CONDITIONS = ("C1", "C2", "C6")


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
    """Return a Tokenizer for ``condition`` (``C1`` / ``C2`` / ``C6``).

    ``cfg`` may include ``model_name`` / ``tokenizer_name`` (for C2) or
    ``base_model_yaml`` pointing at a YAML file with those keys.
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
    raise ValueError(
        f"unknown condition {condition!r}; "
        f"baselines support {BASELINE_CONDITIONS}"
    )


# Alias matching the task wording ``get(...)``.
get = get_tokenizer
