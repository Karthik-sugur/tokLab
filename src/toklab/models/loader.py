"""Load base causal LM + tokenizer from config; record revision."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LoadedModel:
    """Base model bundle used by gate / adapt / finetune."""

    model_name: str
    revision: str
    model: Any
    hf_tokenizer: Any
    tie_word_embeddings: bool
    embedding_rows: int
    embedding_dim: int
    meta: dict[str, Any] = field(default_factory=dict)


def _resolve_revision(model_name: str, hf_tokenizer: Any, model: Any) -> str:
    """Best-effort model revision / commit id."""
    for obj in (hf_tokenizer, model):
        rev = getattr(obj, "name_or_path", None)
        _ = rev
    # Prefer config._name_or_path + commit from config if present.
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in ("_commit_hash", "commit_hash", "revision"):
            val = getattr(cfg, key, None)
            if val:
                return str(val)
        name = getattr(cfg, "_name_or_path", None) or model_name
        return f"{name}@unknown"
    return f"{model_name}@unknown"


def load_model_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load ``configs/models/base_model.yaml`` (or override)."""
    if path is None:
        candidates = [
            Path.cwd() / "configs" / "models" / "base_model.yaml",
            Path(__file__).resolve().parents[3]
            / "configs"
            / "models"
            / "base_model.yaml",
        ]
        for c in candidates:
            if c.is_file():
                path = c
                break
        else:
            path = candidates[0]
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "model_name" not in data:
        raise ValueError(f"{path} must define model_name")
    return data


def load_base_model(
    config: dict[str, Any] | Path | str | None = None,
    *,
    device: str | None = None,
    torch_dtype: str | None = None,
    trust_remote_code: bool = True,
) -> LoadedModel:
    """Load base causal LM and its HF tokenizer; record revision."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config is None or isinstance(config, (str, Path)):
        cfg = load_model_config(config)
    else:
        cfg = dict(config)

    model_name = str(cfg["model_name"])
    tokenizer_name = str(cfg.get("tokenizer_name") or model_name)
    revision = cfg.get("revision")  # optional pin

    tok_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "use_fast": True,
    }
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision:
        tok_kwargs["revision"] = revision
        model_kwargs["revision"] = revision
    if torch_dtype:
        model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

    hf_tok = AutoTokenizer.from_pretrained(tokenizer_name, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    emb = model.get_input_embeddings()
    tie = bool(getattr(model.config, "tie_word_embeddings", False))
    resolved = (
        str(revision)
        if revision
        else _resolve_revision(model_name, hf_tok, model)
    )

    return LoadedModel(
        model_name=model_name,
        revision=resolved,
        model=model,
        hf_tokenizer=hf_tok,
        tie_word_embeddings=tie,
        embedding_rows=int(emb.weight.shape[0]),
        embedding_dim=int(emb.weight.shape[1]),
        meta={
            "tokenizer_name": tokenizer_name,
            "device": device,
            "vocab_size_tokenizer": int(len(hf_tok)),
            "hidden_size": int(
                getattr(model.config, "hidden_size", emb.weight.shape[1])
            ),
        },
    )
