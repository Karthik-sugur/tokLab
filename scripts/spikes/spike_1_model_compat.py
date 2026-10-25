#!/usr/bin/env python3
"""Spike 1: model/tokenizer compatibility for vocab resize + tied embeddings.

Prints observed facts only. Not library code.
"""

from __future__ import annotations

import sys

PREFERRED = "Qwen/Qwen2.5-Coder-0.5B"
FALLBACKS = [
    "Salesforce/codegen-350M-mono",
    "microsoft/CodeGPT-small-py",
]


def _load_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tried: list[str] = []
    for name in [PREFERRED, *FALLBACKS]:
        tried.append(name)
        try:
            print(f"Trying model: {name}")
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                name, trust_remote_code=True
            )
            print(f"Loaded model: {name}")
            return name, tok, model
        except Exception as exc:  # noqa: BLE001 — spike: report load failures
            print(f"Failed to load {name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not load any candidate model. Tried: {tried}")


def _license_str(model_name: str, model) -> str:
    license_val = getattr(getattr(model, "config", None), "license", None)
    if license_val:
        return str(license_val)
    try:
        from huggingface_hub import model_info

        info = model_info(model_name)
        card = getattr(info, "cardData", None) or {}
        if isinstance(card, dict) and card.get("license"):
            return str(card["license"])
        if getattr(info, "license", None):
            return str(info.license)
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__}: {exc})"
    return "unavailable"


def _causal_loss(model, tok, text: str) -> float:
    import torch

    enc = tok(text, return_tensors="pt")
    if enc["input_ids"].shape[-1] < 2:
        raise ValueError("snippet too short for causal LM loss")
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return float(out.loss.item())


def main() -> int:
    try:
        import torch
    except ImportError:
        print(
            "ERROR: torch is required for this spike. "
            'Install with: pip install -e ".[models]"'
        )
        return 1

    model_name, tok, model = _load_model_and_tokenizer()
    cfg = model.config

    print("--- model facts ---")
    print(f"model_name: {model_name}")
    print(f"license: {_license_str(model_name, model)}")
    print(f"tokenizer_class: {type(tok).__name__}")
    print(f"vocab_size (tokenizer): {len(tok)}")
    emb = model.get_input_embeddings()
    print(f"embedding_rows: {emb.weight.shape[0]}")
    print(f"embedding_dim: {emb.weight.shape[1]}")
    print(f"tie_word_embeddings: {getattr(cfg, 'tie_word_embeddings', None)}")
    print(f"hidden_size: {getattr(cfg, 'hidden_size', None)}")
    print(f"num_hidden_layers: {getattr(cfg, 'num_hidden_layers', None)}")
    print(
        "num_key_value_heads: "
        f"{getattr(cfg, 'num_key_value_heads', getattr(cfg, 'n_head', None))}"
    )

    snippet = "def add(a, b):\n    return a + b\n"
    loss_before = _causal_loss(model, tok, snippet)
    print(f"loss_before_resize: {loss_before}")

    n_new = 100
    dummy = [f"<toklab_dummy_{i}>" for i in range(n_new)]
    n_added = tok.add_tokens(dummy)
    print(f"tokens_requested: {n_new}")
    print(f"tokens_added: {n_added}")

    old_rows = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tok))
    new_emb = model.get_input_embeddings()
    new_rows = new_emb.weight.shape[0]
    print(f"embedding_rows_before_resize: {old_rows}")
    print(f"embedding_rows_after_resize: {new_rows}")

    with torch.no_grad():
        mean_vec = new_emb.weight[:old_rows].mean(dim=0)
        new_emb.weight[old_rows:] = mean_vec

    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    still_tied = (
        out_emb is not None
        and in_emb.weight.data_ptr() == out_emb.weight.data_ptr()
    )
    print(f"input_embedding_shape: {tuple(in_emb.weight.shape)}")
    if out_emb is not None:
        print(f"output_embedding_shape: {tuple(out_emb.weight.shape)}")
    else:
        print("output_embedding_shape: None")
    print(f"input_output_embeddings_still_tied: {still_tied}")

    loss_after = _causal_loss(model, tok, snippet)
    print(f"loss_after_resize: {loss_after}")

    print()
    print("FINDINGS:")
    print(f"- Loaded model {model_name} with tokenizer class {type(tok).__name__}.")
    print(
        f"- Vocab size {len(tok)}; embedding rows after resize "
        f"{tuple(in_emb.weight.shape)}."
    )
    print(
        f"- tie_word_embeddings={getattr(cfg, 'tie_word_embeddings', None)}; "
        f"after resize still_tied={still_tied}."
    )
    print(
        f"- Causal LM loss on snippet before={loss_before}, after={loss_after}."
    )
    print(
        f"- Added {n_added} dummy tokens; new rows initialized to mean of "
        "existing embeddings."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
