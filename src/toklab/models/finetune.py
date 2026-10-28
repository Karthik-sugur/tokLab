"""Fine-tune C2..C5 under identical data / schedule (optional LoRA).

Real multi-GPU training is out of scope for the default path: pass
``dry_run=True`` (default for CLI demos) to log the plan without stepping
the optimizer. Callers that opt into training must pass ``dry_run=False``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONDITIONS = ("C2", "C3", "C4", "C5")


@dataclass
class FinetuneConfig:
    """Shared hyper-parameters — identical across C2..C5."""

    seed: int = 0
    n_seeds: int = 1
    max_steps: int = 10
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    max_seq_len: int = 64
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    train_embeddings: bool = True
    dry_run: bool = True
    device: str = "cpu"
    conditions: tuple[str, ...] = CONDITIONS


@dataclass
class FinetuneRunLog:
    condition: str
    seed: int
    dry_run: bool
    steps_planned: int
    steps_executed: int
    use_lora: bool
    train_embeddings: bool
    n_train_bytes: int
    n_train_files: int
    final_loss: float | None
    notes: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _source_bytes(texts: Sequence[str]) -> int:
    return sum(len(t.encode("utf-8")) for t in texts)


def _maybe_attach_lora(model: Any, cfg: FinetuneConfig) -> tuple[Any, str]:
    """Attach LoRA if peft is installed; otherwise leave model unchanged."""
    if not cfg.use_lora:
        return model, "lora_disabled"
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        return model, "peft_not_installed_skipped_lora"
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora), "lora_attached"


def _enable_embedding_grads(model: Any, enabled: bool) -> None:
    emb = model.get_input_embeddings()
    if emb is None:
        return
    for p in emb.parameters():
        p.requires_grad = bool(enabled)
    out = model.get_output_embeddings()
    if out is not None and out is not emb:
        for p in out.parameters():
            p.requires_grad = bool(enabled)


def _batch_ids(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_seq_len: int,
    device: str,
) -> Any:
    import torch

    # Cycle a single packed sequence for the dry-run / tiny path.
    ids = tokenizer.encode(texts[0]).ids[:max_seq_len]
    if len(ids) < 2:
        ids = ids + [0] * (2 - len(ids))
    return torch.tensor([ids], dtype=torch.long, device=device)


def finetune_condition(
    *,
    condition: str,
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    cfg: FinetuneConfig,
    run_dir: Path | str | None = None,
    seed: int | None = None,
) -> FinetuneRunLog:
    """Fine-tune one condition (or dry-run) on the shared ``texts`` bytes."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if not texts:
        raise ValueError("texts must be non-empty (identical data for C2..C5)")

    seed = cfg.seed if seed is None else int(seed)
    _set_seed(seed)
    n_bytes = _source_bytes(texts)
    out_dir = Path(run_dir) if run_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "condition": condition,
        "seed": seed,
        "dry_run": cfg.dry_run,
        "max_steps": cfg.max_steps,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_steps": cfg.warmup_steps,
        "max_seq_len": cfg.max_seq_len,
        "use_lora": cfg.use_lora,
        "train_embeddings": cfg.train_embeddings,
        "n_train_files": len(texts),
        "n_train_bytes": n_bytes,
        "device": cfg.device,
    }
    if out_dir is not None:
        (out_dir / f"finetune_plan_{condition}_seed{seed}.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if cfg.dry_run:
        log = FinetuneRunLog(
            condition=condition,
            seed=seed,
            dry_run=True,
            steps_planned=cfg.max_steps,
            steps_executed=0,
            use_lora=cfg.use_lora,
            train_embeddings=cfg.train_embeddings,
            n_train_bytes=n_bytes,
            n_train_files=len(texts),
            final_loss=None,
            notes="dry_run=True: no optimizer steps executed",
            extras={"plan": plan},
        )
        if out_dir is not None:
            (out_dir / f"finetune_log_{condition}_seed{seed}.json").write_text(
                json.dumps(log.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return log

    # Real training path (tiny models / explicit opt-in only).
    import torch

    model = model.to(cfg.device)
    model, lora_note = _maybe_attach_lora(model, cfg)
    _enable_embedding_grads(model, cfg.train_embeddings)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    opt = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    input_ids = _batch_ids(
        tokenizer,
        texts,
        max_seq_len=cfg.max_seq_len,
        device=cfg.device,
    )
    last_loss: float | None = None
    for step in range(cfg.max_steps):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids)
        loss = out.loss
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().item())
        if out_dir is not None and (step + 1) % max(1, cfg.max_steps // 5) == 0:
            with (out_dir / f"finetune_loss_{condition}_seed{seed}.jsonl").open(
                "a", encoding="utf-8"
            ) as fh:
                fh.write(
                    json.dumps({"step": step + 1, "loss": last_loss}) + "\n"
                )

    log = FinetuneRunLog(
        condition=condition,
        seed=seed,
        dry_run=False,
        steps_planned=cfg.max_steps,
        steps_executed=cfg.max_steps,
        use_lora=cfg.use_lora,
        train_embeddings=cfg.train_embeddings,
        n_train_bytes=n_bytes,
        n_train_files=len(texts),
        final_loss=last_loss,
        notes=lora_note,
        extras={"plan": plan},
    )
    if out_dir is not None:
        (out_dir / f"finetune_log_{condition}_seed{seed}.json").write_text(
            json.dumps(log.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return log


def finetune_grid(
    *,
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    texts: Sequence[str],
    cfg: FinetuneConfig,
    run_dir: Path | str,
) -> list[FinetuneRunLog]:
    """Run C2..C5 with identical ``texts`` / ``cfg`` across seeds.

    ``models`` / ``tokenizers`` must contain every condition in ``cfg.conditions``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "finetune_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Record shared data fingerprint (byte lengths only — no invented metrics).
    fingerprint = {
        "n_files": len(texts),
        "n_bytes": _source_bytes(texts),
        "sha256_concat": __import__("hashlib")
        .sha256("".join(texts).encode("utf-8"))
        .hexdigest(),
    }
    (run_dir / "finetune_data_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logs: list[FinetuneRunLog] = []
    seeds = [cfg.seed + i for i in range(max(1, cfg.n_seeds))]
    for cond in cfg.conditions:
        if cond not in models or cond not in tokenizers:
            raise KeyError(f"missing model/tokenizer for {cond}")
        for seed in seeds:
            logs.append(
                finetune_condition(
                    condition=cond,
                    model=models[cond],
                    tokenizer=tokenizers[cond],
                    texts=texts,
                    cfg=cfg,
                    run_dir=run_dir,
                    seed=seed,
                )
            )
    summary = [log.to_dict() for log in logs]
    (run_dir / "finetune_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return logs


def dry_run_demo(
    *,
    run_dir: Path | str,
    seed: int = 0,
) -> list[FinetuneRunLog]:
    """Dry-run fine-tune plan on the tiny random-init fixture model.

    Does **not** download Qwen or run real training.
    """
    from toklab.models.tiny_fixture import (
        build_tiny_bundle,
        tiny_adapted_vocab,
    )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_tiny_bundle(seed=seed)
    vocab = tiny_adapted_vocab(bundle)
    texts = [
        "def AcmeWidgetFactory(x):\n    return x\n",
        "class HelperTool:\n    pass\n",
    ]
    # One shared model reference per condition for the dry-run plan only.
    models = {c: bundle.model for c in CONDITIONS}
    tokenizers = {c: bundle.tokenizer for c in CONDITIONS}
    cfg = FinetuneConfig(
        seed=seed,
        n_seeds=1,
        max_steps=3,
        dry_run=True,
        use_lora=True,
        train_embeddings=True,
        device="cpu",
    )
    (run_dir / "tiny_fixture_meta.json").write_text(
        json.dumps(
            {
                "vocab_size": bundle.vocab_size,
                "n_adapted": len(vocab),
                "tie_word_embeddings": bundle.tie_word_embeddings,
                "note": "dry_run demo on tiny random-init fixture; not Qwen",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return finetune_grid(
        models=models,
        tokenizers=tokenizers,
        texts=texts,
        cfg=cfg,
        run_dir=run_dir,
    )
