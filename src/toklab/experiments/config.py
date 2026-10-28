"""Pydantic schema for TokLab experiment YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B"
    tokenizer_name: str | None = None
    tie_word_embeddings: bool = True
    d_model: int = 896
    num_hidden_layers: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 64


class DataConfig(BaseModel):
    repos_dir: Path
    manifest_path: Path
    eval_split: str = "test"
    adapt_split: str = "adapt"


class VocabConfig(BaseModel):
    k_values: list[int] = Field(default_factory=lambda: [0, 128])
    min_freq: int = 1
    min_repos: int = 1


class IntrinsicConfig(BaseModel):
    conditions: list[str] = Field(
        default_factory=lambda: ["C2", "C3", "C4", "C5"]
    )
    context_budgets: list[int] = Field(
        default_factory=lambda: [512, 1024, 2048, 4096]
    )
    n_boot: int = 200


class SystemsConfig(BaseModel):
    enabled: bool = False
    conditions: list[str] = Field(default_factory=lambda: ["C2", "C4"])
    seq_lens: list[int] = Field(default_factory=lambda: [64, 128])
    n_warmup: int = 5
    n_repeats: int = 30
    dtype: str = "float32"
    device: str | None = None


class StagesConfig(BaseModel):
    fidelity: bool = True
    intrinsic: bool = True
    factorial: bool = True
    context: bool = True
    vocab_cost: bool = True
    systems: bool = False
    report: bool = True


class ExperimentConfig(BaseModel):
    """Top-level experiment YAML schema."""

    name: str = "mvp"
    seed: int = 0
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    vocab: VocabConfig = Field(default_factory=VocabConfig)
    intrinsic: IntrinsicConfig = Field(default_factory=IntrinsicConfig)
    systems: SystemsConfig = Field(default_factory=SystemsConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    hypotheses_path: Path = Path("configs/hypotheses.yaml")
    runs_root: Path = Path("runs")

    def canonical_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict for hashing (paths as posix strings)."""
        raw = self.model_dump(mode="json")
        return raw


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return ExperimentConfig.model_validate(data)
