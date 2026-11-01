"""Pydantic schema for TokLab experiment YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B"
    tokenizer_name: str | None = None
    tie_word_embeddings: bool = True
    d_model: int = 896
    num_hidden_layers: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 64


class CorpusConfig(BaseModel):
    """Source text read by every stage: ``original`` repos or ``synthetic``."""

    kind: Literal["original", "synthetic"]
    dir: Path
    manifest: Path


class DataConfig(BaseModel):
    # Either ``corpus`` or (legacy) ``repos_dir`` + ``manifest_path``.
    corpus: CorpusConfig | None = None
    repos_dir: Path | None = None
    manifest_path: Path | None = None
    synthetic_dir: Path | None = None
    # Primary evaluation set (hypothesis verdicts use it first).
    eval_split: str = "test"
    # All evaluation sets reported; defaults to ``[eval_split]``.
    eval_splits: list[str] | None = None
    adapt_split: str = "adapt"
    # Split used to choose K* (docs/decisions/005-k-selection.md).
    k_selection_split: str = "val"

    @model_validator(mode="after")
    def _check_source(self) -> DataConfig:
        if self.corpus is None and (
            self.repos_dir is None or self.manifest_path is None
        ):
            raise ValueError(
                "data needs corpus: {kind, dir, manifest} "
                "(or legacy repos_dir + manifest_path)"
            )
        if self.k_selection_split in {self.adapt_split, *self.resolved_eval_splits()}:
            raise ValueError(
                "k_selection_split must differ from adapt and evaluation splits"
            )
        return self

    def resolved_corpus(self) -> CorpusConfig:
        if self.corpus is not None:
            return self.corpus
        assert self.repos_dir is not None and self.manifest_path is not None
        return CorpusConfig(
            kind="original", dir=self.repos_dir, manifest=self.manifest_path
        )

    def resolved_eval_splits(self) -> list[str]:
        splits = list(self.eval_splits or [self.eval_split])
        if self.eval_split not in splits:
            splits.insert(0, self.eval_split)
        if self.adapt_split in splits:
            raise ValueError("the adapt split cannot be an evaluation set")
        return splits


class VocabConfig(BaseModel):
    # Nominal sweep; values above the number of positive-gain candidates are
    # capped (see runner.effective_k_values).
    k_values: list[int] = Field(default_factory=lambda: [0, 128])
    min_freq: int = 1
    min_repos: int = 1

    @model_validator(mode="after")
    def _check_k(self) -> VocabConfig:
        if not self.k_values or any(k < 0 for k in self.k_values):
            raise ValueError("vocab.k_values must be a non-empty list of K >= 0")
        return self


class IntrinsicConfig(BaseModel):
    conditions: list[str] = Field(
        default_factory=lambda: ["C2", "C3", "C4", "C5"]
    )
    context_budgets: list[int] = Field(
        default_factory=lambda: [512, 1024, 2048, 4096]
    )
    n_boot: int = 200


class SystemsConfig(BaseModel):
    """Latency benchmark; runs as its own step (``toklab run systems``) at K*."""

    enabled: bool = False
    conditions: list[str] = Field(default_factory=lambda: ["C2", "C4"])
    seq_lens: list[int] = Field(default_factory=lambda: [64, 128])
    n_warmup: int = 5
    n_repeats: int = 30
    dtype: str = "float32"
    device: str | None = None
    # Eval split whose token streams feed the model (real inputs, not random).
    input_split: str = "test"


class StagesConfig(BaseModel):
    fidelity: bool = True
    intrinsic: bool = True
    factorial: bool = True
    context: bool = True
    vocab_cost: bool = True
    # Unused by the runner: systems is the separate `toklab run systems` step
    # gated by ``systems.enabled``. Kept so older configs still validate.
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
