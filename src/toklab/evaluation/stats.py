"""Statistical helpers (cluster bootstrap by repository)."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate and percentile confidence interval."""

    estimate: float
    low: float
    high: float
    n_repos: int
    n_boot: int
    seed: int
    alpha: float


def cluster_bootstrap_ci(
    repo_to_values: Mapping[str, Sequence[float]],
    *,
    seed: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    statistic: Callable[[list[float]], float] | None = None,
) -> BootstrapCI:
    """Cluster bootstrap CI, resampling **repositories** with replacement.

    For each bootstrap replicate, repos are drawn with replacement; all
    values belonging to drawn repos are pooled; ``statistic`` (default:
    mean) is computed on that pool.
    """
    if not repo_to_values:
        raise ValueError("repo_to_values is empty")
    repos = sorted(repo_to_values.keys())
    for r in repos:
        if len(repo_to_values[r]) == 0:
            raise ValueError(f"repo {r!r} has no values")

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs))

    stat = statistic or _mean

    # Point estimate on the observed (non-resampled) pool.
    observed_pool = [v for r in repos for v in repo_to_values[r]]
    estimate = stat(list(observed_pool))

    rng = random.Random(seed)
    samples: list[float] = []
    n_repos = len(repos)
    for _ in range(n_boot):
        drawn = [repos[rng.randrange(n_repos)] for _ in range(n_repos)]
        pool: list[float] = []
        for r in drawn:
            pool.extend(repo_to_values[r])
        samples.append(stat(pool))

    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return BootstrapCI(
        estimate=estimate,
        low=lo,
        high=hi,
        n_repos=n_repos,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )


def tokens_per_kb_by_repo(
    metrics: Sequence[tuple[str, str, float]],
) -> dict[str, list[float]]:
    """Group file-level tokens_per_kb values by repo.

    ``metrics`` rows are ``(repo, file_id, tokens_per_kb)``.
    """
    out: dict[str, list[float]] = {}
    for repo, _file_id, value in metrics:
        out.setdefault(repo, []).append(float(value))
    return out
