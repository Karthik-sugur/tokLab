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


@dataclass(frozen=True)
class FactorialEffects:
    """2x2 structure × domain factorial contrasts with cluster-bootstrap CIs."""

    metric: str
    k: int
    m_c2: BootstrapCI
    m_c3: BootstrapCI
    m_c4: BootstrapCI
    m_c5: BootstrapCI
    effect_s: BootstrapCI  # approx C4 - C2 on paired repo means
    effect_d: BootstrapCI  # C3 - C2
    interaction: BootstrapCI  # I = (C5-C4) - (C3-C2)
    r_s: float
    r_d: float
    independence_expectation: float
    decision_rule: str
    combined_beats_either: bool | None


def _repo_means(repo_to_values: Mapping[str, Sequence[float]]) -> dict[str, float]:
    return {r: float(np.mean(list(vs))) for r, vs in repo_to_values.items()}


def _bootstrap_paired_diff(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    seed: int,
    n_boot: int,
    alpha: float,
) -> BootstrapCI:
    """Bootstrap mean(a_r - b_r) over repos present in both maps."""
    repos = sorted(set(a) & set(b))
    if not repos:
        raise ValueError("no shared repos for paired contrast")
    diffs = {r: a[r] - b[r] for r in repos}
    return cluster_bootstrap_ci(
        {r: [diffs[r]] for r in repos},
        seed=seed,
        n_boot=n_boot,
        alpha=alpha,
    )


def factorial_analysis(
    condition_repo_values: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    k: int,
    seed: int = 0,
    n_boot: int = 1000,
    alpha: float = 0.05,
    metric: str = "tokens_per_kb",
    decision_rule: str = (
        "Combined beats either alone on a metric only if C5 beats the "
        "better of C3/C4 with a CI excluding zero."
    ),
) -> FactorialEffects:
    """Factorial effects over C2,C3,C4,C5 for one metric.

    Grid: notS/notD=C2, notS/D=C3, S/notD=C4, S/D=C5.
    Interaction ``I = [m(S,D)-m(S,notD)] - [m(notS,D)-m(notS,notD)]``
    i.e. ``(C5 - C4) - (C3 - C2)`` on repo-level means (lower metric =
    fewer tokens/KB).

    Token-reduction rates vs C2: ``r = 1 - m/m_C2`` (using point estimates).
    Independence expectation: ``1 - (1-r_S)(1-r_D)``.
    """
    need = ("C2", "C3", "C4", "C5")
    for c in need:
        if c not in condition_repo_values:
            raise ValueError(f"missing condition {c} in factorial input")

    cis = {
        c: cluster_bootstrap_ci(
            condition_repo_values[c],
            seed=seed + hash(c) % 1000,
            n_boot=n_boot,
            alpha=alpha,
        )
        for c in need
    }
    means = {c: _repo_means(condition_repo_values[c]) for c in need}

    effect_s = _bootstrap_paired_diff(
        means["C4"],
        means["C2"],
        seed=seed + 11,
        n_boot=n_boot,
        alpha=alpha,
    )
    effect_d = _bootstrap_paired_diff(
        means["C3"],
        means["C2"],
        seed=seed + 13,
        n_boot=n_boot,
        alpha=alpha,
    )

    # Interaction on repo means: I_r = (C5_r - C4_r) - (C3_r - C2_r)
    repos = sorted(
        set(means["C2"])
        & set(means["C3"])
        & set(means["C4"])
        & set(means["C5"])
    )
    inter = {
        r: (means["C5"][r] - means["C4"][r])
        - (means["C3"][r] - means["C2"][r])
        for r in repos
    }
    interaction = cluster_bootstrap_ci(
        {r: [inter[r]] for r in repos},
        seed=seed + 17,
        n_boot=n_boot,
        alpha=alpha,
    )

    m2, m3, m4, m5 = (cis[c].estimate for c in need)
    r_s = 0.0 if m2 == 0 else 1.0 - (m4 / m2)
    r_d = 0.0 if m2 == 0 else 1.0 - (m3 / m2)
    indep = 1.0 - (1.0 - r_s) * (1.0 - r_d)

    # Decision rule: C5 beats better of C3/C4 (lower better), CI excludes 0.
    if m3 <= m4:
        beat = _bootstrap_paired_diff(
            means["C5"],
            means["C3"],
            seed=seed + 19,
            n_boot=n_boot,
            alpha=alpha,
        )
    else:
        beat = _bootstrap_paired_diff(
            means["C5"],
            means["C4"],
            seed=seed + 19,
            n_boot=n_boot,
            alpha=alpha,
        )
    combined_beats = beat.estimate < 0 and beat.high < 0
    _ = m5

    return FactorialEffects(
        metric=metric,
        k=k,
        m_c2=cis["C2"],
        m_c3=cis["C3"],
        m_c4=cis["C4"],
        m_c5=cis["C5"],
        effect_s=effect_s,
        effect_d=effect_d,
        interaction=interaction,
        r_s=r_s,
        r_d=r_d,
        independence_expectation=indep,
        decision_rule=decision_rule,
        combined_beats_either=combined_beats,
    )
