"""Post-hoc paired per-repository differences from stored per-file metrics.

Descriptive only: not the pre-registered H1–H5 rule. Reuses
``intrinsic_metrics.parquet``; never re-encodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from toklab.evaluation.stats import cluster_bootstrap_ci

DEFAULT_CONTRASTS = (
    ("C3-C2", "C3", "C2"),
    ("C4-C2", "C4", "C2"),
    ("C5-C2", "C5", "C2"),
    ("C5-min(C3,C4)", "C5", "min(C3,C4)"),
)


def _repo_file_map(
    metrics: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    """Wide table of tokens/KB per (eval_set, k, repo, file_id)."""
    tpk = metrics[metrics["metric"] == "tokens_per_kb"].copy()
    man = manifest[["file_id", "repo"]].drop_duplicates()
    m = tpk.merge(man, on="file_id", how="inner")
    if m["repo"].isna().any():
        raise ValueError("intrinsic_metrics has file_ids missing from the manifest")
    return m


def paired_differences(
    metrics: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    eval_sets: list[str] | None = None,
    k: int | None = None,
    seed: int = 0,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Repo-weighted paired diffs for each eval set at one K (default: max K)."""
    m = _repo_file_map(metrics, manifest)
    if eval_sets is None:
        eval_sets = sorted(m["eval_set"].dropna().unique())
    if k is None:
        k = int(m["k"].max())
    rows: list[dict[str, Any]] = []
    for es in eval_sets:
        sub = m[(m["eval_set"] == es) & (m["k"] == k)]
        if sub.empty:
            continue
        wide = sub.pivot_table(
            index=["repo", "file_id"], columns="condition", values="value"
        )
        rmeans = wide.groupby(level=0).mean()
        for label, a, b in DEFAULT_CONTRASTS:
            if a not in rmeans.columns:
                continue
            if b == "min(C3,C4)":
                if not {"C3", "C4"}.issubset(rmeans.columns):
                    continue
                diff = rmeans[a] - rmeans[["C3", "C4"]].min(axis=1)
            else:
                if b not in rmeans.columns:
                    continue
                diff = rmeans[a] - rmeans[b]
            ci = cluster_bootstrap_ci(
                {r: [float(v)] for r, v in diff.items()},
                seed=seed,
                n_boot=n_boot,
                alpha=alpha,
            )
            # Lower tokens/KB is better: "repos better" = repos where diff < 0
            # when contrasting A vs B with A expected to save tokens.
            n_better = int((diff < 0).sum())
            n_worse = int((diff > 0).sum())
            n_tie = int((diff == 0).sum())
            rows.append(
                {
                    "eval_set": es,
                    "k": k,
                    "contrast": label,
                    "mean": ci.estimate,
                    "ci_low": ci.low,
                    "ci_high": ci.high,
                    "n_repos": ci.n_repos,
                    "n_boot": ci.n_boot,
                    "seed": ci.seed,
                    "alpha": ci.alpha,
                    "repos_better": n_better,
                    "repos_worse": n_worse,
                    "repos_tie": n_tie,
                }
            )
    return pd.DataFrame(rows)


def run_paired_for_experiment(
    run_dir: Path,
    *,
    manifest: Path,
    seed: int = 0,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> Path:
    """Write ``paired_differences.parquet`` under ``run_dir``; return its path."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "intrinsic_metrics.parquet"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing {metrics_path}")
    metrics = pd.read_parquet(metrics_path)
    man = pd.read_parquet(manifest)
    selection = {}
    sel_path = run_dir / "k_selection.json"
    if sel_path.is_file():
        import json

        selection = json.loads(sel_path.read_text(encoding="utf-8"))
    k = selection.get("k_star")
    if k is None and "k" in metrics.columns and not metrics.empty:
        k = int(metrics["k"].max())
    df = paired_differences(
        metrics, man, k=None if k is None else int(k), seed=seed, n_boot=n_boot,
        alpha=alpha,
    )
    out = run_dir / "paired_differences.parquet"
    df.to_parquet(out, index=False)
    meta = {
        "post_hoc": True,
        "note": (
            "Descriptive paired differences added after seeing results; "
            "not the pre-registered rule; does not change any verdict."
        ),
        "seed": seed,
        "n_boot": n_boot,
        "alpha": alpha,
        "k": None if k is None else int(k),
        "n_rows": int(len(df)),
    }
    import json

    (run_dir / "paired_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return out
