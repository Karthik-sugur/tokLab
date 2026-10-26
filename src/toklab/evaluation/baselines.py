"""Run baseline conditions C1, C2, C6: fidelity + intrinsic + CI table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from toklab.core.types import SourceFile
from toklab.data.splits import ManifestView
from toklab.evaluation.fidelity import FidelityReport, check_roundtrip
from toklab.evaluation.intrinsic import measure_files, write_metrics_parquet
from toklab.evaluation.stats import BootstrapCI, cluster_bootstrap_ci
from toklab.tokenizers.registry import BASELINE_CONDITIONS, get_tokenizer


def _load_texts(repos_dir: Path, files: tuple[SourceFile, ...]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for sf in files:
        texts[sf.file_id] = (repos_dir / sf.repo / sf.path).read_text(encoding="utf-8")
    return texts


def _tpk_by_repo(
    files: tuple[SourceFile, ...],
    metrics_df: pd.DataFrame,
    condition: str,
) -> dict[str, list[float]]:
    id_to_repo = {f.file_id: f.repo for f in files}
    sub = metrics_df[
        (metrics_df["condition"] == condition)
        & (metrics_df["metric"] == "tokens_per_kb")
    ]
    out: dict[str, list[float]] = {}
    for row in sub.itertuples(index=False):
        repo = id_to_repo[row.file_id]
        out.setdefault(repo, []).append(float(row.value))
    return out


def format_ci_table(rows: list[tuple[str, BootstrapCI]]) -> str:
    """Plain-text table of mean tokens/KB with CI per condition."""
    lines = [
        f"{'condition':<10} {'mean_tpk':>12} {'ci_low':>12} {'ci_high':>12} "
        f"{'n_repos':>8}",
        "-" * 58,
    ]
    for cond, ci in rows:
        lines.append(
            f"{cond:<10} {ci.estimate:12.4f} {ci.low:12.4f} {ci.high:12.4f} "
            f"{ci.n_repos:8d}"
        )
    return "\n".join(lines)


def run_baselines(
    *,
    manifest: Path | str,
    repos_dir: Path | str,
    out_dir: Path | str,
    split: str = "test",
    seed: int = 0,
    n_boot: int = 1000,
    cfg: dict[str, Any] | None = None,
    conditions: tuple[str, ...] = BASELINE_CONDITIONS,
) -> tuple[str, dict[str, FidelityReport], list[tuple[str, BootstrapCI]]]:
    """Evaluate baseline tokenizers; write metrics parquet; return CI table."""
    repos_dir = Path(repos_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    view = ManifestView.from_parquet(manifest, split)
    files = view.files()
    if not files:
        raise ValueError(f"no files in split {split!r} of {manifest}")

    texts = _load_texts(repos_dir, files)
    all_metrics: list[pd.DataFrame] = []
    fidelity: dict[str, FidelityReport] = {}
    ci_rows: list[tuple[str, BootstrapCI]] = []

    for condition in conditions:
        tok = get_tokenizer(condition, cfg)
        report = check_roundtrip(tok, files, texts=texts)
        fidelity[condition] = report
        if report.n_roundtrip_fail:
            raise RuntimeError(
                f"{condition} failed round-trip on {report.n_roundtrip_fail} "
                f"file(s): {report.failures[:3]}"
            )

        df = measure_files(tok, files, texts=texts, condition=condition)
        all_metrics.append(df)
        by_repo = _tpk_by_repo(files, df, condition)
        ci = cluster_bootstrap_ci(by_repo, seed=seed, n_boot=n_boot)
        ci_rows.append((condition, ci))

    metrics = pd.concat(all_metrics, ignore_index=True)
    write_metrics_parquet(metrics, out_dir / "intrinsic_metrics.parquet")

    # Persist fidelity summary (counts only — no invented rates beyond data).
    fid_rows = [
        {
            "condition": name,
            "n_files": r.n_files,
            "n_roundtrip_ok": r.n_roundtrip_ok,
            "n_roundtrip_fail": r.n_roundtrip_fail,
            "n_orig_parsed": r.n_orig_parsed,
            "n_reparse_ok": r.n_reparse_ok,
            "n_reparse_fail": r.n_reparse_fail,
        }
        for name, r in fidelity.items()
    ]
    pd.DataFrame(fid_rows).to_parquet(out_dir / "fidelity.parquet", index=False)

    table = format_ci_table(ci_rows)
    (out_dir / "baselines_ci_table.txt").write_text(table + "\n", encoding="utf-8")
    return table, fidelity, ci_rows
