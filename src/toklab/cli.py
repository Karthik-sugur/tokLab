"""TokLab command-line interface."""

from pathlib import Path
from typing import Annotated

import typer

from toklab import __version__

app = typer.Typer(help="TokLab research workbench")
data_app = typer.Typer(help="Dataset ingestion and splits")
run_app = typer.Typer(help="Experiment runners")
app.add_typer(data_app, name="data")
app.add_typer(run_app, name="run")


@app.callback()
def _main() -> None:
    """TokLab: measure structure-aware and domain-adapted code tokenizers."""


@app.command("version")
def version_cmd() -> None:
    """Print the installed TokLab version."""
    typer.echo(__version__)


@data_app.command("build")
def data_build(
    repos: Annotated[
        Path,
        typer.Option("--repos", help="Directory of repo checkouts"),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for the manifest"),
    ],
    seed: Annotated[
        int,
        typer.Option("--seed", help="RNG seed for repository splits"),
    ] = 0,
) -> None:
    """Ingest repos, deduplicate, split by repository, write manifest."""
    from toklab.data.splits import build_dataset, split_counts_message

    counts = build_dataset(repos_dir=repos, out_dir=out, seed=seed)
    typer.echo(split_counts_message(counts))
    typer.echo(f"manifest={(out / 'manifest.parquet').resolve()}")
    typer.echo(f"manifest_hash={(out / 'manifest_hash').resolve()}")


@data_app.command("synth")
def data_synth(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Path to manifest.parquet"),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for synth artifacts"),
    ],
    repos: Annotated[
        Path,
        typer.Option(
            "--repos",
            help="Directory of repo checkouts (needed to read source text)",
        ),
    ],
    seed: Annotated[
        int,
        typer.Option("--seed", help="RNG seed for lexicon and drift"),
    ] = 0,
) -> None:
    """Apply enterprise rename (adapt) and drift variants (drift_holdout)."""
    from toklab.data.synth import run_synth

    summary = run_synth(
        manifest=manifest,
        repos_dir=repos,
        out_dir=out,
        seed=seed,
    )
    typer.echo(
        "synth "
        f"adapt_files={summary['adapt_files']} "
        f"drift_holdout_files={summary['drift_holdout_files']} "
        f"seed={summary['seed']}"
    )


@run_app.command("factorial")
def run_factorial_cmd(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Path to manifest.parquet"),
    ],
    repos: Annotated[
        Path,
        typer.Option(
            "--repos",
            help="Directory of repo checkouts (needed to read source text)",
        ),
    ],
    k: Annotated[
        str,
        typer.Option("--k", help="Comma-separated K values, e.g. 0,128,512,2048"),
    ] = "0,128,512,2048",
    split: Annotated[
        str,
        typer.Option("--split", help="Eval split (vocab always from adapt)"),
    ] = "test",
    seed: Annotated[
        int,
        typer.Option("--seed", help="Bootstrap RNG seed"),
    ] = 0,
    n_boot: Annotated[
        int,
        typer.Option("--n-boot", help="Bootstrap replicates"),
    ] = 200,
) -> None:
    """2x2 factorial over C2–C5 for a K sweep; print effects table."""
    from toklab.evaluation.factorial import run_factorial

    k_values = [int(x.strip()) for x in k.split(",") if x.strip()]
    table, _effects = run_factorial(
        manifest=manifest,
        repos_dir=repos,
        k_values=k_values,
        split=split,
        seed=seed,
        n_boot=n_boot,
    )
    typer.echo(table)
    typer.echo(
        "NOTE: results above are from the provided manifest/repos "
        "(fixtures if you used test fixtures)."
    )


@run_app.command("compare")
def run_compare_cmd(
    conditions: Annotated[
        str,
        typer.Option(
            "--conditions",
            help="Comma-separated condition ids, e.g. C2,C4",
        ),
    ],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Path to manifest.parquet"),
    ],
    repos: Annotated[
        Path,
        typer.Option(
            "--repos",
            help="Directory of repo checkouts (needed to read source text)",
        ),
    ],
    split: Annotated[
        str,
        typer.Option("--split", help="Manifest split to evaluate"),
    ] = "test",
) -> None:
    """Print token-count comparison across conditions (no direction asserted)."""
    from toklab.evaluation.compare import compare_conditions

    conds = [c.strip() for c in conditions.split(",") if c.strip()]
    if not conds:
        raise typer.BadParameter("provide at least one condition")
    table = compare_conditions(
        manifest=manifest,
        repos_dir=repos,
        conditions=conds,
        split=split,
    )
    typer.echo(table)


@run_app.command("experiment")
def run_experiment_cmd(
    config: Annotated[
        Path,
        typer.Option("--config", help="Experiment YAML path"),
    ],
) -> None:
    """Run a full experiment from YAML; write runs/<config_hash>/."""
    from toklab.experiments.runner import run_experiment

    status = run_experiment(config)
    typer.echo(f"config_hash={status['config_hash']}")
    typer.echo(f"run_dir={status.get('run_dir')}")
    typer.echo(f"state={status.get('state')}")
    if status.get("report_path"):
        typer.echo(f"report={status['report_path']}")


@run_app.command("finetune-dry")
def run_finetune_dry_cmd(
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for dry-run logs"),
    ] = Path("runs/finetune_dry_tiny"),
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Dry-run C2..C5 fine-tune plan on the tiny fixture (no real training)."""
    from toklab.models.finetune import dry_run_demo

    logs = dry_run_demo(run_dir=out, seed=seed)
    typer.echo(f"out={out.resolve()}")
    typer.echo(f"n_logs={len(logs)}")
    for log in logs:
        typer.echo(
            f"  {log.condition} seed={log.seed} dry_run={log.dry_run} "
            f"steps_executed={log.steps_executed} "
            f"n_train_bytes={log.n_train_bytes}"
        )


@app.command("report")
def report_cmd(
    run: Annotated[
        str,
        typer.Option("--run", help="Config hash (directory under runs/)"),
    ],
    runs_root: Annotated[
        Path,
        typer.Option("--runs-root", help="Runs root directory"),
    ] = Path("runs"),
    hypotheses: Annotated[
        Path,
        typer.Option("--hypotheses", help="hypotheses.yaml path"),
    ] = Path("configs/hypotheses.yaml"),
) -> None:
    """Re-render Markdown report for an existing run hash."""
    from toklab.evaluation.report import render_report

    run_dir = runs_root / run
    if not run_dir.is_dir():
        raise typer.BadParameter(f"run directory not found: {run_dir}")
    path = render_report(run_dir=run_dir, hypotheses_path=hypotheses)
    typer.echo(f"report={path.resolve()}")


@run_app.command("baselines")
def run_baselines_cmd(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Path to manifest.parquet"),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for metrics"),
    ],
    repos: Annotated[
        Path,
        typer.Option(
            "--repos",
            help="Directory of repo checkouts (needed to read source text)",
        ),
    ],
    split: Annotated[
        str,
        typer.Option("--split", help="Manifest split to evaluate"),
    ] = "test",
    seed: Annotated[
        int,
        typer.Option("--seed", help="Bootstrap RNG seed"),
    ] = 0,
    n_boot: Annotated[
        int,
        typer.Option("--n-boot", help="Bootstrap replicates"),
    ] = 1000,
) -> None:
    """Run baseline tokenizers C1, C2, C6; print mean tokens/KB with CI."""
    from toklab.evaluation.baselines import run_baselines

    table, _fid, _cis = run_baselines(
        manifest=manifest,
        repos_dir=repos,
        out_dir=out,
        split=split,
        seed=seed,
        n_boot=n_boot,
    )
    typer.echo(table)


if __name__ == "__main__":
    app()
