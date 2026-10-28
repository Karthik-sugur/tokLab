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
