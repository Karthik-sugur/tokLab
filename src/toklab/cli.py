"""TokLab command-line interface."""

from pathlib import Path
from typing import Annotated

import typer

from toklab import __version__

app = typer.Typer(help="TokLab research workbench")
data_app = typer.Typer(help="Dataset ingestion and splits")
app.add_typer(data_app, name="data")


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


if __name__ == "__main__":
    app()
