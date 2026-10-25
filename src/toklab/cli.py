"""TokLab command-line interface."""

import typer

from toklab import __version__

app = typer.Typer(help="TokLab research workbench")


@app.callback()
def _main() -> None:
    """TokLab: measure structure-aware and domain-adapted code tokenizers."""


@app.command("version")
def version_cmd() -> None:
    """Print the installed TokLab version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
