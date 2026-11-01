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


@app.command("serve-api")
def serve_api_cmd(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
    datasets_root: Annotated[
        Path, typer.Option("--datasets-root")
    ] = Path("tests/fixtures"),
) -> None:
    """Start the FastAPI server (uvicorn)."""
    import uvicorn

    from toklab.api.app import create_app

    api = create_app(runs_root=runs_root, datasets_root=datasets_root)
    uvicorn.run(api, host=host, port=port)


@app.command("serve-ui")
def serve_ui_cmd(
    port: Annotated[int, typer.Option("--port")] = 8501,
    api_url: Annotated[
        str,
        typer.Option(
            "--api-url",
            help="API base URL the UI expects (shown at startup)",
        ),
    ] = "http://127.0.0.1:8000",
) -> None:
    """Start the Streamlit UI (display-only; talks to the API)."""
    import subprocess
    import sys
    from pathlib import Path as P

    # src/toklab/cli.py → parents[2] is the repository root.
    ui_app = P(__file__).resolve().parents[2] / "ui" / "app.py"
    ui_url = f"http://127.0.0.1:{port}"
    typer.echo(f"UI URL:  {ui_url}")
    typer.echo(f"API URL: {api_url.rstrip('/')}  (start with: toklab serve-api)")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_app),
        "--server.port",
        str(port),
    ]
    typer.echo(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


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
    import pandas as pd

    from toklab.core.types import SourceFile
    from toklab.data.splits import (
        build_dataset,
        format_split_detail,
        split_counts_message,
    )

    counts = build_dataset(repos_dir=repos, out_dir=out, seed=seed)
    typer.echo(split_counts_message(counts))
    df = pd.read_parquet(out / "manifest.parquet")
    assigned = [
        SourceFile(
            file_id=row.file_id,
            repo=row.repo,
            path=row.path,
            sha256=row.sha256,
            language=row.language,
            n_bytes=int(row.n_bytes),
            split=row.split,
        )
        for row in df.itertuples(index=False)
    ]
    typer.echo(format_split_detail(assigned))
    typer.echo(f"manifest={(out / 'manifest.parquet').resolve()}")
    digest = (out / "manifest_hash").read_text(encoding="utf-8").strip()
    typer.echo(f"manifest_hash={digest}")
    typer.echo(f"manifest_hash_file={(out / 'manifest_hash').resolve()}")


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
    only: Annotated[
        Path | None,
        typer.Option(
            "--only",
            help="Optional file listing repo/path keys to process (diagnostics)",
        ),
    ] = None,
) -> None:
    """Rename adapt/val/test (lexicon_adapt); drift variants for drift_holdout."""
    from toklab.data.synth import run_synth

    only_keys = None
    if only is not None:
        only_keys = {
            line.strip()
            for line in only.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    summary = run_synth(
        manifest=manifest,
        repos_dir=repos,
        out_dir=out,
        seed=seed,
        only=only_keys,
    )
    typer.echo(
        "synth "
        f"adapt_files={summary['adapt_files']} "
        f"test_inorg_files={summary['test_inorg_files']} "
        f"val_files={summary['val_files']} "
        f"test_files={summary['test_files']} "
        f"drift_holdout_files={summary['drift_holdout_files']} "
        f"seed={summary['seed']} "
        f"manifest_hash={summary['manifest_hash']}"
    )
    for key, v in summary["variants"].items():
        typer.echo(
            f"  {key}: files={v['files']} renamed={v['files_renamed']} "
            f"copied_unchanged={v['files_copied_unchanged']} "
            f"reasons={v['unchanged_reasons']} "
            f"identifiers_renamed={v['identifiers_renamed']}"
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
    typer.echo(f"k_star={status.get('k_star')}")
    typer.echo(f"state={status.get('state')}")
    if status.get("report_path"):
        typer.echo(f"report={status['report_path']}")


@run_app.command("drift")
def run_drift_cmd(
    run: Annotated[
        str,
        typer.Option("--run", help="Experiment config hash under runs/"),
    ],
    runs_root: Annotated[
        Path,
        typer.Option("--runs-root", help="Runs root directory"),
    ] = Path("runs"),
    k_refresh: Annotated[
        int,
        typer.Option("--k-refresh", help="Max new append-only refresh tokens"),
    ] = 4,
    v1_sample_size: Annotated[
        int,
        typer.Option(
            "--v1-sample-size",
            help="How many drifted holdout files to mine for refresh",
        ),
    ] = 1,
) -> None:
    """Vocabulary drift sweep for an existing experiment run."""
    from toklab.evaluation.drift import run_drift_for_experiment_hash

    table, run_dir = run_drift_for_experiment_hash(
        run,
        runs_root=runs_root,
        k_refresh=k_refresh,
        v1_sample_size=v1_sample_size,
    )
    typer.echo(table)
    typer.echo(f"run_dir={run_dir.resolve()}")
    typer.echo(f"drift_table={(run_dir / 'drift_table.txt').resolve()}")
    typer.echo(f"report={(run_dir / 'report.md').resolve()}")


@run_app.command("systems")
def run_systems_cmd(
    run: Annotated[
        str,
        typer.Option("--run", help="Experiment config hash under runs/"),
    ],
    runs_root: Annotated[
        Path,
        typer.Option("--runs-root", help="Runs root directory"),
    ] = Path("runs"),
) -> None:
    """Systems latency benchmark (C2–C5 at the run's K*); needs torch."""
    from toklab.evaluation.systems import run_systems_for_experiment_hash

    df, run_dir = run_systems_for_experiment_hash(run, runs_root=runs_root)
    typer.echo(f"systems_rows={len(df)}")
    typer.echo(f"systems={(run_dir / 'systems.parquet').resolve()}")
    typer.echo(f"report={(run_dir / 'report.md').resolve()}")


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
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            help="md: re-render runs/<hash>/report.md; html: also write a "
            "self-contained HTML copy (opens with no server)",
        ),
    ] = "md",
    out: Annotated[
        Path,
        typer.Option("--out", help="HTML output path (with --format html)"),
    ] = Path("demo_assets/report.html"),
) -> None:
    """Re-render the report for an existing run hash."""
    from toklab.evaluation.report import render_report

    if fmt not in {"md", "html"}:
        raise typer.BadParameter(f"--format must be md or html, got {fmt!r}")
    run_dir = runs_root / run
    if not run_dir.is_dir():
        raise typer.BadParameter(f"run directory not found: {run_dir}")
    path = render_report(run_dir=run_dir, hypotheses_path=hypotheses)
    typer.echo(f"report={path.resolve()}")
    if fmt == "html":
        from toklab.evaluation.report_html import write_report_html

        html_path = write_report_html(path, out)
        typer.echo(f"report_html={html_path.resolve()}")


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
