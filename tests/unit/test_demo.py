"""Demo support: UI defaults, table filters, static HTML report, scripts."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from toklab.cli import app
from toklab.evaluation.report import render_report
from toklab.evaluation.report_html import markdown_to_html

ROOT = Path(__file__).resolve().parents[2]
_UI = ROOT / "ui"
if str(_UI) not in sys.path:
    sys.path.insert(0, str(_UI))

from demo_defaults import (  # noqa: E402
    FALLBACK_SNIPPET,
    default_api_url,
    default_run_id,
    default_snippet,
)
from table_filters import (  # noqa: E402
    eval_sets,
    filter_rows,
    k_label,
    k_star,
    k_values,
)


def _load_write_run():
    spec = importlib.util.spec_from_file_location(
        "_ksel", ROOT / "tests" / "unit" / "test_k_selection.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._write_run


write_run = _load_write_run()


# --- default run id / snippet


def test_default_run_env_overrides_config(tmp_path: Path) -> None:
    cfg = tmp_path / "demo.yaml"
    cfg.write_text("demo_run: fromconfig\n", encoding="utf-8")
    assert default_run_id({"TOKLAB_DEFAULT_RUN": " fromenv "}, cfg) == "fromenv"
    assert default_run_id({}, cfg) == "fromconfig"
    assert default_run_id({}, tmp_path / "missing.yaml") == ""
    assert default_api_url({}) == "http://127.0.0.1:8000"
    assert default_api_url({"TOKLAB_API_URL": "http://x:1"}) == "http://x:1"


def test_demo_config_points_at_a_run_hash_and_a_val_file() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text("utf-8"))
    assert re.fullmatch(r"[0-9a-f]{64}", cfg["demo_run"])
    assert cfg["preview_file"].startswith("data/synthetic/val/")


def test_default_snippet_is_exact_prefix_of_file(tmp_path: Path) -> None:
    src = tmp_path / "f.py"
    src.write_bytes("a = 1\r\nb = 'é'\n\nc = 3\nd = 4\n".encode())
    cfg = tmp_path / "demo.yaml"
    cfg.write_text("preview_file: f.py\npreview_lines: 3\n", encoding="utf-8")
    assert default_snippet(cfg, tmp_path) == "a = 1\r\nb = 'é'\n\n"
    cfg.write_text("preview_file: nope.py\n", encoding="utf-8")
    assert default_snippet(cfg, tmp_path) == FALLBACK_SNIPPET


# --- table filters


def test_table_filters() -> None:
    rows = [
        {"eval_set": "test", "k": 0, "condition": "C2"},
        {"eval_set": "test", "k": 256, "condition": "C3"},
        {"eval_set": "test_inorg", "k": 256, "condition": "C3"},
    ]
    assert eval_sets(rows) == ["test", "test_inorg"]
    assert k_values(rows) == [0, 256]
    assert filter_rows(rows, "test", 256) == [rows[1]]
    assert filter_rows([{"seq_len": 1}], "test", 0) == [{"seq_len": 1}]
    assert k_star([{"k": 0, "k_star": False}, {"k": 256, "k_star": True}]) == 256
    assert k_star([{"k": 0}]) is None
    assert k_label(256, 256) == "256 (K*)" and k_label(0, 256) == "0"


# --- static HTML report


def test_markdown_to_html_subset() -> None:
    md = (
        "# Title\n\n"
        "**K\\* = 16** (ok). Use `a|b` <script>.\n"
        "second line\n\n"
        "| p | \\|V\\| |\n|---|---|\n| 0 | 3 |\n\n"
        "- one\n  continued\n- two\n"
    )
    html = markdown_to_html(md)
    assert "<h1>Title</h1>" in html
    assert (
        "<p><strong>K* = 16</strong> (ok). Use <code>a|b</code> &lt;script&gt;."
        " second line</p>"
    ) in html
    assert "<th>|V|</th>" in html and "<td>3</td>" in html
    assert "<li>one continued</li><li>two</li>" in html
    assert "<script>" not in html and "<pre>" not in html


def test_cli_report_html_writes_self_contained_file(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_run(runs / "abc", k_star=0)
    out = tmp_path / "demo_assets" / "report.html"
    res = CliRunner().invoke(
        app,
        ["report", "--run", "abc", "--runs-root", str(runs), "--format", "html",
         "--hypotheses", str(ROOT / "configs" / "hypotheses.yaml"),
         "--out", str(out)],
    )
    assert res.exit_code == 0, res.output
    assert f"report_html={out.resolve()}" in res.output
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "<h2>Limitations</h2>" in html and "<table>" in html
    assert "http" not in html  # no external assets: opens offline

    bad = CliRunner().invoke(
        app, ["report", "--run", "abc", "--runs-root", str(runs), "--format", "pdf"]
    )
    assert bad.exit_code != 0


def test_drift_proportions_come_from_run_data(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run(run_dir, k_star=0)
    hypo = ROOT / "configs" / "hypotheses.yaml"
    text = render_report(run_dir=run_dir, hypotheses_path=hypo).read_text("utf-8")
    assert "{0, 10, 25, 50}" not in text
    (run_dir / "drift_meta.json").write_text(
        json.dumps({"proportions": [0, 5], "k_static": 0}), encoding="utf-8"
    )
    text = render_report(run_dir=run_dir, hypotheses_path=hypo).read_text("utf-8")
    assert "Proportions p ∈ {0, 5}" in text


# --- README / DEMO.md carry no hand-written numbers


def _sync_module():
    spec = importlib.util.spec_from_file_location(
        "_sync", ROOT / "scripts" / "sync_readme_limitations.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_sync_readme_limitations_copies_section_verbatim(tmp_path: Path) -> None:
    sync = _sync_module()
    report = tmp_path / "report.md"
    report.write_text(
        "# R\n\n## Other\n\nx\n\n## Limitations\n\n- a 0.5\n- b\n", encoding="utf-8"
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        f"top\n{sync.START}\nold\n{sync.END}\nbottom\n", encoding="utf-8"
    )
    out = sync.sync(readme, report, "h")
    assert out.startswith("top\n") and out.endswith("\nbottom\n")
    assert "- a 0.5\n- b\n" + sync.END in out and "old" not in out
    with pytest.raises(ValueError, match="Limitations"):
        sync.limitations_section("# no section\n")


def test_readme_limitations_match_generated_report() -> None:
    sync = _sync_module()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split(sync.START, 1)[1].split(sync.END, 1)[0]
    run = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text("utf-8"))
    report = ROOT / "runs" / run["demo_run"] / "report.md"
    if not report.is_file():
        pytest.skip(f"{report} not present (runs/ is not committed)")
    section = sync.limitations_section(report.read_text(encoding="utf-8"))
    assert section in block


@pytest.mark.parametrize("doc", ["README.md", "docs/DEMO.md"])
def test_docs_have_no_decimal_results(doc: str) -> None:
    sync = _sync_module()
    text = (ROOT / doc).read_text(encoding="utf-8")
    if sync.START in text:
        head, rest = text.split(sync.START, 1)
        text = head + rest.split(sync.END, 1)[1]
    found = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", text)) - {"3.11"}
    assert not found, found


# --- scripts


@pytest.mark.parametrize("script", ["demo.sh", "export_demo_bundle.sh"])
def test_scripts_parse(script: str) -> None:
    path = ROOT / "scripts" / script
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert "set -euo pipefail" in path.read_text(encoding="utf-8")


def test_demo_script_contract() -> None:
    s = (ROOT / "scripts" / "demo.sh").read_text(encoding="utf-8")
    assert "source .venv/bin/activate" in s
    assert "API_WAIT_SECS=20" in s and '"$API_URL/datasets"' in s
    assert "--server.headless true" in s
    assert 'open "$UI_URL"' in s
    assert "trap cleanup EXIT INT TERM" in s
    assert "export HF_HUB_OFFLINE=1" in s
    assert "tokenizer not cached, Preview needs internet once" in s
    assert "TOKLAB_DEFAULT_RUN" in s
