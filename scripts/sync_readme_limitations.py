#!/usr/bin/env python3
"""Copy the "## Limitations" section of a generated report into README.md.

    python scripts/sync_readme_limitations.py            # run from configs/demo.yaml
    python scripts/sync_readme_limitations.py --run <hash>

Replaces the text between the README markers verbatim, so the README never
carries hand-written limitation numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- limitations:start -->"
END = "<!-- limitations:end -->"


def limitations_section(report_md: str) -> str:
    """Body of ``## Limitations`` up to the next level-1/2 heading or EOF."""
    lines = report_md.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "## Limitations")
    except StopIteration as exc:
        raise ValueError("report has no '## Limitations' section") from exc
    body: list[str] = []
    for ln in lines[start + 1 :]:
        if ln.startswith("# ") or ln.startswith("## "):
            break
        body.append(ln)
    return "\n".join(body).strip("\n")


def render_block(run_hash: str, section: str) -> str:
    return (
        f"{START}\n"
        f"_Copied verbatim from `runs/{run_hash}/report.md` by "
        "`scripts/sync_readme_limitations.py`; re-run it after a new run._\n\n"
        f"{section}\n"
        f"{END}"
    )


def sync(readme: Path, report: Path, run_hash: str) -> str:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"{readme} lacks {START} / {END} markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    section = limitations_section(report.read_text(encoding="utf-8"))
    new = head + render_block(run_hash, section) + tail
    readme.write_text(new, encoding="utf-8")
    return new


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="run hash (default: configs/demo.yaml)")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = ap.parse_args(argv)
    run_hash = args.run or yaml.safe_load(
        (ROOT / "configs" / "demo.yaml").read_text(encoding="utf-8")
    )["demo_run"]
    report = args.runs_root / run_hash / "report.md"
    if not report.is_file():
        print(f"ERROR: {report} not found", file=sys.stderr)
        return 1
    sync(args.readme, report, run_hash)
    print(f"README limitations synced from {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
