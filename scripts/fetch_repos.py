#!/usr/bin/env python3
"""Clone and check out pinned commits from configs/repos.yaml.

Does not invent SHAs: entries still marked TODO are refused.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required (pulled in via project deps). "
        f"Import failed: {exc}"
    ) from exc


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_repos(config_path: Path) -> list[dict]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "repos" not in data:
        raise ValueError(f"{config_path} must contain a top-level 'repos' list")
    repos = data["repos"]
    if not isinstance(repos, list) or not repos:
        raise ValueError(f"{config_path}: 'repos' must be a non-empty list")
    return repos


def fetch_one(entry: dict, dest_root: Path) -> None:
    name = entry.get("name")
    url = entry.get("url")
    sha = entry.get("commit_sha")
    license_ = entry.get("license")
    if not name or not url or not sha:
        raise ValueError(f"repo entry missing name/url/commit_sha: {entry!r}")
    if "TODO" in str(url) or str(sha).upper() == "TODO":
        raise SystemExit(
            f"refusing to fetch {name!r}: url or commit_sha is still TODO "
            f"(license={license_!r}). Pin a real commit first."
        )
    dest = dest_root / name
    if dest.exists():
        _run(["git", "-C", str(dest), "fetch", "--all"])
    else:
        _run(["git", "clone", url, str(dest)])
    _run(["git", "-C", str(dest), "checkout", "--detach", str(sha)])
    print(f"checked out {name} @ {sha} (license={license_})", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/repos.yaml"),
        help="Path to repos.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/repos"),
        help="Directory to place cloned repositories",
    )
    args = parser.parse_args(argv)
    repos = load_repos(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    for entry in repos:
        fetch_one(entry, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
