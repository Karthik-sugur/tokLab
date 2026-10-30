#!/usr/bin/env python3
"""Clone repos from configs/repos.yaml and pin real commit SHAs.

If commit_sha is missing/null, shallow-clone, read ``git rev-parse HEAD``,
detect the LICENSE SPDX id, and write both back into the YAML.
Does not invent SHAs.
"""

from __future__ import annotations

import argparse
import re
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

ALLOWED_LICENSES = frozenset(
    {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC"}
)

_LICENSE_FILE_NAMES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "LICENSE.rst",
    "LICENCE",
    "LICENCE.txt",
    "COPYING",
    "COPYING.txt",
)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_out(cmd: list[str], cwd: Path | None = None) -> str:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def load_config(config_path: Path) -> dict:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "repos" not in data:
        raise ValueError(f"{config_path} must contain a top-level 'repos' list")
    repos = data["repos"]
    if not isinstance(repos, list) or not repos:
        raise ValueError(f"{config_path}: 'repos' must be a non-empty list")
    return data


def save_config(config_path: Path, data: dict) -> None:
    header = (
        "# Repository pins for TokLab data collection.\n"
        "# commit_sha filled by scripts/fetch_repos.py from git rev-parse HEAD.\n"
        "# Do not invent commit SHAs.\n\n"
    )
    body = yaml.safe_dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    config_path.write_text(header + body, encoding="utf-8")


def needs_pin(entry: dict) -> bool:
    sha = entry.get("commit_sha")
    if sha is None:
        return True
    if isinstance(sha, str) and sha.strip().lower() in {"", "null", "none", "todo"}:
        return True
    return False


def detect_license(repo_dir: Path) -> str:
    """Return SPDX id from LICENSE text, or UNKNOWN."""
    candidates: list[Path] = []
    for name in _LICENSE_FILE_NAMES:
        path = repo_dir / name
        if path.is_file():
            candidates.append(path)
    for path in sorted(repo_dir.glob("LICENSE*")):
        if path.is_file() and path not in candidates:
            candidates.append(path)
    if not candidates:
        return "UNKNOWN"

    text = candidates[0].read_text(encoding="utf-8", errors="replace")
    # Explicit SPDX tag anywhere in the file.
    spdx = re.search(
        r"SPDX-License-Identifier:\s*([A-Za-z0-9.+-]+)",
        text,
        flags=re.IGNORECASE,
    )
    if spdx:
        return spdx.group(1).strip()

    lower = text.lower()
    # Specific checks before generic phrases.
    if "apache license" in lower and "version 2.0" in lower:
        return "Apache-2.0"
    if re.search(r"\bapache-2\.0\b", lower):
        return "Apache-2.0"
    if "isc license" in lower or (
        "permission to use, copy, modify, and/or distribute" in lower
        and "isc" in lower
    ):
        return "ISC"
    if "bsd 3-clause" in lower or "the 3-clause bsd license" in lower:
        return "BSD-3-Clause"
    if "bsd 2-clause" in lower or "simplified bsd" in lower:
        return "BSD-2-Clause"
    # Classic BSD body heuristics.
    if "redistribution and use in source and binary forms" in lower:
        if "neither the name" in lower or "the names of" in lower:
            return "BSD-3-Clause"
        return "BSD-2-Clause"
    if "mit license" in lower or (
        "permission is hereby granted, free of charge" in lower
        and 'the software is provided "as is"' in lower
    ):
        return "MIT"
    return "UNKNOWN"


def count_py_files(repo_dir: Path) -> int:
    n = 0
    for path in repo_dir.rglob("*.py"):
        if ".git" in path.parts:
            continue
        n += 1
    return n


def apply_include_flag(entry: dict) -> None:
    lic = entry.get("license")
    if lic in ALLOWED_LICENSES:
        entry["include"] = True
    else:
        entry["include"] = False


def pin_missing_sha(entry: dict, dest_root: Path) -> Path:
    """Shallow-clone (if needed), read HEAD SHA + license into entry."""
    name = entry.get("name")
    url = entry.get("url")
    if not name or not url:
        raise ValueError(f"repo entry missing name/url: {entry!r}")
    if "TODO" in str(url):
        raise SystemExit(f"refusing to fetch {name!r}: url is still TODO")

    dest = dest_root / str(name)
    if not dest.exists():
        _run(["git", "clone", "--depth", "1", str(url), str(dest)])
    sha = _run_out(["git", "-C", str(dest), "rev-parse", "HEAD"])
    if not sha or not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        raise RuntimeError(f"unexpected rev-parse output for {name}: {sha!r}")
    entry["commit_sha"] = sha
    entry["license"] = detect_license(dest)
    apply_include_flag(entry)
    print(
        f"pinned {name} @ {sha} (license={entry['license']}, "
        f"include={entry['include']})",
        flush=True,
    )
    return dest


def fetch_one(entry: dict, dest_root: Path) -> Path:
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
    dest = dest_root / str(name)
    if dest.exists():
        _run(["git", "-C", str(dest), "fetch", "--all"])
    else:
        _run(["git", "clone", str(url), str(dest)])
    _run(["git", "-C", str(dest), "checkout", "--detach", str(sha)])
    print(f"checked out {name} @ {sha} (license={license_})", flush=True)
    return dest


def print_table(rows: list[tuple[str, str, str, int, bool]]) -> None:
    headers = ("repo", "SHA", "license", ".py files", "include")
    widths = [len(h) for h in headers]
    str_rows: list[tuple[str, str, str, str, str]] = []
    for name, sha, lic, n_py, include in rows:
        cells = (name, sha, lic, str(n_py), str(include))
        str_rows.append(cells)
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: tuple[str, ...]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(fmt(headers), flush=True)
    print("-+-".join("-" * w for w in widths), flush=True)
    for cells in str_rows:
        print(fmt(cells), flush=True)


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
    data = load_config(args.config)
    repos: list[dict] = data["repos"]
    args.out.mkdir(parents=True, exist_ok=True)

    table_rows: list[tuple[str, str, str, int, bool]] = []
    for entry in repos:
        if needs_pin(entry):
            dest = pin_missing_sha(entry, args.out)
            save_config(args.config, data)
        else:
            apply_include_flag(entry)
            dest = fetch_one(entry, args.out)
            save_config(args.config, data)
        n_py = count_py_files(dest)
        table_rows.append(
            (
                str(entry["name"]),
                str(entry["commit_sha"]),
                str(entry.get("license")),
                n_py,
                bool(entry.get("include", False)),
            )
        )

    print("\n=== repo pin summary ===", flush=True)
    print_table(table_rows)

    excluded = [
        (name, lic)
        for name, _sha, lic, _n, include in table_rows
        if not include
    ]
    if excluded:
        print("\nExcluded (include: false):", flush=True)
        for name, lic in excluded:
            reason = (
                "license is UNKNOWN"
                if lic == "UNKNOWN"
                else f"license {lic!r} not in {sorted(ALLOWED_LICENSES)}"
            )
            print(f"  - {name}: {reason}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
