"""Capture reproducible environment metadata for a run."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_sha(cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


def collect_env_log(
    *,
    seed: int,
    manifest_hash: str,
    vocab_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    """Return env dict: git SHA, versions, CPU/GPU, seeds, hashes."""
    gpu = "none"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        else:
            gpu = "cpu"
    except ImportError:
        gpu = "torch_not_installed"

    return {
        "git_sha": _git_sha(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu,
        "seed": seed,
        "manifest_hash": manifest_hash,
        "vocab_hash": vocab_hash,
        "config_hash": config_hash,
        "library_versions": {
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
            "transformers": _pkg_version("transformers"),
            "tokenizers": _pkg_version("tokenizers"),
            "torch": _pkg_version("torch"),
            "libcst": _pkg_version("libcst"),
            "tree_sitter": _pkg_version("tree-sitter"),
        },
    }
