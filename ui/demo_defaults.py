"""Demo defaults for the UI: run id and Preview snippet (no metrics)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_CONFIG = REPO_ROOT / "configs" / "demo.yaml"
FALLBACK_SNIPPET = "def AcmeWidgetFactory(x):\n    return x\n"


def load_demo_config(path: Path = DEMO_CONFIG) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def default_run_id(
    env: Mapping[str, str] | None = None, config_path: Path = DEMO_CONFIG
) -> str:
    """TOKLAB_DEFAULT_RUN, else ``demo_run`` from configs/demo.yaml, else ""."""
    env = os.environ if env is None else env
    from_env = env.get("TOKLAB_DEFAULT_RUN", "").strip()
    if from_env:
        return from_env
    return str(load_demo_config(config_path).get("demo_run") or "").strip()


def default_api_url(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    return env.get("TOKLAB_API_URL", "").strip() or "http://127.0.0.1:8000"


def default_snippet(
    config_path: Path = DEMO_CONFIG, repo_root: Path = REPO_ROOT
) -> str:
    """First ``preview_lines`` lines of ``preview_file``; fallback if absent."""
    cfg = load_demo_config(config_path)
    rel = cfg.get("preview_file")
    if not rel:
        return FALLBACK_SNIPPET
    path = repo_root / str(rel)
    if not path.is_file():
        return FALLBACK_SNIPPET
    n = int(cfg.get("preview_lines") or 30)
    with path.open(encoding="utf-8", newline="") as fh:
        lines = [line for _, line in zip(range(n), fh, strict=False)]
    return "".join(lines) or FALLBACK_SNIPPET
