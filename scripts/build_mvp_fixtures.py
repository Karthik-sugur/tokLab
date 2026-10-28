#!/usr/bin/env python3
"""Build tests/fixtures/mvp repos + manifest for the mvp_py experiment."""

from __future__ import annotations

from pathlib import Path

from toklab.data.splits import build_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "mvp"
REPOS = OUT / "repos"
MANIFEST = OUT / "manifest"


def main() -> None:
    bodies = {
        "repo_a": (
            "class AcmePaymentSettlementBatchProcessor:\n"
            "    def run(self):\n"
            "        return AcmePaymentSettlementBatchProcessor()\n"
        ),
        "repo_b": (
            "def AcmePaymentSettlementBatchProcessor():\n"
            "    x = AcmePaymentSettlementBatchProcessor\n"
            "    return x\n"
        ),
        "repo_c": "def helper_alpha(n):\n    return n + 1\n",
        "repo_d": "class WidgetFactory:\n    def make(self):\n        return 1\n",
        "repo_e": "print('ok')\n",
    }
    for name, body in bodies.items():
        path = REPOS / name / "mod.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    MANIFEST.mkdir(parents=True, exist_ok=True)
    counts = build_dataset(REPOS, MANIFEST, seed=0)
    print(f"wrote {MANIFEST / 'manifest.parquet'} counts={counts}")


if __name__ == "__main__":
    main()
