"""Validate a synthetic layer directory against its synth manifest.

Usage:
    python scripts/validate_synth.py --repos data/repos --synthetic data/synthetic

Re-reads every output file listed in ``synth_manifest.parquet``:

* ``renamed=true``  -> output must differ from the original and be
  AST alpha-equivalent to it
* ``renamed=false`` -> output must equal the original byte for byte

Exits 1 if any row fails.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from toklab.data.synth.validate import alpha_equivalent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", type=Path, default=Path("data/repos"))
    ap.add_argument("--synthetic", type=Path, default=Path("data/synthetic"))
    args = ap.parse_args()

    df = pd.read_parquet(args.synthetic / "synth_manifest.parquet")
    failures: list[str] = []
    passed_renamed: Counter[str] = Counter()
    for row in df.itertuples(index=False):
        orig = (args.repos / row.repo / row.path).read_bytes()
        out = (args.synthetic / row.output).read_bytes()
        key = row.split if row.variant == "adapt_lexicon" else row.variant
        if row.renamed:
            if out == orig:
                failures.append(f"{row.output}: renamed=true but output unchanged")
                continue
            try:
                ok = alpha_equivalent(orig.decode("utf-8"), out.decode("utf-8"))
            except SyntaxError as exc:
                ok = False
                failures.append(f"{row.output}: SyntaxError {exc.msg}")
                continue
            if not ok:
                failures.append(f"{row.output}: not alpha-equivalent")
                continue
            passed_renamed[key] += 1
        elif out != orig:
            failures.append(f"{row.output}: renamed=false but output differs")

    print(f"synth_manifest_rows={len(df)}")
    for key, g in df.groupby(
        df["split"].where(df["variant"] == "adapt_lexicon", df["variant"]),
        sort=False,
    ):
        n_ren = int(g["renamed"].sum())
        reasons = g.loc[~g["renamed"], "reason"].value_counts().sort_index()
        reason_txt = ", ".join(f"{r}={n}" for r, n in reasons.items()) or "-"
        pct = 100.0 * passed_renamed[key] / n_ren if n_ren else float("nan")
        print(
            f"{key:>10}: files={len(g)} renamed={n_ren} "
            f"copied_unchanged={len(g) - n_ren} ({reason_txt}) "
            f"identifiers_renamed={int(g['n_identifiers'].sum())} "
            f"renamed_alpha_pass={passed_renamed[key]}/{n_ren} ({pct:.1f}%)"
        )
    print(f"validation_failures={len(failures)}")
    for line in failures[:50]:
        print(f"  {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
