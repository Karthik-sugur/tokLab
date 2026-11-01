"""Validate a synthetic layer directory against its synth manifest.

Usage:
    python scripts/validate_synth.py --repos data/repos --synthetic data/synthetic

Re-reads every output file listed in ``synth_manifest.parquet`` as bytes.
Gate rule (exit 0 only if both hold for every row):

* ``renamed=true``  -> output differs from the original and is AST
  alpha-equivalent to it
* ``renamed=false`` -> output is byte-identical to the original
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from toklab.data.synth.validate import alpha_equivalent


def _first_diff(a: bytes, b: bytes) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def check_row(orig: bytes, out: bytes, renamed: bool) -> str | None:
    """Return a failure message, or None if the row passes the gate."""
    if not renamed:
        if out == orig:
            return None
        off = _first_diff(orig, out)
        return (
            f"renamed=false but output differs from the original "
            f"(bytes {len(orig)} vs {len(out)}, first diff at offset {off}: "
            f"{orig[max(0, off - 10):off + 10]!r} vs "
            f"{out[max(0, off - 10):off + 10]!r})"
        )
    if out == orig:
        return "renamed=true but output unchanged"
    try:
        ok = alpha_equivalent(orig.decode("utf-8"), out.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return f"renamed=true but cannot check: {type(exc).__name__}: {exc}"
    return None if ok else "renamed=true but not alpha-equivalent"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", type=Path, default=Path("data/repos"))
    ap.add_argument("--synthetic", type=Path, default=Path("data/synthetic"))
    args = ap.parse_args()

    df = pd.read_parquet(args.synthetic / "synth_manifest.parquet")
    failures: list[str] = []
    passed_renamed: Counter[str] = Counter()
    unchanged_identical: Counter[str] = Counter()
    for row in df.itertuples(index=False):
        orig = (args.repos / row.repo / row.path).read_bytes()
        out = (args.synthetic / row.output).read_bytes()
        key = row.split if row.variant == "adapt_lexicon" else row.variant
        problem = check_row(orig, out, bool(row.renamed))
        if problem is not None:
            failures.append(f"{row.output}: {problem}")
        elif row.renamed:
            passed_renamed[key] += 1
        else:
            unchanged_identical[key] += 1

    print(f"synth_manifest_rows={len(df)}")
    group_key = df["split"].where(df["variant"] == "adapt_lexicon", df["variant"])
    for key, g in df.groupby(group_key, sort=False):
        n_ren = int(g["renamed"].sum())
        n_unch = len(g) - n_ren
        reasons = g.loc[~g["renamed"], "reason"].value_counts().sort_index()
        reason_txt = ", ".join(f"{r}={n}" for r, n in reasons.items()) or "-"
        pct = (
            f"{100.0 * passed_renamed[key] / n_ren:.1f}%" if n_ren else "n/a"
        )
        print(
            f"{key:>10}: files={len(g)} renamed={n_ren} "
            f"copied_unchanged={n_unch} ({reason_txt}) "
            f"identifiers_renamed={int(g['n_identifiers'].sum())} "
            f"renamed_alpha_pass={passed_renamed[key]}/{n_ren} ({pct}) "
            f"unchanged_byte_identical={unchanged_identical[key]}/{n_unch}"
        )
    print(f"validation_failures={len(failures)}")
    for line in failures[:50]:
        print(f"  {line}")
    print(f"gate={'PASS' if not failures else 'FAIL'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
