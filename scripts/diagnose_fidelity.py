"""Find files whose C2 round trip fails and show where the bytes differ.

Print-only diagnosis; changes nothing on disk.

    python scripts/diagnose_fidelity.py --split test --corpus data/repos
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

from toklab.core.source_io import read_source_text
from toklab.data.ingest import load_ingest_config
from toklab.data.splits import ManifestView
from toklab.tokenizers.registry import get_tokenizer

UNUSUAL_CATEGORIES = {"Cc", "Cf", "Co", "Cs", "Cn", "Zl", "Zp"}
CR = b"\r"
BOM = b"\xef\xbb\xbf"


def _first_diff(a: bytes, b: bytes) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def _unusual_chars(text: str) -> list[str]:
    seen: dict[str, int] = {}
    for ch in text:
        if ch in "\n\t\r":
            continue
        cat = unicodedata.category(ch)
        if cat in UNUSUAL_CATEGORIES or (ord(ch) > 0x7F and cat.startswith("Z")):
            seen[ch] = seen.get(ch, 0) + 1
    return [
        f"U+{ord(c):04X} {unicodedata.name(c, '?')} [{unicodedata.category(c)}] "
        f"x{n}"
        for c, n in sorted(seen.items())
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/processed/manifest.parquet")
    ap.add_argument("--corpus", default="data/repos")
    ap.add_argument(
        "--flat",
        action="store_true",
        help="corpus is <dir>/<repo>/<path> (default for data/repos)",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--condition", default="C2")
    ap.add_argument(
        "--apply-exclusions",
        action="store_true",
        help="skip configs/ingest.yaml excluded_files (as a rebuilt manifest would)",
    )
    args = ap.parse_args()

    corpus = Path(args.corpus)
    files = sorted(
        ManifestView.from_parquet(args.manifest, args.split).files(),
        key=lambda f: f.file_id,
    )
    if args.apply_exclusions:
        excluded = load_ingest_config()["excluded_files"]
        skipped = [f for f in files if f"{f.repo}/{f.path}" in excluded]
        files = [f for f in files if f"{f.repo}/{f.path}" not in excluded]
        for f in skipped:
            print(f"excluded {f.repo}/{f.path}: {excluded[f'{f.repo}/{f.path}']}")
    tok = get_tokenizer(args.condition, {"attach_leading_space": True})
    n_fail = 0
    for sf in files:
        root = corpus if args.flat else corpus / args.split
        path = root / sf.repo / sf.path
        text = read_source_text(path)
        decoded = tok.decode(tok.encode(text).ids)
        if decoded == text:
            continue
        n_fail += 1
        a = text.encode("utf-8")
        b = decoded.encode("utf-8")
        off = _first_diff(a, b)
        lo = max(0, off - 20)
        hi = off + 20
        print(f"FAIL {args.condition} {sf.repo}/{sf.path} (file_id={sf.file_id})")
        print(f"  path={path}")
        print(f"  bytes original={len(a)} decoded={len(b)}")
        print(f"  first differing byte offset={off}")
        print(f"  original[{lo}:{hi}]={a[lo:hi]!r}")
        print(f"  decoded [{lo}:{hi}]={b[lo:hi]!r}")
        has_cr = CR in a
        has_bom = a.startswith(BOM)
        print(f"  has CR={has_cr} has BOM={has_bom}")
        unusual = _unusual_chars(text)
        print(f"  unusual unicode: {unusual if unusual else 'none'}")
        print(
            f"  text is NFC={unicodedata.is_normalized('NFC', text)} "
            f"decoded is NFC={unicodedata.is_normalized('NFC', decoded)} "
            f"NFC(original)==decoded={unicodedata.normalize('NFC', text) == decoded}"
        )
    print(f"split={args.split} condition={args.condition} files={len(files)} "
          f"roundtrip_fail={n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
