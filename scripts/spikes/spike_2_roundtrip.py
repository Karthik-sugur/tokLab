#!/usr/bin/env python3
"""Spike 2: lossless round-trip and leading-space / add_tokens behavior.

Prints observed facts only. Not library code.
"""

from __future__ import annotations

import sys
from pathlib import Path

PREFERRED = "Qwen/Qwen2.5-Coder-0.5B"
FALLBACKS = [
    "Salesforce/codegen-350M-mono",
    "microsoft/CodeGPT-small-py",
]


def _load_code_tokenizer():
    from transformers import AutoTokenizer

    tried: list[str] = []
    for name in [PREFERRED, *FALLBACKS]:
        tried.append(name)
        try:
            print(f"Trying tokenizer: {name}")
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            print(f"Loaded tokenizer: {name}")
            return name, tok
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load {name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not load any tokenizer. Tried: {tried}")


def _collect_stdlib_py(limit: int = 100) -> list[Path]:
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    root = Path(sys.base_prefix) / "lib" / ver
    if not root.is_dir():
        # macOS framework / homebrew layouts
        candidates = list(Path(sys.base_prefix).rglob("os.py"))
        if not candidates:
            raise RuntimeError(f"Could not find stdlib under {sys.base_prefix}")
        root = candidates[0].parent
    files = sorted(p for p in root.rglob("*.py") if p.is_file())
    # Prefer smaller top-level modules first for faster IO
    top = sorted(root.glob("*.py"))
    rest = [p for p in files if p not in top]
    ordered = top + rest
    selected = ordered[:limit]
    print(f"stdlib_root: {root}")
    print(f"stdlib_py_files_selected: {len(selected)}")
    return selected


def _roundtrip(tok, text: str) -> bool:
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, clean_up_tokenization_spaces=False)
    return decoded == text


def _show_tokens(label: str, tok, text: str) -> None:
    ids = tok.encode(text, add_special_tokens=False)
    pieces = tok.convert_ids_to_tokens(ids)
    print(f"  [{label!r}] ids={ids}")
    print(f"  [{label!r}] tokens={pieces}")
    decoded = tok.decode(ids, clean_up_tokenization_spaces=False)
    print(f"  [{label!r}] decode==input: {decoded == text} (decoded={decoded!r})")


def main() -> int:
    from transformers import AutoTokenizer

    code_name, code_tok = _load_code_tokenizer()
    print("Loading GPT-2 tokenizer...")
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    print(f"gpt2_tokenizer_class: {type(gpt2).__name__}")

    files = _collect_stdlib_py(100)
    print()
    print("--- round-trip decode(encode(x)) == x ---")
    for label, tok in (("code", code_tok), ("gpt2", gpt2)):
        ok = 0
        failures: list[tuple[str, str]] = []
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                failures.append((str(path), f"read_error: {exc}"))
                continue
            if _roundtrip(tok, text):
                ok += 1
            else:
                ids = tok.encode(text, add_special_tokens=False)
                decoded = tok.decode(ids, clean_up_tokenization_spaces=False)
                detail = (
                    f"len_in={len(text)} len_out={len(decoded)} "
                    f"ids={len(ids)}"
                )
                failures.append((str(path), detail))
        print(f"{label} ({code_name if label == 'code' else 'gpt2'}): "
              f"{ok}/{len(files)} exact round-trips")
        print(f"{label}_failures: {len(failures)}")
        for path, detail in failures[:10]:
            print(f"  FAIL {path}: {detail}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    print()
    print("--- leading spaces ---")
    for label, tok in (("code", code_tok), ("gpt2", gpt2)):
        print(f"{label}:")
        for s in ("foo", " foo", "x = foo"):
            _show_tokens(s, tok, s)

    print()
    print("--- add_tokens long identifier matching ---")
    long_id = "AcmePaymentSettlementBatchProcessor"
    # Fresh copy of code tokenizer for mutation
    from transformers import AutoTokenizer as AT

    mut = AT.from_pretrained(code_name, trust_remote_code=True)
    before = mut.encode(long_id, add_special_tokens=False)
    print(f"before_add tokens for {long_id!r}: {mut.convert_ids_to_tokens(before)}")
    n_added = mut.add_tokens([long_id])
    print(f"add_tokens returned: {n_added}")
    after = mut.encode(long_id, add_special_tokens=False)
    print(f"after_add tokens for {long_id!r}: {mut.convert_ids_to_tokens(after)}")

    examples = [
        long_id,
        f"{long_id}Helper",
        f"pre{long_id}",
        f" {long_id}",
        f"x = {long_id}()",
        f"class {long_id}:",
    ]
    for ex in examples:
        ids = mut.encode(ex, add_special_tokens=False)
        pieces = mut.convert_ids_to_tokens(ids)
        print(f"  text={ex!r}")
        print(f"    tokens={pieces}")
        # Did the single new token id appear?
        new_id = mut.convert_tokens_to_ids(long_id)
        print(f"    contains_new_token_id({new_id}): {new_id in ids}")

    print()
    print("FINDINGS:")
    print(
        f"- Round-trip checked on {len(files)} stdlib .py files for "
        f"{code_name} and gpt2 (see counts above)."
    )
    print(
        "- Leading-space tokenization for 'foo' vs ' foo' vs 'x = foo' "
        "printed above for both tokenizers."
    )
    print(
        f"- After add_tokens([{long_id!r}]), encoding of standalone and "
        "embedded forms printed above; check whether the new token matches "
        "inside larger words."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
