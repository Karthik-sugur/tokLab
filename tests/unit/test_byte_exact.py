"""Source bytes must survive ingest, synth copy-through and corpus reads."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd

from toklab.core.source_io import read_source_text
from toklab.data.ingest import ingest_repos
from toklab.data.splits import MIN_REPOS_DISABLED, build_dataset
from toklab.data.synth.pipeline import run_synth

BOM = b"\xef\xbb\xbf"


def _crlf_bom_east_asian(i: int) -> bytes:
    # BOM + CRLF + East Asian wide characters; no renameable definitions.
    text = (
        f'x{i} = "中文字符串 日本語のテキスト 한국어 {i}"\r\n'
        "print(len(x0) if False else 0)\r\n"
    )
    return BOM + text.encode("utf-8")


def _crlf_renameable(i: int) -> bytes:
    text = (
        f"class Widget{i}:\r\n"
        "    def run(self):\r\n"
        f'        return "東アジアの幅 {i}"\r\n'
    )
    return text.encode("utf-8")


def _make_repos(root: Path) -> dict[tuple[str, str], bytes]:
    written: dict[tuple[str, str], bytes] = {}
    for i in range(5):
        repo = f"repo_{i}"
        for rel, data in (
            ("pkg/east_asian.py", _crlf_bom_east_asian(i)),
            ("pkg/widget.py", _crlf_renameable(i)),
        ):
            p = root / repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            written[(repo, rel)] = data
    return written


def test_read_source_text_keeps_crlf_and_bom(tmp_path: Path) -> None:
    p = tmp_path / "f.py"
    data = _crlf_bom_east_asian(0)
    p.write_bytes(data)
    text = read_source_text(p)
    assert text.startswith("\ufeff")
    assert "\r\n" in text
    assert text.encode("utf-8") == data


def test_manifest_bytes_and_hash_match_disk(tmp_path: Path) -> None:
    repos = tmp_path / "repos"
    written = _make_repos(repos)
    files = list(ingest_repos(repos))
    assert {(f.repo, f.path) for f in files} == set(written)
    for f in files:
        data = written[(f.repo, f.path)]
        assert f.n_bytes == len(data)
        assert f.sha256 == hashlib.sha256(data).hexdigest()
        # The text every corpus reader tokenizes has exactly these bytes.
        assert read_source_text(repos / f.repo / f.path).encode("utf-8") == data


def test_synth_copy_through_is_byte_identical(tmp_path: Path) -> None:
    repos = tmp_path / "repos"
    written = _make_repos(repos)
    manifest_dir = tmp_path / "manifest"
    build_dataset(repos, manifest_dir, seed=0, min_repos=MIN_REPOS_DISABLED)
    out = tmp_path / "synth"
    run_synth(
        manifest=manifest_dir / "manifest.parquet", repos_dir=repos, out_dir=out
    )
    df = pd.read_parquet(out / "synth_manifest.parquet")
    assert len(df) > 0
    unchanged = df[~df["renamed"]]
    renamed = df[df["renamed"]]
    assert len(unchanged) > 0
    assert len(renamed) > 0
    for row in unchanged.itertuples(index=False):
        got = (out / row.output).read_bytes()
        assert got == written[(row.repo, row.path)], row.output
    for row in renamed.itertuples(index=False):
        orig = written[(row.repo, row.path)]
        got = (out / row.output).read_bytes()
        assert got != orig
        # Renamed outputs keep the original line endings.
        assert got.count(b"\r\n") == orig.count(b"\r\n")
        assert b"\n" not in got.replace(b"\r\n", b"")


def _load_validate_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "validate_synth.py"
    spec = importlib.util.spec_from_file_location("validate_synth", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_validate_gate_rejects_newline_translated_copy() -> None:
    mod = _load_validate_script()
    orig = _crlf_bom_east_asian(0)
    translated = orig.replace(b"\r\n", b"\n")
    assert mod.check_row(orig, orig, renamed=False) is None
    problem = mod.check_row(orig, translated, renamed=False)
    assert problem is not None and "differs" in problem
    assert mod.check_row(orig, orig, renamed=True) is not None
    src = b"def foo():\r\n    return foo\r\n"
    good = b"def AcmeFoo():\r\n    return AcmeFoo\r\n"
    bad = b"def AcmeFoo():\r\n    return foo\r\n"
    assert mod.check_row(src, good, renamed=True) is None
    assert mod.check_row(src, bad, renamed=True) is not None
