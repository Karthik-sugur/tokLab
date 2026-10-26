"""Tests for baseline tokenizers, fidelity, intrinsic metrics, and bootstrap."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from toklab.core.types import SourceFile
from toklab.evaluation.fidelity import check_roundtrip, check_roundtrip_text
from toklab.evaluation.intrinsic import (
    compression_ratio,
    measure_file,
    original_n_bytes,
    tokens_per_kb,
)
from toklab.evaluation.stats import cluster_bootstrap_ci
from toklab.tokenizers.byte_level import ByteLevelTokenizer
from toklab.tokenizers.native_hf import NativeHFTokenizer
from toklab.tokenizers.registry import get_tokenizer

_GPT2: NativeHFTokenizer | None = None


def _gpt2() -> NativeHFTokenizer:
    global _GPT2
    if _GPT2 is None:
        _GPT2 = NativeHFTokenizer.from_pretrained("gpt2", name="C1")
    return _GPT2


@pytest.fixture(scope="module")
def gpt2_tok() -> NativeHFTokenizer:
    return _gpt2()


@pytest.fixture(scope="module")
def byte_tok() -> ByteLevelTokenizer:
    return ByteLevelTokenizer()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mini_repos(root: Path) -> Path:
    """Tiny multi-repo layout so the test split is non-empty."""
    repos = root / "repos"
    for name, body in [
        ("repo_a", "def a():\n    return 1\n"),
        ("repo_b", "def b():\n    return 2\n"),
        ("repo_c", "def c():\n    return 3\n"),
        ("repo_d", "def d():\n    return 4\n"),
        ("repo_e", "def e():\n    return 5\n"),
    ]:
        _write(repos / name / "mod.py", body)
    return repos


def test_byte_level_compression_ratio_is_exactly_one(
    byte_tok: ByteLevelTokenizer,
) -> None:
    text = "def café(x):\n    return x 🚀\n"
    n_bytes = original_n_bytes(text)
    n_tokens = len(byte_tok.encode(text).ids)
    assert n_bytes == n_tokens
    assert compression_ratio(n_bytes, n_tokens) == 1.0
    rows = {r["metric"]: r["value"] for r in measure_file(byte_tok, "f1", text)}
    assert rows["compression_ratio"] == 1.0
    assert rows["n_bytes"] == n_bytes
    assert rows["n_bytes"] != len(text)  # multi-byte chars present


def test_compression_ratio_uses_original_bytes(gpt2_tok: NativeHFTokenizer) -> None:
    text = "αβγ"  # 3 chars, 6 UTF-8 bytes
    assert len(text) == 3
    n_bytes = original_n_bytes(text)
    assert n_bytes == 6
    n_tokens = len(gpt2_tok.encode(text).ids)
    assert compression_ratio(n_bytes, n_tokens) == pytest.approx(n_bytes / n_tokens)
    rows = measure_file(gpt2_tok, "id", text, condition="C1")
    by_metric = {r["metric"]: r["value"] for r in rows}
    assert by_metric["n_bytes"] == 6
    assert by_metric["compression_ratio"] == pytest.approx(6 / by_metric["n_tokens"])


def test_byte_spans_align_with_characters(byte_tok: ByteLevelTokenizer) -> None:
    text = "a🚀b"
    result = byte_tok.encode(text)
    assert len(result.ids) == len(result.spans) == original_n_bytes(text)
    for start, end in result.spans:
        assert 0 <= start < end <= len(text)
        assert end == start + 1
    # Every character index appears at least once.
    covered = {s for s, _e in result.spans}
    assert covered == set(range(len(text)))
    assert byte_tok.decode(result.ids) == text


def test_hf_spans_match_ids_and_are_in_range(gpt2_tok: NativeHFTokenizer) -> None:
    text = "x = foo\ndef bar():\n    return 1\n"
    result = gpt2_tok.encode(text)
    assert len(result.ids) == len(result.spans)
    for start, end in result.spans:
        assert 0 <= start <= end <= len(text)
    assert gpt2_tok.decode(result.ids) == text


@given(st.text(min_size=0, max_size=64))
@settings(max_examples=40, deadline=None)
def test_byte_level_roundtrip_hypothesis(text: str) -> None:
    tok = ByteLevelTokenizer()
    assert tok.decode(tok.encode(text).ids) == text


@given(
    st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),
            blacklist_characters="\x00",
        ),
        min_size=0,
        max_size=48,
    )
)
@settings(max_examples=25, deadline=None)
def test_gpt2_roundtrip_hypothesis(text: str) -> None:
    tok = _gpt2()
    assert tok.decode(tok.encode(text).ids) == text


@pytest.mark.parametrize(
    "code",
    [
        "x = 1\n",
        "def foo(a, b):\n    return a + b\n",
        "class A:\n    pass\n",
        "# café\nprint('🚀')\n",
    ],
)
def test_roundtrip_on_real_code(
    gpt2_tok: NativeHFTokenizer,
    byte_tok: ByteLevelTokenizer,
    code: str,
) -> None:
    assert check_roundtrip_text(gpt2_tok, [code]) == []
    assert check_roundtrip_text(byte_tok, [code]) == []


def test_fidelity_reparse(tmp_path: Path, byte_tok: ByteLevelTokenizer) -> None:
    text = "def ok():\n    return 1\n"
    sf = SourceFile("fid", "r", "ok.py", "x", "python", len(text.encode()), "test")
    report = check_roundtrip(byte_tok, [sf], texts={"fid": text})
    assert report.n_roundtrip_ok == 1
    assert report.n_orig_parsed == 1
    assert report.n_reparse_ok == 1
    assert report.n_reparse_fail == 0


def test_bootstrap_ci_deterministic_for_seed() -> None:
    data = {
        "repo_a": [10.0, 12.0],
        "repo_b": [20.0],
        "repo_c": [15.0, 16.0, 14.0],
    }
    a = cluster_bootstrap_ci(data, seed=7, n_boot=200)
    b = cluster_bootstrap_ci(data, seed=7, n_boot=200)
    c = cluster_bootstrap_ci(data, seed=8, n_boot=200)
    assert a == b
    assert a.estimate == b.estimate
    assert (a.low, a.high) == (b.low, b.high)
    assert (a.low, a.high) != (c.low, c.high) or a.estimate == c.estimate


def test_registry_c1_c6() -> None:
    c1 = get_tokenizer("C1", {})
    c6 = get_tokenizer("C6", {})
    assert c1.name == "C1"
    assert c6.name == "C6"
    assert c6.vocab_size == 256


def test_tokens_per_kb_formula() -> None:
    assert tokens_per_kb(100, 1024) == pytest.approx(100.0)
    assert tokens_per_kb(50, 512) == pytest.approx(100.0)


def test_run_baselines_on_fixture(tmp_path: Path) -> None:
    from toklab.data.splits import build_dataset
    from toklab.evaluation.baselines import run_baselines

    repos = _mini_repos(tmp_path)
    manifest_dir = tmp_path / "manifest"
    build_dataset(repos, manifest_dir, seed=0)
    out = tmp_path / "base_out"
    # C1+C6 here for speed; full C1/C2/C6 exercised via CLI acceptance.
    table, fidelity, cis = run_baselines(
        manifest=manifest_dir / "manifest.parquet",
        repos_dir=repos,
        out_dir=out,
        split="test",
        seed=0,
        n_boot=50,
        conditions=("C1", "C6"),
    )
    assert "C1" in table and "C6" in table
    assert fidelity["C6"].all_roundtrips_ok
    assert fidelity["C1"].all_roundtrips_ok
    assert (out / "intrinsic_metrics.parquet").is_file()
    assert len(cis) == 2


def test_registry_c2_loads_from_yaml() -> None:
    cfg_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "models"
        / "base_model.yaml"
    )
    tok = get_tokenizer("C2", {"base_model_yaml": str(cfg_path)})
    assert tok.name == "C2"
    assert tok.vocab_size > 0
    assert tok.decode(tok.encode("x = 1\n").ids) == "x = 1\n"
