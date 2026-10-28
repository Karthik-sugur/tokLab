"""Tests for C3/C5 adapted vocab, selection, and 2x2 grid invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from toklab.core.types import SourceFile
from toklab.data.splits import ManifestView, build_dataset
from toklab.tokenizers.native_hf import NativeHFTokenizer
from toklab.tokenizers.registry import get_tokenizer
from toklab.vocab.adapted_vocab import AdaptedVocab
from toklab.vocab.build import build_adapted_vocab
from toklab.vocab.mining import mine_candidates, mine_from_files
from toklab.vocab.selection import select_vocab


@pytest.fixture(scope="module")
def base() -> NativeHFTokenizer:
    return NativeHFTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", name="C2"
    )


def _enterprise_fixture(root: Path) -> Path:
    """Repos with long ids so adapted vocab has something to gain."""
    repos = root / "repos"
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
        "repo_c": "def other():\n    return 1\n",
        "repo_d": "class Z:\n    pass\n",
        "repo_e": "print(0)\n",
    }
    for name, body in bodies.items():
        p = repos / name / "mod.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return repos


def test_k0_c3_matches_c2_ids(base: NativeHFTokenizer) -> None:
    text = "def foo(a):\n    return a + 1\n"
    c2 = get_tokenizer("C2", {"base_tokenizer": base})
    c3 = get_tokenizer(
        "C3",
        {"base_tokenizer": base, "adapted_vocab": {}},
    )
    assert c2.encode(text).ids == c3.encode(text).ids


def test_k0_c5_matches_c4_ids(base: NativeHFTokenizer) -> None:
    text = "def foo(a):\n    return a + 1\n"
    cfg = {"base_tokenizer": base, "attach_leading_space": True}
    c4 = get_tokenizer("C4", cfg)
    c5 = get_tokenizer("C5", {**cfg, "adapted_vocab": {}})
    assert c4.encode(text).ids == c5.encode(text).ids


def test_adapted_ids_append_only(base: NativeHFTokenizer, tmp_path: Path) -> None:
    repos = _enterprise_fixture(tmp_path)
    man = tmp_path / "m"
    build_dataset(repos, man, seed=0)
    view = ManifestView.from_parquet(man / "manifest.parquet", "adapt")
    vocab = build_adapted_vocab(
        view,
        repos_dir=repos,
        base=base,
        mode="native",
        k=8,
        min_freq=1,
        min_repos=1,
        base_model="Qwen/Qwen2.5-Coder-0.5B",
    )
    assert vocab.id_offset == base.vocab_size
    for e in vocab.entries:
        assert e.id >= base.vocab_size
    # Existing base ids unchanged: encode a short token only via base path.
    c3 = get_tokenizer(
        "C3", {"base_tokenizer": base, "adapted_vocab": vocab}
    )
    plain = "x = 1\n"
    # Any id that is not adapted should be < id_offset
    for tid in c3.encode(plain).ids:
        if tid not in vocab.reverse_dict():
            assert tid < base.vocab_size


def test_mine_test_split_raises(base: NativeHFTokenizer, tmp_path: Path) -> None:
    repos = _enterprise_fixture(tmp_path)
    man = tmp_path / "m"
    build_dataset(repos, man, seed=0)
    test_view = ManifestView.from_parquet(man / "manifest.parquet", "test")
    with pytest.raises(PermissionError, match="adapt"):
        mine_candidates(
            test_view,
            repos_dir=repos,
            mode="native",
            base_tokenizer=base,
        )
    files = [
        SourceFile("f", "r", "a.py", "h", "python", 1, "test"),
    ]
    with pytest.raises(PermissionError, match="adapt"):
        mine_from_files(
            files,
            {"f": "x=1\n"},
            mode="native",
            base_tokenizer=base,
        )


def test_roundtrip_c3_c5(base: NativeHFTokenizer, tmp_path: Path) -> None:
    repos = _enterprise_fixture(tmp_path)
    man = tmp_path / "m"
    build_dataset(repos, man, seed=0)
    adapt = ManifestView.from_parquet(man / "manifest.parquet", "adapt")
    vocab_n = build_adapted_vocab(
        adapt, repos_dir=repos, base=base, mode="native", k=16
    )
    vocab_s = build_adapted_vocab(
        adapt, repos_dir=repos, base=base, mode="structural", k=16
    )
    cfg = {"base_tokenizer": base}
    c3 = get_tokenizer("C3", {**cfg, "adapted_vocab": vocab_n})
    c5 = get_tokenizer("C5", {**cfg, "adapted_vocab": vocab_s})
    for sf in adapt.files():
        text = (repos / sf.repo / sf.path).read_text(encoding="utf-8")
        assert c3.decode(c3.encode(text).ids) == text
        assert c5.decode(c5.encode(text).ids) == text


def test_selection_deterministic(base: NativeHFTokenizer, tmp_path: Path) -> None:
    repos = _enterprise_fixture(tmp_path)
    man = tmp_path / "m"
    build_dataset(repos, man, seed=0)
    adapt = ManifestView.from_parquet(man / "manifest.parquet", "adapt")
    cands = mine_candidates(
        adapt, repos_dir=repos, mode="native", base_tokenizer=base
    )
    a = select_vocab(cands, base, k=5)
    b = select_vocab(cands, base, k=5)
    assert [x.token for x in a] == [x.token for x in b]
    assert [x.gain for x in a] == [x.gain for x in b]


def test_c3_c5_share_base_config(base: NativeHFTokenizer) -> None:
    cfg = {"base_tokenizer": base, "model_name": "Qwen/Qwen2.5-Coder-0.5B"}
    c2 = get_tokenizer("C2", cfg)
    c3 = get_tokenizer("C3", {**cfg, "adapted_vocab": {}})
    c4 = get_tokenizer("C4", cfg)
    c5 = get_tokenizer("C5", {**cfg, "adapted_vocab": {}})
    # Same underlying HF tokenizer object when base_tokenizer is shared.
    assert c3.base is base
    assert c4.base is base
    assert c5.base is base
    assert c2 is base or c2._tok is base._tok


def test_adapted_vocab_save_load(tmp_path: Path, base: NativeHFTokenizer) -> None:
    from toklab.vocab.selection import ScoredCandidate

    selected = [
        ScoredCandidate("FooBar", 3, 2, 4, 9.0),
        ScoredCandidate("BazQux", 2, 1, 3, 4.0),
    ]
    v = AdaptedVocab.from_selected(
        selected, id_offset=base.vocab_size, mode="native", base_model="x"
    )
    path = tmp_path / "v.json"
    v.save(path)
    v2 = AdaptedVocab.load(path)
    assert v.content_hash() == v2.content_hash()
    assert v.as_dict() == v2.as_dict()
    assert all(i >= base.vocab_size for i in v.as_dict().values())


def test_factorial_runs_on_fixture(base: NativeHFTokenizer, tmp_path: Path) -> None:
    from toklab.evaluation.factorial import run_factorial

    repos = _enterprise_fixture(tmp_path)
    man = tmp_path / "m"
    build_dataset(repos, man, seed=0)
    table, effects = run_factorial(
        manifest=man / "manifest.parquet",
        repos_dir=repos,
        k_values=[0, 8],
        split="test",
        seed=0,
        n_boot=20,
        cfg={"base_tokenizer": base, "model_name": "Qwen/Qwen2.5-Coder-0.5B"},
    )
    assert "source=fixtures_only" in table
    assert len(effects) == 2
    # K=0: C3 should match C2 metric closely (same encoding).
    e0 = effects[0]
    assert e0.k == 0
    assert abs(e0.m_c2.estimate - e0.m_c3.estimate) < 1e-9
    print(table)
