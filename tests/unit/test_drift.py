"""Phase 9: vocabulary drift — unseen assert, refresh ids, coverage, p=0."""

from __future__ import annotations

from pathlib import Path

import pytest

from toklab.core.types import SourceFile
from toklab.data.splits import build_dataset
from toklab.data.synth.drift import apply_drift
from toklab.data.synth.lexicon import EnterpriseLexicon
from toklab.evaluation.drift import (
    DriftLeakError,
    adapted_coverage,
    assert_drift_names_unseen,
    refresh_vocab_append_only,
    run_drift,
)
from toklab.tokenizers.native_hf import NativeHFTokenizer
from toklab.tokenizers.registry import get_tokenizer
from toklab.vocab.adapted_vocab import AdaptedEntry, AdaptedVocab, empty_adapted_vocab
from toklab.vocab.selection import ScoredCandidate


@pytest.fixture(scope="module")
def base() -> NativeHFTokenizer:
    return NativeHFTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", name="C2"
    )


def _repos(root: Path) -> Path:
    repos = root / "repos"
    bodies = {
        "repo_a": (
            "class AcmePaymentSettlementBatchProcessor:\n"
            "    def run(self):\n"
            "        return AcmePaymentSettlementBatchProcessor()\n"
        ),
        "repo_b": (
            "def AcmePaymentSettlementBatchProcessor():\n"
            "    return AcmePaymentSettlementBatchProcessor\n"
        ),
        "repo_c": "def helper_alpha(n):\n    return n + 1\n",
        "repo_d": (
            "class WidgetFactory:\n"
            "    def make(self):\n"
            "        return WidgetFactory()\n"
        ),
        "repo_e": "print(0)\n",
    }
    for name, body in bodies.items():
        p = repos / name / "mod.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return repos


def test_unseen_assertion_fires_on_leak() -> None:
    with pytest.raises(DriftLeakError, match="adapt corpus"):
        assert_drift_names_unseen(
            {"LeakedName", "Other"},
            {"LeakedName", "AdaptOnly"},
        )
    assert_drift_names_unseen({"FreshDrift"}, {"AdaptOnly"})


def test_refresh_keeps_old_ids(base: NativeHFTokenizer) -> None:
    static = AdaptedVocab(
        id_offset=base.vocab_size,
        entries=[
            AdaptedEntry(
                token="OldTokenAlpha",
                id=base.vocab_size,
                gain=2.0,
                freq=2,
                n_repos=1,
            )
        ],
        mode="native",
        base_model="x",
    )
    old = static.token_to_id_snapshot()
    sf = SourceFile("f1", "r", "a.py", "h", "python", 10, "adapt")
    texts = {"f1": "def BrandNewDriftToken():\n    return BrandNewDriftToken\n"}
    refreshed = refresh_vocab_append_only(
        static,
        sample_files=[sf],
        sample_texts=texts,
        base=base,
        mode="native",
        k_refresh=4,
    )
    new = refreshed.token_to_id_snapshot()
    assert new["OldTokenAlpha"] == old["OldTokenAlpha"]
    assert len(refreshed) >= len(static)


def test_coverage_zero_when_vocab_empty(base: NativeHFTokenizer) -> None:
    empty = empty_adapted_vocab(base.vocab_size, mode="native")
    tok = get_tokenizer(
        "C3",
        {"base_tokenizer": base, "adapted_vocab": empty},
    )
    text = "def WidgetFactory():\n    return 1\n"
    cover, hits, n = adapted_coverage(tok, text, set())
    assert n > 0
    assert hits == 0
    assert cover == 0.0


def test_p0_equals_no_drift_results(base: NativeHFTokenizer, tmp_path: Path) -> None:
    repos = _repos(tmp_path)
    man = tmp_path / "m"
    build_dataset(repos, man, seed=0)
    table, df, rows = run_drift(
        manifest=man / "manifest.parquet",
        repos_dir=repos,
        out_dir=tmp_path / "drift_out",
        seed=0,
        k_static=4,
        k_refresh=2,
        proportions=(0,),
        cfg={"base_tokenizer": base, "model_name": "Qwen/Qwen2.5-Coder-0.5B"},
    )
    assert "STATIC_C3" in table
    # p=0: apply_drift is identity — C2 tokens on holdout match raw files.
    from toklab.data.splits import ManifestView

    hold = ManifestView.from_parquet(man / "manifest.parquet", "drift_holdout")
    raw_texts = [
        (repos / sf.repo / sf.path).read_text(encoding="utf-8")
        for sf in sorted(hold.files(), key=lambda f: f.file_id)
    ]
    tok_c2 = get_tokenizer("C2", {"base_tokenizer": base})
    n_raw = sum(len(tok_c2.encode(t).ids) for t in raw_texts)
    row_c2 = next(r for r in rows if r.variant == "C2" and r.proportion == 0)
    assert row_c2.n_tokens == n_raw
    # Identity drift → STATIC and REFRESH share baseline comparison inputs.
    lex = EnterpriseLexicon.from_yaml(seed=0)
    for sf in hold.files():
        raw = (repos / sf.repo / sf.path).read_text(encoding="utf-8")
        assert apply_drift(raw, lex, 0, seed=0).source == raw


def test_run_drift_sweep_writes_artifacts(
    base: NativeHFTokenizer, tmp_path: Path
) -> None:
    repos = _repos(tmp_path)
    man = tmp_path / "m"
    build_dataset(repos, man, seed=0)
    out = tmp_path / "out"
    table, df, rows = run_drift(
        manifest=man / "manifest.parquet",
        repos_dir=repos,
        out_dir=out,
        seed=0,
        k_static=4,
        k_refresh=2,
        proportions=(0, 10, 25, 50),
        cfg={"base_tokenizer": base, "model_name": "Qwen/Qwen2.5-Coder-0.5B"},
    )
    assert (out / "drift_metrics.parquet").is_file()
    assert (out / "drift_table.txt").is_file()
    assert set(df["proportion"]) == {0, 10, 25, 50}
    assert all(r.bpb is None for r in rows)
    assert all(r.bpb_status == "not run" for r in rows)
    # Empty-vocab path: k_static=0 → coverage 0 for STATIC.
    _t, df0, rows0 = run_drift(
        manifest=man / "manifest.parquet",
        repos_dir=repos,
        seed=0,
        k_static=0,
        k_refresh=0,
        proportions=(50,),
        cfg={"base_tokenizer": base, "model_name": "Qwen/Qwen2.5-Coder-0.5B"},
    )
    static = [r for r in rows0 if r.variant.startswith("STATIC")]
    assert static
    assert all(r.coverage == 0.0 for r in static)
    assert all(r.n_vocab == 0 for r in static)


def test_extend_append_only_preserves_ids() -> None:
    v = AdaptedVocab(
        id_offset=100,
        entries=[AdaptedEntry("A", 100, 1.0, 1, 1)],
        mode="native",
    )
    v2 = v.extend_append_only(
        [ScoredCandidate("B", 1, 1, 3, 2.0)]
    )
    assert v2.as_dict()["A"] == 100
    assert v2.as_dict()["B"] == 101
