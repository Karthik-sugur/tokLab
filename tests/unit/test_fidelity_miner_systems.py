"""NFC exclusions, whole-identifier native mining, C3 fast path, systems step."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from toklab.data.ingest import ingest_repos, load_ingest_config
from toklab.tokenizers.composed import encode_native_longest_match
from toklab.tokenizers.native_hf import NativeHFTokenizer
from toklab.tokenizers.registry import get_tokenizer
from toklab.vocab.mining import _join_native_identifiers, _native_candidates

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def base() -> NativeHFTokenizer:
    return NativeHFTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", name="C2")


# --- Task 3: base tokenizer cannot round-trip non-NFC text -> exclusion list


def test_base_tokenizer_nfc_normalises_so_roundtrip_fails(
    base: NativeHFTokenizer,
) -> None:
    """Why the excluded files exist: the base tokenizer itself changes them."""
    c2 = get_tokenizer("C2", {"base_tokenizer": base})
    for ch, nfc in (("\uf96e", "\u8449"), ("\u2001", "\u2003"), ("\u212a", "K"),
                    ("e\u0301", "\u00e9")):
        text = f"x = '{ch}'\n"
        assert c2.decode(c2.encode(text).ids) == f"x = '{nfc}'\n" != text
    plain = "x = '葉 café'\n"  # already NFC: exact round trip
    assert c2.decode(c2.encode(plain).ids) == plain


def test_ingest_config_lists_failing_files_with_reasons() -> None:
    excluded = load_ingest_config(ROOT / "configs" / "ingest.yaml")["excluded_files"]
    assert "pydantic/tests/test_networks.py" in excluded
    assert all(reason.startswith("NFC normalizer") for reason in excluded.values())


def test_excluded_files_skipped_at_ingest(tmp_path: Path) -> None:
    repos = tmp_path / "repos"
    for name in ("keep.py", "drop.py"):
        p = repos / "r" / "pkg" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n", encoding="utf-8")
    cfg = tmp_path / "ingest.yaml"
    cfg.write_text(
        "skip_dir_names: [vendor]\nskip_file_names: []\n"
        "excluded_files:\n  - path: r/pkg/drop.py\n    reason: test reason\n",
        encoding="utf-8",
    )
    assert [f.path for f in ingest_repos(repos, config_path=cfg)] == ["pkg/keep.py"]

    cfg.write_text(
        "skip_dir_names: [vendor]\nskip_file_names: []\n"
        "excluded_files:\n  - path: r/pkg/drop.py\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'path' and 'reason'"):
        load_ingest_config(cfg)


# --- Task 4: native miner returns whole identifiers


def test_native_miner_keeps_leading_pretoken(base: NativeHFTokenizer) -> None:
    src = (
        "x = some_rather_long_text_value\n"
        "foo.StarkCatalogAccountController(AB890X)\n"
        "if TYPE_CHECKING:\n"
        "    map_imperatively(eq_)\n"
    )
    cands = _native_candidates(src, base._tok)
    for whole in ("some_rather_long_text_value", "StarkCatalogAccountController",
                  "AB890X", "TYPE_CHECKING", "map_imperatively", "eq_"):
        assert whole in cands
    for fragment in ("ome_rather_long_text_value", "_rather_long_text_value",
                     "archCatalogAccountController", "_CHECKING", "_imperatively"):
        assert fragment not in cands
    for c in cands:
        start = src.find(c)
        assert start >= 0
        assert start == 0 or not (src[start - 1].isalnum() or src[start - 1] == "_")


def test_join_spans_with_leading_space_and_punctuation() -> None:
    text = " some_value (Other)"
    spans = [(0, 5), (5, 11), (11, 13), (13, 18), (18, 19)]
    assert _join_native_identifiers(text, spans) == ["some_value", "Other"]


# --- C3 fast path is the same encoder as the exhaustive scan

_ALPHABET = st.sampled_from(list("ab_c1 .()\n\té"))


@settings(max_examples=150, deadline=None)
@given(st.text(_ALPHABET, max_size=40))
def test_c3_fast_path_equals_exhaustive_scan(text: str) -> None:
    adapted = {"ab": 900001, "ab_c": 900002, "c1": 900003, "é_a": 900004}

    class FakeBase:
        def encode(self, piece: str):  # one id per char keeps spans exact
            from toklab.core.types import EncodeResult

            return EncodeResult(
                ids=[ord(ch) for ch in piece],
                spans=[(i, i + 1) for i in range(len(piece))],
            )

    fast = encode_native_longest_match(text, FakeBase(), adapted,
                                       plain_identifiers=True)
    slow = encode_native_longest_match(text, FakeBase(), adapted,
                                       plain_identifiers=False)
    assert fast == slow


# --- determinism: bootstrap seeds no longer depend on salted str hash()


def test_factorial_ci_identical_across_hash_seeds() -> None:
    code = (
        "from toklab.evaluation.stats import factorial_analysis\n"
        "v={c:{f'r{i}':[float(i+j+len(c)) for j in range(3)] for i in range(6)}"
        " for c in ('C2','C3','C4','C5')}\n"
        "e=factorial_analysis(v,k=1,seed=0,n_boot=50)\n"
        "print(e.m_c2.low,e.m_c3.high,e.m_c5.low,e.interaction.high)\n"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(outs) == 1


# --- Task 6: systems on real token streams


def test_token_stream_source_bytes(base: NativeHFTokenizer) -> None:
    from toklab.core.types import SourceFile
    from toklab.evaluation.systems import build_token_stream

    c2 = get_tokenizer("C2", {"base_tokenizer": base})
    texts = {"a": "def f():\n    return 'é'\n", "b": "x = 1\n"}
    files = [SourceFile(f, "r", f"{f}.py", "h", "python", 1, "test") for f in texts]
    stream = build_token_stream(c2, files, texts, max_tokens=10_000)
    n_a = len(c2.encode(texts["a"]).ids)
    assert len(stream.ids) == n_a + len(c2.encode(texts["b"]).ids)
    assert stream.source_bytes(n_a) == len(texts["a"].encode("utf-8"))
    assert stream.source_bytes(len(stream.ids)) == sum(
        len(t.encode("utf-8")) for t in texts.values()
    )
    assert 0 < stream.source_bytes(1) < len(texts["a"].encode("utf-8"))


def test_systems_benchmark_real_inputs_and_quality_flags() -> None:
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock, patch

    from toklab.evaluation.systems import (
        MIN_REPEATS,
        TokenStream,
        measure_forward_latency,
        run_systems_benchmark,
    )

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = MagicMock(vocab_size=50)
            self.emb = torch.nn.Embedding(50, 4)

        def forward(self, input_ids=None, **_kw):  # noqa: ANN001
            return self.emb(input_ids).mean()

    stream = TokenStream([1, 2, 3, 4, 5, 6], ["f"], [6], [12],
                         [[2, 4, 6, 8, 10, 12]])
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=Tiny()
    ):
        df = run_systems_benchmark(
            model_name="dummy",
            conditions=["C2", "C3", "C4", "C5"],
            seq_lens=[4],
            device="cpu",
            n_warmup=2,
            n_repeats=10,
            token_streams=dict.fromkeys(("C2", "C3", "C4", "C5"), stream),
            k=256,
        )
    by = df.set_index("condition")
    assert set(by.index) == {"C2", "C3", "C4", "C5"}
    assert not by.loc["C3", "quality_valid"] and not by.loc["C5", "quality_valid"]
    assert by.loc["C2", "quality_valid"] and by.loc["C4", "quality_valid"]
    assert (df["source_bytes"] == 8).all() and (df["k"] == 256).all()
    assert (df["device"] == "cpu").all() and (df["dtype"] == "float32").all()
    assert df["notes"].str.contains("real_inputs").all()
    assert (df["n_repeats"] == 10).all() and (df["n_warmup"] == 2).all()
    with pytest.raises(ValueError, match=f">= {MIN_REPEATS}"):
        measure_forward_latency(Tiny(), torch.tensor([[1, 2]]), condition="C2",
                                seq_len=2, device="cpu", dtype="float32",
                                n_warmup=1, n_repeats=MIN_REPEATS - 1)


def test_pipeline_script_has_skippable_systems_step() -> None:
    script = (ROOT / "scripts" / "run_real_pipeline.sh").read_text(encoding="utf-8")
    assert 'SKIP_SYSTEMS="${SKIP_SYSTEMS:-0}"' in script
    assert "toklab run systems --run" in script
    assert 'echo "manifest_hash=$MANIFEST_HASH"' in script
    assert script.rstrip().endswith('echo "run_hash=$RUN_HASH"')
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / "run_real_pipeline.sh")],
                   check=True)
