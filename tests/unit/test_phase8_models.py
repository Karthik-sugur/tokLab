"""Phase 8: model gate, adapt embeddings, BPB, quality suppression."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from toklab.core.types import EncodeResult
from toklab.evaluation.quality import (
    QualitySuppressedError,
    assert_quality_allowed,
    bits_per_byte,
    evaluate_quality,
)
from toklab.models.adapt_embeddings import adapt_embeddings
from toklab.models.finetune import FinetuneConfig, dry_run_demo, finetune_grid
from toklab.models.gate import GateVerdict, ModelVariant, gate
from toklab.models.tiny_fixture import (
    ASCII_BASE,
    build_tiny_bundle,
    tiny_adapted_vocab,
)


@pytest.fixture
def bundle():
    return build_tiny_bundle(seed=0, tie_word_embeddings=True)


def test_gate_incompatible_when_token_id_ge_rows(bundle) -> None:
    class HighIdTok:
        name = "high"

        def encode(self, text: str) -> EncodeResult:
            rows = bundle.model.get_input_embeddings().weight.shape[0]
            return EncodeResult(ids=[rows], spans=[(0, 1)])

        def decode(self, ids: list[int]) -> str:
            return "x"

    variant = ModelVariant(
        model=bundle.model,
        n_embedding_rows=int(
            bundle.model.get_input_embeddings().weight.shape[0]
        ),
        tie_word_embeddings=True,
    )
    result = gate(HighIdTok(), variant, sample_texts=["x"])
    assert result.verdict == GateVerdict.INCOMPATIBLE
    assert result.max_token_id is not None
    assert result.max_token_id >= result.embedding_rows


def test_gate_untrained_new_rows(bundle) -> None:
    vocab = tiny_adapted_vocab(bundle)
    adapted = adapt_embeddings(
        bundle.model, vocab, bundle.hf_tokenizer, mark_trained=False
    )
    tok = bundle.tokenizer.with_adapted(vocab)
    variant = ModelVariant(
        model=adapted.model,
        init_record=adapted.init_record,
        has_new_vocab_rows=True,
        trained_new_rows=False,
        tie_word_embeddings=True,
    )
    text = "AcmeWidgetFactory()\n"
    result = gate(tok, variant, sample_texts=[text])
    assert result.verdict == GateVerdict.UNTRAINED_NEW_ROWS
    assert result.init_recorded is True
    assert result.quality_allowed is False


def test_tied_embedding_resize_keeps_tie(bundle) -> None:
    vocab = tiny_adapted_vocab(bundle)
    before_in = bundle.model.get_input_embeddings()
    before_out = bundle.model.get_output_embeddings()
    assert before_in.weight.data_ptr() == before_out.weight.data_ptr()

    adapted = adapt_embeddings(bundle.model, vocab, bundle.hf_tokenizer)
    in_emb = adapted.model.get_input_embeddings()
    out_emb = adapted.model.get_output_embeddings()
    assert adapted.init_record.still_tied_after is True
    assert in_emb.weight.shape[0] == bundle.vocab_size + len(vocab)
    assert out_emb.weight.shape == in_emb.weight.shape
    assert in_emb.weight.data_ptr() == out_emb.weight.data_ptr()
    assert adapted.init_record.method == "mean_of_base_bpe_decomposition"
    entry = vocab.entries[0]
    bpe = bundle.hf_tokenizer.encode(entry.token)
    bpe_valid = [i for i in bpe if i < bundle.vocab_size]
    expected = in_emb.weight[: bundle.vocab_size][bpe_valid].mean(dim=0)
    assert torch.allclose(in_emb.weight[entry.id], expected, atol=1e-5)


def test_bpb_formula_handmade() -> None:
    assert bits_per_byte(nll_nats_sum=math.log(2.0), n_bytes=1) == pytest.approx(
        1.0
    )
    assert bits_per_byte(
        nll_nats_sum=2.0 * math.log(2.0), n_bytes=4
    ) == pytest.approx(0.5)


def test_quality_refuses_gated_out(bundle) -> None:
    vocab = tiny_adapted_vocab(bundle)
    adapted = adapt_embeddings(
        bundle.model, vocab, bundle.hf_tokenizer, mark_trained=False
    )
    tok = bundle.tokenizer.with_adapted(vocab)
    variant = ModelVariant(
        model=adapted.model,
        init_record=adapted.init_record,
        has_new_vocab_rows=True,
        trained_new_rows=False,
    )
    g = gate(tok, variant, sample_texts=["AcmeWidgetFactory\n"])
    assert g.verdict == GateVerdict.UNTRAINED_NEW_ROWS
    with pytest.raises(QualitySuppressedError, match="UNTRAINED_NEW_ROWS"):
        assert_quality_allowed(g)
    with pytest.raises(QualitySuppressedError):
        evaluate_quality(
            condition="C3",
            model=adapted.model,
            tokenizer=tok,
            gate_result=g,
            texts=["AcmeWidgetFactory\n"],
        )


def test_quality_ok_native_runs(bundle) -> None:
    variant = ModelVariant(
        model=bundle.model,
        has_new_vocab_rows=False,
        uses_boundary_shift=False,
        tie_word_embeddings=True,
    )
    text = "def f(x):\n    return x\n"
    g = gate(bundle.tokenizer, variant, sample_texts=[text])
    assert g.verdict == GateVerdict.OK_NATIVE
    report = evaluate_quality(
        condition="C2",
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        gate_result=g,
        texts=[text],
    )
    assert report.bpb > 0
    assert report.n_bytes == len(text.encode("utf-8"))
    assert report.gate_verdict == "OK_NATIVE"


def test_gate_boundary_shift(bundle) -> None:
    variant = ModelVariant(
        model=bundle.model,
        has_new_vocab_rows=False,
        uses_boundary_shift=True,
        tie_word_embeddings=True,
    )
    g = gate(bundle.tokenizer, variant, sample_texts=["abc\n"])
    assert g.verdict == GateVerdict.BOUNDARY_SHIFT_UNVALIDATED
    assert g.quality_allowed is True


def test_dry_run_finetune_demo(tmp_path: Path) -> None:
    logs = dry_run_demo(run_dir=tmp_path / "ft", seed=0)
    assert len(logs) == 4
    assert all(log.dry_run for log in logs)
    assert all(log.steps_executed == 0 for log in logs)
    assert (tmp_path / "ft" / "finetune_summary.json").is_file()
    assert (tmp_path / "ft" / "finetune_data_fingerprint.json").is_file()
    assert len({log.n_train_bytes for log in logs}) == 1


def test_finetune_grid_identical_data_fingerprint(tmp_path: Path, bundle) -> None:
    texts = ["def a():\n    return 1\n", "def b():\n    return 2\n"]
    cfg = FinetuneConfig(dry_run=True, max_steps=2, n_seeds=1, seed=0)
    models = {c: bundle.model for c in ("C2", "C3", "C4", "C5")}
    toks = {c: bundle.tokenizer for c in ("C2", "C3", "C4", "C5")}
    logs = finetune_grid(
        models=models,
        tokenizers=toks,
        texts=texts,
        cfg=cfg,
        run_dir=tmp_path / "grid",
    )
    assert {log.condition for log in logs} == {"C2", "C3", "C4", "C5"}
    assert len({log.n_train_bytes for log in logs}) == 1


def test_ascii_base_constant() -> None:
    assert ASCII_BASE == 128
