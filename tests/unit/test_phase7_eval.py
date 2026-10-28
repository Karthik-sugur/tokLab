"""Phase 7: context util, vocab_cost, systems, store, report, determinism."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from toklab.core.types import EncodeResult, SourceFile
from toklab.data.splits import build_dataset
from toklab.evaluation.intrinsic import context_utilisation
from toklab.evaluation.report import evaluate_hypotheses, render_report
from toklab.evaluation.vocab_cost import (
    delta_bytes,
    delta_params,
    flops_per_token_estimate,
    kv_cache_bytes,
)


def _mini_repos(root: Path) -> Path:
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
        "repo_c": "def c():\n    return 3\n",
        "repo_d": "def d():\n    return 4\n",
        "repo_e": "print(0)\n",
    }
    for name, body in bodies.items():
        p = repos / name / "mod.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return repos


def test_context_utilisation_greedy_packing() -> None:
    files = [
        ("a", 10, 100),
        ("b", 20, 200),
        ("c", 5, 50),
    ]
    df = context_utilisation(files, [15, 35, 100])
    assert list(df["budget_tokens"]) == [15, 35, 100]
    # budget 15: a(10) fits, b(20) skip, c(5) fits → 2 files, 150 bytes, 15 tok
    row15 = df[df["budget_tokens"] == 15].iloc[0]
    assert int(row15["files_fit"]) == 2
    assert int(row15["bytes_fit"]) == 150
    assert int(row15["tokens_used"]) == 15
    row35 = df[df["budget_tokens"] == 35].iloc[0]
    assert int(row35["files_fit"]) == 3
    assert int(row35["tokens_used"]) == 35


def test_vocab_cost_hand_computed() -> None:
    # delta_V=128, d=896, tied → 128*896*1 = 114688
    assert delta_params(128, 896, tie_word_embeddings=True) == 114688
    # untied → *2
    assert delta_params(128, 896, tie_word_embeddings=False) == 229376
    assert delta_bytes(114688, bytes_per_param=2) == 229376
    # kv: 2*24*2*64*128*2 = 1572864
    assert (
        kv_cache_bytes(
            layers=24, kv_heads=2, head_dim=64, seq_len=128, bytes_per_elem=2
        )
        == 1_572_864
    )
    fl = flops_per_token_estimate(d_model=896, layers=24, seq_len=128)
    assert fl.label == "ESTIMATE"
    d, L, T = 896, 24, 128
    expected_mlp = L * 2 * d * (4.0 * d)
    expected_attn = L * (4 * d * d + 2 * T * d)
    assert fl.flops_mlp_per_token == pytest.approx(expected_mlp)
    assert fl.flops_attn_per_token == pytest.approx(expected_attn)
    assert fl.flops_per_token_estimate == pytest.approx(
        expected_mlp + expected_attn
    )


def test_systems_timing_median_iqr_repeats() -> None:
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(32, 8)
            self.lin = torch.nn.Linear(8, 8)

        def forward(self, input_ids=None, **_kw):  # noqa: ANN001
            x = self.emb(input_ids)
            return self.lin(x.mean(dim=1))

    from toklab.evaluation.systems import measure_forward_latency

    model = Tiny()
    ids = torch.randint(0, 32, (1, 16))
    row = measure_forward_latency(
        model,
        ids,
        condition="C2",
        seq_len=16,
        device="cpu",
        dtype="float32",
        n_warmup=5,
        n_repeats=30,
        source_bytes=16,
        quality_valid=True,
    )
    assert row.n_repeats == 30
    assert row.n_warmup == 5
    assert row.latency_ms_median >= 0
    assert row.latency_ms_iqr >= 0
    assert row.device == "cpu"
    assert row.dtype == "float32"


def test_systems_c3_c5_quality_valid_false() -> None:
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock, patch

    from toklab.evaluation.systems import run_systems_benchmark

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = MagicMock(vocab_size=50)
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, input_ids=None, **_kw):  # noqa: ANN001
            return self.lin(input_ids.float().mean(dim=1, keepdim=True).expand(-1, 4))

        def to(self, *_a, **_k):  # noqa: ANN001
            return self

        def eval(self):  # noqa: ANN201
            return self

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=Tiny(),
    ):
        df = run_systems_benchmark(
            model_name="dummy",
            conditions=["C2", "C3", "C5"],
            seq_lens=[8],
            device="cpu",
            dtype="float32",
            n_warmup=5,
            n_repeats=30,
        )
    assert set(df["condition"]) == {"C2", "C3", "C5"}
    assert bool(df[df["condition"] == "C2"]["quality_valid"].iloc[0]) is True
    assert bool(df[df["condition"] == "C3"]["quality_valid"].iloc[0]) is False
    assert bool(df[df["condition"] == "C5"]["quality_valid"].iloc[0]) is False


def test_rerun_identical_token_metrics(tmp_path: Path) -> None:
    from toklab.experiments.config import ExperimentConfig
    from toklab.experiments.runner import run_experiment

    repos = _mini_repos(tmp_path)
    man = tmp_path / "manifest"
    build_dataset(repos, man, seed=0)
    runs = tmp_path / "runs"

    def make_cfg() -> ExperimentConfig:
        return ExperimentConfig.model_validate(
            {
                "name": "det",
                "seed": 0,
                "model": {
                    "model_name": "Qwen/Qwen2.5-Coder-0.5B",
                    "tokenizer_name": "Qwen/Qwen2.5-Coder-0.5B",
                },
                "data": {
                    "repos_dir": str(repos),
                    "manifest_path": str(man / "manifest.parquet"),
                },
                "vocab": {"k_values": [0, 4], "min_freq": 1, "min_repos": 1},
                "intrinsic": {
                    "conditions": ["C2", "C3", "C4", "C5"],
                    "context_budgets": [64, 128],
                    "n_boot": 20,
                },
                "systems": {"enabled": False},
                "stages": {
                    "fidelity": True,
                    "intrinsic": True,
                    "factorial": True,
                    "context": True,
                    "vocab_cost": True,
                    "systems": False,
                    "report": True,
                },
                "hypotheses_path": str(
                    Path(__file__).resolve().parents[2]
                    / "configs"
                    / "hypotheses.yaml"
                ),
                "runs_root": str(runs),
            }
        )

    s1 = run_experiment(make_cfg())
    path1 = Path(s1["run_dir"]) / "intrinsic_metrics.parquet"
    bytes1 = path1.read_bytes()
    s2 = run_experiment(make_cfg())
    path2 = Path(s2["run_dir"]) / "intrinsic_metrics.parquet"
    bytes2 = path2.read_bytes()
    assert s1["config_hash"] == s2["config_hash"]
    assert bytes1 == bytes2


def test_report_negative_interaction(tmp_path: Path) -> None:
    """Report renders and states negative synergy when I > 0 on tokens/KB."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Fabricate effects with positive interaction (bad for synergy claim).
    effects = pd.DataFrame(
        [
            {
                "k": 8,
                "metric": "tokens_per_kb",
                "c2": 100.0,
                "c3": 90.0,
                "c4": 85.0,
                "c5": 95.0,
                "c2_low": 98.0,
                "c2_high": 102.0,
                "c3_low": 88.0,
                "c3_high": 92.0,
                "c4_low": 83.0,
                "c4_high": 87.0,
                "c5_low": 93.0,
                "c5_high": 97.0,
                "effect_s": -15.0,
                "effect_d": -10.0,
                "interaction": 20.0,
                "interaction_low": 10.0,
                "interaction_high": 30.0,
                "r_s": 0.15,
                "r_d": 0.10,
                "independence_expectation": 0.235,
                "combined_beats_either": False,
                "decision_rule": "C5 beats better of C3/C4",
            }
        ]
    )
    effects.to_parquet(run_dir / "factorial_effects.parquet", index=False)
    pd.DataFrame(
        [
            {
                "condition": "C2",
                "n_files": 1,
                "n_roundtrip_ok": 1,
                "n_roundtrip_fail": 0,
                "n_orig_parsed": 1,
                "n_reparse_ok": 1,
                "n_reparse_fail": 0,
                "valid": True,
                "status": "OK",
            }
        ]
    ).to_parquet(run_dir / "fidelity.parquet", index=False)
    (run_dir / "env.json").write_text(
        '{"seed":0,"git_sha":"x","python_version":"3","platform":"t",'
        '"cpu":"c","gpu":"none","manifest_hash":"m","vocab_hash":"v",'
        '"config_hash":"h","library_versions":{}}',
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        '{"name":"neg","model":{"model_name":"x"}}', encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        '{"config_hash":"h","name":"neg"}', encoding="utf-8"
    )
    hypo = (
        Path(__file__).resolve().parents[2] / "configs" / "hypotheses.yaml"
    )
    path = render_report(run_dir=run_dir, hypotheses_path=hypo)
    text = path.read_text(encoding="utf-8")
    assert "Interaction" in text or "interaction" in text.lower()
    assert "worse than additive" in text or "negative" in text.lower()
    assert "not supported" in text


def test_report_invalid_broken_tokenizer(tmp_path: Path) -> None:
    class BrokenTok:
        name = "C2"

        def encode(self, text: str) -> EncodeResult:
            return EncodeResult(ids=[1, 2], spans=[(0, 1), (1, 2)])

        def decode(self, ids: list[int]) -> str:
            return "NOT_THE_ORIGINAL"

    from toklab.evaluation.fidelity import check_roundtrip

    sf = SourceFile("f1", "r", "a.py", "h", "python", 3, "test")
    report = check_roundtrip(BrokenTok(), [sf], texts={"f1": "abc"})
    assert report.n_roundtrip_fail == 1

    run_dir = tmp_path / "inv"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "condition": "C2",
                "n_files": 1,
                "n_roundtrip_ok": 0,
                "n_roundtrip_fail": 1,
                "n_orig_parsed": 0,
                "n_reparse_ok": 0,
                "n_reparse_fail": 0,
                "valid": False,
                "status": "INVALID",
            },
            {
                "condition": "C4",
                "n_files": 1,
                "n_roundtrip_ok": 1,
                "n_roundtrip_fail": 0,
                "n_orig_parsed": 1,
                "n_reparse_ok": 1,
                "n_reparse_fail": 0,
                "valid": True,
                "status": "OK",
            },
        ]
    ).to_parquet(run_dir / "fidelity.parquet", index=False)
    (run_dir / "env.json").write_text(
        '{"seed":0,"git_sha":"x","python_version":"3","platform":"t",'
        '"cpu":"c","gpu":"none","manifest_hash":"m","vocab_hash":"v",'
        '"config_hash":"h","library_versions":{}}',
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        '{"name":"inv","model":{"model_name":"x"}}', encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        '{"config_hash":"h","name":"inv"}', encoding="utf-8"
    )
    hypo = (
        Path(__file__).resolve().parents[2] / "configs" / "hypotheses.yaml"
    )
    path = render_report(run_dir=run_dir, hypotheses_path=hypo)
    text = path.read_text(encoding="utf-8")
    assert "INVALID" in text
    assert "C2" in text


def test_hypothesis_verdicts_from_yaml_rules() -> None:
    effects = pd.DataFrame(
        [
            {
                "k": 8,
                "metric": "tokens_per_kb",
                "c2": 100.0,
                "c3": 90.0,
                "c4": 85.0,
                "c5": 95.0,
                "c2_low": 98.0,
                "c2_high": 102.0,
                "c3_low": 88.0,
                "c3_high": 92.0,
                "c4_low": 83.0,
                "c4_high": 87.0,
                "c5_low": 93.0,
                "c5_high": 97.0,
                "effect_s": -15.0,
                "effect_d": -10.0,
                "interaction": 20.0,
                "interaction_low": 10.0,
                "interaction_high": 30.0,
                "r_s": 0.15,
                "r_d": 0.10,
                "independence_expectation": 0.235,
                "combined_beats_either": False,
                "decision_rule": "rule",
            }
        ]
    )
    hypo = {
        "decision_rule": "rule",
        "hypotheses": {
            "H4": {"claim": "combined beats", "contrast": "C5 vs max"},
            "H5": {"claim": "interaction nonzero", "contrast": "I"},
        },
    }
    verdicts = evaluate_hypotheses(effects, hypo, invalid_conditions=set())
    by_id = {v["id"]: v for v in verdicts}
    assert by_id["H4"]["verdict"] == "not supported"
    assert by_id["H5"]["verdict"] == "supported"

