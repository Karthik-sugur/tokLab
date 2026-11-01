"""K sweep cap, K* on val (C3 only), and report fixes judged at K*."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from toklab.evaluation.report import (
    evaluate_hypotheses,
    interaction_note,
    render_report,
)
from toklab.experiments.runner import effective_k_values, select_k_star

HYPO = Path(__file__).resolve().parents[2] / "configs" / "hypotheses.yaml"


def test_k_sweep_capped_at_positive_gain_candidates() -> None:
    nominal = [0, 256, 1024, 4096, 16384]
    assert effective_k_values(nominal, {"native": 65469, "structural": 48230}) == (
        nominal
    )
    assert effective_k_values(nominal, {"native": 900, "structural": 3000}) == [
        0, 256, 1024, 3000,
    ]
    assert effective_k_values(nominal, {"native": 0, "structural": 0}) == [0]


def test_k_star_argmin_ties_smaller_and_eligibility() -> None:
    per_k = {0: 250.0, 256: 240.0, 1024: 238.0, 4096: 238.0}
    assert select_k_star(per_k, dict.fromkeys(per_k, True)) == 1024
    assert select_k_star(per_k, {0: True, 256: True, 1024: False, 4096: True}) == 4096
    assert select_k_star({0: 1.0, 8: 2.0}, {0: True, 8: True}) == 0
    assert select_k_star(per_k, dict.fromkeys(per_k, False)) is None


def _effect_row(k: int, *, interaction: tuple[float, float, float], beats: bool,
                c3: float, eval_set: str) -> dict:
    i, lo, hi = interaction
    return {
        "eval_set": eval_set, "k": k, "metric": "tokens_per_kb",
        "c2": 100.0, "c2_low": 98.0, "c2_high": 102.0,
        "c3": c3, "c3_low": c3 - 0.5, "c3_high": c3 + 0.5,
        "c4": 90.0, "c4_low": 89.0, "c4_high": 91.0,
        "c5": 85.0, "c5_low": 84.0, "c5_high": 86.0,
        "effect_s": -10.0, "effect_d": c3 - 100.0,
        "interaction": i, "interaction_low": lo, "interaction_high": hi,
        "r_s": 0.1, "r_d": 0.0, "independence_expectation": 0.1,
        "combined_beats_either": beats, "decision_rule": "rule",
    }


def _write_run(run_dir: Path, *, k_star: int | None) -> None:
    run_dir.mkdir(parents=True)
    sets = ("test", "test_inorg")
    fid, eff, effects, ctx = [], [], [], []
    for s in sets:
        for k in (0, 256):
            for c, est in (("C2", 100.0), ("C3", 100.0 if k == 0 else 97.0),
                           ("C4", 90.0), ("C5", 90.0 if k == 0 else 85.0)):
                fid.append({"eval_set": s, "k": k, "condition": c, "n_files": 3,
                            "n_roundtrip_ok": 3, "n_roundtrip_fail": 0,
                            "n_orig_parsed": 3, "n_reparse_ok": 3,
                            "n_reparse_fail": 0, "valid": True, "status": "OK"})
                eff.append({"eval_set": s, "k": k, "condition": c,
                            "metric": "tokens_per_kb", "estimate": est,
                            "ci_low": est - 1, "ci_high": est + 1, "n_repos": 5,
                            "n_boot": 10, "seed": 0, "alpha": 0.05, "valid": True})
        effects.append(_effect_row(0, interaction=(0.0, -0.5, 0.5), beats=False,
                                   c3=100.0, eval_set=s))
        effects.append(_effect_row(256, interaction=(-2.0, -3.0, -1.0), beats=True,
                                   c3=97.0, eval_set=s))
        for c in ("C2", "C3", "C4", "C5"):
            ctx.append({"eval_set": s, "k": k_star or 256, "condition": c,
                        "budget_tokens": 64, "files_fit": 1, "bytes_fit": 10,
                        "tokens_used": 5})
    pd.DataFrame(fid).to_parquet(run_dir / "fidelity.parquet", index=False)
    pd.DataFrame(eff).to_parquet(run_dir / "token_efficiency_ci.parquet", index=False)
    pd.DataFrame(effects).to_parquet(run_dir / "factorial_effects.parquet", index=False)
    pd.DataFrame(ctx).to_parquet(run_dir / "context_utilisation.parquet", index=False)
    pd.DataFrame(
        [
            {"k": k, "condition": c, "baseline": b, "delta_v": k,
             "delta_params": k * 896, "delta_bytes": k * 896 * 2, "eval_set": "test",
             "tokens_saved_per_kb": 0.0 if k == 0 else 3.0}
            for k in (0, 256) for c, b in (("C3", "C2"), ("C5", "C4"))
        ]
    ).to_parquet(run_dir / "vocab_cost_by_k.parquet", index=False)
    (run_dir / "config.json").write_text(json.dumps({
        "name": "ksel", "data": {"eval_split": "test",
                                 "eval_splits": ["test", "test_inorg"]},
        "vocab": {"min_freq": 1, "min_repos": 1, "k_values": [0, 256]},
    }))
    (run_dir / "status.json").write_text(json.dumps({"config_hash": "h", "name": "k"}))
    (run_dir / "env.json").write_text("{}")
    (run_dir / "k_selection.json").write_text(json.dumps({
        "split": "val", "condition": "C3", "metric": "tokens_per_kb", "rule": "argmin",
        "n_files": 4, "k_star": k_star,
        "status": "ok" if k_star is not None else "not run: no files in split 'val'",
        "per_k": [{"k": 0, "tokens_per_kb": 101.0, "roundtrip_fail": 0,
                   "eligible": True, "vocab_size": 0},
                  {"k": 256, "tokens_per_kb": 99.0, "roundtrip_fail": 0,
                   "eligible": True, "vocab_size": 256}],
    }))
    (run_dir / "corpus.json").write_text(json.dumps({
        "kind": "synthetic", "dir": "data/synthetic", "manifest": "m",
        "manifest_hash": "abc", "synthetic_hash": "def",
        "files_per_split": {"adapt": 10, "test": 3},
        "repos_per_split": {"adapt": ["a1", "a2"],
                            "test": ["t1", "t2", "t3", "t4", "t5"]},
        "excluded_files": [{"path": "r/x.py", "reason": "NFC normalizer"}],
        "variants": [{"split": "adapt", "variant": "adapt_lexicon", "files": 10,
                      "files_renamed": 8, "files_copied_unchanged": 2,
                      "unchanged_reasons": {"ast_parse_error": 2}}],
    }))
    (run_dir / "name_overlap.json").write_text(json.dumps({
        s: {"identifier_occurrences": 100, "occurrences_in_adapt": 82,
            "occurrence_fraction": 0.82, "distinct_identifiers": 40,
            "distinct_in_adapt": 16, "distinct_fraction": 0.4}
        for s in sets
    }))


def test_hypotheses_judged_at_k_star_only() -> None:
    hypo = {"decision_rule": "rule", "hypotheses": {
        "H2": {"claim": "c3", "contrast": "C3 vs C2"},
        "H4": {"claim": "combined", "contrast": "C5 vs max"},
        "H5": {"claim": "interaction", "contrast": "I"},
    }}
    effects = pd.DataFrame([
        _effect_row(0, interaction=(0.0, -0.5, 0.5), beats=False, c3=100.0,
                    eval_set="test"),
        _effect_row(256, interaction=(-2.0, -3.0, -1.0), beats=True, c3=97.0,
                    eval_set="test"),
    ])
    at0 = {v["id"]: v["verdict"] for v in
           evaluate_hypotheses(effects, hypo, invalid_conditions=set(), k=0)}
    at256 = {v["id"]: v["verdict"] for v in
             evaluate_hypotheses(effects, hypo, invalid_conditions=set(), k=256)}
    assert at0 == {"H2": "inconclusive", "H4": "not supported", "H5": "inconclusive"}
    assert at256 == {"H2": "supported", "H4": "supported", "H5": "supported"}


def test_interaction_sentence_driven_by_ci() -> None:
    effects = pd.DataFrame([
        _effect_row(0, interaction=(0.4, -0.5, 1.3), beats=False, c3=100.0,
                    eval_set="test"),
        _effect_row(8, interaction=(-2.0, -3.0, -1.0), beats=True, c3=97.0,
                    eval_set="test"),
    ])
    note0 = interaction_note(effects, 0)
    assert note0.startswith("No evidence of an interaction (CI includes zero)")
    assert "beats" not in note0
    assert "Decision rule not satisfied" in note0
    note8 = interaction_note(effects, 8)
    assert "CI excludes zero" in note8 and "beats" in note8


def test_report_renders_every_set_k_and_all_hypotheses(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, k_star=0)
    text = render_report(run_dir=run_dir, hypotheses_path=HYPO).read_text("utf-8")

    for s in ("test", "test_inorg"):
        block = text.split(f"## Token efficiency [{s}]")[1].split("## 2×2")[0]
        assert "not run" not in block
        for k in ("0 (K*)", "256"):
            for c in ("C2", "C3", "C4", "C5"):
                assert f"| {k} | {c} |" in block, (s, k, c)
        section = text.split(f"## 2×2 interaction [{s}]")[1].split("| K |")[0]
        summary = section.split("**Interaction summary:**")[1]
        assert summary.strip().startswith(
            "No evidence of an interaction (CI includes zero)"
        )
        assert "beats" not in summary
        for h in ("H1", "H2", "H3", "H4", "H5"):
            assert f"### {h} [{s}] — " in text
    assert "Hypotheses verdicts [test] (primary) at K*=0" in text
    assert "### H4 [test] — not supported" in text

    assert "## Corpus" in text and "| kind | **synthetic** |" in text
    assert "| adapt | adapt_lexicon | 10 | 8 | 2 | ast_parse_error=2 |" in text
    assert "**K\\* = 0**" in text
    assert "## Vocab overhead per K (ESTIMATE)" in text
    assert "| 256 | C5 vs C4 | 256 | 229376 | 458752 | 3.0000 |" in text

    lim = text.split("## Limitations")[1]
    assert "Shared Zipf lexicon" in lim
    assert "82 of 100 identifier occurrences (0.8200)" in lim
    assert "16 of 40 distinct identifiers (0.4000)" in lim
    assert "Related repositories in different splits" in lim
    assert "Only 5 test repositories" in lim
    assert "FILE-level hold-out" in lim
    assert "Alpha-equivalence is not runtime equivalence" in lim
    assert "K chosen on val: K*=0" in lim
    assert "Quality not measured" in lim
    assert "r/x.py (NFC normalizer)" in lim


def test_report_without_k_star_marks_hypotheses_not_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, k_star=None)
    text = render_report(run_dir=run_dir, hypotheses_path=HYPO).read_text("utf-8")
    for h in ("H1", "H2", "H3", "H4", "H5"):
        assert f"### {h} [test] — not run" in text
    assert "K\\* = not selected" in text
