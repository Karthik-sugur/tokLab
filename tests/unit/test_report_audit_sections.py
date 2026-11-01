"""Report sections added in the audit follow-up (method notes, paired, baselines)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from toklab.evaluation.paired import paired_differences
from toklab.evaluation.report import (
    build_limitations,
    ci_level_label,
    interaction_note,
    method_notes,
    render_report,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_run():
    spec = importlib.util.spec_from_file_location(
        "_ksel_audit", ROOT / "tests" / "unit" / "test_k_selection.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._write_run


def test_ci_level_derived_from_alpha() -> None:
    assert ci_level_label(0.05) == "95%"
    assert ci_level_label(0.1) == "90%"
    note = interaction_note(
        pd.DataFrame(
            [
                {
                    "k": 0,
                    "interaction": 0.4,
                    "interaction_low": -0.5,
                    "interaction_high": 1.3,
                    "combined_beats_either": False,
                }
            ]
        ),
        0,
        alpha=0.1,
    )
    assert "90% CI" in note and "95% CI" not in note


def test_method_notes_state_n_boot_and_h1_h3_rule() -> None:
    notes = method_notes(n_boot=50, alpha=0.05, test_repos=5)
    text = " ".join(notes)
    assert "n_boot=50" in text and "alpha=0.05" in text and "95%" in text
    assert "H1–H3" in text and "overlap" in text
    assert "H4 and H5 use paired" in text
    assert "does not specify" in text
    assert "126 distinct bootstrap resamples" in text


def test_quality_limitation_omits_systems_when_not_run() -> None:
    with_sys = build_limitations(
        corpus={},
        overlap={},
        selection={},
        cfg={},
        primary="test",
        any_invalid=False,
        test_repos=5,
        systems_ran=True,
    )
    without = build_limitations(
        corpus={},
        overlap={},
        selection={},
        cfg={},
        primary="test",
        any_invalid=False,
        test_repos=5,
        systems_ran=False,
    )
    assert any("quality_valid=False" in x for x in with_sys)
    assert not any("quality_valid=False" in x for x in without)
    assert any("Quality not measured" in x for x in without)


def test_paired_differences_equal_repo_means() -> None:
    metrics = pd.DataFrame(
        [
            {
                "eval_set": "test",
                "k": 8,
                "condition": c,
                "file_id": f,
                "metric": "tokens_per_kb",
                "value": v,
            }
            for c, vals in (
                ("C2", [("a1", 100.0), ("a2", 110.0), ("b1", 200.0)]),
                ("C3", [("a1", 90.0), ("a2", 100.0), ("b1", 170.0)]),
                ("C4", [("a1", 130.0), ("a2", 140.0), ("b1", 260.0)]),
                ("C5", [("a1", 118.0), ("a2", 128.0), ("b1", 220.0)]),
            )
            for f, v in vals
        ]
    )
    manifest = pd.DataFrame(
        [
            {"file_id": "a1", "repo": "ra"},
            {"file_id": "a2", "repo": "ra"},
            {"file_id": "b1", "repo": "rb"},
        ]
    )
    df = paired_differences(metrics, manifest, k=8, seed=0, n_boot=50)
    by = df.set_index("contrast")
    assert by.loc["C3-C2", "mean"] == pytest.approx(-20.0)
    assert by.loc["C3-C2", "repos_better"] == 2


def test_report_renders_paired_baselines_method_and_context_note(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_run()(run_dir, k_star=0)
    pd.DataFrame(
        [
            {
                "eval_set": "test",
                "k": 0,
                "contrast": "C3-C2",
                "mean": -1.0,
                "ci_low": -2.0,
                "ci_high": -0.5,
                "n_repos": 5,
                "n_boot": 1000,
                "seed": 0,
                "alpha": 0.05,
                "repos_better": 5,
                "repos_worse": 0,
                "repos_tie": 0,
            }
        ]
    ).to_parquet(run_dir / "paired_differences.parquet", index=False)
    (run_dir / "paired_meta.json").write_text(
        json.dumps(
            {
                "post_hoc": True,
                "note": (
                    "Descriptive paired differences added after seeing results; "
                    "not the pre-registered rule; does not change any verdict."
                ),
                "n_boot": 1000,
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )
    bdir = run_dir / "baselines"
    bdir.mkdir()
    pd.DataFrame(
        [
            {
                "condition": c,
                "n_files": 3,
                "n_roundtrip_ok": 3,
                "n_roundtrip_fail": 0,
                "n_orig_parsed": 3,
                "n_reparse_ok": 3,
                "n_reparse_fail": 0,
                "valid": True,
                "status": "OK",
            }
            for c in ("C1", "C2", "C6")
        ]
    ).to_parquet(bdir / "fidelity.parquet", index=False)
    pd.DataFrame(
        [
            {
                "condition": c,
                "metric": "tokens_per_kb",
                "estimate": e,
                "ci_low": e - 1,
                "ci_high": e + 1,
                "n_repos": 2,
                "n_boot": 50,
                "seed": 0,
                "alpha": 0.05,
                "valid": True,
                "status": "OK",
            }
            for c, e in (("C1", 400.0), ("C2", 240.0), ("C6", 1000.0))
        ]
    ).to_parquet(bdir / "token_efficiency_ci.parquet", index=False)

    text = render_report(
        run_dir=run_dir, hypotheses_path=ROOT / "configs" / "hypotheses.yaml"
    ).read_text(encoding="utf-8")
    assert "## Method notes (CIs and hypotheses)" in text
    assert "n_boot=" in text and "H1–H3" in text
    assert (
        "## Descriptive paired differences (post-hoc, not the pre-registered rule)"
        in text
    )
    assert "does **not** change any verdict" in text
    assert "| test | 0 | C3-C2 |" in text
    assert "## Intrinsic baselines (not part of the 2×2)" in text
    assert "| C1 | **OK** |" in text and "| C6 | **OK** |" in text
    assert "not monotonic" in text
    assert "quality_valid=False" not in text
    assert "How these columns are averaged" in text
