"""2×2 effects follow the documented (repository-weighted) definition."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from toklab.evaluation.report import render_report
from toklab.evaluation.stats import factorial_analysis

ROOT = Path(__file__).resolve().parents[2]

# Unequal file counts per repo, so file- and repo-weighted means differ.
VALUES = {
    "C2": {"r1": [100.0, 110.0, 120.0], "r2": [200.0]},
    "C3": {"r1": [90.0, 100.0, 104.0], "r2": [170.0]},
    "C4": {"r1": [130.0, 140.0, 150.0], "r2": [260.0]},
    "C5": {"r1": [118.0, 128.0, 138.0], "r2": [220.0]},
}


def _pooled(c: str) -> float:
    return float(np.mean([v for vs in VALUES[c].values() for v in vs]))


def _repo(c: str, r: str) -> float:
    return float(np.mean(VALUES[c][r]))


def test_effects_are_repo_weighted_cells_are_file_weighted() -> None:
    eff = factorial_analysis(VALUES, k=1, seed=0, n_boot=50)

    for c, ci in (("C2", eff.m_c2), ("C3", eff.m_c3), ("C4", eff.m_c4),
                  ("C5", eff.m_c5)):
        assert ci.estimate == pytest.approx(_pooled(c))

    def repo_avg(f) -> float:
        return float(np.mean([f(r) for r in ("r1", "r2")]))

    d = repo_avg(lambda r: _repo("C3", r) - _repo("C2", r))
    s = repo_avg(lambda r: _repo("C4", r) - _repo("C2", r))
    i = repo_avg(lambda r: (_repo("C5", r) - _repo("C4", r))
                 - (_repo("C3", r) - _repo("C2", r)))
    assert eff.effect_d.estimate == pytest.approx(d)
    assert eff.effect_s.estimate == pytest.approx(s)
    assert eff.interaction.estimate == pytest.approx(i)

    cell_d = _pooled("C3") - _pooled("C2")
    cell_i = (_pooled("C5") - _pooled("C4")) - (_pooled("C3") - _pooled("C2"))
    assert eff.effect_d.estimate != pytest.approx(cell_d)
    assert eff.interaction.estimate != pytest.approx(cell_i)

    # Decision rule: better of C3/C4 picked by cell means (C3 here), contrast
    # is the repo-weighted paired difference C5 - C3 (> 0, so not beaten).
    beat = repo_avg(lambda r: _repo("C5", r) - _repo("C3", r))
    assert beat > 0 and eff.combined_beats_either is False


def _write_run():
    spec = importlib.util.spec_from_file_location(
        "_ksel_def", ROOT / "tests" / "unit" / "test_k_selection.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._write_run


def test_report_states_how_effects_are_averaged(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run()(run_dir, k_star=0)
    text = render_report(
        run_dir=run_dir, hypotheses_path=ROOT / "configs" / "hypotheses.yaml"
    ).read_text(encoding="utf-8")
    for s in ("test", "test_inorg"):
        section = text.split(f"## 2×2 interaction [{s}]")[1].split("## Context")[0]
        note = section.split("**How these columns are averaged:**")[1]
        assert "each file weighted equally" in note
        assert "mean over the 5 repositories of per-repository differences" in note
        assert "differ from differences of the printed cell means" in note
        # Fixture row at K*=0: c2=100, c3=100, c4=90, c5=85; effect_s=-10,
        # effect_d=0, interaction=0.
        assert (
            "at K=0 the cell means give D=+0.0000, S=-10.0000, I=-5.0000; "
            "the table shows D=+0.0000, S=-10.0000, I=+0.0000"
        ) in note
        assert "decision-rule column uses the same repository-weighted" in note
