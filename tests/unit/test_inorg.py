"""test_inorg: file-level hold-out inside adapt repos, with no mining leakage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from toklab.core.types import SourceFile
from toklab.data.dedup import DuplicateCluster
from toklab.data.splits import (
    INORG_SPLIT,
    MIN_REPOS_DISABLED,
    ManifestView,
    build_dataset,
    hold_out_inorg_files,
)
from toklab.evaluation.report import load_run_label, render_report
from toklab.vocab.mining import mine_candidates, mine_from_files

HYPO = Path(__file__).resolve().parents[2] / "configs" / "hypotheses.yaml"
V1_HASH = "0250decc37f179b589bec3bf9f1f190b84bdc77f12fad58d10b6879bae319a0f"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_repos(root: Path, n_repos: int = 12, n_files: int = 10) -> Path:
    """Each file defines an identifier unique to that file; each repo also
    holds an exact in-repo duplicate pair (one duplicate cluster)."""
    repos = root / "repos"

    def filler(tag: str) -> str:
        # Distinct content per file so MinHash never links unrelated files.
        words = [
            hashlib.sha256(f"{tag}:{k}".encode()).hexdigest()[:12]
            for k in range(40)
        ]
        return "".join(f"w_{w} = {k}\n" for k, w in enumerate(words))

    for i in range(n_repos):
        repo = f"repo_{i:02d}"
        for j in range(n_files):
            _write(
                repos / repo / f"m{j}.py",
                f"def only_{repo}_file{j}():\n    return {i * 100 + j}\n"
                + filler(f"{repo}/{j}"),
            )
        dup = f"def dup_{repo}():\n    return '{repo}'\n" + filler(f"{repo}/dup")
        _write(repos / repo / "dup_a.py", dup)
        _write(repos / repo / "dup_b.py", dup)
    return repos


def _manifest(tmp_path: Path, **kw) -> Path:
    repos = _make_repos(tmp_path, **kw)
    out = tmp_path / "manifest"
    build_dataset(repos, out, seed=0)
    return out / "manifest.parquet"


def test_inorg_is_seeded_fraction_of_each_adapt_repo(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    df = pd.read_parquet(manifest)
    adapt_repos = set(df.loc[df["split"] == "adapt", "repo"])
    inorg = df[df["split"] == INORG_SPLIT]
    assert set(inorg["repo"]) == adapt_repos
    for repo in adapt_repos:
        n_total = int((df["repo"] == repo).sum())
        n_held = int(((df["repo"] == repo) & (df["split"] == INORG_SPLIT)).sum())
        assert 0 < n_held <= round(0.20 * n_total)
    # Same inputs + seed -> identical hold-out.
    again = _manifest(tmp_path / "again")
    assert pd.read_parquet(again).sort_values("file_id")["split"].tolist() == (
        df.sort_values("file_id")["split"].tolist()
    )


def test_duplicate_cluster_never_spans_mining_and_inorg(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    df = pd.read_parquet(manifest)
    assert df["repo"].nunique() == 12
    for repo, g in df.groupby("repo"):
        assert len(g) == 12, repo
        dups = g[g["path"].isin(["dup_a.py", "dup_b.py"])]
        assert len(dups) == 2, repo
        assert dups["split"].nunique() == 1, f"{repo}: {dups['split'].tolist()}"


def test_hold_out_moves_near_dup_cluster_as_unit() -> None:
    files = [
        SourceFile(f"f{i}", "r", f"p{i}.py", "h", "python", 1, "adapt")
        for i in range(10)
    ]
    near = DuplicateCluster("c1", ("f0", "f1", "f2"))
    for seed in range(20):
        out = hold_out_inorg_files(files, [near], seed=seed, fraction=0.2)
        splits = {f.file_id: f.split for f in out}
        assert len({splits["f0"], splits["f1"], splits["f2"]}) == 1
        assert sum(s == INORG_SPLIT for s in splits.values()) <= 2
        assert any(s == "adapt" for s in splits.values())


def test_mining_never_sees_inorg_files(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    repos = tmp_path / "repos"
    adapt = ManifestView.from_parquet(manifest, "adapt")
    inorg = ManifestView.from_parquet(manifest, INORG_SPLIT)
    assert len(inorg) > 0
    mined = {
        c.token
        for c in mine_candidates(
            adapt, repos_dir=repos, mode="structural", base_tokenizer=None
        )
    }
    adapt_ids = {f.file_id for f in adapt}
    for f in inorg:
        assert f.file_id not in adapt_ids
        unique = f"only_{f.repo}_{Path(f.path).stem.replace('m', 'file', 1)}"
        if f.path.startswith("m"):
            assert unique not in mined
        else:
            assert f"dup_{f.repo}" not in mined


def test_selection_inputs_are_adapt_only(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    inorg = ManifestView.from_parquet(manifest, INORG_SPLIT)
    with pytest.raises(PermissionError):
        mine_candidates(
            inorg, repos_dir=tmp_path / "repos", mode="structural",
            base_tokenizer=None,
        )
    from toklab.vocab.build import build_adapted_vocab

    with pytest.raises(PermissionError):
        build_adapted_vocab(
            inorg, repos_dir=tmp_path / "repos", base=None, mode="structural", k=4
        )
    files = list(inorg.files())
    with pytest.raises(PermissionError):
        mine_from_files(
            files,
            {f.file_id: "" for f in files},
            mode="structural",
            base_tokenizer=None,
        )


def _write_run(run_dir: Path, *, with_eval_set: bool, config_hash: str) -> None:
    run_dir.mkdir(parents=True)
    sets = ["test", "test_inorg"] if with_eval_set else [None]
    fid, eff, eff_rows, ctx = [], [], [], []
    for s in sets:
        tag = {"eval_set": s} if s else {}
        for c in ("C2", "C3", "C4", "C5"):
            fid.append(
                {**tag, "condition": c, "n_files": 3, "n_roundtrip_ok": 3,
                 "n_roundtrip_fail": 0, "n_orig_parsed": 3, "n_reparse_ok": 3,
                 "n_reparse_fail": 0, "valid": True, "status": "OK"}
            )
            eff.append(
                {**tag, "condition": c, "metric": "tokens_per_kb",
                 "estimate": 1.0, "ci_low": 0.5, "ci_high": 1.5, "n_repos": 2,
                 "n_boot": 10, "seed": 0, "alpha": 0.05}
            )
            ctx.append(
                {**tag, "condition": c, "budget_tokens": 64, "files_fit": 1,
                 "bytes_fit": 10, "tokens_used": 5}
            )
        row = {"k": 8, "metric": "tokens_per_kb", "decision_rule": "r",
               "combined_beats_either": False, "r_s": 0.0, "r_d": 0.0,
               "independence_expectation": 0.0}
        for key in ("c2", "c3", "c4", "c5", "effect_s", "effect_d",
                    "interaction"):
            row[key] = 1.0
        for key in ("c2", "c3", "c4", "c5", "interaction"):
            row[f"{key}_low"], row[f"{key}_high"] = 0.5, 1.5
        eff_rows.append({**tag, **row})
    pd.DataFrame(fid).to_parquet(run_dir / "fidelity.parquet", index=False)
    pd.DataFrame(eff).to_parquet(run_dir / "token_efficiency_ci.parquet", index=False)
    pd.DataFrame(eff_rows).to_parquet(
        run_dir / "factorial_effects.parquet", index=False
    )
    pd.DataFrame(ctx).to_parquet(run_dir / "context_utilisation.parquet", index=False)
    data = {"eval_split": "test"}
    if with_eval_set:
        data["eval_splits"] = ["test", "test_inorg"]
    (run_dir / "config.json").write_text(
        json.dumps({"name": "x", "data": data, "vocab": {"min_repos": 1}}),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"config_hash": config_hash, "name": "x"}), encoding="utf-8"
    )
    (run_dir / "env.json").write_text("{}", encoding="utf-8")


def test_report_shows_both_test_sets(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, with_eval_set=True, config_hash="h")
    text = render_report(run_dir=run_dir, hypotheses_path=HYPO).read_text("utf-8")
    for s in ("test", "test_inorg"):
        assert f"# Eval set `{s}`" in text
        assert f"## Token efficiency [{s}]" in text
        assert f"## 2×2 interaction [{s}]" in text
        assert f"## Hypotheses verdicts [{s}]" in text
    assert "new codebase" in text
    assert "new files in known codebases" in text
    assert "min_repos=1" in text and "overfit" in text


def test_legacy_run_marks_inorg_not_run_and_v1_label(tmp_path: Path) -> None:
    run_dir = tmp_path / V1_HASH
    _write_run(run_dir, with_eval_set=False, config_hash=V1_HASH)
    text = render_report(run_dir=run_dir, hypotheses_path=HYPO).read_text("utf-8")
    assert "Run status: preliminary v1" in text
    assert "## Token efficiency [test]" in text
    assert "# Eval set `test_inorg`" in text
    assert "Not run for this evaluation set" in text
    assert load_run_label("unknown-hash") == {}


def test_runner_reports_both_eval_sets(tmp_path: Path) -> None:
    from toklab.experiments.config import ExperimentConfig
    from toklab.experiments.runner import run_experiment

    repos = _make_repos(tmp_path, n_repos=5, n_files=4)
    man = tmp_path / "manifest"
    build_dataset(repos, man, seed=0, min_repos=MIN_REPOS_DISABLED)
    assert len(ManifestView.from_parquet(man / "manifest.parquet", INORG_SPLIT))
    cfg = ExperimentConfig.model_validate(
        {
            "name": "inorg",
            "seed": 0,
            "model": {
                "model_name": "Qwen/Qwen2.5-Coder-0.5B",
                "tokenizer_name": "Qwen/Qwen2.5-Coder-0.5B",
            },
            "data": {
                "repos_dir": str(repos),
                "manifest_path": str(man / "manifest.parquet"),
                "eval_split": "test",
                "eval_splits": ["test", "test_inorg"],
            },
            "vocab": {"k_values": [0, 2]},
            "intrinsic": {
                "conditions": ["C2", "C3", "C4", "C5"],
                "context_budgets": [64],
                "n_boot": 5,
            },
            "systems": {"enabled": False},
            "stages": {"systems": False},
            "hypotheses_path": str(HYPO),
            "runs_root": str(tmp_path / "runs"),
        }
    )
    status = run_experiment(cfg)
    run_dir = Path(status["run_dir"])
    for name in (
        "fidelity.parquet",
        "intrinsic_metrics.parquet",
        "token_efficiency_ci.parquet",
        "factorial_effects.parquet",
        "context_utilisation.parquet",
    ):
        df = pd.read_parquet(run_dir / name)
        assert set(df["eval_set"]) == {"test", "test_inorg"}, name
    text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Token efficiency [test]" in text
    assert "## Token efficiency [test_inorg]" in text
