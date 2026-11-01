"""Run hash must change when manifest / synthetic layer hashes change."""

from __future__ import annotations

from pathlib import Path

from toklab.experiments.config import ExperimentConfig
from toklab.experiments.runner import run_identity_dict
from toklab.experiments.store import compute_config_hash


def _cfg(tmp_path: Path, *, man_hash: str, synth_hash: str) -> ExperimentConfig:
    man = tmp_path / "manifest"
    man.mkdir(parents=True, exist_ok=True)
    (man / "manifest.parquet").write_bytes(b"parquet-placeholder")
    (man / "manifest_hash").write_text(man_hash + "\n", encoding="utf-8")
    synth = tmp_path / "synthetic"
    synth.mkdir(parents=True, exist_ok=True)
    (synth / "rename_map.json").write_text(
        f'{{"marker": "{synth_hash}"}}\n', encoding="utf-8"
    )
    (synth / "synth_summary.json").write_text("{}\n", encoding="utf-8")
    return ExperimentConfig.model_validate(
        {
            "name": "id_test",
            "seed": 0,
            "data": {
                "repos_dir": str(tmp_path / "repos"),
                "manifest_path": str(man / "manifest.parquet"),
                "synthetic_dir": str(synth),
            },
            "runs_root": str(tmp_path / "runs"),
        }
    )


def test_run_hash_includes_manifest_and_synthetic(tmp_path: Path) -> None:
    a = run_identity_dict(_cfg(tmp_path / "a", man_hash="aaa", synth_hash="s1"))
    b = run_identity_dict(_cfg(tmp_path / "b", man_hash="bbb", synth_hash="s1"))
    c = run_identity_dict(_cfg(tmp_path / "c", man_hash="aaa", synth_hash="s2"))
    ha = compute_config_hash(a)
    hb = compute_config_hash(b)
    hc = compute_config_hash(c)
    assert a["manifest_hash"] == "aaa"
    assert a["synthetic_hash"] is not None
    assert ha != hb  # different manifest
    assert ha != hc  # different synthetic payload
    assert hb != hc
