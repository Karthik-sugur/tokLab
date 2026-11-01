"""Experiment reads data.corpus: synthetic layer checks, readers, drift files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from toklab.core.source_io import read_source_text
from toklab.data.corpus import CorpusError, open_corpus, verify_synthetic_corpus
from toklab.data.splits import build_dataset, read_manifest_digest
from toklab.data.synth.drift import DRIFT_PROPORTIONS
from toklab.data.synth.pipeline import ADAPT_LEXICON_SPLITS, SYNTH_MANIFEST
from toklab.evaluation.drift import DriftLeakError, run_drift
from toklab.experiments.config import ExperimentConfig
from toklab.tokenizers.native_hf import NativeHFTokenizer
from toklab.tokenizers.registry import get_tokenizer

HYPO = Path(__file__).resolve().parents[2] / "configs" / "hypotheses.yaml"
SYNTH_NAME = "AcmeLedgerReconciliationService"
LEAK_NAME = "LeakyEvalOnlyIdentifier"


@pytest.fixture(scope="module")
def base() -> NativeHFTokenizer:
    return NativeHFTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", name="C2")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repos(root: Path, n_repos: int = 12, n_files: int = 6) -> Path:
    repos = root / "repos"
    for i in range(n_repos):
        for j in range(n_files):
            words = [
                hashlib.sha256(f"{i}:{j}:{k}".encode()).hexdigest()[:10]
                for k in range(30)
            ]
            body = "".join(f"w_{w} = {k}\n" for k, w in enumerate(words))
            _write(
                repos / f"repo_{i:02d}" / f"m{j}.py",
                f"def helper(x):\n    return helper(x) + {i * 10 + j}\n" + body,
            )
    return repos


def _fake_synth(
    manifest: Path, repos: Path, out: Path, *, h: str | None = None
) -> None:
    """Synthetic layer with text that differs from the original.

    ``helper`` -> ``AcmeLedgerReconciliationService`` everywhere; non-adapt
    splits also get an identifier that never appears in adapt.
    """
    man = pd.read_parquet(manifest)
    rows = []
    for r in man.itertuples(index=False):
        text = read_source_text(repos / r.repo / r.path)
        renamed = text.replace("helper", SYNTH_NAME)
        if r.split in ADAPT_LEXICON_SPLITS:
            if r.split != "adapt":
                renamed += f"{LEAK_NAME} = 1\n"
            _write(out / r.split / r.repo / r.path, renamed)
            rows.append((r.split, "adapt_lexicon", r, f"{r.split}/{r.repo}/{r.path}"))
        else:
            for p in DRIFT_PROPORTIONS:
                _write(
                    out / "drift" / f"p{p}" / r.repo / r.path,
                    text if p == 0 else renamed,
                )
                rows.append(
                    (r.split, f"drift_p{p}", r, f"drift/p{p}/{r.repo}/{r.path}")
                )
    pd.DataFrame(
        [
            {
                "split": split, "variant": variant, "file_id": r.file_id,
                "repo": r.repo, "path": r.path, "output": output,
                "renamed": variant != "drift_p0", "reason": "",
                "n_identifiers": 1,
                "manifest_hash": h or read_manifest_digest(manifest),
                "subset": False,
            }
            for split, variant, r, output in rows
        ]
    ).to_parquet(out / SYNTH_MANIFEST, index=False)
    drift_maps = {str(p): {} for p in DRIFT_PROPORTIONS}
    (out / "rename_map.json").write_text(json.dumps({"drift": drift_maps}))
    (out / "synth_summary.json").write_text("{}\n")


@pytest.fixture
def layer(tmp_path: Path) -> tuple[Path, Path, Path]:
    repos = _repos(tmp_path)
    man_dir = tmp_path / "manifest"
    build_dataset(repos, man_dir, seed=0)
    manifest = man_dir / "manifest.parquet"
    synth = tmp_path / "synthetic"
    _fake_synth(manifest, repos, synth)
    return repos, manifest, synth


def test_synthetic_corpus_errors_are_clear(layer, tmp_path: Path) -> None:
    repos, manifest, synth = layer
    verify_synthetic_corpus(synth, manifest)

    df = pd.read_parquet(synth / SYNTH_MANIFEST)
    df.assign(manifest_hash="0" * 64).to_parquet(synth / SYNTH_MANIFEST, index=False)
    with pytest.raises(CorpusError, match="was built from manifest_hash"):
        open_corpus("synthetic", synth, manifest)

    df.drop(columns=["manifest_hash"]).to_parquet(synth / SYNTH_MANIFEST, index=False)
    with pytest.raises(CorpusError, match="records no manifest_hash"):
        open_corpus("synthetic", synth, manifest)

    (synth / SYNTH_MANIFEST).unlink()
    with pytest.raises(CorpusError, match="synth_manifest.parquet is missing"):
        open_corpus("synthetic", synth, manifest)


def test_real_synth_records_manifest_hash(tmp_path: Path) -> None:
    from toklab.data.synth import run_synth

    repos = _repos(tmp_path, n_repos=12, n_files=2)
    build_dataset(repos, tmp_path / "m", seed=0)
    manifest = tmp_path / "m" / "manifest.parquet"
    summary = run_synth(manifest=manifest, repos_dir=repos, out_dir=tmp_path / "s")
    digest = read_manifest_digest(manifest)
    assert summary["manifest_hash"] == digest
    assert set(pd.read_parquet(tmp_path / "s" / SYNTH_MANIFEST)["manifest_hash"]) == {
        digest
    }
    verify_synthetic_corpus(tmp_path / "s", manifest)


def test_mining_reads_synthetic_adapt_only(layer, base: NativeHFTokenizer) -> None:
    from toklab.experiments.runner import AdaptMining

    _repos_dir, manifest, synth = layer
    corpus = open_corpus("synthetic", synth, manifest)
    cfg = ExperimentConfig.model_validate(
        {"data": {"corpus": {"kind": "synthetic", "dir": str(synth),
                             "manifest": str(manifest)}}}
    )
    mining = AdaptMining(corpus, cfg, base, base_model="x")
    for mode in ("native", "structural"):
        names = {c.token for c in mining.raw[mode]}
        assert SYNTH_NAME in names, mode
        assert "helper" not in names, mode  # original-only name
        assert LEAK_NAME not in names, mode  # only in non-adapt splits


def test_token_counts_equal_counts_on_synthetic_text(
    layer, tmp_path: Path, base: NativeHFTokenizer
) -> None:
    from toklab.experiments.runner import run_experiment

    repos, manifest, synth = layer
    cfg = ExperimentConfig.model_validate(
        {
            "name": "synthetic_corpus",
            "seed": 0,
            "model": {"model_name": "Qwen/Qwen2.5-Coder-0.5B"},
            "data": {
                "corpus": {"kind": "synthetic", "dir": str(synth),
                           "manifest": str(manifest)},
                "eval_split": "test",
                "eval_splits": ["test", "test_inorg"],
            },
            "vocab": {"k_values": [0, 2]},
            "intrinsic": {"conditions": ["C2", "C3", "C4", "C5"],
                          "context_budgets": [64], "n_boot": 5},
            "hypotheses_path": str(HYPO),
            "runs_root": str(tmp_path / "runs"),
        }
    )
    status = run_experiment(cfg)
    run_dir = Path(status["run_dir"])
    corpus_meta = json.loads((run_dir / "corpus.json").read_text())
    assert corpus_meta["kind"] == "synthetic"
    assert corpus_meta["synthetic_hash"]

    metrics = pd.read_parquet(run_dir / "intrinsic_metrics.parquet")
    man = pd.read_parquet(manifest)
    c2 = get_tokenizer("C2", {"base_tokenizer": base})
    n_differ = 0
    for split in ("test", "test_inorg"):
        sub = metrics[
            (metrics["eval_set"] == split)
            & (metrics["condition"] == "C2")
            & (metrics["metric"] == "n_tokens")
            & (metrics["k"] == 0)
        ]
        got = dict(zip(sub["file_id"], sub["value"].astype(int), strict=True))
        files = man[man["split"] == split]
        assert set(got) == set(files["file_id"])
        for r in files.itertuples(index=False):
            synthetic_text = read_source_text(synth / split / r.repo / r.path)
            original_text = read_source_text(repos / r.repo / r.path)
            assert synthetic_text != original_text
            n_syn = len(c2.encode(synthetic_text).ids)
            assert got[r.file_id] == n_syn
            n_differ += n_syn != len(c2.encode(original_text).ids)
    assert n_differ > 0

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Corpus" in report and "| kind | **synthetic** |" in report


def test_drift_reads_pregenerated_files(layer, base: NativeHFTokenizer) -> None:
    _repos_dir, manifest, synth = layer
    corpus = open_corpus("synthetic", synth, manifest)
    drift_files = sorted(
        corpus.view("drift_holdout").files(), key=lambda f: f.file_id
    )
    marked = drift_files[0]
    path = corpus.drift_root(25) / marked.repo / marked.path
    path.write_text(read_source_text(path) + "PregeneratedDriftMarker = 1\n")

    _table, _df, rows = run_drift(
        manifest=manifest,
        corpus=corpus,
        seed=0,
        k_static=2,
        k_refresh=0,
        proportions=(0, 25),
        cfg={"base_tokenizer": base, "model_name": "Qwen/Qwen2.5-Coder-0.5B"},
    )
    c2 = get_tokenizer("C2", {"base_tokenizer": base})
    for p in (0, 25):
        expect = sum(
            len(c2.encode(read_source_text(corpus.drift_root(p) / f.repo / f.path)).ids)
            for f in drift_files
        )
        row = next(r for r in rows if r.variant == "C2" and r.proportion == p)
        assert row.n_tokens == expect

    maps = json.loads((synth / "rename_map.json").read_text())
    maps["drift"]["25"] = {f"{marked.repo}/{marked.path}": {"old": SYNTH_NAME}}
    (synth / "rename_map.json").write_text(json.dumps(maps))
    with pytest.raises(DriftLeakError, match="adapt corpus"):
        run_drift(
            manifest=manifest,
            corpus=corpus,
            seed=0,
            k_static=0,
            k_refresh=0,
            proportions=(25,),
            cfg={"base_tokenizer": base, "model_name": "Qwen/Qwen2.5-Coder-0.5B"},
        )


def test_mvp_config_reads_synthetic_and_sweeps_k() -> None:
    from toklab.experiments.config import load_experiment_config

    root = Path(__file__).resolve().parents[2]
    cfg = load_experiment_config(root / "configs" / "experiments" / "mvp_py.yaml")
    corpus = cfg.data.resolved_corpus()
    assert (corpus.kind, corpus.dir.as_posix()) == ("synthetic", "data/synthetic")
    assert corpus.manifest.as_posix() == "data/processed/manifest.parquet"
    assert cfg.vocab.k_values == [0, 256, 1024, 4096, 16384]
    assert cfg.data.k_selection_split == "val"
    s = cfg.systems
    assert s.enabled and s.conditions == ["C2", "C3", "C4", "C5"]
    assert (s.seq_lens, s.n_warmup, s.n_repeats) == ([512, 1024, 2048], 2, 10)
    assert (s.device, s.dtype) == ("cpu", "float32")
