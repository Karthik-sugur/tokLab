"""Unit tests for ingest, dedup, repository splits, and ManifestView."""

from __future__ import annotations

from pathlib import Path

import pytest

from toklab.core.types import SourceFile
from toklab.data.dedup import cluster_duplicates, merge_repos_for_clusters
from toklab.data.ingest import ingest_repos, load_ingest_config
from toklab.data.splits import (
    ManifestView,
    assign_splits,
    build_dataset,
    manifest_hash,
    write_manifest,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _near_dup_body(tag: str) -> str:
    """Long similar bodies so MinHash near-dup triggers across repos."""
    lines = [f"# module preamble {i}\n" for i in range(80)]
    lines.append(f"VALUE = {tag!r}\n")
    lines.append("def helper(x):\n    return x + 1\n")
    return "".join(lines)


def make_fixture_repos(root: Path) -> Path:
    """Create small multi-repo fixture under ``root/repos``."""
    repos = root / "repos"

    # repo_a / repo_b share an exact duplicate (forces repo merge).
    shared = "def shared():\n    return 1\n"
    _write(repos / "repo_a" / "a.py", "def a_only():\n    return 'a'\n")
    _write(repos / "repo_a" / "shared.py", shared)
    _write(repos / "repo_b" / "b.py", "def b_only():\n    return 'b'\n")
    _write(repos / "repo_b" / "shared.py", shared)

    # Near-duplicates spanning repo_c and repo_d.
    _write(repos / "repo_c" / "near.py", _near_dup_body("c"))
    _write(repos / "repo_d" / "near.py", _near_dup_body("d"))

    # Independent repos for remaining splits.
    _write(repos / "repo_e" / "e.py", "def e_only():\n    return 'e'\n")
    _write(repos / "repo_f" / "f.py", "def f_only():\n    return 'f'\n")
    _write(repos / "repo_g" / "g.py", "def g_only():\n    return 'g'\n")

    # Skips: vendor, generated marker, binary, oversized.
    _write(repos / "repo_e" / "vendor" / "skip_me.py", "x = 1\n")
    _write(
        repos / "repo_e" / "gen.py",
        "# @generated — do not edit\nauto = True\n",
    )
    (repos / "repo_e" / "binary.py").write_bytes(b"print(1)\x00\xff")
    big = repos / "repo_e" / "too_big.py"
    big.parent.mkdir(parents=True, exist_ok=True)
    big.write_bytes(b"a = 1\n" * (200_000))  # > 1 MiB

    return repos


def test_ingest_skips_vendor_generated_binary_and_large(tmp_path: Path) -> None:
    repos = make_fixture_repos(tmp_path)
    files = list(ingest_repos(repos))
    paths = {(f.repo, f.path) for f in files}
    assert ("repo_e", "vendor/skip_me.py") not in paths
    assert ("repo_e", "gen.py") not in paths
    assert ("repo_e", "binary.py") not in paths
    assert ("repo_e", "too_big.py") not in paths
    assert ("repo_e", "e.py") in paths
    assert all(f.language == "python" for f in files)
    assert all(f.split == "" for f in files)


def test_ingest_config_lists_docs_examples_and_hook_files() -> None:
    cfg = load_ingest_config()
    dirs = cfg["skip_dir_names"]
    files = cfg["skip_file_names"]
    for name in (
        "docs_src",
        "docs",
        "examples",
        "example",
        "tutorials",
        "benchmarks",
        ".tox",
        "node_modules",
        "vendor",
    ):
        assert name in dirs
    assert "conftest.py" in files
    assert "setup.py" in files


def test_ingest_skips_docs_examples_conftest_setup(tmp_path: Path) -> None:
    repos = tmp_path / "repos"
    keep = "def keep():\n    return 1\n"
    _write(repos / "demo" / "pkg" / "core.py", keep)
    _write(repos / "demo" / "docs" / "guide.py", "x = 1\n")
    _write(repos / "demo" / "docs_src" / "page.py", "x = 2\n")
    _write(repos / "demo" / "examples" / "demo.py", "x = 3\n")
    _write(repos / "demo" / "example" / "tiny.py", "x = 4\n")
    _write(repos / "demo" / "tutorials" / "t.py", "x = 5\n")
    _write(repos / "demo" / "benchmarks" / "b.py", "x = 6\n")
    _write(repos / "demo" / ".tox" / "py.py", "x = 7\n")
    _write(repos / "demo" / "node_modules" / "x.py", "x = 8\n")
    _write(repos / "demo" / "tests" / "conftest.py", "x = 9\n")
    _write(repos / "demo" / "setup.py", "x = 10\n")
    _write(repos / "demo" / "pkg" / "conftest.py", "x = 11\n")

    paths = {(f.repo, f.path) for f in ingest_repos(repos)}
    assert paths == {("demo", "pkg/core.py")}


def test_same_inputs_and_seed_identical_manifest_hash(tmp_path: Path) -> None:
    repos = make_fixture_repos(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    build_dataset(repos, out1, seed=0)
    build_dataset(repos, out2, seed=0)
    h1 = (out1 / "manifest_hash").read_text(encoding="utf-8").strip()
    h2 = (out2 / "manifest_hash").read_text(encoding="utf-8").strip()
    assert h1 == h2
    assert len(h1) == 64


def test_no_repo_appears_in_two_splits(tmp_path: Path) -> None:
    repos = make_fixture_repos(tmp_path)
    out = tmp_path / "out"
    build_dataset(repos, out, seed=0)
    from toklab.data.splits import SPLITS

    repo_to_splits: dict[str, set[str]] = {}
    for split in SPLITS:
        for f in ManifestView.from_parquet(out / "manifest.parquet", split):
            repo_to_splits.setdefault(f.repo, set()).add(f.split)
    for repo, splits in repo_to_splits.items():
        assert len(splits) == 1, f"repo {repo} in splits {splits}"
    total = sum(
        len(ManifestView.from_parquet(out / "manifest.parquet", s)) for s in SPLITS
    )
    assert total > 0


def test_no_duplicate_cluster_spans_two_splits(tmp_path: Path) -> None:
    repos = make_fixture_repos(tmp_path)
    files = list(ingest_repos(repos))
    texts = {
        f.file_id: (repos / f.repo / f.path).read_text(encoding="utf-8") for f in files
    }
    clusters = cluster_duplicates(files, texts)
    assigned = assign_splits(files, clusters, seed=0)
    id_to_split = {f.file_id: f.split for f in assigned}
    for cluster in clusters:
        splits = {id_to_split[fid] for fid in cluster.file_ids}
        assert len(splits) == 1, f"cluster {cluster.cluster_id} spans {splits}"


def test_manifest_view_raises_on_other_split(tmp_path: Path) -> None:
    repos = make_fixture_repos(tmp_path)
    out = tmp_path / "out"
    build_dataset(repos, out, seed=0)
    view = ManifestView.from_parquet(out / "manifest.parquet", "adapt")
    assert view.split == "adapt"
    view.access_split("adapt")  # ok
    with pytest.raises(PermissionError, match="refusing access"):
        view.access_split("test")


def test_exact_dup_merges_repos(tmp_path: Path) -> None:
    repos = make_fixture_repos(tmp_path)
    files = list(ingest_repos(repos))
    texts = {
        f.file_id: (repos / f.repo / f.path).read_text(encoding="utf-8") for f in files
    }
    clusters = cluster_duplicates(files, texts)
    groups = merge_repos_for_clusters(files, clusters)
    # repo_a and repo_b share shared.py exact hash → same group.
    ab = next(g for g in groups if "repo_a" in g or "repo_b" in g)
    assert "repo_a" in ab and "repo_b" in ab


def test_near_dup_merges_repos(tmp_path: Path) -> None:
    repos = make_fixture_repos(tmp_path)
    files = list(ingest_repos(repos))
    texts = {
        f.file_id: (repos / f.repo / f.path).read_text(encoding="utf-8") for f in files
    }
    clusters = cluster_duplicates(files, texts)
    groups = merge_repos_for_clusters(files, clusters)
    cd = [g for g in groups if "repo_c" in g or "repo_d" in g]
    assert len(cd) == 1
    assert "repo_c" in cd[0] and "repo_d" in cd[0]


def test_write_manifest_roundtrip(tmp_path: Path) -> None:
    files = [
        SourceFile("id1", "r", "a.py", "abc", "python", 3, "adapt"),
        SourceFile("id2", "r", "b.py", "def", "python", 3, "test"),
    ]
    digest = write_manifest(files, tmp_path / "m")
    assert digest == manifest_hash(files)
    adapt = ManifestView.from_parquet(tmp_path / "m" / "manifest.parquet", "adapt")
    assert len(adapt) == 1
    assert adapt.files()[0].path == "a.py"
