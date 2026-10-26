"""Repository-level dataset splits and single-split ManifestView."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from toklab.core.hashing import config_hash, sha256_text
from toklab.core.types import SourceFile
from toklab.data.dedup import (
    DuplicateCluster,
    cluster_duplicates,
    merge_repos_for_clusters,
)
from toklab.data.ingest import ingest_repos

SPLITS = ("adapt", "val", "test", "drift_holdout")
# Fractions over repository *groups* (after near-dup merges).
DEFAULT_RATIOS: dict[str, float] = {
    "adapt": 0.50,
    "val": 0.15,
    "test": 0.20,
    "drift_holdout": 0.15,
}


class ManifestView:
    """Read-only view of a manifest restricted to exactly one split.

    Vocabulary and other leakage-sensitive code must receive
    ``ManifestView(..., split="adapt")``, never the full manifest.
    """

    def __init__(self, files: Sequence[SourceFile], split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        self._split = split
        self._files = tuple(f for f in files if f.split == split)

    @classmethod
    def from_parquet(cls, path: Path | str, split: str) -> ManifestView:
        df = pd.read_parquet(path)
        files = [
            SourceFile(
                file_id=row.file_id,
                repo=row.repo,
                path=row.path,
                sha256=row.sha256,
                language=row.language,
                n_bytes=int(row.n_bytes),
                split=row.split,
            )
            for row in df.itertuples(index=False)
        ]
        return cls(files, split)

    @property
    def split(self) -> str:
        return self._split

    def files(self) -> tuple[SourceFile, ...]:
        """Return files belonging to this view's split only."""
        return self._files

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self):
        return iter(self._files)

    def access_split(self, split: str) -> tuple[SourceFile, ...]:
        """Return files for ``split``; raises if ``split`` is not this view."""
        if split != self._split:
            raise PermissionError(
                f"ManifestView is bound to split {self._split!r}; "
                f"refusing access to {split!r}"
            )
        return self._files


def _assign_groups_to_splits(
    groups: list[frozenset[str]],
    seed: int,
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    """Map each repo name to a split label."""
    ratios = dict(ratios or DEFAULT_RATIOS)
    missing = [s for s in SPLITS if s not in ratios]
    if missing:
        raise ValueError(f"ratios missing splits: {missing}")
    total = sum(ratios[s] for s in SPLITS)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    ordered = sorted(groups, key=lambda g: (min(g), len(g)))
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    # Largest-remainder method keeps small-N assignments closer to ratios.
    raw = [n * ratios[s] for s in SPLITS]
    floors = [int(x) for x in raw]
    left = n - sum(floors)
    order = sorted(
        range(len(SPLITS)),
        key=lambda i: (-(raw[i] - floors[i]), i),
    )
    targets_list = list(floors)
    for i in order[:left]:
        targets_list[i] += 1
    targets = dict(zip(SPLITS, targets_list, strict=True))

    repo_to_split: dict[str, str] = {}
    idx = 0
    for split_name in SPLITS:
        for _ in range(targets[split_name]):
            if idx >= n:
                break
            for repo in ordered[idx]:
                repo_to_split[repo] = split_name
            idx += 1
    return repo_to_split


def _load_texts(repos_dir: Path, files: list[SourceFile]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for f in files:
        path = repos_dir / f.repo / f.path
        texts[f.file_id] = path.read_text(encoding="utf-8")
    return texts


def assign_splits(
    files: list[SourceFile],
    clusters: list[DuplicateCluster],
    seed: int,
    ratios: dict[str, float] | None = None,
) -> list[SourceFile]:
    """Assign each file a split; clusters spanning repos force repo merges."""
    groups = merge_repos_for_clusters(files, clusters)
    repo_to_split = _assign_groups_to_splits(groups, seed=seed, ratios=ratios)
    return [replace(f, split=repo_to_split[f.repo]) for f in files]


def manifest_hash(files: Sequence[SourceFile]) -> str:
    """Canonical hash of a manifest (order-independent via sorted keys)."""
    rows = [
        asdict(f)
        for f in sorted(files, key=lambda x: (x.file_id, x.repo, x.path))
    ]
    return config_hash({"files": rows})


def write_manifest(files: Sequence[SourceFile], out_dir: Path | str) -> str:
    """Write ``manifest.parquet`` and ``manifest_hash``; return the hash."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ordered = sorted(files, key=lambda x: (x.file_id, x.repo, x.path))
    df = pd.DataFrame([asdict(f) for f in ordered])
    parquet_path = out / "manifest.parquet"
    df.to_parquet(parquet_path, index=False)
    digest = manifest_hash(ordered)
    (out / "manifest_hash").write_text(digest + "\n", encoding="utf-8")
    return digest


def build_dataset(
    repos_dir: Path | str,
    out_dir: Path | str,
    seed: int = 0,
    ratios: dict[str, float] | None = None,
) -> dict[str, int]:
    """Ingest, dedup, split by repository, and write the manifest.

    Returns a mapping of split name -> file count.
    """
    repos_dir = Path(repos_dir)
    files = list(ingest_repos(repos_dir))
    if not files:
        raise ValueError(f"no eligible .py files under {repos_dir}")

    texts = _load_texts(repos_dir, files)
    clusters = cluster_duplicates(files, texts)
    assigned = assign_splits(files, clusters, seed=seed, ratios=ratios)
    write_manifest(assigned, out_dir)

    counts = {s: 0 for s in SPLITS}
    for f in assigned:
        counts[f.split] += 1
    return counts


def split_counts_message(counts: dict[str, int]) -> str:
    """Human-readable split counts for CLI output."""
    parts = [f"{s}={counts.get(s, 0)}" for s in SPLITS]
    parts.append(f"total={sum(counts.values())}")
    return "split_counts " + " ".join(parts)


# Re-export helper used by tests / hashing of seed configs.
def run_fingerprint(repos_dir: Path, seed: int) -> str:
    """Hash of (relative paths + seed) for debugging determinism."""
    files = sorted(
        (f.repo, f.path, f.sha256) for f in ingest_repos(repos_dir)
    )
    return sha256_text(repr((files, seed)))
