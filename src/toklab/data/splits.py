"""Repository-level dataset splits and single-split ManifestView."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from toklab.core.hashing import config_hash, sha256_text
from toklab.core.types import SourceFile
from toklab.data.dedup import (
    DuplicateCluster,
    cluster_duplicates,
    drop_cross_repo_duplicates,
)
from toklab.data.ingest import ingest_repos, load_ingest_config

SPLITS = ("adapt", "val", "test", "drift_holdout")
# Fractions over *repositories* (after dropping cross-repo dup files).
DEFAULT_RATIOS: dict[str, float] = {
    "adapt": 0.60,
    "val": 0.10,
    "test": 0.20,
    "drift_holdout": 0.10,
}
DEFAULT_MIN_REPOS: dict[str, int] = {
    "val": 3,
    "test": 3,
    "drift_holdout": 3,
}
# For tiny unit-test fixtures that cannot meet production floors.
MIN_REPOS_DISABLED: dict[str, int] = {s: 0 for s in SPLITS}


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


def _ratios_and_mins_from_config() -> tuple[dict[str, float], dict[str, int]]:
    try:
        cfg = load_ingest_config()
    except FileNotFoundError:
        return dict(DEFAULT_RATIOS), dict(DEFAULT_MIN_REPOS)
    ratios = dict(DEFAULT_RATIOS)
    ratios.update(cfg.get("split_ratios") or {})
    mins = dict(DEFAULT_MIN_REPOS)
    mins.update(cfg.get("min_repos_per_split") or {})
    return ratios, mins


def compute_repo_targets(
    n_repos: int,
    ratios: dict[str, float],
    min_repos: dict[str, int],
) -> dict[str, int]:
    """Largest-remainder repo counts, then enforce per-split minima."""
    missing = [s for s in SPLITS if s not in ratios]
    if missing:
        raise ValueError(f"ratios missing splits: {missing}")
    total = sum(ratios[s] for s in SPLITS)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    required = sum(max(0, int(min_repos.get(s, 0))) for s in SPLITS)
    if n_repos < required:
        raise ValueError(
            f"cannot satisfy min_repos_per_split={dict(min_repos)}: "
            f"need at least {required} repositories, got {n_repos}"
        )

    raw = [n_repos * ratios[s] for s in SPLITS]
    floors = [int(x) for x in raw]
    left = n_repos - sum(floors)
    order = sorted(
        range(len(SPLITS)),
        key=lambda i: (-(raw[i] - floors[i]), i),
    )
    targets_list = list(floors)
    for i in order[:left]:
        targets_list[i] += 1
    targets = dict(zip(SPLITS, targets_list, strict=True))

    for split_name in SPLITS:
        need = int(min_repos.get(split_name, 0)) - targets[split_name]
        if need <= 0:
            continue
        # Steal from adapt first, then any split above its minimum.
        donors = ["adapt"] + [s for s in SPLITS if s != "adapt" and s != split_name]
        for donor in donors:
            if need <= 0:
                break
            donor_floor = int(min_repos.get(donor, 0))
            give = min(need, max(0, targets[donor] - donor_floor))
            targets[donor] -= give
            targets[split_name] += give
            need -= give
        if need > 0:
            raise ValueError(
                f"cannot give split {split_name!r} "
                f"{min_repos.get(split_name, 0)} repos "
                f"(targets after ratios={targets}, min_repos={dict(min_repos)}, "
                f"n_repos={n_repos})"
            )

    if sum(targets.values()) != n_repos:
        raise RuntimeError(
            f"internal error: targets sum {sum(targets.values())} != {n_repos}"
        )
    for split_name in SPLITS:
        floor = int(min_repos.get(split_name, 0))
        if targets[split_name] < floor:
            raise ValueError(
                f"split {split_name!r} got {targets[split_name]} repos "
                f"but min_repos_per_split requires {floor}"
            )
    return targets


def _assign_repos_to_splits(
    repo_file_counts: dict[str, int],
    seed: int,
    ratios: dict[str, float],
    min_repos: dict[str, int],
) -> dict[str, str]:
    """Map each repo to a split; balance file mass under fixed repo quotas."""
    repos = sorted(repo_file_counts)
    n = len(repos)
    targets = compute_repo_targets(n, ratios, min_repos)
    total_files = sum(repo_file_counts.values()) or 1
    target_files = {s: total_files * ratios[s] for s in SPLITS}
    current_files = {s: 0 for s in SPLITS}
    slots = dict(targets)
    rng = random.Random(seed)

    # Largest repos first; name tie-break for determinism.
    ordered = sorted(repos, key=lambda r: (-repo_file_counts[r], r))
    repo_to_split: dict[str, str] = {}
    for repo in ordered:
        candidates = [s for s in SPLITS if slots[s] > 0]
        if not candidates:
            raise RuntimeError("no split slots remaining")
        best_def = max(target_files[s] - current_files[s] for s in candidates)
        tied = [
            s
            for s in candidates
            if abs((target_files[s] - current_files[s]) - best_def) < 1e-9
        ]
        tied.sort()
        pick = tied[rng.randrange(len(tied))] if len(tied) > 1 else tied[0]
        repo_to_split[repo] = pick
        slots[pick] -= 1
        current_files[pick] += repo_file_counts[repo]
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
    min_repos: dict[str, int] | None = None,
) -> list[SourceFile]:
    """Drop cross-repo dup copies, then assign each remaining repo a split."""
    cfg_ratios, cfg_mins = _ratios_and_mins_from_config()
    ratios = dict(cfg_ratios if ratios is None else ratios)
    mins = dict(cfg_mins if min_repos is None else min_repos)

    kept = drop_cross_repo_duplicates(files, clusters)
    if not kept:
        raise ValueError("no files left after dropping cross-repo duplicates")

    repo_file_counts = Counter(f.repo for f in kept)
    repo_to_split = _assign_repos_to_splits(
        dict(repo_file_counts),
        seed=seed,
        ratios=ratios,
        min_repos=mins,
    )
    return [replace(f, split=repo_to_split[f.repo]) for f in kept]


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


def format_split_detail(files: Sequence[SourceFile]) -> str:
    """Repos and file counts per split, with repo names."""
    lines: list[str] = []
    for split_name in SPLITS:
        subset = [f for f in files if f.split == split_name]
        repos = sorted({f.repo for f in subset})
        lines.append(
            f"{split_name}: repos={len(repos)} files={len(subset)}"
        )
        lines.append(f"  {', '.join(repos) if repos else '(none)'}")
    return "\n".join(lines)


def build_dataset(
    repos_dir: Path | str,
    out_dir: Path | str,
    seed: int = 0,
    ratios: dict[str, float] | None = None,
    min_repos: dict[str, int] | None = None,
) -> dict[str, int]:
    """Ingest, dedup (drop cross-repo copies), split by repository, write manifest.

    Returns a mapping of split name -> file count.
    """
    repos_dir = Path(repos_dir)
    files = list(ingest_repos(repos_dir))
    if not files:
        raise ValueError(f"no eligible .py files under {repos_dir}")

    texts = _load_texts(repos_dir, files)
    clusters = cluster_duplicates(files, texts)
    assigned = assign_splits(
        files, clusters, seed=seed, ratios=ratios, min_repos=min_repos
    )
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
