"""Exact-hash and MinHash near-duplicate clustering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

from toklab.core.hashing import sha256_text
from toklab.core.types import SourceFile

NUM_PERM = 128
NEAR_DUP_THRESHOLD = 0.85
SHINGLE_SIZE = 5


@dataclass(frozen=True)
class DuplicateCluster:
    """A set of files linked by exact or near duplication."""

    cluster_id: str
    file_ids: tuple[str, ...]


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self._parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Deterministic parent: lexicographically smaller root wins.
        if ra < rb:
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for x in self._parent:
            groups[self.find(x)].append(x)
        return {k: sorted(v) for k, v in groups.items()}


def _shingles(text: str) -> set[str]:
    if len(text) < SHINGLE_SIZE:
        return {text} if text else set()
    return {text[i : i + SHINGLE_SIZE] for i in range(len(text) - SHINGLE_SIZE + 1)}


def _minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for s in _shingles(text):
        m.update(s.encode("utf-8"))
    return m


def cluster_duplicates(
    files: list[SourceFile],
    texts: dict[str, str],
    *,
    threshold: float = NEAR_DUP_THRESHOLD,
) -> list[DuplicateCluster]:
    """Cluster files by exact content hash and MinHash near-duplicates.

    ``texts`` maps ``file_id`` -> source text (read-only; never written back).
    Singleton files still appear as one-member clusters.
    """
    uf = _UnionFind()
    for f in files:
        uf.add(f.file_id)

    # Exact duplicates by content sha256.
    by_hash: dict[str, list[str]] = defaultdict(list)
    for f in files:
        by_hash[f.sha256].append(f.file_id)
    for ids in by_hash.values():
        if len(ids) < 2:
            continue
        head = ids[0]
        for other in ids[1:]:
            uf.union(head, other)

    # Near-duplicates via MinHash LSH.
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    signatures: dict[str, MinHash] = {}
    for f in sorted(files, key=lambda x: x.file_id):
        text = texts[f.file_id]
        mh = _minhash(text)
        signatures[f.file_id] = mh
        # Query before insert so we only link to earlier indexed items.
        for other_id in lsh.query(mh):
            uf.union(f.file_id, other_id)
        lsh.insert(f.file_id, mh)

    clusters: list[DuplicateCluster] = []
    for _root, members in sorted(uf.components().items(), key=lambda kv: kv[0]):
        cluster_id = sha256_text("cluster:" + ",".join(members))[:16]
        clusters.append(
            DuplicateCluster(cluster_id=cluster_id, file_ids=tuple(members))
        )
    # Stable order by cluster_id
    clusters.sort(key=lambda c: c.cluster_id)
    return clusters


def merge_repos_for_clusters(
    files: list[SourceFile],
    clusters: list[DuplicateCluster],
) -> list[frozenset[str]]:
    """Return repo groups that must share a split.

    If a duplicate cluster spans multiple repositories, those repos are
    unioned. Every repository appears in exactly one group.
    """
    id_to_repo = {f.file_id: f.repo for f in files}
    uf = _UnionFind()
    for f in files:
        uf.add(f.repo)

    for cluster in clusters:
        repos = sorted({id_to_repo[fid] for fid in cluster.file_ids})
        if len(repos) < 2:
            continue
        head = repos[0]
        for r in repos[1:]:
            uf.union(head, r)

    groups = [frozenset(members) for members in uf.components().values()]
    groups.sort(key=lambda g: min(g))
    return groups
