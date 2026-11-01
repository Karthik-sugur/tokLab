"""Where experiment source text is read from.

``original``: ``<dir>/<repo>/<path>`` (the cloned repositories).
``synthetic``: the synthetic enterprise layer written by ``toklab data synth``:
``<dir>/<split>/<repo>/<path>`` for adapt / test_inorg / val / test and
``<dir>/drift/p<P>/<repo>/<path>`` for the drift_holdout variants.

Every reader in the experiment (fidelity, intrinsic, context packing, mining,
systems inputs, factorial, drift) goes through :class:`Corpus`, so all
conditions see exactly the same text.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from toklab.core.hashing import config_hash, sha256_text
from toklab.core.source_io import read_source_text
from toklab.core.types import SourceFile
from toklab.data.splits import SPLITS, ManifestView, read_manifest_digest
from toklab.data.synth.drift import DRIFT_PROPORTIONS
from toklab.data.synth.pipeline import (
    ADAPT_LEXICON_SPLITS,
    DRIFT_SPLIT,
    SYNTH_MANIFEST,
)

CORPUS_KINDS = ("original", "synthetic")
SYNTH_LAYER_FILES = ("rename_map.json", "synth_summary.json", "lexicon_meta.json")


class CorpusError(RuntimeError):
    """The configured corpus is missing or does not match the manifest."""


def synthetic_layer_hash(root: Path | str | None) -> str | None:
    """Hash of the synthetic layer's key artifacts (``None`` if absent)."""
    if root is None:
        return None
    root = Path(root)
    if not root.is_dir():
        return None
    parts: dict[str, str] = {}
    for name in SYNTH_LAYER_FILES:
        path = root / name
        if path.is_file():
            parts[name] = sha256_text(path.read_text(encoding="utf-8"))
    if not parts:
        return None
    return config_hash(parts)


def verify_synthetic_corpus(root: Path | str, manifest: Path | str) -> pd.DataFrame:
    """Check the synthetic layer was built from ``manifest``; return its rows.

    Raises:
        CorpusError: ``synth_manifest.parquet`` is missing, predates manifest
            hashing, was built from another manifest or a subset, or does not
            cover every manifest file.
    """
    root = Path(root)
    manifest = Path(manifest)
    synth_manifest = root / SYNTH_MANIFEST
    rerun = (
        f"re-run `toklab data synth --manifest {manifest} --out {root}` "
        "(scripts/run_real_pipeline.sh step 2)"
    )
    if not synth_manifest.is_file():
        raise CorpusError(
            f"synthetic corpus {root}: {SYNTH_MANIFEST} is missing; {rerun}"
        )
    df = pd.read_parquet(synth_manifest)
    if "manifest_hash" not in df.columns:
        raise CorpusError(
            f"synthetic corpus {root}: {SYNTH_MANIFEST} records no manifest_hash "
            f"(written by an older synth); {rerun}"
        )
    expected = read_manifest_digest(manifest)
    found = sorted(set(df["manifest_hash"].astype(str)))
    if found != [expected]:
        raise CorpusError(
            f"synthetic corpus {root} was built from manifest_hash={found} but "
            f"{manifest} has manifest_hash={expected}; {rerun}"
        )
    if "subset" in df.columns and bool(df["subset"].any()):
        raise CorpusError(
            f"synthetic corpus {root} was built for a file subset (--only); {rerun}"
        )
    man = pd.read_parquet(manifest)
    for split in (*ADAPT_LEXICON_SPLITS, DRIFT_SPLIT):
        want = set(man.loc[man["split"] == split, "file_id"])
        have = set(df.loc[df["split"] == split, "file_id"])
        if want - have:
            raise CorpusError(
                f"synthetic corpus {root} lacks {len(want - have)} {split} "
                f"file(s) of {manifest}; {rerun}"
            )
    return df


@dataclass(frozen=True)
class Corpus:
    """Resolves manifest files to paths in the configured corpus."""

    kind: str
    root: Path
    manifest: Path

    def __post_init__(self) -> None:
        if self.kind not in CORPUS_KINDS:
            raise ValueError(f"corpus kind must be one of {CORPUS_KINDS}")

    @property
    def is_synthetic(self) -> bool:
        return self.kind == "synthetic"

    def split_root(self, split: str) -> Path:
        """Directory holding ``<repo>/<path>`` for files of ``split``."""
        if not self.is_synthetic:
            return self.root
        if split == DRIFT_SPLIT:
            raise ValueError(
                "drift_holdout files live under drift/p<P>; use drift_root()"
            )
        if split not in ADAPT_LEXICON_SPLITS:
            raise ValueError(f"no synthetic directory for split {split!r}")
        return self.root / split

    def drift_root(self, proportion: int) -> Path:
        """Pre-generated drift variant directory (synthetic corpora only)."""
        if not self.is_synthetic:
            raise ValueError("drift variants are pre-generated only for synthetic")
        if proportion not in DRIFT_PROPORTIONS:
            raise ValueError(f"unsupported drift proportion {proportion}")
        return self.root / "drift" / f"p{proportion}"

    def path(self, sf: SourceFile) -> Path:
        return self.split_root(sf.split) / sf.repo / sf.path

    def read(self, sf: SourceFile) -> str:
        return read_source_text(self.path(sf))

    def load_texts(self, files: Iterable[SourceFile]) -> dict[str, str]:
        return {sf.file_id: self.read(sf) for sf in files}

    def view(self, split: str) -> ManifestView:
        return ManifestView.from_parquet(self.manifest, split)


def open_corpus(
    kind: str, root: Path | str, manifest: Path | str, *, verify: bool = True
) -> Corpus:
    """Build a :class:`Corpus`; synthetic corpora are verified against the manifest."""
    corpus = Corpus(kind=kind, root=Path(root), manifest=Path(manifest))
    if not corpus.root.is_dir():
        raise CorpusError(f"corpus directory not found: {corpus.root}")
    if not corpus.manifest.is_file():
        raise CorpusError(f"manifest not found: {corpus.manifest}")
    if corpus.is_synthetic and verify:
        verify_synthetic_corpus(corpus.root, corpus.manifest)
    return corpus


def corpus_summary(corpus: Corpus) -> dict[str, Any]:
    """Data-derived description of the corpus for the report's Corpus block."""
    from toklab.data.ingest import load_ingest_config

    man = pd.read_parquet(corpus.manifest)
    present = [s for s in SPLITS if (man["split"] == s).any()]
    files_per_split = {s: int((man["split"] == s).sum()) for s in present}
    repos_per_split = {
        s: sorted(set(man.loc[man["split"] == s, "repo"])) for s in present
    }
    try:
        excluded = load_ingest_config()["excluded_files"]
    except (FileNotFoundError, ValueError):
        excluded = {}
    out: dict[str, Any] = {
        "kind": corpus.kind,
        "dir": corpus.root.as_posix(),
        "manifest": corpus.manifest.as_posix(),
        "manifest_hash": read_manifest_digest(corpus.manifest),
        "synthetic_hash": (
            synthetic_layer_hash(corpus.root) if corpus.is_synthetic else None
        ),
        "files_per_split": files_per_split,
        "repos_per_split": repos_per_split,
        "excluded_files": [{"path": p, "reason": r} for p, r in excluded.items()],
        "variants": [],
    }
    if not corpus.is_synthetic:
        return out
    df = pd.read_parquet(corpus.root / SYNTH_MANIFEST)
    for (split, variant), grp in df.groupby(["split", "variant"], sort=True):
        unchanged = grp[~grp["renamed"].astype(bool)]
        out["variants"].append(
            {
                "split": str(split),
                "variant": str(variant),
                "files": int(len(grp)),
                "files_renamed": int(grp["renamed"].astype(bool).sum()),
                "files_copied_unchanged": int(len(unchanged)),
                "unchanged_reasons": {
                    str(k): int(v)
                    for k, v in sorted(unchanged["reason"].value_counts().items())
                },
            }
        )
    return out
