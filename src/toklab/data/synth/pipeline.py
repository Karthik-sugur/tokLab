"""Orchestrate the synthetic enterprise layer over every split of a manifest.

* adapt / test_inorg / val / test: renamed with ``lexicon_adapt``
* drift_holdout: drift variants p{0,10,25,50} with ``lexicon_drift`` only
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from toklab.core.source_io import read_source_text, write_source_text
from toklab.data.splits import ManifestView, read_manifest_digest
from toklab.data.synth.drift import DRIFT_PROPORTIONS, apply_drift
from toklab.data.synth.lexicon import EnterpriseLexicon
from toklab.data.synth.rename import rename_file_with_lexicon

ADAPT_LEXICON_SPLITS = ("adapt", "test_inorg", "val", "test")
DRIFT_SPLIT = "drift_holdout"
SYNTH_MANIFEST = "synth_manifest.parquet"


def _emit(src: Path, dst: Path, *, renamed: bool, text: str) -> None:
    """Write a synthetic output; unchanged files are copied byte for byte."""
    if renamed:
        write_source_text(dst, text)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _variant_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    renamed = [r for r in rows if r["renamed"]]
    reasons = Counter(r["reason"] for r in rows if not r["renamed"])
    return {
        "files": len(rows),
        "files_renamed": len(renamed),
        "files_copied_unchanged": len(rows) - len(renamed),
        "unchanged_reasons": dict(sorted(reasons.items())),
        "identifiers_renamed": sum(r["n_identifiers"] for r in rows),
    }


def run_synth(
    *,
    manifest: Path | str,
    repos_dir: Path | str,
    out_dir: Path | str,
    seed: int = 0,
    lexicon_yaml: Path | str | None = None,
    only: set[str] | None = None,
) -> dict[str, Any]:
    """Apply the synthetic layer to all splits.

    Writes:
      ``out/{adapt,test_inorg,val,test}/{repo}/{path}`` (lexicon_adapt)
      ``out/drift/p{0,10,25,50}/{repo}/{path}`` (lexicon_drift)
      ``out/rename_map.json``, ``out/synth_manifest.parquet``,
      ``out/lexicon_meta.json``, ``out/synth_summary.json``

    ``only``: optional set of ``"repo/path"`` keys to restrict processing
    (diagnostic re-runs). Name allocation order then differs from a full run.
    """
    repos_dir = Path(repos_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_hash = read_manifest_digest(manifest)

    lexicon = EnterpriseLexicon.from_yaml(lexicon_yaml, seed=seed)
    assert not (
        lexicon.lexicon_adapt.all_terms() & lexicon.lexicon_drift.all_terms()
    )

    def files_of(split: str):
        for sf in ManifestView.from_parquet(manifest, split).files():
            if only is None or f"{sf.repo}/{sf.path}" in only:
                yield sf

    rows: list[dict[str, Any]] = []
    maps: dict[str, Any] = {}
    adapt_new_names: set[str] = set()
    for split in ADAPT_LEXICON_SPLITS:
        maps[split] = {}
        for sf in files_of(split):
            src = repos_dir / sf.repo / sf.path
            text = read_source_text(src)
            result = rename_file_with_lexicon(text, lexicon, which="adapt")
            rel_key = f"{sf.repo}/{sf.path}"
            maps[split][rel_key] = result.rename_map
            adapt_new_names.update(result.rename_map.values())
            _emit(
                src,
                out_dir / split / sf.repo / sf.path,
                renamed=result.renamed,
                text=result.source,
            )
            rows.append(
                {
                    "split": split,
                    "variant": "adapt_lexicon",
                    "file_id": sf.file_id,
                    "repo": sf.repo,
                    "path": sf.path,
                    "output": f"{split}/{rel_key}",
                    "renamed": result.renamed,
                    "reason": result.reason,
                    "n_identifiers": len(result.rename_map),
                }
            )

    drift_maps: dict[str, dict[str, dict[str, str]]] = {}
    drift_new_names: set[str] = set()
    drift_files = list(files_of(DRIFT_SPLIT))
    for proportion in DRIFT_PROPORTIONS:
        variant = f"p{proportion}"
        drift_maps[str(proportion)] = {}
        for sf in drift_files:
            src = repos_dir / sf.repo / sf.path
            text = read_source_text(src)
            # Distinct sub-seed per file+proportion for stability.
            file_seed = seed + 1009 * proportion + int(sf.file_id[:8], 16) % 10_000
            result = apply_drift(text, lexicon, proportion, seed=file_seed)
            rel_key = f"{sf.repo}/{sf.path}"
            drift_maps[str(proportion)][rel_key] = result.rename_map
            drift_new_names.update(result.rename_map.values())
            _emit(
                src,
                out_dir / "drift" / variant / sf.repo / sf.path,
                renamed=result.renamed,
                text=result.source,
            )
            rows.append(
                {
                    "split": DRIFT_SPLIT,
                    "variant": f"drift_{variant}",
                    "file_id": sf.file_id,
                    "repo": sf.repo,
                    "path": sf.path,
                    "output": f"drift/{variant}/{rel_key}",
                    "renamed": result.renamed,
                    "reason": result.reason,
                    "n_identifiers": len(result.rename_map),
                }
            )

    overlap = adapt_new_names & drift_new_names
    if overlap:
        raise AssertionError(
            "drift names appeared in lexicon_adapt output: "
            f"{sorted(overlap)[:10]}"
        )

    maps["drift"] = drift_maps
    (out_dir / "rename_map.json").write_text(
        json.dumps(maps, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "lexicon_meta.json").write_text(
        json.dumps(lexicon.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    columns = [
        "split", "variant", "file_id", "repo", "path", "output",
        "renamed", "reason", "n_identifiers",
    ]
    synth_df = pd.DataFrame(rows, columns=columns)
    synth_df["manifest_hash"] = source_manifest_hash
    synth_df["subset"] = only is not None
    synth_df.to_parquet(out_dir / SYNTH_MANIFEST, index=False)

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key = r["split"] if r["variant"] == "adapt_lexicon" else r["variant"]
        by_variant.setdefault(key, []).append(r)
    summary: dict[str, Any] = {
        "seed": seed,
        "manifest_hash": source_manifest_hash,
        "drift_proportions": list(DRIFT_PROPORTIONS),
        "subset": only is not None,
        "variants": {k: _variant_summary(v) for k, v in by_variant.items()},
        "adapt_files": len(maps["adapt"]),
        "test_inorg_files": len(maps["test_inorg"]),
        "val_files": len(maps["val"]),
        "test_files": len(maps["test"]),
        "drift_holdout_files": len(drift_files),
    }
    (out_dir / "synth_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
