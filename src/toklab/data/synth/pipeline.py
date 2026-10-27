"""Orchestrate adapt rename + drift variants from a manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toklab.data.splits import ManifestView
from toklab.data.synth.drift import DRIFT_PROPORTIONS, apply_drift
from toklab.data.synth.lexicon import EnterpriseLexicon
from toklab.data.synth.rename import rename_file_with_lexicon


def _read(repos_dir: Path, repo: str, rel: str) -> str:
    return (repos_dir / repo / rel).read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_synth(
    *,
    manifest: Path | str,
    repos_dir: Path | str,
    out_dir: Path | str,
    seed: int = 0,
    lexicon_yaml: Path | str | None = None,
) -> dict[str, Any]:
    """Rename adapt-split files; emit drift variants for drift_holdout.

    Writes:
      ``out/adapt/{repo}/{path}``
      ``out/rename_map.json`` (adapt + per-proportion drift maps)
      ``out/drift/p{0,10,25,50}/{repo}/{path}``
      ``out/lexicon_meta.json``
    """
    repos_dir = Path(repos_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lexicon = EnterpriseLexicon.from_yaml(lexicon_yaml, seed=seed)
    # Assert disjoint term inventories (also enforced in EnterpriseLexicon).
    assert not (
        lexicon.lexicon_adapt.all_terms() & lexicon.lexicon_drift.all_terms()
    )

    adapt_view = ManifestView.from_parquet(manifest, "adapt")
    drift_view = ManifestView.from_parquet(manifest, "drift_holdout")

    adapt_maps: dict[str, dict[str, str]] = {}
    adapt_new_names: set[str] = set()
    for sf in adapt_view.files():
        text = _read(repos_dir, sf.repo, sf.path)
        result = rename_file_with_lexicon(text, lexicon, which="adapt")
        rel_key = f"{sf.repo}/{sf.path}"
        adapt_maps[rel_key] = result.rename_map
        adapt_new_names.update(result.rename_map.values())
        _write(out_dir / "adapt" / sf.repo / sf.path, result.source)

    drift_maps: dict[str, dict[str, dict[str, str]]] = {}
    drift_new_names: set[str] = set()
    for proportion in DRIFT_PROPORTIONS:
        drift_maps[str(proportion)] = {}
        for sf in drift_view.files():
            text = _read(repos_dir, sf.repo, sf.path)
            # Distinct sub-seed per file+proportion for stability.
            file_seed = seed + 1009 * proportion + int(sf.file_id[:8], 16) % 10_000
            result = apply_drift(text, lexicon, proportion, seed=file_seed)
            rel_key = f"{sf.repo}/{sf.path}"
            drift_maps[str(proportion)][rel_key] = result.rename_map
            drift_new_names.update(result.rename_map.values())
            _write(
                out_dir / "drift" / f"p{proportion}" / sf.repo / sf.path,
                result.source,
            )

    overlap = adapt_new_names & drift_new_names
    if overlap:
        raise AssertionError(
            "drift names appeared in lexicon_adapt output: "
            f"{sorted(overlap)[:10]}"
        )

    combined = {"adapt": adapt_maps, "drift": drift_maps}
    (out_dir / "rename_map.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "lexicon_meta.json").write_text(
        json.dumps(lexicon.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "adapt_files": len(adapt_maps),
        "drift_holdout_files": len(drift_view),
        "drift_proportions": list(DRIFT_PROPORTIONS),
        "seed": seed,
    }
    (out_dir / "synth_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
