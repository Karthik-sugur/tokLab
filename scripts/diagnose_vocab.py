"""Diagnose why the adapted vocabulary is tiny / ineffective (print only).

Usage:
    python scripts/diagnose_vocab.py --corpus data/synthetic \
        --config configs/experiments/mvp_py.yaml

``--corpus`` is a directory laid out as ``{split}/{repo}/{path}``
(``data/synthetic``) or ``{repo}/{path}`` (``data/repos``, pass ``--flat``).
Uses the same mining (``mine_candidates``) and selection
(``score_candidates`` / ``select_vocab``) code as the experiment runner.
Test-split files are only *counted* here (the adapt-overlap ceiling); they
are never used to build anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from toklab.core.source_io import read_source_text
from toklab.data.splits import ManifestView
from toklab.experiments.config import load_experiment_config
from toklab.experiments.runner import _model_cfg
from toklab.parsing.segmenters import TreeSitterSegmenter
from toklab.vocab.build import shared_base_from_cfg
from toklab.vocab.mining import _structural_candidates, mine_candidates
from toklab.vocab.selection import score_candidates, select_vocab


def _root(corpus: Path, split: str, flat: bool) -> Path:
    return corpus if flat else corpus / split


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--flat", action="store_true", help="corpus is {repo}/{path}")
    ap.add_argument(
        "--config", type=Path, default=Path("configs/experiments/mvp_py.yaml")
    )
    ap.add_argument("--modes", default="structural,native")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    run_corpus = cfg.data.resolved_corpus()
    manifest = Path(run_corpus.manifest)
    min_freq, min_repos = cfg.vocab.min_freq, cfg.vocab.min_repos
    k_values = list(cfg.vocab.k_values)
    print(f"corpus={args.corpus} manifest={manifest}")
    print(f"config: min_freq={min_freq} min_repos={min_repos} k_values={k_values}")
    print(
        f"config: data.corpus kind={run_corpus.kind} dir={run_corpus.dir} "
        "(the corpus the runner reads)"
    )

    base = shared_base_from_cfg(_model_cfg(cfg))
    adapt_view = ManifestView.from_parquet(manifest, "adapt")
    adapt_root = _root(args.corpus, "adapt", args.flat)

    adapt_idents: set[str] = set()
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        t0 = time.time()
        cands = mine_candidates(
            adapt_view, repos_dir=adapt_root, mode=mode, base_tokenizer=base
        )
        total = sum(c.freq for c in cands)
        pass_filters = [
            c for c in cands if c.freq >= min_freq and c.n_repos >= min_repos
        ]
        multi_repo = sum(1 for c in cands if c.n_repos >= 2)
        scored = score_candidates(
            cands, base, min_freq=min_freq, min_repos=min_repos
        )
        print(f"\n=== [{mode}] ADAPT split mining ({time.time() - t0:.1f}s) ===")
        print(f"distinct identifiers={len(cands)} total occurrences={total}")
        print(
            f"pass freq>={min_freq} and repos>={min_repos}: {len(pass_filters)}  "
            f"(for reference: repos>=2: {multi_repo}, repos>=3: "
            f"{sum(1 for c in cands if c.n_repos >= 3)})"
        )
        print(f"positive gain (n_base_tokens>1) after filters: {len(scored)}")
        for k in k_values:
            sel = select_vocab(cands, base, k=k, min_freq=min_freq, min_repos=min_repos)
            print(f"  K={k}: vocabulary size produced={len(sel)}")
        print("top 20 by gain: token | freq | repos | base_tokens | gain")
        for s in scored[:20]:
            print(
                f"  {s.token} | {s.freq} | {s.n_repos} | {s.n_base_tokens} | "
                f"{s.gain:.0f}"
            )
        if mode == "structural":
            adapt_idents = {c.token for c in cands}

    if not args.flat:
        maps = json.loads((args.corpus / "rename_map.json").read_text("utf-8"))
        repos_of: dict[str, set[str]] = defaultdict(set)
        files_of: Counter[str] = Counter()
        olds_of: dict[str, set[str]] = defaultdict(set)
        for rel, mapping in maps.get("adapt", {}).items():
            repo = rel.split("/", 1)[0]
            for old, new in mapping.items():
                repos_of[new].add(repo)
                files_of[new] += 1
                olds_of[new].add(old)
        n_pairs = sum(len(m) for m in maps.get("adapt", {}).values())
        print("\n=== renamed identifiers (adapt rename_map) ===")
        print(f"(file, old->new) pairs={n_pairs} distinct new names={len(repos_of)}")
        if repos_of:
            reps = [len(v) for v in repos_of.values()]
            files = list(files_of.values())
            print(
                f"repos per new name: min={min(reps)} "
                f"median={statistics.median(reps)} max={max(reps)}"
            )
            print(
                f"files per new name: min={min(files)} "
                f"median={statistics.median(files)} max={max(files)}"
            )
            print(
                "new names used for >1 distinct original name: "
                f"{sum(1 for v in olds_of.values() if len(v) > 1)}"
            )

    if adapt_idents:
        test_view = ManifestView.from_parquet(manifest, "test")
        test_root = _root(args.corpus, "test", args.flat)
        ts = TreeSitterSegmenter()
        occ = hit = 0
        distinct: set[str] = set()
        t0 = time.time()
        for sf in test_view.files():
            text = read_source_text(test_root / sf.repo / sf.path)
            for ident in _structural_candidates(text, ts):
                occ += 1
                distinct.add(ident)
                hit += ident in adapt_idents
        d_hit = len(distinct & adapt_idents)
        elapsed = time.time() - t0
        print(f"\n=== TEST split overlap with adapt identifiers ({elapsed:.1f}s) ===")
        print(
            f"test identifier occurrences={occ} also in adapt={hit} "
            f"fraction={hit / occ:.4f}" if occ else "no test identifiers"
        )
        print(
            f"test distinct identifiers={len(distinct)} also in adapt={d_hit} "
            f"fraction={d_hit / len(distinct):.4f}" if distinct else ""
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
