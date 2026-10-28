"""Vocabulary drift experiments (STATIC vs REFRESH vs baselines).

For each drift proportion ``p ∈ {0, 10, 25, 50}``:

1. Build drifted ``drift_holdout`` texts via ``data.synth.drift.apply_drift``.
2. Assert every new drift identifier is absent from the adapt corpus.
3. Tokenize with baselines (C2, C4), STATIC (frozen adapt vocab), and
   REFRESH (append-only new entries from a small v1 sample; old ids fixed).
4. Report token reduction vs baseline, adapted-token coverage, and BPB
   only for gate-valid fine-tuned variants (else ``not run``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from toklab.core.types import SourceFile
from toklab.data.splits import ManifestView
from toklab.data.synth.drift import DRIFT_PROPORTIONS, apply_drift
from toklab.data.synth.lexicon import EnterpriseLexicon
from toklab.evaluation.intrinsic import original_n_bytes
from toklab.parsing.segmenters import TreeSitterSegmenter
from toklab.tokenizers.registry import get_tokenizer, load_base_model_config
from toklab.vocab.adapted_vocab import AdaptedVocab
from toklab.vocab.build import build_adapted_vocab, shared_base_from_cfg
from toklab.vocab.mining import mine_from_files
from toklab.vocab.selection import select_vocab


class DriftLeakError(AssertionError):
    """Raised when a drift name appears in the adapt corpus."""


def collect_identifiers(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, name)`` for structural identifier leaves."""
    seg = TreeSitterSegmenter()
    out: list[tuple[int, int, str]] = []
    for s in seg.segment(text):
        if s.kind != "ident":
            continue
        name = text[s.start : s.end]
        if name.isidentifier():
            out.append((s.start, s.end, name))
    return out


def adapt_corpus_identifiers(
    files: Sequence[SourceFile],
    texts: Mapping[str, str],
) -> set[str]:
    """All identifier strings appearing in the adapt corpus."""
    names: set[str] = set()
    for sf in files:
        for _a, _b, name in collect_identifiers(texts[sf.file_id]):
            names.add(name)
    return names


def assert_drift_names_unseen(
    new_names: set[str],
    adapt_idents: set[str],
) -> None:
    """Assert every new drift identifier is absent from the adapt corpus."""
    leak = sorted(new_names & adapt_idents)
    if leak:
        raise DriftLeakError(
            "drift identifier(s) present in adapt corpus: "
            f"{leak[:20]}"
        )


def adapted_coverage(
    tokenizer: Any,
    text: str,
    adapted_ids: set[int],
) -> tuple[float, int, int]:
    """Share of identifier occurrences that encode as one adapted token.

    Returns ``(coverage, hits, n_idents)``. Coverage is ``0.0`` when there are
    identifiers but no adapted hits (including empty vocab). ``nan`` only when
    the text has zero identifiers.
    """
    idents = collect_identifiers(text)
    if not idents:
        return float("nan"), 0, 0
    if not adapted_ids:
        return 0.0, 0, len(idents)
    enc = tokenizer.encode(text)
    hits = 0
    for start, end, _name in idents:
        matched = [
            i
            for i, (s, e) in enumerate(enc.spans)
            if s == start and e == end
        ]
        if len(matched) == 1 and enc.ids[matched[0]] in adapted_ids:
            hits += 1
    return hits / len(idents), hits, len(idents)


def token_reduction(n_baseline: int, n_variant: int) -> float:
    """``1 - n_variant / n_baseline`` (higher = more compression vs baseline)."""
    if n_baseline <= 0:
        return float("nan")
    return 1.0 - (n_variant / n_baseline)


def refresh_vocab_append_only(
    static: AdaptedVocab,
    *,
    sample_files: Sequence[SourceFile],
    sample_texts: Mapping[str, str],
    base: Any,
    mode: str,
    k_refresh: int,
    min_freq: int = 1,
    min_repos: int = 1,
) -> AdaptedVocab:
    """Mine a v1 sample and append new entries; preserve all old token→id."""
    old_map = static.token_to_id_snapshot()
    # Tag sample files as adapt so mine_from_files accepts them for refresh
    # mining (explicit refresh path — not the primary adapt-only mine API).
    tagged = [
        SourceFile(
            file_id=sf.file_id,
            repo=sf.repo,
            path=sf.path,
            sha256=sf.sha256,
            language=sf.language,
            n_bytes=sf.n_bytes,
            split="adapt",
        )
        for sf in sample_files
    ]
    if k_refresh <= 0 or not tagged:
        refreshed = static.extend_append_only([])
    else:
        cands = mine_from_files(
            list(tagged),
            dict(sample_texts),
            mode=mode,  # type: ignore[arg-type]
            base_tokenizer=base,
            required_split="adapt",
        )
        # Prefer tokens not already in static vocab.
        novel = [c for c in cands if c.token not in old_map]
        selected = select_vocab(
            novel if novel else cands,
            base,
            k=k_refresh,
            min_freq=min_freq,
            min_repos=min_repos,
        )
        # Drop any that somehow still collide.
        selected = [s for s in selected if s.token not in old_map]
        refreshed = static.extend_append_only(selected)

    # Hard assert: every old binding unchanged.
    new_map = refreshed.token_to_id_snapshot()
    for tok, tid in old_map.items():
        if tok not in new_map or new_map[tok] != tid:
            raise AssertionError(
                f"refresh changed old id for {tok!r}: "
                f"was {tid}, now {new_map.get(tok)}"
            )
    return refreshed


def _count_tokens(tok: Any, texts: Sequence[str]) -> int:
    return sum(len(tok.encode(t).ids) for t in texts)


def _mean_coverage(
    tok: Any, texts: Sequence[str], adapted_ids: set[int]
) -> float:
    hits = 0
    n = 0
    for t in texts:
        _c, h, ni = adapted_coverage(tok, t, adapted_ids)
        hits += h
        n += ni
    if n == 0:
        return float("nan")
    return hits / n


@dataclass(frozen=True)
class DriftRow:
    proportion: int
    variant: str
    baseline: str
    n_tokens: int
    n_tokens_baseline: int
    token_reduction: float
    coverage: float
    n_bytes: int
    bpb: float | None
    bpb_status: str
    n_vocab: int
    notes: str = ""


def format_drift_table(rows: Sequence[DriftRow]) -> str:
    lines = [
        "drift  metric=tokens  reduction=1-n_var/n_base  "
        "coverage=adapted_ident_hit_rate",
        f"{'p':>4} {'variant':<14} {'base':<4} {'n_tok':>8} {'n_base':>8} "
        f"{'reduc':>8} {'cover':>8} {'bpb':>10} {'|V|':>5}",
        "-" * 78,
    ]
    for r in rows:
        bpb_s = (
            "not run"
            if r.bpb is None
            else f"{r.bpb:.4f}"
        )
        cover = "nan" if r.coverage != r.coverage else f"{r.coverage:.3f}"
        reduc = (
            "nan"
            if r.token_reduction != r.token_reduction
            else f"{r.token_reduction:.3f}"
        )
        lines.append(
            f"{r.proportion:4d} {r.variant:<14} {r.baseline:<4} "
            f"{r.n_tokens:8d} {r.n_tokens_baseline:8d} "
            f"{reduc:>8} {cover:>8} {bpb_s:>10} {r.n_vocab:5d}"
        )
    return "\n".join(lines)


def build_drift_texts(
    *,
    drift_files: Sequence[SourceFile],
    repos_dir: Path,
    lexicon: EnterpriseLexicon,
    proportion: int,
    seed: int,
) -> tuple[dict[str, str], set[str], dict[str, dict[str, str]]]:
    """Apply drift to holdout files; return texts, new names, rename maps."""
    texts: dict[str, str] = {}
    new_names: set[str] = set()
    maps: dict[str, dict[str, str]] = {}
    for sf in drift_files:
        raw = (repos_dir / sf.repo / sf.path).read_text(encoding="utf-8")
        file_seed = (
            seed + 1009 * proportion + int(sf.file_id[:8], 16) % 10_000
        )
        result = apply_drift(raw, lexicon, proportion, seed=file_seed)
        texts[sf.file_id] = result.source
        new_names.update(result.rename_map.values())
        maps[f"{sf.repo}/{sf.path}"] = dict(result.rename_map)
    return texts, new_names, maps


def run_drift(
    *,
    manifest: Path | str,
    repos_dir: Path | str,
    out_dir: Path | str | None = None,
    seed: int = 0,
    k_static: int = 8,
    k_refresh: int = 4,
    v1_sample_size: int = 1,
    proportions: Sequence[int] = DRIFT_PROPORTIONS,
    cfg: dict[str, Any] | None = None,
    gate_valid_finetuned: Mapping[str, bool] | None = None,
    bpb_fn: Any | None = None,
) -> tuple[str, pd.DataFrame, list[DriftRow]]:
    """Run the full drift sweep; optionally write parquet/table under ``out_dir``.

    ``gate_valid_finetuned`` maps variant name → whether BPB is allowed.
    When absent or false, ``bpb`` is left ``None`` with status ``not run``.
    """
    repos_dir = Path(repos_dir)
    manifest = Path(manifest)
    cfg = dict(cfg or {})
    try:
        cfg = {**load_base_model_config(cfg.get("base_model_yaml")), **cfg}
    except (FileNotFoundError, ValueError, OSError):
        pass
    if "model_name" not in cfg:
        cfg["model_name"] = "Qwen/Qwen2.5-Coder-0.5B"

    adapt_view = ManifestView.from_parquet(manifest, "adapt")
    drift_view = ManifestView.from_parquet(manifest, "drift_holdout")
    adapt_files = adapt_view.files()
    drift_files = tuple(sorted(drift_view.files(), key=lambda f: f.file_id))
    if not drift_files:
        raise ValueError("no files in drift_holdout split")

    adapt_texts = {
        sf.file_id: (repos_dir / sf.repo / sf.path).read_text(encoding="utf-8")
        for sf in adapt_files
    }
    adapt_idents = adapt_corpus_identifiers(adapt_files, adapt_texts)

    if cfg.get("base_tokenizer") is not None:
        base = cfg["base_tokenizer"]
    else:
        base = shared_base_from_cfg(cfg)
    shared = {
        **{k: v for k, v in cfg.items() if k != "base_tokenizer"},
        "base_tokenizer": base,
        "attach_leading_space": True,
    }
    vocab_n = build_adapted_vocab(
        adapt_view,
        repos_dir=repos_dir,
        base=base,
        mode="native",
        k=k_static,
        min_freq=1,
        min_repos=1,
        base_model=str(cfg.get("model_name", "")),
    )
    vocab_s = build_adapted_vocab(
        adapt_view,
        repos_dir=repos_dir,
        base=base,
        mode="structural",
        k=k_static,
        min_freq=1,
        min_repos=1,
        base_model=str(cfg.get("model_name", "")),
    )

    lexicon = EnterpriseLexicon.from_yaml(seed=seed)
    gate_map = dict(gate_valid_finetuned or {})
    rows: list[DriftRow] = []

    for p in proportions:
        if p not in DRIFT_PROPORTIONS:
            raise ValueError(f"unsupported proportion {p}")
        drift_texts, new_names, _maps = build_drift_texts(
            drift_files=drift_files,
            repos_dir=repos_dir,
            lexicon=lexicon,
            proportion=p,
            seed=seed,
        )
        assert_drift_names_unseen(new_names, adapt_idents)

        ordered_ids = [sf.file_id for sf in drift_files]
        texts_list = [drift_texts[i] for i in ordered_ids]
        n_bytes = sum(original_n_bytes(t) for t in texts_list)

        # v1 sample = first N holdout files (deterministic order).
        sample_files = drift_files[: max(0, v1_sample_size)]
        sample_texts = {sf.file_id: drift_texts[sf.file_id] for sf in sample_files}

        refresh_n = refresh_vocab_append_only(
            vocab_n,
            sample_files=sample_files,
            sample_texts=sample_texts,
            base=base,
            mode="native",
            k_refresh=k_refresh,
        )
        refresh_s = refresh_vocab_append_only(
            vocab_s,
            sample_files=sample_files,
            sample_texts=sample_texts,
            base=base,
            mode="structural",
            k_refresh=k_refresh,
        )

        tok_c2 = get_tokenizer("C2", shared)
        tok_c4 = get_tokenizer("C4", shared)
        tok_static_c3 = get_tokenizer(
            "C3", {**shared, "adapted_vocab": vocab_n}
        )
        tok_static_c5 = get_tokenizer(
            "C5", {**shared, "adapted_vocab": vocab_s}
        )
        tok_refresh_c3 = get_tokenizer(
            "C3", {**shared, "adapted_vocab": refresh_n}
        )
        tok_refresh_c5 = get_tokenizer(
            "C5", {**shared, "adapted_vocab": refresh_s}
        )

        n_c2 = _count_tokens(tok_c2, texts_list)
        n_c4 = _count_tokens(tok_c4, texts_list)

        variants: list[tuple[str, str, Any, AdaptedVocab | None, int]] = [
            ("C2", "C2", tok_c2, None, n_c2),
            ("C4", "C4", tok_c4, None, n_c4),
            ("STATIC_C3", "C2", tok_static_c3, vocab_n, n_c2),
            ("STATIC_C5", "C4", tok_static_c5, vocab_s, n_c4),
            ("REFRESH_C3", "C2", tok_refresh_c3, refresh_n, n_c2),
            ("REFRESH_C5", "C4", tok_refresh_c5, refresh_s, n_c4),
        ]

        for name, base_name, tok, vocab, n_base in variants:
            n_tok = _count_tokens(tok, texts_list)
            ids = (
                set(vocab.as_dict().values())
                if vocab is not None
                else set()
            )
            cover = _mean_coverage(tok, texts_list, ids)
            bpb: float | None = None
            bpb_status = "not run"
            if gate_map.get(name) and bpb_fn is not None:
                bpb = float(bpb_fn(tok, texts_list))
                bpb_status = "measured"
            elif gate_map.get(name):
                bpb_status = "gate_ok_but_bpb_fn_missing"
            rows.append(
                DriftRow(
                    proportion=p,
                    variant=name,
                    baseline=base_name,
                    n_tokens=n_tok,
                    n_tokens_baseline=n_base,
                    token_reduction=token_reduction(n_base, n_tok),
                    coverage=0.0 if (vocab is not None and len(vocab) == 0) else cover,
                    n_bytes=n_bytes,
                    bpb=bpb,
                    bpb_status=bpb_status,
                    n_vocab=0 if vocab is None else len(vocab),
                    notes=(
                        "empty_vocab_coverage_forced_0"
                        if vocab is not None and len(vocab) == 0
                        else ""
                    ),
                )
            )

    table = format_drift_table(rows)
    df = pd.DataFrame([asdict(r) for r in rows])
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "drift_metrics.parquet", index=False)
        (out / "drift_table.txt").write_text(table + "\n", encoding="utf-8")
        (out / "drift_meta.json").write_text(
            json.dumps(
                {
                    "proportions": list(proportions),
                    "k_static": k_static,
                    "k_refresh": k_refresh,
                    "v1_sample_size": v1_sample_size,
                    "seed": seed,
                    "n_drift_files": len(drift_files),
                    "n_adapt_idents": len(adapt_idents),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return table, df, rows


def run_drift_for_experiment_hash(
    config_hash: str,
    *,
    runs_root: Path | str = "runs",
    k_static: int | None = None,
    k_refresh: int = 4,
    v1_sample_size: int = 1,
) -> tuple[str, Path]:
    """Load ``runs/<hash>/config.json``, run drift, re-render report."""
    from toklab.evaluation.report import render_report

    runs_root = Path(runs_root)
    run_dir = runs_root / config_hash
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"no config.json in {run_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    data = cfg.get("data", {})
    vocab_cfg = cfg.get("vocab", {})
    model = cfg.get("model", {})
    k = k_static
    if k is None:
        ks = vocab_cfg.get("k_values") or [8]
        k = max(int(x) for x in ks)

    model_cfg = {
        "model_name": model.get("model_name", "Qwen/Qwen2.5-Coder-0.5B"),
        "tokenizer_name": model.get("tokenizer_name")
        or model.get("model_name"),
    }
    table, _df, _rows = run_drift(
        manifest=data["manifest_path"],
        repos_dir=data["repos_dir"],
        out_dir=run_dir,
        seed=int(cfg.get("seed", 0)),
        k_static=int(k),
        k_refresh=k_refresh,
        v1_sample_size=v1_sample_size,
        cfg=model_cfg,
        gate_valid_finetuned={},  # no fine-tune in default path → BPB not run
    )
    hypo = cfg.get("hypotheses_path", "configs/hypotheses.yaml")
    render_report(run_dir=run_dir, hypotheses_path=Path(hypo))
    return table, run_dir
