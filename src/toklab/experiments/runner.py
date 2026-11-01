"""Execute experiment stages in order and persist artifacts.

Every stage reads source text through one :class:`~toklab.data.corpus.Corpus`
(``data.corpus``). Vocabulary is mined once per mode from the ADAPT split,
then sliced for each K of the sweep. K* is chosen on the val split for C3
only (docs/decisions/005-k-selection.md); hypotheses are judged at K*, the
rest of the sweep is descriptive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from toklab.core.hashing import config_hash as hash_dict
from toklab.core.types import SourceFile
from toklab.data.corpus import (
    Corpus,
    corpus_summary,
    open_corpus,
    synthetic_layer_hash,
)
from toklab.data.splits import read_manifest_digest
from toklab.evaluation.fidelity import roundtrip_with_counts, source_parses
from toklab.evaluation.intrinsic import (
    compression_ratio,
    context_utilisation,
    original_n_bytes,
    tokens_per_kb,
)
from toklab.evaluation.stats import (
    BootstrapCI,
    cluster_bootstrap_ci,
    factorial_analysis,
)
from toklab.evaluation.vocab_cost import (
    as_plain_dict,
    estimate_kv_cache,
    estimate_vocab_cost,
    flops_per_token_estimate,
)
from toklab.experiments.config import ExperimentConfig, load_experiment_config
from toklab.experiments.env_log import collect_env_log
from toklab.experiments.store import RunStore, compute_config_hash
from toklab.parsing.segmenters import TreeSitterSegmenter
from toklab.tokenizers.registry import get_tokenizer, load_base_model_config
from toklab.vocab.adapted_vocab import AdaptedVocab
from toklab.vocab.build import shared_base_from_cfg, vocab_from_scored
from toklab.vocab.mining import CandidateStats, _structural_candidates, mine_candidates
from toklab.vocab.selection import ScoredCandidate, score_candidates

GRID = ("C2", "C3", "C4", "C5")
K_SELECTION_CONDITION = "C3"
K_SELECTION_DOC = "docs/decisions/005-k-selection.md"
VOCAB_MODES = {"C3": "native", "C5": "structural"}


def _ci_to_row(condition: str, ci: BootstrapCI) -> dict[str, Any]:
    return {
        "condition": condition,
        "metric": "tokens_per_kb",
        "estimate": ci.estimate,
        "ci_low": ci.low,
        "ci_high": ci.high,
        "n_repos": ci.n_repos,
        "n_boot": ci.n_boot,
        "seed": ci.seed,
        "alpha": ci.alpha,
    }


def _effects_to_row(e: Any) -> dict[str, Any]:
    return {
        "k": e.k,
        "metric": e.metric,
        "c2": e.m_c2.estimate,
        "c3": e.m_c3.estimate,
        "c4": e.m_c4.estimate,
        "c5": e.m_c5.estimate,
        "c2_low": e.m_c2.low,
        "c2_high": e.m_c2.high,
        "c3_low": e.m_c3.low,
        "c3_high": e.m_c3.high,
        "c4_low": e.m_c4.low,
        "c4_high": e.m_c4.high,
        "c5_low": e.m_c5.low,
        "c5_high": e.m_c5.high,
        "effect_s": e.effect_s.estimate,
        "effect_s_low": e.effect_s.low,
        "effect_s_high": e.effect_s.high,
        "effect_d": e.effect_d.estimate,
        "effect_d_low": e.effect_d.low,
        "effect_d_high": e.effect_d.high,
        "interaction": e.interaction.estimate,
        "interaction_low": e.interaction.low,
        "interaction_high": e.interaction.high,
        "r_s": e.r_s,
        "r_d": e.r_d,
        "independence_expectation": e.independence_expectation,
        "combined_beats_either": e.combined_beats_either,
        "decision_rule": e.decision_rule,
    }


def _model_cfg(cfg: ExperimentConfig) -> dict[str, Any]:
    model_cfg: dict[str, Any] = {
        "model_name": cfg.model.model_name,
        "tokenizer_name": cfg.model.tokenizer_name or cfg.model.model_name,
        "tie_word_embeddings": cfg.model.tie_word_embeddings,
        "d_model": cfg.model.d_model,
        "num_hidden_layers": cfg.model.num_hidden_layers,
        "num_key_value_heads": cfg.model.num_key_value_heads,
        "head_dim": cfg.model.head_dim,
    }
    try:
        model_cfg = {**load_base_model_config(), **model_cfg}
    except (FileNotFoundError, ValueError, OSError):
        pass
    return model_cfg


def effective_k_values(
    k_values: Sequence[int], n_positive: Mapping[str, int]
) -> list[int]:
    """Nominal K sweep capped at the number of positive-gain candidates.

    The cap is the larger of the native / structural candidate counts; each
    mode's vocabulary at K holds ``min(K, n_positive[mode])`` entries.
    Duplicates created by the cap are removed.
    """
    cap = max(n_positive.values(), default=0)
    return sorted({min(int(k), cap) for k in k_values})


def open_experiment_corpus(config: ExperimentConfig) -> Corpus:
    c = config.data.resolved_corpus()
    return open_corpus(c.kind, c.dir, c.manifest)


def run_identity_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Canonical config plus data-layer digests (manifest + synthetic).

    Including these digests ensures identical YAML on different corpora
    yields different ``runs/<config_hash>/`` directories.
    """
    cfg_dict = config.canonical_dict()
    corpus = config.data.resolved_corpus()
    manifest = Path(corpus.manifest)
    manifest_digest = read_manifest_digest(manifest) if manifest.is_file() else None
    synth_dir = (
        corpus.dir if corpus.kind == "synthetic" else config.data.synthetic_dir
    )
    identity = dict(cfg_dict)
    identity["manifest_hash"] = manifest_digest
    identity["synthetic_hash"] = synthetic_layer_hash(synth_dir)
    return identity


class AdaptMining:
    """Adapt-only candidates per vocab mode, mined once for the whole sweep."""

    def __init__(
        self,
        corpus: Corpus,
        config: ExperimentConfig,
        base: Any,
        base_model: str,
    ) -> None:
        view = corpus.view(config.data.adapt_split)
        root = corpus.split_root(config.data.adapt_split)
        self.base_model = base_model
        self.id_offset = int(base.vocab_size)
        self.raw: dict[str, list[CandidateStats]] = {}
        self.scored: dict[str, list[ScoredCandidate]] = {}
        for mode in ("native", "structural"):
            raw = mine_candidates(
                view, repos_dir=root, mode=mode, base_tokenizer=base  # type: ignore[arg-type]
            )
            self.raw[mode] = raw
            self.scored[mode] = score_candidates(
                raw,
                base,
                min_freq=config.vocab.min_freq,
                min_repos=config.vocab.min_repos,
            )

    def n_positive(self) -> dict[str, int]:
        return {m: len(s) for m, s in self.scored.items()}

    def vocab(self, mode: str, k: int) -> AdaptedVocab:
        return vocab_from_scored(
            self.scored[mode],
            k=k,
            id_offset=self.id_offset,
            mode=mode,  # type: ignore[arg-type]
            base_model=self.base_model,
        )


def _grid_tokenizers(
    shared: dict[str, Any], vocabs: Mapping[str, AdaptedVocab]
) -> dict[str, Any]:
    return {
        "C2": get_tokenizer("C2", shared),
        "C3": get_tokenizer("C3", {**shared, "adapted_vocab": vocabs["native"]}),
        "C4": get_tokenizer("C4", shared),
        "C5": get_tokenizer("C5", {**shared, "adapted_vocab": vocabs["structural"]}),
    }


def _file_tpk(
    files: Sequence[SourceFile], counts: Mapping[str, int], n_bytes: Mapping[str, int]
) -> dict[str, list[float]]:
    by_repo: dict[str, list[float]] = {}
    for sf in files:
        by_repo.setdefault(sf.repo, []).append(
            tokens_per_kb(counts[sf.file_id], n_bytes[sf.file_id])
        )
    return by_repo


def select_k_star(
    per_k: Mapping[int, float], eligible: Mapping[int, bool]
) -> int | None:
    """argmin over eligible K of val tokens/KB; ties go to the smaller K."""
    cands = [k for k in sorted(per_k) if eligible.get(k, False)]
    if not cands:
        return None
    best = min(per_k[k] for k in cands)
    return min(k for k in cands if per_k[k] == best)


def _identifier_overlap(
    texts: Mapping[str, str], adapt_names: set[str], ts: TreeSitterSegmenter
) -> dict[str, Any]:
    occ = occ_in = 0
    distinct: set[str] = set()
    for text in texts.values():
        for name in _structural_candidates(text, ts):
            occ += 1
            distinct.add(name)
            if name in adapt_names:
                occ_in += 1
    n_distinct_in = len(distinct & adapt_names)
    return {
        "identifier_occurrences": occ,
        "occurrences_in_adapt": occ_in,
        "occurrence_fraction": (occ_in / occ) if occ else None,
        "distinct_identifiers": len(distinct),
        "distinct_in_adapt": n_distinct_in,
        "distinct_fraction": (n_distinct_in / len(distinct)) if distinct else None,
    }


def load_k_selection(run_dir: Path | str) -> dict[str, Any]:
    import json

    path = Path(run_dir) / "k_selection.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_kstar_vocabs(run_dir: Path | str) -> tuple[int, dict[str, AdaptedVocab]]:
    """K* and its native / structural vocabularies saved by the runner.

    Raises:
        FileNotFoundError: the run has no recorded K* vocabulary.
    """
    run_dir = Path(run_dir)
    sel = load_k_selection(run_dir)
    k_star = sel.get("k_star")
    files = sel.get("vocab_files") or {}
    if k_star is None or not files:
        raise FileNotFoundError(f"no K* vocabulary recorded in {run_dir}")
    vocabs = {mode: AdaptedVocab.load(run_dir / rel) for mode, rel in files.items()}
    return int(k_star), vocabs


def run_experiment(
    config: ExperimentConfig | Path | str,
) -> dict[str, Any]:
    """Execute stages; write ``runs/<config_hash>/``; return status summary."""
    if not isinstance(config, ExperimentConfig):
        config = load_experiment_config(config)

    corpus = open_experiment_corpus(config)

    cfg_dict = config.canonical_dict()
    identity = run_identity_dict(config)
    chash = compute_config_hash(identity)
    store = RunStore(config.runs_root)
    store.ensure_run_dir(chash)
    store.write_json(chash, "config.json", cfg_dict)
    store.write_json(chash, "run_identity.json", identity)
    run_dir = store.run_dir(chash)

    status: dict[str, Any] = {
        "config_hash": chash,
        "name": config.name,
        "stages": {},
        "state": "running",
        "run_dir": str(run_dir.resolve()),
    }
    store.upsert_run(
        config_hash=chash, name=config.name, status="running", seed=config.seed
    )
    store.write_status(chash, status)

    manifest_digest = identity["manifest_hash"] or read_manifest_digest(
        corpus.manifest
    )
    store.write_json(chash, "corpus.json", corpus_summary(corpus))

    model_cfg = _model_cfg(config)
    base = shared_base_from_cfg(model_cfg)
    shared = {**model_cfg, "base_tokenizer": base, "attach_leading_space": True}
    mining = AdaptMining(
        corpus, config, base, base_model=str(model_cfg.get("model_name", ""))
    )
    n_positive = mining.n_positive()
    sweep = effective_k_values(config.vocab.k_values, n_positive)
    vocabs = {
        k: {m: mining.vocab(m, k) for m in ("native", "structural")} for k in sweep
    }
    toks_by_k = {k: _grid_tokenizers(shared, vocabs[k]) for k in sweep}
    store.write_json(
        chash,
        "k_sweep.json",
        {
            "nominal_k_values": list(config.vocab.k_values),
            "n_positive_gain_candidates": n_positive,
            "k_values": sweep,
            "vocab_sizes": {
                str(k): {m: len(v) for m, v in vocabs[k].items()} for k in sweep
            },
        },
    )
    vocab_hash = hash_dict(
        {
            str(k): {m: v.content_hash() for m, v in vocabs[k].items()}
            for k in sweep
        }
    )

    env = collect_env_log(
        seed=config.seed,
        manifest_hash=manifest_digest,
        vocab_hash=vocab_hash,
        config_hash=chash,
    )
    store.write_json(chash, "env.json", env)
    store.upsert_run(
        config_hash=chash,
        name=config.name,
        status="running",
        seed=config.seed,
        manifest_hash=manifest_digest,
        vocab_hash=vocab_hash,
    )

    conditions = [c for c in config.intrinsic.conditions if c in GRID]

    # K selection: val split, C3 only.
    sel_split = config.data.k_selection_split
    val_files = tuple(
        sorted(corpus.view(sel_split).files(), key=lambda f: f.file_id)
    )
    selection: dict[str, Any] = {
        "split": sel_split,
        "condition": K_SELECTION_CONDITION,
        "metric": "tokens_per_kb (mean over files, lower is better)",
        "rule": "argmin over K; ties -> smaller K; K with any C3 round-trip "
        "failure on the selection split is not eligible",
        "doc": K_SELECTION_DOC,
        "n_files": len(val_files),
        "per_k": [],
        "k_star": None,
    }
    if val_files:
        val_texts = corpus.load_texts(val_files)
        val_bytes = {fid: original_n_bytes(t) for fid, t in val_texts.items()}
        per_k: dict[int, float] = {}
        eligible: dict[int, bool] = {}
        for k in sweep:
            rep, counts = roundtrip_with_counts(
                toks_by_k[k][K_SELECTION_CONDITION],
                val_files,
                val_texts,
                orig_parses={fid: True for fid in val_texts},
            )
            vals = [tokens_per_kb(counts[f], val_bytes[f]) for f in val_texts]
            per_k[k] = float(np.mean(vals))
            eligible[k] = rep.all_roundtrips_ok
            selection["per_k"].append(
                {
                    "k": k,
                    "tokens_per_kb": per_k[k],
                    "roundtrip_fail": rep.n_roundtrip_fail,
                    "eligible": eligible[k],
                    "vocab_size": len(vocabs[k]["native"]),
                }
            )
        selection["k_star"] = select_k_star(per_k, eligible)
        if selection["k_star"] is None:
            selection["status"] = "not run: no eligible K (round-trip failures)"
        else:
            selection["status"] = "ok"
    else:
        selection["status"] = f"not run: no files in split {sel_split!r}"
    k_star = selection["k_star"]
    # Descriptive tables (context, cost, systems, drift) need one K even when
    # K* could not be chosen; hypotheses are then reported as not run.
    k_report = int(k_star) if k_star is not None else max(sweep)
    selection["k_report"] = k_report
    vocab_dir = run_dir / "vocab"
    selection["vocab_files"] = {}
    for mode, v in vocabs[k_report].items():
        rel = f"vocab/{mode}_k{k_report}.json"
        v.save(vocab_dir / f"{mode}_k{k_report}.json")
        selection["vocab_files"][mode] = rel
    store.write_json(chash, "k_selection.json", selection)
    status["k_star"] = k_star
    store.write_status(chash, status)

    # Evaluation sets.
    eval_sets: dict[str, tuple[tuple[SourceFile, ...], dict[str, str]]] = {}
    status["eval_sets"] = {}
    for split in config.data.resolved_eval_splits():
        files = tuple(sorted(corpus.view(split).files(), key=lambda f: f.file_id))
        if not files:
            if split == config.data.eval_split:
                raise ValueError(f"no files in eval split {split!r}")
            status["eval_sets"][split] = "not run: no files in split"
            continue
        eval_sets[split] = (files, corpus.load_texts(files))
        status["eval_sets"][split] = f"{len(files)} files"
    store.write_status(chash, status)

    # One encode (+ decode for fidelity) per (eval set, condition, K, file).
    counts: dict[tuple[str, str, int], dict[str, int]] = {}
    fid_rows: list[dict[str, Any]] = []
    fidelity_ok: dict[tuple[str, str, int], bool] = {}
    n_bytes: dict[str, dict[str, int]] = {}
    for name, (files, texts) in eval_sets.items():
        n_bytes[name] = {fid: original_n_bytes(t) for fid, t in texts.items()}
        parses = {fid: source_parses(t) for fid, t in texts.items()}
        for cond in conditions:
            k_list = sweep if cond in VOCAB_MODES else [sweep[0]]
            for k in k_list:
                rep, cnt = roundtrip_with_counts(
                    toks_by_k[k][cond], files, texts, orig_parses=parses
                )
                for kk in (sweep if cond not in VOCAB_MODES else [k]):
                    counts[(name, cond, kk)] = cnt
                    fidelity_ok[(name, cond, kk)] = rep.all_roundtrips_ok
                    fid_rows.append(
                        {
                            "eval_set": name,
                            "k": kk,
                            "condition": cond,
                            "n_files": rep.n_files,
                            "n_roundtrip_ok": rep.n_roundtrip_ok,
                            "n_roundtrip_fail": rep.n_roundtrip_fail,
                            "n_orig_parsed": rep.n_orig_parsed,
                            "n_reparse_ok": rep.n_reparse_ok,
                            "n_reparse_fail": rep.n_reparse_fail,
                            "valid": rep.all_roundtrips_ok,
                            "status": "OK" if rep.all_roundtrips_ok else "INVALID",
                        }
                    )

    if config.stages.fidelity:
        fid = pd.DataFrame(fid_rows)
        if not fid.empty:
            fid = fid.sort_values(["eval_set", "k", "condition"]).reset_index(
                drop=True
            )
        store.write_parquet(chash, "fidelity.parquet", fid)
        status["stages"]["fidelity"] = "ok"
        store.write_status(chash, status)

    if config.stages.intrinsic:
        rows: list[dict[str, Any]] = []
        ci_rows: list[dict[str, Any]] = []
        for name, (files, _texts) in eval_sets.items():
            for k in sweep:
                for cond in conditions:
                    cnt = counts[(name, cond, k)]
                    for sf in files:
                        nt = cnt[sf.file_id]
                        nb = n_bytes[name][sf.file_id]
                        cr = compression_ratio(nb, nt) if nt else float("nan")
                        for metric, value in (
                            ("n_tokens", nt),
                            ("n_bytes", nb),
                            ("compression_ratio", cr),
                            ("tokens_per_kb", tokens_per_kb(nt, nb)),
                        ):
                            rows.append(
                                {
                                    "eval_set": name,
                                    "k": k,
                                    "condition": cond,
                                    "file_id": sf.file_id,
                                    "metric": metric,
                                    "value": float(value),
                                }
                            )
                    ci = cluster_bootstrap_ci(
                        _file_tpk(files, cnt, n_bytes[name]),
                        seed=config.seed,
                        n_boot=config.intrinsic.n_boot,
                    )
                    ci_rows.append(
                        {
                            "eval_set": name,
                            "k": k,
                            **_ci_to_row(cond, ci),
                            "valid": fidelity_ok[(name, cond, k)],
                        }
                    )
        metrics = pd.DataFrame(rows)
        metrics = metrics.sort_values(
            ["eval_set", "k", "condition", "file_id", "metric"]
        ).reset_index(drop=True)
        store.write_parquet(chash, "intrinsic_metrics.parquet", metrics)
        store.write_parquet(chash, "token_efficiency_ci.parquet", pd.DataFrame(ci_rows))
        status["stages"]["intrinsic"] = "ok"
        store.write_status(chash, status)

    if config.stages.factorial and set(conditions) >= set(GRID):
        hypo_path = Path(config.hypotheses_path)
        decision_rule = ""
        if hypo_path.is_file():
            hypo = yaml.safe_load(hypo_path.read_text(encoding="utf-8"))
            decision_rule = str(hypo.get("decision_rule", "")).strip()
        effect_rows: list[dict[str, Any]] = []
        for name, (files, _texts) in eval_sets.items():
            for k in sweep:
                cond_vals = {
                    c: _file_tpk(files, counts[(name, c, k)], n_bytes[name])
                    for c in GRID
                }
                effects = factorial_analysis(
                    cond_vals,
                    k=k,
                    seed=config.seed,
                    n_boot=config.intrinsic.n_boot,
                    decision_rule=decision_rule,
                )
                effect_rows.append(
                    {
                        "eval_set": name,
                        **_effects_to_row(effects),
                        "k_star": k == k_star,
                    }
                )
        store.write_parquet(
            chash, "factorial_effects.parquet", pd.DataFrame(effect_rows)
        )
        status["stages"]["factorial"] = "ok"
        store.write_status(chash, status)

    if config.stages.context:
        ctx_frames: list[pd.DataFrame] = []
        for name, (files, _texts) in eval_sets.items():
            for cond in conditions:
                cnt = counts[(name, cond, k_report)]
                packed = [
                    (sf.file_id, cnt[sf.file_id], n_bytes[name][sf.file_id])
                    for sf in files
                ]
                df = context_utilisation(packed, config.intrinsic.context_budgets)
                df.insert(0, "condition", cond)
                df.insert(0, "k", k_report)
                df.insert(0, "eval_set", name)
                ctx_frames.append(df)
        if ctx_frames:
            ctx = pd.concat(ctx_frames, ignore_index=True)
            ctx = ctx.sort_values(
                ["eval_set", "condition", "budget_tokens"]
            ).reset_index(drop=True)
            store.write_parquet(chash, "context_utilisation.parquet", ctx)
        status["stages"]["context"] = "ok"
        store.write_status(chash, status)

    if config.stages.vocab_cost:
        _write_vocab_cost(
            store, chash, config, sweep, vocabs, k_report, counts, eval_sets, n_bytes
        )
        status["stages"]["vocab_cost"] = "ok"
        store.write_status(chash, status)

    # Measured name overlap for the limitations section (counts only; nothing
    # is built from eval files).
    ts = TreeSitterSegmenter()
    adapt_names = {c.token for c in mining.raw["structural"]}
    store.write_json(
        chash,
        "name_overlap.json",
        {
            name: _identifier_overlap(texts, adapt_names, ts)
            for name, (_files, texts) in eval_sets.items()
        },
    )

    if config.systems.enabled:
        status["stages"]["systems"] = (
            f"pending: separate step `toklab run systems --run {chash}`"
        )
    else:
        status["stages"]["systems"] = "skipped (systems.enabled=false)"

    if config.stages.report:
        from toklab.evaluation.report import render_report

        report_path = render_report(
            run_dir=run_dir,
            hypotheses_path=Path(config.hypotheses_path),
        )
        status["stages"]["report"] = "ok"
        status["report_path"] = str(report_path.resolve())
        store.write_status(chash, status)

    status["state"] = "complete"
    store.write_status(chash, status)
    store.upsert_run(
        config_hash=chash,
        name=config.name,
        status="complete",
        seed=config.seed,
        manifest_hash=manifest_digest,
        vocab_hash=vocab_hash,
    )
    return status


def _write_vocab_cost(
    store: RunStore,
    chash: str,
    config: ExperimentConfig,
    sweep: Sequence[int],
    vocabs: Mapping[int, Mapping[str, AdaptedVocab]],
    k_report: int,
    counts: Mapping[tuple[str, str, int], Mapping[str, int]],
    eval_sets: Mapping[str, tuple[tuple[SourceFile, ...], dict[str, str]]],
    n_bytes: Mapping[str, Mapping[str, int]],
) -> None:
    """Per-K overhead (ESTIMATE) and tokens saved per KB on the primary set."""
    primary = config.data.eval_split

    def mean_tpk(cond: str, k: int) -> float | None:
        if primary not in eval_sets or (primary, cond, k) not in counts:
            return None
        files = eval_sets[primary][0]
        cnt = counts[(primary, cond, k)]
        return float(
            np.mean([tokens_per_kb(cnt[f.file_id], n_bytes[primary][f.file_id])
                     for f in files])
        )

    by_k: list[dict[str, Any]] = []
    for k in sweep:
        for cond, baseline in (("C3", "C2"), ("C5", "C4")):
            delta_v = len(vocabs[k][VOCAB_MODES[cond]])
            vc = estimate_vocab_cost(
                delta_v=delta_v,
                d_model=config.model.d_model,
                tie_word_embeddings=config.model.tie_word_embeddings,
            )
            m_base, m_cond = mean_tpk(baseline, k), mean_tpk(cond, k)
            by_k.append(
                {
                    "k": k,
                    "condition": cond,
                    "baseline": baseline,
                    "delta_v": delta_v,
                    "delta_params": vc.delta_params,
                    "delta_bytes": vc.delta_bytes,
                    "eval_set": primary,
                    "tokens_saved_per_kb": (
                        None if m_base is None or m_cond is None else m_base - m_cond
                    ),
                }
            )
    store.write_parquet(chash, "vocab_cost_by_k.parquet", pd.DataFrame(by_k))

    delta_v = len(vocabs[k_report]["native"])
    vc = estimate_vocab_cost(
        delta_v=delta_v,
        d_model=config.model.d_model,
        tie_word_embeddings=config.model.tie_word_embeddings,
    )
    seq_lens = sorted(
        set(list(config.intrinsic.context_budgets[:2]) + list(config.systems.seq_lens))
    ) or [128]
    cost_rows: list[dict[str, Any]] = []
    for t in seq_lens:
        kv = estimate_kv_cache(
            layers=config.model.num_hidden_layers,
            kv_heads=config.model.num_key_value_heads,
            head_dim=config.model.head_dim,
            seq_len=t,
        )
        fl = flops_per_token_estimate(
            d_model=config.model.d_model,
            layers=config.model.num_hidden_layers,
            seq_len=t,
        )
        row: dict[str, Any] = {"seq_len": t, "k": k_report, "delta_v": delta_v}
        for key, val in as_plain_dict(vc).items():
            row[f"vocab_{key}"] = val
        for key, val in as_plain_dict(kv).items():
            row[f"kv_{key}"] = val
        for key, val in as_plain_dict(fl).items():
            row[f"flops_{key}"] = val
        cost_rows.append(row)
    store.write_parquet(chash, "vocab_cost.parquet", pd.DataFrame(cost_rows))
