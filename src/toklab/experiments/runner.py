"""Execute experiment stages in order and persist artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from toklab.core.hashing import config_hash as hash_dict
from toklab.core.hashing import sha256_text
from toklab.data.splits import ManifestView
from toklab.evaluation.fidelity import check_roundtrip
from toklab.evaluation.intrinsic import (
    context_utilisation_for_tokenizer,
    measure_files,
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
from toklab.tokenizers.registry import get_tokenizer, load_base_model_config
from toklab.vocab.build import build_adapted_vocab, shared_base_from_cfg


def _load_texts(repos_dir: Path, files: tuple) -> dict[str, str]:
    return {
        sf.file_id: (repos_dir / sf.repo / sf.path).read_text(encoding="utf-8")
        for sf in files
    }


def _tpk_by_repo(
    files: tuple, metrics_df: pd.DataFrame, condition: str
) -> dict[str, list[float]]:
    id_to_repo = {f.file_id: f.repo for f in files}
    sub = metrics_df[
        (metrics_df["condition"] == condition)
        & (metrics_df["metric"] == "tokens_per_kb")
    ]
    out: dict[str, list[float]] = {}
    for row in sub.itertuples(index=False):
        repo = id_to_repo[row.file_id]
        out.setdefault(repo, []).append(float(row.value))
    return out


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
        "effect_d": e.effect_d.estimate,
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


def _build_tokenizers(
    cfg: ExperimentConfig,
    *,
    repos_dir: Path,
    manifest: Path,
    k: int,
) -> tuple[dict[str, Any], str, Any, Any]:
    """Build C1–C6 for one K; return toks, vocab_hash, native/structural vocabs."""
    model_cfg = _model_cfg(cfg)
    base = shared_base_from_cfg(model_cfg)
    adapt_view = ManifestView.from_parquet(manifest, cfg.data.adapt_split)
    vocab_n = build_adapted_vocab(
        adapt_view,
        repos_dir=repos_dir,
        base=base,
        mode="native",
        k=k,
        min_freq=cfg.vocab.min_freq,
        min_repos=cfg.vocab.min_repos,
        base_model=str(model_cfg.get("model_name", "")),
    )
    vocab_s = build_adapted_vocab(
        adapt_view,
        repos_dir=repos_dir,
        base=base,
        mode="structural",
        k=k,
        min_freq=cfg.vocab.min_freq,
        min_repos=cfg.vocab.min_repos,
        base_model=str(model_cfg.get("model_name", "")),
    )
    shared = {
        **model_cfg,
        "base_tokenizer": base,
        "attach_leading_space": True,
    }
    toks: dict[str, Any] = {
        "C1": get_tokenizer("C1", shared),
        "C2": get_tokenizer("C2", shared),
        "C3": get_tokenizer("C3", {**shared, "adapted_vocab": vocab_n}),
        "C4": get_tokenizer("C4", shared),
        "C5": get_tokenizer("C5", {**shared, "adapted_vocab": vocab_s}),
        "C6": get_tokenizer("C6", shared),
    }
    vocab_hash = hash_dict(
        {
            "native": vocab_n.content_hash(),
            "structural": vocab_s.content_hash(),
            "k": k,
        }
    )
    return toks, vocab_hash, vocab_n, vocab_s


def run_experiment(
    config: ExperimentConfig | Path | str,
) -> dict[str, Any]:
    """Execute stages; write ``runs/<config_hash>/``; return status summary."""
    if not isinstance(config, ExperimentConfig):
        config = load_experiment_config(config)

    cfg_dict = config.canonical_dict()
    chash = compute_config_hash(cfg_dict)
    store = RunStore(config.runs_root)
    store.ensure_run_dir(chash)
    store.write_json(chash, "config.json", cfg_dict)

    status: dict[str, Any] = {
        "config_hash": chash,
        "name": config.name,
        "stages": {},
        "state": "running",
        "run_dir": str(store.run_dir(chash).resolve()),
    }
    store.upsert_run(
        config_hash=chash, name=config.name, status="running", seed=config.seed
    )
    store.write_status(chash, status)

    repos_dir = Path(config.data.repos_dir)
    manifest = Path(config.data.manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    hash_file = manifest.parent / "manifest_hash"
    if hash_file.is_file():
        manifest_digest = hash_file.read_text(encoding="utf-8").strip()
    else:
        manifest_digest = sha256_text(manifest.read_bytes().hex())

    k_primary = max(config.vocab.k_values) if config.vocab.k_values else 0
    toks, vocab_hash, vocab_n, _vocab_s = _build_tokenizers(
        config, repos_dir=repos_dir, manifest=manifest, k=k_primary
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

    eval_view = ManifestView.from_parquet(manifest, config.data.eval_split)
    eval_files = tuple(sorted(eval_view.files(), key=lambda f: f.file_id))
    if not eval_files:
        raise ValueError(f"no files in eval split {config.data.eval_split!r}")
    texts = _load_texts(repos_dir, eval_files)

    conditions = list(config.intrinsic.conditions)
    fidelity_ok: dict[str, bool] = {}

    if config.stages.fidelity:
        fid_rows: list[dict[str, Any]] = []
        for cond in conditions:
            report = check_roundtrip(toks[cond], eval_files, texts=texts)
            ok = report.all_roundtrips_ok
            fidelity_ok[cond] = ok
            fid_rows.append(
                {
                    "condition": cond,
                    "n_files": report.n_files,
                    "n_roundtrip_ok": report.n_roundtrip_ok,
                    "n_roundtrip_fail": report.n_roundtrip_fail,
                    "n_orig_parsed": report.n_orig_parsed,
                    "n_reparse_ok": report.n_reparse_ok,
                    "n_reparse_fail": report.n_reparse_fail,
                    "valid": ok,
                    "status": "OK" if ok else "INVALID",
                }
            )
        store.write_parquet(chash, "fidelity.parquet", pd.DataFrame(fid_rows))
        status["stages"]["fidelity"] = "ok"
        store.write_status(chash, status)
    else:
        for cond in conditions:
            fidelity_ok[cond] = True

    valid_conditions = [c for c in conditions if fidelity_ok.get(c, True)]

    if config.stages.intrinsic:
        frames: list[pd.DataFrame] = []
        ci_rows: list[dict[str, Any]] = []
        for cond in conditions:
            df = measure_files(
                toks[cond], eval_files, texts=texts, condition=cond
            )
            df = df.sort_values(
                ["condition", "file_id", "metric"]
            ).reset_index(drop=True)
            frames.append(df)
            if cond in valid_conditions:
                by_repo = _tpk_by_repo(eval_files, df, cond)
                ci = cluster_bootstrap_ci(
                    by_repo, seed=config.seed, n_boot=config.intrinsic.n_boot
                )
                ci_rows.append(_ci_to_row(cond, ci))
        metrics = pd.concat(frames, ignore_index=True)
        metrics = metrics.sort_values(
            ["condition", "file_id", "metric"]
        ).reset_index(drop=True)
        store.write_parquet(chash, "intrinsic_metrics.parquet", metrics)
        store.write_parquet(
            chash, "token_efficiency_ci.parquet", pd.DataFrame(ci_rows)
        )
        status["stages"]["intrinsic"] = "ok"
        store.write_status(chash, status)

    if config.stages.factorial and set(conditions) >= {"C2", "C3", "C4", "C5"}:
        hypo_path = Path(config.hypotheses_path)
        decision_rule = ""
        if hypo_path.is_file():
            hypo = yaml.safe_load(hypo_path.read_text(encoding="utf-8"))
            decision_rule = str(hypo.get("decision_rule", "")).strip()

        effect_rows: list[dict[str, Any]] = []
        for k in config.vocab.k_values:
            toks_k, _, _, _ = _build_tokenizers(
                config, repos_dir=repos_dir, manifest=manifest, k=k
            )
            cond_vals: dict[str, dict[str, list[float]]] = {}
            for c in ("C2", "C3", "C4", "C5"):
                tok = toks_k[c]
                by_repo: dict[str, list[float]] = {}
                for sf in eval_files:
                    text = texts[sf.file_id]
                    n_ids = len(tok.encode(text).ids)
                    n_bytes = len(text.encode("utf-8"))
                    by_repo.setdefault(sf.repo, []).append(
                        tokens_per_kb(n_ids, n_bytes)
                    )
                cond_vals[c] = by_repo
            effects = factorial_analysis(
                cond_vals,
                k=k,
                seed=config.seed,
                n_boot=config.intrinsic.n_boot,
                decision_rule=decision_rule,
            )
            effect_rows.append(_effects_to_row(effects))
        store.write_parquet(
            chash, "factorial_effects.parquet", pd.DataFrame(effect_rows)
        )
        status["stages"]["factorial"] = "ok"
        store.write_status(chash, status)

    if config.stages.context:
        ctx_frames: list[pd.DataFrame] = []
        for cond in conditions:
            df = context_utilisation_for_tokenizer(
                toks[cond],
                eval_files,
                texts,
                config.intrinsic.context_budgets,
                condition=cond,
            )
            ctx_frames.append(df)
        ctx = pd.concat(ctx_frames, ignore_index=True)
        ctx = ctx.sort_values(
            ["condition", "budget_tokens"]
        ).reset_index(drop=True)
        store.write_parquet(chash, "context_utilisation.parquet", ctx)
        status["stages"]["context"] = "ok"
        store.write_status(chash, status)

    if config.stages.vocab_cost:
        delta_v = len(vocab_n)
        vc = estimate_vocab_cost(
            delta_v=delta_v,
            d_model=config.model.d_model,
            tie_word_embeddings=config.model.tie_word_embeddings,
        )
        seq_lens = sorted(
            set(
                list(config.intrinsic.context_budgets[:2])
                + list(config.systems.seq_lens)
            )
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
            row: dict[str, Any] = {"seq_len": t, "delta_v": delta_v}
            for key, val in as_plain_dict(vc).items():
                row[f"vocab_{key}"] = val
            for key, val in as_plain_dict(kv).items():
                row[f"kv_{key}"] = val
            for key, val in as_plain_dict(fl).items():
                row[f"flops_{key}"] = val
            cost_rows.append(row)
        store.write_parquet(
            chash, "vocab_cost.parquet", pd.DataFrame(cost_rows)
        )
        status["stages"]["vocab_cost"] = "ok"
        store.write_status(chash, status)

    if config.stages.systems and config.systems.enabled:
        from toklab.evaluation.systems import run_systems_benchmark

        sys_df = run_systems_benchmark(
            model_name=config.model.model_name,
            conditions=list(config.systems.conditions),
            seq_lens=list(config.systems.seq_lens),
            device=config.systems.device,
            dtype=config.systems.dtype,
            n_warmup=config.systems.n_warmup,
            n_repeats=config.systems.n_repeats,
        )
        store.write_parquet(chash, "systems.parquet", sys_df)
        status["stages"]["systems"] = "ok"
        store.write_status(chash, status)
    else:
        status["stages"]["systems"] = "skipped"

    if config.stages.report:
        from toklab.evaluation.report import render_report

        report_path = render_report(
            run_dir=store.run_dir(chash),
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
