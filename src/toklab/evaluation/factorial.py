"""Run C2–C5 factorial over a K sweep; print effects table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from toklab.data.splits import ManifestView
from toklab.evaluation.intrinsic import tokens_per_kb
from toklab.evaluation.stats import FactorialEffects, factorial_analysis
from toklab.tokenizers.registry import get_tokenizer, load_base_model_config
from toklab.vocab.build import build_adapted_vocab, shared_base_from_cfg


def _load_hypotheses() -> dict[str, Any]:
    candidates = [
        Path.cwd() / "configs" / "hypotheses.yaml",
        Path(__file__).resolve().parents[3] / "configs" / "hypotheses.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "decision_rule": (
            "Combined beats either alone on a metric only if C5 beats the "
            "better of C3/C4 with a CI excluding zero."
        )
    }


def _tpk_by_repo(
    files: tuple,
    texts: dict[str, str],
    tok: Any,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for sf in files:
        text = texts[sf.file_id]
        n_ids = len(tok.encode(text).ids)
        n_bytes = len(text.encode("utf-8"))
        val = tokens_per_kb(n_ids, n_bytes)
        out.setdefault(sf.repo, []).append(val)
    return out


def format_factorial_table(effects: list[FactorialEffects]) -> str:
    lines = [
        "source=fixtures_only  metric=tokens_per_kb  (lower is better)",
        f"{'k':>6} {'C2':>10} {'C3':>10} {'C4':>10} {'C5':>10} "
        f"{'eff_S':>10} {'eff_D':>10} {'I':>10} {'indep':>8} {'C5>best':>8}",
        "-" * 96,
    ]
    for e in effects:
        lines.append(
            f"{e.k:6d} "
            f"{e.m_c2.estimate:10.2f} "
            f"{e.m_c3.estimate:10.2f} "
            f"{e.m_c4.estimate:10.2f} "
            f"{e.m_c5.estimate:10.2f} "
            f"{e.effect_s.estimate:10.2f} "
            f"{e.effect_d.estimate:10.2f} "
            f"{e.interaction.estimate:10.2f} "
            f"{e.independence_expectation:8.3f} "
            f"{str(e.combined_beats_either):>8}"
        )
        lines.append(
            f"{'':6} "
            f"[{e.m_c2.low:.1f},{e.m_c2.high:.1f}] "
            f"[{e.m_c3.low:.1f},{e.m_c3.high:.1f}] "
            f"[{e.m_c4.low:.1f},{e.m_c4.high:.1f}] "
            f"[{e.m_c5.low:.1f},{e.m_c5.high:.1f}] "
            f"I_CI=[{e.interaction.low:.2f},{e.interaction.high:.2f}]"
        )
    if effects:
        lines.append("")
        lines.append("decision_rule: " + effects[0].decision_rule.strip())
    return "\n".join(lines)


def run_factorial(
    *,
    manifest: Path | str,
    repos_dir: Path | str,
    k_values: list[int],
    split: str = "test",
    seed: int = 0,
    n_boot: int = 200,
    min_freq: int = 1,
    min_repos: int = 1,
    cfg: dict[str, Any] | None = None,
) -> tuple[str, list[FactorialEffects]]:
    """Build adapt vocabs per K; evaluate C2–C5 on ``split``; return table."""
    repos_dir = Path(repos_dir)
    cfg = dict(cfg or {})
    cfg = {**load_base_model_config(cfg.get("base_model_yaml")), **cfg}
    hypo = _load_hypotheses()
    decision_rule = str(hypo.get("decision_rule", "")).strip()

    adapt_view = ManifestView.from_parquet(manifest, "adapt")
    eval_view = ManifestView.from_parquet(manifest, split)
    eval_files = eval_view.files()
    if not eval_files:
        raise ValueError(f"no files in split {split!r}")

    texts = {
        sf.file_id: (repos_dir / sf.repo / sf.path).read_text(encoding="utf-8")
        for sf in (*adapt_view.files(), *eval_files)
    }
    base = shared_base_from_cfg(cfg)
    base_model = str(cfg.get("model_name", ""))

    results: list[FactorialEffects] = []
    for k in k_values:
        vocab_n = build_adapted_vocab(
            adapt_view,
            repos_dir=repos_dir,
            base=base,
            mode="native",
            k=k,
            min_freq=min_freq,
            min_repos=min_repos,
            base_model=base_model,
        )
        vocab_s = build_adapted_vocab(
            adapt_view,
            repos_dir=repos_dir,
            base=base,
            mode="structural",
            k=k,
            min_freq=min_freq,
            min_repos=min_repos,
            base_model=base_model,
        )
        shared_cfg = {
            **cfg,
            "base_tokenizer": base,
            "attach_leading_space": True,
        }
        toks = {
            "C2": get_tokenizer("C2", shared_cfg),
            "C3": get_tokenizer(
                "C3", {**shared_cfg, "adapted_vocab": vocab_n}
            ),
            "C4": get_tokenizer("C4", shared_cfg),
            "C5": get_tokenizer(
                "C5", {**shared_cfg, "adapted_vocab": vocab_s}
            ),
        }
        cond_vals = {
            c: _tpk_by_repo(eval_files, texts, toks[c])
            for c in ("C2", "C3", "C4", "C5")
        }
        effects = factorial_analysis(
            cond_vals,
            k=k,
            seed=seed,
            n_boot=n_boot,
            decision_rule=decision_rule,
        )
        results.append(effects)

    table = format_factorial_table(results)
    return table, results
