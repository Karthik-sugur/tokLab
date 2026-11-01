"""Render Markdown experiment reports from a run directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

EVAL_SET_LABELS: dict[str, str] = {
    "test": (
        "new codebase: held-out repositories (repo-level split, never "
        "seen by vocabulary mining)"
    ),
    "test_inorg": (
        "new files in known codebases: seeded fraction of files held out "
        "from each adapt repository (excluded from vocabulary mining)"
    ),
}
DEFAULT_EVAL_SETS = ("test", "test_inorg")
GRID = ("C2", "C3", "C4", "C5")


def _repo_root_candidates(*parts: str) -> list[Path]:
    return [
        Path.cwd().joinpath(*parts),
        Path(__file__).resolve().parents[3].joinpath(*parts),
    ]


def _template_dir() -> Path:
    candidates = _repo_root_candidates("reports", "templates")
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def load_run_label(config_hash: str, path: Path | None = None) -> dict[str, str]:
    """Return ``{"label", "note"}`` for ``config_hash`` from run_labels.yaml."""
    paths = [path] if path is not None else _repo_root_candidates(
        "configs", "run_labels.yaml"
    )
    for p in paths:
        if p.is_file():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            entry = (data.get("runs") or {}).get(config_hash)
            if isinstance(entry, dict) and entry.get("label"):
                return {
                    "label": str(entry["label"]),
                    "note": str(entry.get("note", "")).strip(),
                }
            return {}
    return {}


def _eval_set_frame(
    df: pd.DataFrame | None, name: str, legacy_set: str
) -> pd.DataFrame | None:
    """Rows for one eval set; runs without ``eval_set`` are one legacy set."""
    if df is None:
        return None
    if "eval_set" not in df.columns:
        return df if name == legacy_set else None
    return df[df["eval_set"] == name].reset_index(drop=True)


def _read_parquet(run_dir: Path, name: str) -> pd.DataFrame | None:
    path = run_dir / name
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def _read_json(run_dir: Path, name: str) -> dict[str, Any]:
    path = run_dir / name
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_hypotheses(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"hypotheses": {}, "decision_rule": ""}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"hypotheses": {}}


def _ci_excludes_zero(low: float, high: float) -> bool:
    return low > 0 or high < 0


def _pick_row(effects: pd.DataFrame, k: int | None) -> pd.Series | None:
    if k is None:
        return effects.sort_values("k").iloc[-1]
    sub = effects[effects["k"] == k]
    return None if sub.empty else sub.iloc[0]


def _all_verdicts(
    hypos: dict[str, Any], verdict: str, reason: str
) -> list[dict[str, str]]:
    return [
        {
            "id": str(hid),
            "claim": str(meta.get("claim", "")).strip(),
            "contrast": str(meta.get("contrast", "")).strip(),
            "verdict": verdict,
            "reason": reason,
        }
        for hid, meta in hypos.items()
    ]


def evaluate_hypotheses(
    effects: pd.DataFrame | None,
    hypotheses: dict[str, Any],
    *,
    invalid_conditions: set[str],
    k: int | None = None,
) -> list[dict[str, str]]:
    """Compute supported / not supported / inconclusive from YAML rules.

    Uses the factorial row at ``k`` (K*); with ``k=None`` (legacy runs
    without K selection) the largest-K row. Claims involving INVALID
    conditions are inconclusive. Every hypothesis in the YAML gets a row.
    """
    hypos = hypotheses.get("hypotheses") or {}
    decision_rule = str(hypotheses.get("decision_rule", "")).strip()

    if effects is None or effects.empty:
        return _all_verdicts(hypos, "inconclusive", "no factorial effects recorded")
    row = _pick_row(effects, k)
    if row is None:
        return _all_verdicts(
            hypos, "inconclusive", f"no factorial row at K*={k}"
        )
    if invalid_conditions & set(GRID):
        return _all_verdicts(
            hypos,
            "inconclusive",
            "one or more C2–C5 conditions marked INVALID "
            "(fidelity failure); excluded from claims",
        )

    verdicts: list[dict[str, str]] = []
    for hid, meta in hypos.items():
        claim = str(meta.get("claim", "")).strip()
        contrast = str(meta.get("contrast", "")).strip()

        if hid in ("H1", "H2", "H3"):
            cond = {"H1": "c4", "H2": "c3", "H3": "c5"}[hid]
            est = (
                float(row["effect_s"]) if hid == "H1"
                else float(row["effect_d"]) if hid == "H2"
                else float(row["c5"]) - float(row["c2"])
            )
            c2_lo, c2_hi = float(row["c2_low"]), float(row["c2_high"])
            lo, hi = float(row[f"{cond}_low"]), float(row[f"{cond}_high"])
            name = cond.upper()
            if est < 0 and hi < c2_lo:
                verd = "supported"
                reason = f"{name} tokens/KB CI entirely below C2"
            elif est > 0 and lo > c2_hi:
                verd = "not supported"
                reason = f"{name} tokens/KB CI entirely above C2"
            else:
                verd = "inconclusive"
                reason = f"{name} vs C2 CI overlap or zero not excluded"
        elif hid == "H4":
            beats = row.get("combined_beats_either")
            if beats is None or (isinstance(beats, float) and pd.isna(beats)):
                verd = "inconclusive"
                reason = "combined_beats_either unknown"
            elif bool(beats):
                verd = "supported"
                reason = decision_rule or "combined_beats_either=True"
            else:
                verd = "not supported"
                reason = (
                    "C5 does not beat better of C3/C4 with CI excluding zero"
                )
        elif hid == "H5":
            ilo = float(row["interaction_low"])
            ihi = float(row["interaction_high"])
            iest = float(row["interaction"])
            if _ci_excludes_zero(ilo, ihi):
                verd = "supported"
                reason = (
                    "interaction CI excludes zero "
                    f"(I={iest:.4f}, [{ilo:.4f},{ihi:.4f}])"
                )
            else:
                verd = "inconclusive"
                reason = (
                    f"interaction CI includes zero "
                    f"(I={iest:.4f}, [{ilo:.4f},{ihi:.4f}])"
                )
        else:
            verd = "inconclusive"
            reason = f"unknown hypothesis id {hid}"

        verdicts.append(
            {
                "id": str(hid),
                "claim": claim,
                "contrast": contrast,
                "verdict": verd,
                "reason": reason,
            }
        )
    return verdicts


def _df_records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


def interaction_note(effects: pd.DataFrame | None, k: int | None) -> str:
    """One CI-driven sentence about the 2×2 interaction at ``k``."""
    if effects is None or effects.empty:
        return ""
    row = _pick_row(effects, k)
    if row is None:
        return f"No factorial row at K*={k}."
    iest = float(row["interaction"])
    ilo, ihi = float(row["interaction_low"]), float(row["interaction_high"])
    stat = f"I={iest:.4f}, 95% CI [{ilo:.4f}, {ihi:.4f}] at K={int(row['k'])}"
    if not _ci_excludes_zero(ilo, ihi):
        text = f"No evidence of an interaction (CI includes zero): {stat}."
    elif iest > 0:
        text = (
            f"Sub-additive interaction (CI excludes zero): {stat}. On tokens/KB "
            "the domain vocabulary saves fewer tokens when structure is on, so "
            "the combination is worse than additive."
        )
    else:
        text = (
            f"Super-additive interaction (CI excludes zero): {stat}. On tokens/KB "
            "the domain vocabulary saves more tokens when structure is on."
        )
    beats = row.get("combined_beats_either")
    if beats is not None and not (isinstance(beats, float) and pd.isna(beats)):
        if bool(beats):
            text += (
                " Decision rule satisfied: C5 beats the better of C3/C4 with a "
                "CI excluding zero."
            )
        else:
            text += (
                " Decision rule not satisfied: C5 is not better than the better "
                "of C3/C4 with a CI excluding zero."
            )
    return text


def effects_averaging_note(
    effects: pd.DataFrame | None,
    efficiency: pd.DataFrame | None,
    k: int | None,
) -> str:
    """How the 2×2 columns are averaged (matches ``stats.factorial_analysis``)."""
    if effects is None or effects.empty:
        return ""
    row = _pick_row(effects, k)
    if row is None:
        return ""
    k_row = int(row["k"])
    n_repos = ""
    if efficiency is not None and not efficiency.empty and "n_repos" in efficiency:
        sub = efficiency
        if "k" in sub.columns:
            sub = sub[sub["k"] == k_row]
        if not sub.empty:
            n_repos = f"{int(sub['n_repos'].iloc[0])} "
    c2, c3, c4, c5 = (float(row[c]) for c in ("c2", "c3", "c4", "c5"))
    cell_d, cell_s = c3 - c2, c4 - c2
    cell_i = (c5 - c4) - (c3 - c2)
    return (
        "C2–C5 are the cell means of the Token efficiency table: the mean over "
        "all files, each file weighted equally. effect_S (C4−C2), effect_D "
        "(C3−C2) and I = (C5−C4) − (C3−C2) are the mean over the "
        f"{n_repos}repositories of per-repository differences of repository "
        "means, each repository weighted equally, with cluster-bootstrap CIs "
        "over repositories. They therefore differ from differences of the "
        f"printed cell means: at K={k_row} the cell means give "
        f"D={cell_d:+.4f}, S={cell_s:+.4f}, I={cell_i:+.4f}; the table shows "
        f"D={float(row['effect_d']):+.4f}, S={float(row['effect_s']):+.4f}, "
        f"I={float(row['interaction']):+.4f}. The decision-rule column uses the "
        "same repository-weighted paired difference (C5 minus the better of "
        "C3/C4, the better one picked by the cell means) and is True only if "
        "that difference and its whole CI are below zero."
    )


def _with_k(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for r in records:
        k = r.get("k")
        r["k"] = None if k is None or (isinstance(k, float) and pd.isna(k)) else int(k)
    return records


def _eval_set_section(
    name: str,
    *,
    primary: bool,
    fidelity: pd.DataFrame | None,
    efficiency: pd.DataFrame | None,
    effects: pd.DataFrame | None,
    context: pd.DataFrame | None,
    hypo: dict[str, Any],
    status_note: str,
    k_star: int | None,
    k_judged: int | None,
    hypotheses_blocked: str,
) -> dict[str, Any]:
    fidelity_rows = _with_k(_df_records(fidelity))
    invalid: set[str] = set()
    for r in fidelity_rows:
        at_judged = r["k"] is None or k_judged is None or r["k"] == k_judged
        if not bool(r.get("valid", True)) and at_judged:
            invalid.add(str(r["condition"]))

    eff_rows: list[dict[str, Any]] = []
    for r in _with_k(_df_records(efficiency)):
        valid = bool(r.get("valid", str(r["condition"]) not in invalid))
        eff_rows.append(
            {
                "k": r["k"],
                "condition": str(r["condition"]),
                "estimate": float(r["estimate"]),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
                "n_repos": int(r["n_repos"]),
                "excluded": not valid,
                "k_star": r["k"] is not None and r["k"] == k_star,
            }
        )
    eff_rows.sort(key=lambda r: (-1 if r["k"] is None else r["k"], r["condition"]))

    effect_rows = _with_k(_df_records(effects))
    for r in effect_rows:
        r["is_k_star"] = r["k"] is not None and r["k"] == k_star

    ran = any(
        df is not None and not df.empty
        for df in (fidelity, efficiency, effects, context)
    )
    if hypotheses_blocked:
        verdicts = _all_verdicts(
            hypo.get("hypotheses") or {}, "not run", hypotheses_blocked
        )
    elif not ran:
        verdicts = _all_verdicts(
            hypo.get("hypotheses") or {}, "not run", "evaluation set not run"
        )
    else:
        verdicts = evaluate_hypotheses(
            effects, hypo, invalid_conditions=invalid, k=k_judged
        )
    return {
        "name": name,
        "label": EVAL_SET_LABELS.get(name, name),
        "primary": primary,
        "ran": ran,
        "status_note": status_note or ("" if ran else "not run"),
        "fidelity": fidelity_rows,
        "any_invalid": bool(invalid),
        "invalid_conditions": sorted(invalid),
        "efficiency": eff_rows,
        "effects": effect_rows,
        "effects_note": effects_averaging_note(
            effects, efficiency, k_judged if k_judged is not None else k_star
        ),
        "interaction_note": (
            "" if hypotheses_blocked else interaction_note(effects, k_judged)
        ),
        "context": _with_k(_df_records(context)),
        "hypotheses": verdicts,
    }


def _eval_set_names(cfg: dict[str, Any]) -> tuple[str, list[str]]:
    data = cfg.get("data") or {}
    primary = str(data.get("eval_split") or "test")
    names = [primary, *(data.get("eval_splits") or []), *DEFAULT_EVAL_SETS]
    ordered: list[str] = []
    for n in names:
        if n not in ordered:
            ordered.append(str(n))
    return primary, ordered


def _fmt_frac(x: Any) -> str:
    return "n/a" if x is None else f"{float(x):.4f}"


def build_limitations(
    *,
    corpus: dict[str, Any],
    overlap: dict[str, Any],
    selection: dict[str, Any],
    cfg: dict[str, Any],
    primary: str,
    any_invalid: bool,
    test_repos: int | None,
) -> list[str]:
    """Limitations, filled in from run data where the run recorded it."""
    vocab = cfg.get("vocab") or {}
    min_repos = vocab.get("min_repos", "n/a")
    min_freq = vocab.get("min_freq", "n/a")
    synthetic = corpus.get("kind") == "synthetic"
    out: list[str] = []

    if overlap:
        parts = [
            f"{name}: {o['occurrences_in_adapt']} of {o['identifier_occurrences']} "
            f"identifier occurrences ({_fmt_frac(o['occurrence_fraction'])}) and "
            f"{o['distinct_in_adapt']} of {o['distinct_identifiers']} distinct "
            f"identifiers ({_fmt_frac(o['distinct_fraction'])}) also occur in adapt"
            for name, o in sorted(overlap.items())
        ]
        lead = (
            "Shared Zipf lexicon: adapt and evaluation splits are renamed from "
            "one lexicon, so company-style names overlap heavily between adapt "
            "and the evaluation sets (measured, structural identifiers): "
            if synthetic
            else "Name overlap between adapt and the evaluation sets (measured, "
            "structural identifiers): "
        )
        out.append(lead + "; ".join(parts) + ".")
    else:
        out.append("Name overlap between adapt and evaluation sets: not measured.")

    repos = corpus.get("repos_per_split") or {}
    if repos:
        listing = "; ".join(
            f"{s}: {', '.join(r)}" for s, r in repos.items()
        )
        out.append(
            "Related repositories in different splits: repositories are split "
            "as whole units, but related projects (same ecosystem, shared "
            "dependencies or test fixtures) can sit in different splits; "
            f"relatedness is not controlled. Split membership: {listing}."
        )
    else:
        out.append(
            "Related repositories may sit in different splits; split membership "
            "not recorded for this run."
        )

    if test_repos is not None:
        out.append(
            f"Only {test_repos} test repositories: CIs are cluster bootstraps "
            f"over {test_repos} repositories and are coarse."
        )

    out.append(
        "test_inorg is a FILE-level hold-out inside adapt repositories, an "
        "explicit exception to repository-level splitting: it measures reuse "
        "of a known codebase's names, not generalisation to new codebases. "
        f"Hypothesis claims use the primary set ({primary}) first."
    )
    if synthetic:
        out.append(
            "Alpha-equivalence is not runtime equivalence: the synthetic renamer "
            "checks that renamed code has the same syntax tree up to consistent "
            "identifier renaming; it does not prove the code behaves the same "
            "(string-based lookups such as getattr, reflection or external APIs "
            "can break)."
        )

    k_star = selection.get("k_star")
    if selection:
        if k_star is not None:
            out.append(
                f"K chosen on val: K*={k_star} minimises C3 tokens/KB on the "
                f"{selection.get('split')} split ({selection.get('n_files')} "
                "files), chosen with C3 only (docs/decisions/005-k-selection.md). "
                "Hypotheses are judged at K* only; other K values are "
                "descriptive. Selecting K on val still tunes the setup to this "
                "corpus."
            )
        else:
            out.append(
                f"K* not selected ({selection.get('status')}); hypotheses not "
                "judged; descriptive tables use the largest K."
            )
    else:
        out.append(
            "K was not chosen on val in this run (legacy run: largest K used)."
        )

    out.append(
        f"Selection uses min_freq={min_freq} and min_repos={min_repos}. With "
        "min_repos=1, entries frequent in a single adapt repository can be "
        "selected: they inflate gains on test_inorg (same repositories) and "
        "overfit relative to test (new repositories)."
    )
    out.append(
        "Quality not measured: no fine-tuning was run, so there is no "
        "bits-per-byte result; C3/C5 add untrained embedding rows and any "
        "quality figure for them is suppressed (systems rows tagged "
        "quality_valid=False)."
    )
    excluded = corpus.get("excluded_files") or []
    if excluded:
        out.append(
            f"{len(excluded)} file(s) excluded at ingest for every condition "
            "because the base tokenizer cannot round-trip them (NFC "
            "normalisation): "
            + "; ".join(f"{e['path']} ({e['reason']})" for e in excluded)
            + "."
        )
    out.append(
        "FLOPs / KV / param overhead figures are ESTIMATE labels, not measured."
    )
    out.append(
        "Drift REFRESH may append tokens mined from a small drifted v1 sample; "
        "old adapted ids stay unchanged (append-only). Drift BPB is reported "
        "only for gate-valid fine-tuned variants; otherwise marked not run."
    )
    out.append(
        "Conditions failing fidelity are marked INVALID and excluded "
        "from hypothesis claims for that evaluation set."
        if any_invalid
        else "All evaluated conditions passed fidelity on this run."
    )
    return out


def build_report_context(
    run_dir: Path,
    *,
    hypotheses_path: Path,
    run_labels_path: Path | None = None,
) -> dict[str, Any]:
    """Assemble template context from parquet/json in ``run_dir``.

    Fidelity, efficiency, 2x2 and context tables are reported per evaluation
    set (``test`` = new codebase, ``test_inorg`` = new files in a known
    codebase) and per K. Hypotheses are judged at K* (``k_selection.json``);
    legacy runs without it use the largest K. Runs without an ``eval_set``
    column are treated as a single set named by ``data.eval_split``.
    """
    run_dir = Path(run_dir)
    env = _read_json(run_dir, "env.json")
    cfg = _read_json(run_dir, "config.json")
    status = _read_json(run_dir, "status.json")
    corpus = _read_json(run_dir, "corpus.json")
    selection = _read_json(run_dir, "k_selection.json")
    sweep = _read_json(run_dir, "k_sweep.json")
    overlap = _read_json(run_dir, "name_overlap.json")
    fidelity = _read_parquet(run_dir, "fidelity.parquet")
    efficiency = _read_parquet(run_dir, "token_efficiency_ci.parquet")
    effects = _read_parquet(run_dir, "factorial_effects.parquet")
    context = _read_parquet(run_dir, "context_utilisation.parquet")
    systems = _read_parquet(run_dir, "systems.parquet")
    vocab_cost = _read_parquet(run_dir, "vocab_cost.parquet")
    vocab_cost_by_k = _read_parquet(run_dir, "vocab_cost_by_k.parquet")
    drift = _read_parquet(run_dir, "drift_metrics.parquet")
    drift_meta = _read_json(run_dir, "drift_meta.json")
    hypo = _load_hypotheses(hypotheses_path)

    k_star = selection.get("k_star") if selection else None
    if selection:
        k_judged = k_star
        blocked = (
            "" if k_star is not None
            else f"K* not selected ({selection.get('status', 'unknown')})"
        )
    else:
        k_judged, blocked = None, ""

    primary, names = _eval_set_names(cfg)
    status_notes = status.get("eval_sets") or {}
    sections = [
        _eval_set_section(
            name,
            primary=name == primary,
            fidelity=_eval_set_frame(fidelity, name, primary),
            efficiency=_eval_set_frame(efficiency, name, primary),
            effects=_eval_set_frame(effects, name, primary),
            context=_eval_set_frame(context, name, primary),
            hypo=hypo,
            status_note=str(status_notes.get(name, "")),
            k_star=k_star,
            k_judged=k_judged,
            hypotheses_blocked=blocked,
        )
        for name in names
    ]
    any_invalid = any(s["any_invalid"] for s in sections)
    config_hash = status.get("config_hash") or run_dir.name
    test_repos = len((corpus.get("repos_per_split") or {}).get("test", [])) or None

    systems_rows = _df_records(systems)
    return {
        "run_dir": str(run_dir.resolve()),
        "config_hash": config_hash,
        "name": status.get("name") or cfg.get("name", ""),
        "run_label": load_run_label(str(config_hash), run_labels_path),
        "env": env,
        "config": cfg,
        "corpus": corpus,
        "k_selection": selection,
        "k_sweep": sweep,
        "k_star": k_star,
        "primary_eval_set": primary,
        "eval_sets": sections,
        "any_invalid": any_invalid,
        "systems": systems_rows,
        "systems_status": str((status.get("stages") or {}).get("systems", "")),
        "vocab_cost": _with_k(_df_records(vocab_cost)),
        "vocab_cost_by_k": _with_k(_df_records(vocab_cost_by_k)),
        "drift": _df_records(drift),
        "drift_meta": drift_meta,
        "decision_rule": str(hypo.get("decision_rule", "")).strip(),
        "limitations": build_limitations(
            corpus=corpus,
            overlap=overlap,
            selection=selection,
            cfg=cfg,
            primary=primary,
            any_invalid=any_invalid,
            test_repos=test_repos,
        ),
    }


def render_report(
    *,
    run_dir: Path | str,
    hypotheses_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """Render ``report.md`` into the run directory (or ``out_path``)."""
    run_dir = Path(run_dir)
    if hypotheses_path is None:
        hypotheses_path = Path("configs/hypotheses.yaml")
    hypotheses_path = Path(hypotheses_path)

    ctx = build_report_context(run_dir, hypotheses_path=hypotheses_path)
    env = Environment(
        loader=FileSystemLoader(str(_template_dir())),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("report.md.j2")
    text = template.render(**ctx)
    dest = Path(out_path) if out_path is not None else run_dir / "report.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest
