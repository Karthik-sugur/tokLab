"""Render Markdown experiment reports from a run directory."""

from __future__ import annotations

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
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _load_hypotheses(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"hypotheses": {}, "decision_rule": ""}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"hypotheses": {}}


def _ci_excludes_zero(low: float, high: float) -> bool:
    return low > 0 or high < 0


def evaluate_hypotheses(
    effects: pd.DataFrame | None,
    hypotheses: dict[str, Any],
    *,
    invalid_conditions: set[str],
) -> list[dict[str, str]]:
    """Compute supported / not supported / inconclusive from YAML rules.

    Uses the largest-K factorial row when multiple K values exist.
    Claims involving INVALID conditions are inconclusive.
    """
    hypos = hypotheses.get("hypotheses") or {}
    decision_rule = str(hypotheses.get("decision_rule", "")).strip()
    verdicts: list[dict[str, str]] = []

    if effects is None or effects.empty:
        for hid, meta in hypos.items():
            verdicts.append(
                {
                    "id": str(hid),
                    "claim": str(meta.get("claim", "")).strip(),
                    "contrast": str(meta.get("contrast", "")),
                    "verdict": "inconclusive",
                    "reason": "no factorial effects recorded",
                }
            )
        return verdicts

    row = effects.sort_values("k").iloc[-1]
    grid_invalid = bool(invalid_conditions & {"C2", "C3", "C4", "C5"})

    for hid, meta in hypos.items():
        claim = str(meta.get("claim", "")).strip()
        contrast = str(meta.get("contrast", "")).strip()
        if grid_invalid:
            verdicts.append(
                {
                    "id": str(hid),
                    "claim": claim,
                    "contrast": contrast,
                    "verdict": "inconclusive",
                    "reason": (
                        "one or more C2–C5 conditions marked INVALID "
                        "(fidelity failure); excluded from claims"
                    ),
                }
            )
            continue

        if hid == "H1":
            est = float(row["effect_s"])
            c2_hi = float(row["c2_high"])
            c4_lo = float(row["c4_low"])
            c4_hi = float(row["c4_high"])
            c2_lo = float(row["c2_low"])
            if est < 0 and c4_hi < c2_lo:
                verd = "supported"
                reason = "C4 tokens/KB CI entirely below C2"
            elif est > 0 and c4_lo > c2_hi:
                verd = "not supported"
                reason = "C4 tokens/KB CI entirely above C2"
            else:
                verd = "inconclusive"
                reason = "C4 vs C2 CI overlap or zero not excluded"
        elif hid == "H2":
            est = float(row["effect_d"])
            c2_hi = float(row["c2_high"])
            c3_lo = float(row["c3_low"])
            c3_hi = float(row["c3_high"])
            c2_lo = float(row["c2_low"])
            if est < 0 and c3_hi < c2_lo:
                verd = "supported"
                reason = "C3 tokens/KB CI entirely below C2"
            elif est > 0 and c3_lo > c2_hi:
                verd = "not supported"
                reason = "C3 tokens/KB CI entirely above C2"
            else:
                verd = "inconclusive"
                reason = "C3 vs C2 CI overlap or zero not excluded"
        elif hid == "H3":
            est = float(row["c5"]) - float(row["c2"])
            c2_hi = float(row["c2_high"])
            c5_lo = float(row["c5_low"])
            c5_hi = float(row["c5_high"])
            c2_lo = float(row["c2_low"])
            if est < 0 and c5_hi < c2_lo:
                verd = "supported"
                reason = "C5 tokens/KB CI entirely below C2"
            elif est > 0 and c5_lo > c2_hi:
                verd = "not supported"
                reason = "C5 tokens/KB CI entirely above C2"
            else:
                verd = "inconclusive"
                reason = "C5 vs C2 CI overlap or zero not excluded"
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


def _interaction_note(effects: pd.DataFrame | None) -> str:
    if effects is None or effects.empty:
        return ""
    iest = float(effects.sort_values("k").iloc[-1]["interaction"])
    # On tokens/KB (lower better): positive I means sub-additive gains.
    if iest > 0:
        return (
            "Interaction point estimate is positive on tokens/KB "
            f"(I={iest:.4f}): combined gains are worse than additive "
            "independence expectation (negative result for synergy)."
        )
    if iest < 0:
        return (
            "Interaction point estimate is negative on tokens/KB "
            f"(I={iest:.4f}): combined beats additive independence "
            "expectation on the point estimate."
        )
    return "Interaction point estimate is exactly zero."


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
) -> dict[str, Any]:
    invalid: set[str] = set()
    fidelity_rows: list[dict[str, Any]] = []
    if fidelity is not None:
        for row in fidelity.itertuples(index=False):
            d = {
                "condition": row.condition,
                "n_files": int(row.n_files),
                "n_roundtrip_ok": int(row.n_roundtrip_ok),
                "n_roundtrip_fail": int(row.n_roundtrip_fail),
                "status": str(row.status),
                "valid": bool(row.valid),
            }
            if not d["valid"]:
                invalid.add(str(d["condition"]))
            fidelity_rows.append(d)

    # Exclude invalid from efficiency claims display (keep row but flag).
    eff_rows: list[dict[str, Any]] = []
    if efficiency is not None:
        for row in efficiency.itertuples(index=False):
            cond = str(row.condition)
            eff_rows.append(
                {
                    "condition": cond,
                    "estimate": float(row.estimate),
                    "ci_low": float(row.ci_low),
                    "ci_high": float(row.ci_high),
                    "n_repos": int(row.n_repos),
                    "excluded": cond in invalid,
                }
            )

    verdicts = evaluate_hypotheses(effects, hypo, invalid_conditions=invalid)
    ran = any(
        df is not None and not df.empty
        for df in (fidelity, efficiency, effects, context)
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
        "effects": _df_records(effects),
        "interaction_note": _interaction_note(effects),
        "context": _df_records(context),
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


def build_report_context(
    run_dir: Path,
    *,
    hypotheses_path: Path,
    run_labels_path: Path | None = None,
) -> dict[str, Any]:
    """Assemble template context from parquet/json in ``run_dir``.

    Fidelity, efficiency, 2x2 and context tables are reported per evaluation
    set (``test`` = new codebase, ``test_inorg`` = new files in a known
    codebase). Runs without an ``eval_set`` column are treated as a single
    set named by ``data.eval_split``.
    """
    run_dir = Path(run_dir)
    env = _read_json(run_dir, "env.json")
    cfg = _read_json(run_dir, "config.json")
    status = _read_json(run_dir, "status.json")
    fidelity = _read_parquet(run_dir, "fidelity.parquet")
    efficiency = _read_parquet(run_dir, "token_efficiency_ci.parquet")
    effects = _read_parquet(run_dir, "factorial_effects.parquet")
    context = _read_parquet(run_dir, "context_utilisation.parquet")
    systems = _read_parquet(run_dir, "systems.parquet")
    vocab_cost = _read_parquet(run_dir, "vocab_cost.parquet")
    drift = _read_parquet(run_dir, "drift_metrics.parquet")
    hypo = _load_hypotheses(hypotheses_path)

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
        )
        for name in names
    ]
    any_invalid = any(s["any_invalid"] for s in sections)
    config_hash = status.get("config_hash") or run_dir.name
    vocab = cfg.get("vocab") or {}
    min_repos = vocab.get("min_repos", "n/a")
    min_freq = vocab.get("min_freq", "n/a")

    return {
        "run_dir": str(run_dir.resolve()),
        "config_hash": config_hash,
        "name": status.get("name") or cfg.get("name", ""),
        "run_label": load_run_label(str(config_hash), run_labels_path),
        "env": env,
        "config": cfg,
        "primary_eval_set": primary,
        "eval_sets": sections,
        "any_invalid": any_invalid,
        "systems": _df_records(systems),
        "vocab_cost": _df_records(vocab_cost),
        "drift": _df_records(drift),
        "decision_rule": str(hypo.get("decision_rule", "")).strip(),
        "limitations": [
            "Vocabulary mined from the adapt split only; test and test_inorg "
            "files are never used for mining.",
            "test_inorg is a FILE-level hold-out inside adapt repositories, an "
            "explicit exception to repository-level splitting: it measures reuse "
            "of a known codebase's names, not generalisation to new codebases. "
            f"Hypothesis claims use the primary set ({primary}) first.",
            f"Selection uses min_freq={min_freq} and min_repos={min_repos}. With "
            "min_repos=1, entries frequent in a single adapt repository can be "
            "selected: they inflate gains on test_inorg (same repositories) and "
            "overfit relative to test (new repositories). K was not chosen on "
            "the val split in this run.",
            "Quality / BPB suppressed for conditions with untrained new vocab rows "
            "(C3/C5): systems rows tagged quality_valid=False when present.",
            "FLOPs / KV / param overhead figures are ESTIMATE labels, not measured.",
            "Drift REFRESH may append tokens mined from a small drifted v1 sample; "
            "old adapted ids stay unchanged (append-only).",
            "Drift BPB is reported only for gate-valid fine-tuned variants; "
            "otherwise marked not run.",
            "Fixture-scale corpora: do not generalise beyond the evaluated repos.",
            (
                "Conditions failing fidelity are marked INVALID and excluded "
                "from hypothesis claims for that evaluation set."
                if any_invalid
                else "All evaluated conditions passed fidelity on this run."
            ),
        ],
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
