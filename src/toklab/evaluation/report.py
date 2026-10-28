"""Render Markdown experiment reports from a run directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape


def _template_dir() -> Path:
    candidates = [
        Path.cwd() / "reports" / "templates",
        Path(__file__).resolve().parents[3] / "reports" / "templates",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


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


def build_report_context(
    run_dir: Path,
    *,
    hypotheses_path: Path,
) -> dict[str, Any]:
    """Assemble template context from parquet/json in ``run_dir``."""
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

    interaction_negative = False
    interaction_note = ""
    effects_rows = _df_records(effects)
    if effects is not None and not effects.empty:
        last = effects.sort_values("k").iloc[-1]
        iest = float(last["interaction"])
        interaction_negative = iest > 0
        # On tokens/KB (lower better): positive I means structure+domain
        # interaction increases tokens relative to additivity of reductions
        # i.e. sub-additive gains. Report plainly either way.
        if iest > 0:
            interaction_note = (
                "Interaction point estimate is positive on tokens/KB "
                f"(I={iest:.4f}): combined gains are worse than additive "
                "independence expectation (negative result for synergy)."
            )
        elif iest < 0:
            interaction_note = (
                "Interaction point estimate is negative on tokens/KB "
                f"(I={iest:.4f}): combined beats additive independence "
                "expectation on the point estimate."
            )
        else:
            interaction_note = "Interaction point estimate is exactly zero."

    hypo = _load_hypotheses(hypotheses_path)
    verdicts = evaluate_hypotheses(effects, hypo, invalid_conditions=invalid)

    any_invalid = bool(invalid)

    return {
        "run_dir": str(run_dir.resolve()),
        "config_hash": status.get("config_hash") or run_dir.name,
        "name": status.get("name") or cfg.get("name", ""),
        "env": env,
        "config": cfg,
        "fidelity": fidelity_rows,
        "any_invalid": any_invalid,
        "invalid_conditions": sorted(invalid),
        "efficiency": eff_rows,
        "effects": effects_rows,
        "interaction_note": interaction_note,
        "interaction_negative_synergy": interaction_negative,
        "context": _df_records(context),
        "systems": _df_records(systems),
        "vocab_cost": _df_records(vocab_cost),
        "hypotheses": verdicts,
        "decision_rule": str(hypo.get("decision_rule", "")).strip(),
        "limitations": [
            "Vocabulary mined from the adapt split only; test never used for mining.",
            "Quality / BPB suppressed for conditions with untrained new vocab rows "
            "(C3/C5): systems rows tagged quality_valid=False when present.",
            "FLOPs / KV / param overhead figures are ESTIMATE labels, not measured.",
            "Fixture-scale corpora: do not generalise beyond the evaluated repos.",
            (
                "Conditions failing fidelity are marked INVALID and excluded "
                "from hypothesis claims."
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
