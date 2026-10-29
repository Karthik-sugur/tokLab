"""FastAPI application factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import ValidationError

from toklab.api.schemas import (
    DatasetOut,
    ExperimentCreateRequest,
    ExperimentCreateResponse,
    ExperimentStatusResponse,
    RunAcceptedResponse,
    TokenizePreviewRequest,
    TokenizePreviewResponse,
)
from toklab.api.worker import ExperimentWorker
from toklab.experiments.config import ExperimentConfig
from toklab.experiments.preview import MAX_PREVIEW_CHARS, preview_tokenize
from toklab.experiments.store import (
    RunStore,
    compute_config_hash,
    list_prepared_datasets,
)

MAX_CONFIG_BYTES = 512_000


def _http(status: int, detail: str, code: str | None = None) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"detail": detail, "code": code} if code else detail,
    )


def _df_records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def create_app(
    *,
    runs_root: Path | str = "runs",
    datasets_root: Path | str = "tests/fixtures",
    runner: Any | None = None,
) -> FastAPI:
    """Build the TokLab API (injectable store roots + optional runner)."""
    store = RunStore(runs_root)
    worker = ExperimentWorker(store, runner=runner)
    datasets_root = Path(datasets_root)

    app = FastAPI(title="TokLab API", version="0.1.0")
    app.state.store = store
    app.state.worker = worker
    app.state.datasets_root = datasets_root

    @app.exception_handler(ValidationError)
    async def _validation_handler(
        _request: Request, exc: ValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/tokenize/preview", response_model=TokenizePreviewResponse)
    def tokenize_preview(body: TokenizePreviewRequest) -> TokenizePreviewResponse:
        if len(body.text) > MAX_PREVIEW_CHARS:
            raise _http(
                413,
                f"snippet too large ({len(body.text)} > {MAX_PREVIEW_CHARS})",
                "too_large",
            )
        cfg: dict[str, Any] = {}
        if body.model_name:
            cfg["model_name"] = body.model_name
        if body.tokenizer_name:
            cfg["tokenizer_name"] = body.tokenizer_name
        try:
            raw = preview_tokenize(body.text, body.conditions, cfg=cfg or None)
        except ValueError as exc:
            msg = str(exc)
            if "too large" in msg:
                raise _http(413, msg, "too_large") from exc
            raise _http(422, msg, "invalid") from exc
        except Exception as exc:  # noqa: BLE001
            raise _http(422, f"{type(exc).__name__}: {exc}", "invalid") from exc
        return TokenizePreviewResponse.model_validate(raw)

    @app.get("/datasets", response_model=list[DatasetOut])
    def datasets() -> list[DatasetOut]:
        rows = list_prepared_datasets(app.state.datasets_root)
        return [DatasetOut.model_validate(r) for r in rows]

    @app.post("/experiments", response_model=ExperimentCreateResponse)
    def create_experiment(body: ExperimentCreateRequest) -> ExperimentCreateResponse:
        payload = json.dumps(body.config)
        if len(payload.encode("utf-8")) > MAX_CONFIG_BYTES:
            raise _http(413, "config JSON too large", "too_large")
        try:
            config = ExperimentConfig.model_validate(body.config)
        except ValidationError as exc:
            raise _http(422, str(exc), "invalid") from exc
        # Force runs_root to this API's store.
        config = config.model_copy(update={"runs_root": Path(store.runs_root)})
        cfg_dict = config.canonical_dict()
        chash = compute_config_hash(cfg_dict)
        store.ensure_run_dir(chash)
        store.write_json(chash, "config.json", cfg_dict)
        status = {
            "config_hash": chash,
            "name": config.name,
            "state": "created",
            "stages": {},
            "run_dir": str(store.run_dir(chash).resolve()),
        }
        store.write_status(chash, status)
        store.upsert_run(
            config_hash=chash,
            name=config.name,
            status="created",
            seed=config.seed,
        )
        return ExperimentCreateResponse(
            id=chash,
            name=config.name,
            status="created",
            run_dir=str(store.run_dir(chash).resolve()),
        )

    def _require_run(exp_id: str) -> dict[str, Any]:
        row = store.get_run(exp_id)
        status = store.load_status(exp_id)
        if row is None and status is None and not store.run_dir(exp_id).is_dir():
            raise _http(404, f"experiment {exp_id} not found", "missing")
        if row is None and status is None:
            # Directory exists without index — still 404 for unknown hash.
            if not (store.run_dir(exp_id) / "config.json").is_file():
                raise _http(404, f"experiment {exp_id} not found", "missing")
        return {
            "row": row,
            "status": status or {},
        }

    @app.post("/experiments/{exp_id}/run", response_model=RunAcceptedResponse)
    def run_experiment_endpoint(exp_id: str) -> RunAcceptedResponse:
        info = _require_run(exp_id)
        cfg = store.read_json(exp_id, "config.json")
        if cfg is None:
            raise _http(404, f"experiment {exp_id} config missing", "missing")
        try:
            config = ExperimentConfig.model_validate(cfg)
        except ValidationError as exc:
            raise _http(422, str(exc), "invalid") from exc
        # Keep runs_root from the stored config so the hash matches ``exp_id``.
        if compute_config_hash(config.canonical_dict()) != exp_id:
            raise _http(422, "config hash mismatch with experiment id", "invalid")
        if worker.busy:
            raise _http(
                409,
                f"worker busy with {worker.current_id}",
                "busy",
            )
        st = (info["status"] or {}).get("state") or (
            info["row"] or {}
        ).get("status")
        if st == "running":
            raise _http(409, "experiment already running", "busy")
        try:
            worker.start(exp_id, config)
        except RuntimeError as exc:
            if str(exc) == "busy":
                raise _http(409, "worker busy", "busy") from exc
            raise
        return RunAcceptedResponse(
            id=exp_id, status="running", message="experiment started"
        )

    @app.get("/experiments/{exp_id}", response_model=ExperimentStatusResponse)
    def get_experiment(exp_id: str) -> ExperimentStatusResponse:
        info = _require_run(exp_id)
        status = info["status"] or {}
        row = info["row"] or {}
        return ExperimentStatusResponse(
            id=exp_id,
            name=str(status.get("name") or row.get("name") or ""),
            status=str(row.get("status") or status.get("state") or "unknown"),
            state=status.get("state"),
            stages=dict(status.get("stages") or {}),
            run_dir=status.get("run_dir") or row.get("run_dir"),
            report_path=status.get("report_path"),
            error=status.get("error"),
        )

    @app.get("/experiments/{exp_id}/results")
    def get_results(
        exp_id: str,
        metric: str | None = Query(None),
        condition: str | None = Query(None),
        table: str = Query(
            "intrinsic_metrics",
            description="Artifact stem without .parquet",
        ),
    ) -> dict[str, Any]:
        _require_run(exp_id)
        status = store.load_status(exp_id) or {}
        state = status.get("state")
        if state not in {"complete", "failed"} and not store.artifact_exists(
            exp_id, f"{table}.parquet"
        ):
            # Allow reading partial artifacts if present; else 409.
            if not store.artifact_exists(exp_id, f"{table}.parquet"):
                raise _http(409, "results not ready", "not_ready")
        name = table if table.endswith(".parquet") else f"{table}.parquet"
        df = store.try_read_parquet(exp_id, name)
        if df is None:
            raise _http(409, f"artifact {name} not ready", "not_ready")
        if metric and "metric" in df.columns:
            df = df[df["metric"] == metric]
        if condition and "condition" in df.columns:
            df = df[df["condition"] == condition]
        return {"id": exp_id, "table": name, "rows": _df_records(df)}

    @app.get("/experiments/{exp_id}/factorial")
    def get_factorial(exp_id: str) -> dict[str, Any]:
        _require_run(exp_id)
        df = store.try_read_parquet(exp_id, "factorial_effects.parquet")
        if df is None or df.empty:
            raise _http(409, "factorial results not ready", "not_ready")
        need = {"c2", "c3", "c4", "c5"}
        cols = {c.lower() for c in df.columns}
        if not need.issubset(cols):
            raise _http(
                409,
                "factorial table missing one or more C2–C5 cells",
                "not_ready",
            )
        # Per-row NaN cell → 409
        for col in ("c2", "c3", "c4", "c5"):
            if df[col].isna().any():
                raise _http(
                    409,
                    f"factorial cell {col} missing (NaN)",
                    "not_ready",
                )
        return {"id": exp_id, "rows": _df_records(df)}

    @app.get("/experiments/{exp_id}/drift")
    def get_drift(exp_id: str) -> dict[str, Any]:
        _require_run(exp_id)
        df = store.try_read_parquet(exp_id, "drift_metrics.parquet")
        if df is None:
            raise _http(409, "drift results not ready", "not_ready")
        return {"id": exp_id, "rows": _df_records(df)}

    @app.get("/experiments/{exp_id}/report")
    def get_report(
        exp_id: str,
        format: str = Query("md", pattern="^(md|html)$"),
    ) -> Any:
        _require_run(exp_id)
        md = store.read_text(exp_id, "report.md")
        if md is None:
            raise _http(409, "report not ready", "not_ready")
        if format == "md":
            return PlainTextResponse(md, media_type="text/markdown")
        # Minimal HTML wrap — no metric computation.
        escaped = (
            md.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>TokLab report</title></head><body>"
            f"<pre>{escaped}</pre></body></html>"
        )
        return HTMLResponse(html)

    @app.get("/experiments/{exp_id}/fidelity")
    def get_fidelity(exp_id: str) -> dict[str, Any]:
        """Stored fidelity table (UI greys out failed conditions)."""
        _require_run(exp_id)
        df = store.try_read_parquet(exp_id, "fidelity.parquet")
        if df is None:
            raise _http(409, "fidelity not ready", "not_ready")
        return {"id": exp_id, "rows": _df_records(df)}

    return app


app = create_app()
