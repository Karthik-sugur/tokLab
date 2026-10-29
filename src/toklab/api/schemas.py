"""Pydantic request/response models for the TokLab API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TokenizePreviewRequest(BaseModel):
    text: str = Field(..., min_length=1)
    conditions: list[str] = Field(default_factory=lambda: ["C2", "C4", "C5"])
    model_name: str | None = None
    tokenizer_name: str | None = None


class SpanOut(BaseModel):
    start: int
    end: int


class ConditionTokensOut(BaseModel):
    condition: str
    n_tokens: int
    ids: list[int]
    spans: list[SpanOut]


class TokenizePreviewResponse(BaseModel):
    text: str
    n_chars: int
    n_bytes: int
    conditions: list[ConditionTokensOut]


class DatasetOut(BaseModel):
    id: str
    path: str
    manifest_path: str
    manifest_hash: str | None = None


class ExperimentCreateRequest(BaseModel):
    config: dict[str, Any]


class ExperimentCreateResponse(BaseModel):
    id: str
    name: str
    status: str
    run_dir: str


class ExperimentStatusResponse(BaseModel):
    id: str
    name: str
    status: str
    state: str | None = None
    stages: dict[str, Any] = Field(default_factory=dict)
    run_dir: str | None = None
    report_path: str | None = None
    error: str | None = None


class RunAcceptedResponse(BaseModel):
    id: str
    status: str
    message: str


class ErrorBody(BaseModel):
    detail: str
    code: str | None = None
