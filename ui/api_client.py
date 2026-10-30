"""HTTP client for the TokLab API. UI must not compute metrics itself."""

from __future__ import annotations

from typing import Any

import httpx


def api_unreachable_message(base_url: str) -> str:
    return (
        f"Cannot reach the API at {base_url}. "
        "Start it with: toklab serve-api"
    )


class ApiUnreachableError(RuntimeError):
    """Raised when the API host refuses / times out (no raw traceback to UI)."""


class TokLabClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def _raise_unreachable(self, exc: Exception) -> None:
        raise ApiUnreachableError(
            api_unreachable_message(self.base_url)
        ) from exc

    def _get(self, path: str, **params: Any) -> Any:
        try:
            with httpx.Client(base_url=self.base_url, timeout=120.0) as client:
                params_f = {k: v for k, v in params.items() if v is not None}
                r = client.get(path, params=params_f)
                r.raise_for_status()
                if "application/json" in r.headers.get("content-type", ""):
                    return r.json()
                return r.text
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            self._raise_unreachable(exc)

    def _post(self, path: str, json_body: dict[str, Any] | None = None) -> Any:
        try:
            with httpx.Client(base_url=self.base_url, timeout=120.0) as client:
                r = client.post(path, json=json_body)
                r.raise_for_status()
                return r.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            self._raise_unreachable(exc)

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def datasets(self) -> list[dict[str, Any]]:
        return self._get("/datasets")

    def preview(
        self, text: str, conditions: list[str] | None = None
    ) -> dict[str, Any]:
        return self._post(
            "/tokenize/preview",
            {
                "text": text,
                "conditions": conditions or ["C2", "C4", "C5"],
            },
        )

    def experiment(self, exp_id: str) -> dict[str, Any]:
        return self._get(f"/experiments/{exp_id}")

    def results(
        self,
        exp_id: str,
        *,
        table: str = "token_efficiency_ci",
        metric: str | None = None,
        condition: str | None = None,
    ) -> dict[str, Any]:
        return self._get(
            f"/experiments/{exp_id}/results",
            table=table,
            metric=metric,
            condition=condition,
        )

    def factorial(self, exp_id: str) -> dict[str, Any]:
        return self._get(f"/experiments/{exp_id}/factorial")

    def drift(self, exp_id: str) -> dict[str, Any]:
        return self._get(f"/experiments/{exp_id}/drift")

    def fidelity(self, exp_id: str) -> dict[str, Any]:
        return self._get(f"/experiments/{exp_id}/fidelity")

    def report_md(self, exp_id: str) -> str:
        return self._get(f"/experiments/{exp_id}/report", format="md")
