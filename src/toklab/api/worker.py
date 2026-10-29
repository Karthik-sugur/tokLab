"""Single-worker background runner (one experiment at a time)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from toklab.experiments.config import ExperimentConfig
from toklab.experiments.runner import run_experiment
from toklab.experiments.store import RunStore


class ExperimentWorker:
    """Thread-backed single slot: 409 if a run is already in progress."""

    def __init__(
        self,
        store: RunStore,
        *,
        runner: Callable[[ExperimentConfig], dict[str, Any]] | None = None,
    ) -> None:
        self.store = store
        self._runner = runner or run_experiment
        self._lock = threading.Lock()
        self._busy = False
        self._current_id: str | None = None
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def current_id(self) -> str | None:
        with self._lock:
            return self._current_id

    def start(self, config_hash: str, config: ExperimentConfig) -> None:
        with self._lock:
            if self._busy:
                raise RuntimeError("busy")
            self._busy = True
            self._current_id = config_hash

        def _target() -> None:
            try:
                status = self.store.load_status(config_hash) or {
                    "config_hash": config_hash,
                    "name": config.name,
                    "stages": {},
                }
                status["state"] = "running"
                status["status"] = "running"
                self.store.write_status(config_hash, status)
                self.store.upsert_run(
                    config_hash=config_hash,
                    name=config.name,
                    status="running",
                    seed=config.seed,
                )
                result = self._runner(config)
                # Runner already writes complete status; refresh index.
                self.store.upsert_run(
                    config_hash=config_hash,
                    name=config.name,
                    status=str(result.get("state", "complete")),
                    seed=config.seed,
                )
            except Exception as exc:  # noqa: BLE001 — record failure for API
                err_status = {
                    "config_hash": config_hash,
                    "name": config.name,
                    "state": "failed",
                    "stages": {},
                    "error": f"{type(exc).__name__}: {exc}",
                    "run_dir": str(self.store.run_dir(config_hash).resolve()),
                }
                self.store.write_status(config_hash, err_status)
                self.store.upsert_run(
                    config_hash=config_hash,
                    name=config.name,
                    status="failed",
                    seed=config.seed,
                )
            finally:
                with self._lock:
                    self._busy = False
                    self._current_id = None
                    self._thread = None

        thread = threading.Thread(
            target=_target, name=f"toklab-run-{config_hash[:8]}", daemon=True
        )
        with self._lock:
            self._thread = thread
        thread.start()
