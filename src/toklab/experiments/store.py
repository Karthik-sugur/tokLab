"""Run directory layout: ``runs/<config_hash>/`` + SQLite run index."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd

from toklab.core.hashing import config_hash as hash_config

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    config_hash TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    run_dir TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    seed INTEGER,
    manifest_hash TEXT,
    vocab_hash TEXT
);
"""


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Stable short hash of the experiment config (full SHA-256 hex)."""
    return hash_config(config_dict)


class RunStore:
    """Filesystem + SQLite index for experiment runs."""

    def __init__(self, runs_root: Path | str) -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.runs_root / "index.sqlite"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def run_dir(self, config_hash: str) -> Path:
        return self.runs_root / config_hash

    def ensure_run_dir(self, config_hash: str) -> Path:
        path = self.run_dir(config_hash)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, config_hash: str, name: str, obj: Any) -> Path:
        path = self.run_dir(config_hash) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def write_parquet(
        self, config_hash: str, name: str, df: pd.DataFrame
    ) -> Path:
        path = self.run_dir(config_hash) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic column order for byte-identical re-runs.
        ordered = df.reindex(sorted(df.columns), axis=1)
        ordered.to_parquet(path, index=False)
        return path

    def write_text(self, config_hash: str, name: str, text: str) -> Path:
        path = self.run_dir(config_hash) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def read_parquet(self, config_hash: str, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.run_dir(config_hash) / name)

    def upsert_run(
        self,
        *,
        config_hash: str,
        name: str,
        status: str,
        seed: int | None = None,
        manifest_hash: str | None = None,
        vocab_hash: str | None = None,
    ) -> None:
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        run_dir = str(self.run_dir(config_hash).resolve())
        with self._connect() as conn:
            row = conn.execute(
                "SELECT created_at FROM runs WHERE config_hash = ?",
                (config_hash,),
            ).fetchone()
            created = row["created_at"] if row else now
            conn.execute(
                """
                INSERT INTO runs (
                    config_hash, name, status, run_dir, created_at, updated_at,
                    seed, manifest_hash, vocab_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(config_hash) DO UPDATE SET
                    name=excluded.name,
                    status=excluded.status,
                    run_dir=excluded.run_dir,
                    updated_at=excluded.updated_at,
                    seed=excluded.seed,
                    manifest_hash=excluded.manifest_hash,
                    vocab_hash=excluded.vocab_hash
                """,
                (
                    config_hash,
                    name,
                    status,
                    run_dir,
                    created,
                    now,
                    seed,
                    manifest_hash,
                    vocab_hash,
                ),
            )

    def get_run(self, config_hash: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE config_hash = ?",
                (config_hash,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def read_json(self, config_hash: str, name: str) -> dict[str, Any] | None:
        path = self.run_dir(config_hash) / name
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read_text(self, config_hash: str, name: str) -> str | None:
        path = self.run_dir(config_hash) / name
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def artifact_exists(self, config_hash: str, name: str) -> bool:
        return (self.run_dir(config_hash) / name).is_file()

    def try_read_parquet(
        self, config_hash: str, name: str
    ) -> pd.DataFrame | None:
        path = self.run_dir(config_hash) / name
        if not path.is_file():
            return None
        return pd.read_parquet(path)

    def write_status(self, config_hash: str, status: dict[str, Any]) -> Path:
        return self.write_json(config_hash, "status.json", status)

    def load_status(self, config_hash: str) -> dict[str, Any] | None:
        return self.read_json(config_hash, "status.json")


def list_prepared_datasets(datasets_root: Path | str) -> list[dict[str, Any]]:
    """List directories that contain ``manifest.parquet`` (prepared datasets)."""
    root = Path(datasets_root)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    # Direct children and one-level nested (e.g. tests/fixtures/mvp/manifest).
    candidates: list[Path] = []
    for path in sorted(root.rglob("manifest.parquet")):
        candidates.append(path.parent)
    seen: set[str] = set()
    for man_dir in candidates:
        key = str(man_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        hash_path = man_dir / "manifest_hash"
        digest = (
            hash_path.read_text(encoding="utf-8").strip()
            if hash_path.is_file()
            else None
        )
        out.append(
            {
                "id": man_dir.name,
                "path": str(man_dir.resolve()),
                "manifest_path": str((man_dir / "manifest.parquet").resolve()),
                "manifest_hash": digest,
            }
        )
    return out
