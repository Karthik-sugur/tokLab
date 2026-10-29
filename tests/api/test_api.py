"""TestClient coverage for TokLab API endpoints and error codes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from toklab.api.app import MAX_CONFIG_BYTES, create_app
from toklab.experiments.config import ExperimentConfig
from toklab.experiments.preview import MAX_PREVIEW_CHARS
from toklab.experiments.store import RunStore, compute_config_hash


def _mini_cfg(repos: Path, manifest: Path, runs_root: Path) -> dict[str, Any]:
    return {
        "name": "api_test",
        "seed": 0,
        "model": {"model_name": "Qwen/Qwen2.5-Coder-0.5B"},
        "data": {
            "repos_dir": str(repos),
            "manifest_path": str(manifest),
        },
        "vocab": {"k_values": [0]},
        "stages": {
            "fidelity": False,
            "intrinsic": False,
            "factorial": False,
            "context": False,
            "vocab_cost": False,
            "systems": False,
            "report": False,
        },
        "runs_root": str(runs_root),
    }


@pytest.fixture
def mvp_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    repos = root / "tests" / "fixtures" / "mvp" / "repos"
    manifest = root / "tests" / "fixtures" / "mvp" / "manifest" / "manifest.parquet"
    assert manifest.is_file()
    return repos, manifest


@pytest.fixture
def client(tmp_path: Path, mvp_paths: tuple[Path, Path]) -> TestClient:
    runs = tmp_path / "runs"
    fixtures = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

    def fake_runner(config: ExperimentConfig) -> dict[str, Any]:
        store = RunStore(config.runs_root)
        chash = compute_config_hash(config.canonical_dict())
        store.ensure_run_dir(chash)
        # Write artifacts BEFORE marking complete (tests poll on state).
        pd.DataFrame(
            [
                {
                    "condition": "C2",
                    "metric": "tokens_per_kb",
                    "estimate": 100.0,
                    "ci_low": 90.0,
                    "ci_high": 110.0,
                    "n_repos": 1,
                    "n_boot": 10,
                    "seed": 0,
                    "alpha": 0.05,
                }
            ]
        ).to_parquet(
            store.run_dir(chash) / "token_efficiency_ci.parquet", index=False
        )
        pd.DataFrame(
            [
                {
                    "k": 0,
                    "metric": "tokens_per_kb",
                    "c2": 1.0,
                    "c3": 1.0,
                    "c4": 1.0,
                    "c5": 1.0,
                    "c2_low": 1.0,
                    "c2_high": 1.0,
                    "c3_low": 1.0,
                    "c3_high": 1.0,
                    "c4_low": 1.0,
                    "c4_high": 1.0,
                    "c5_low": 1.0,
                    "c5_high": 1.0,
                    "effect_s": 0.0,
                    "effect_d": 0.0,
                    "interaction": 0.0,
                    "interaction_low": 0.0,
                    "interaction_high": 0.0,
                    "r_s": 0.0,
                    "r_d": 0.0,
                    "independence_expectation": 0.0,
                    "combined_beats_either": False,
                    "decision_rule": "x",
                }
            ]
        ).to_parquet(
            store.run_dir(chash) / "factorial_effects.parquet", index=False
        )
        pd.DataFrame(
            [{"proportion": 0, "variant": "C2", "token_reduction": 0.0}]
        ).to_parquet(store.run_dir(chash) / "drift_metrics.parquet", index=False)
        (store.run_dir(chash) / "report.md").write_text("# ok\n", encoding="utf-8")
        pd.DataFrame(
            [
                {
                    "condition": "C2",
                    "n_files": 1,
                    "n_roundtrip_ok": 1,
                    "n_roundtrip_fail": 0,
                    "valid": True,
                    "status": "OK",
                }
            ]
        ).to_parquet(store.run_dir(chash) / "fidelity.parquet", index=False)
        status = {
            "config_hash": chash,
            "name": config.name,
            "state": "complete",
            "stages": {"intrinsic": "ok"},
            "run_dir": str(store.run_dir(chash).resolve()),
        }
        store.write_status(chash, status)
        store.upsert_run(
            config_hash=chash, name=config.name, status="complete", seed=0
        )
        return status

    app = create_app(
        runs_root=runs,
        datasets_root=fixtures,
        runner=fake_runner,
    )
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_datasets_list(client: TestClient) -> None:
    r = client.get("/datasets")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert any("manifest" in d["path"] for d in body)


def test_tokenize_preview_ok(client: TestClient) -> None:
    r = client.post(
        "/tokenize/preview",
        json={"text": "x = 1\n", "conditions": ["C6"]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["conditions"][0]["condition"] == "C6"
    assert data["conditions"][0]["n_tokens"] == data["n_bytes"]


def test_tokenize_preview_413(client: TestClient) -> None:
    r = client.post(
        "/tokenize/preview",
        json={"text": "a" * (MAX_PREVIEW_CHARS + 1), "conditions": ["C6"]},
    )
    assert r.status_code == 413


def test_tokenize_preview_422_empty_conditions(client: TestClient) -> None:
    r = client.post(
        "/tokenize/preview",
        json={"text": "hi", "conditions": []},
    )
    assert r.status_code == 422


def test_create_experiment_and_get(
    client: TestClient, mvp_paths: tuple[Path, Path], tmp_path: Path
) -> None:
    repos, manifest = mvp_paths
    runs = Path(client.app.state.store.runs_root)
    cfg = _mini_cfg(repos, manifest, runs)
    r = client.post("/experiments", json={"config": cfg})
    assert r.status_code == 200
    body = r.json()
    exp_id = body["id"]
    assert body["status"] == "created"
    g = client.get(f"/experiments/{exp_id}")
    assert g.status_code == 200
    assert g.json()["id"] == exp_id


def test_create_experiment_422(client: TestClient) -> None:
    r = client.post("/experiments", json={"config": {"name": "x"}})
    assert r.status_code == 422


def test_create_experiment_413(client: TestClient) -> None:
    huge = {"data": {"repos_dir": "/x", "manifest_path": "/y"}, "pad": "z" * (
        MAX_CONFIG_BYTES
    )}
    r = client.post("/experiments", json={"config": huge})
    assert r.status_code == 413


def test_get_experiment_404(client: TestClient) -> None:
    r = client.get("/experiments/" + "0" * 64)
    assert r.status_code == 404


def test_run_and_results(
    client: TestClient, mvp_paths: tuple[Path, Path]
) -> None:
    repos, manifest = mvp_paths
    runs = Path(client.app.state.store.runs_root)
    cfg = _mini_cfg(repos, manifest, runs)
    exp_id = client.post("/experiments", json={"config": cfg}).json()["id"]
    # Results before run → 409
    early = client.get(
        f"/experiments/{exp_id}/results",
        params={"table": "token_efficiency_ci"},
    )
    assert early.status_code == 409
    started = client.post(f"/experiments/{exp_id}/run")
    assert started.status_code == 200
    # Wait for fake runner thread
    import time

    for _ in range(100):
        st = client.get(f"/experiments/{exp_id}").json()
        if st.get("state") == "complete" or st.get("status") == "complete":
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"run did not complete: {st}")
    store: RunStore = client.app.state.store
    assert store.artifact_exists(exp_id, "token_efficiency_ci.parquet")
    res = client.get(
        f"/experiments/{exp_id}/results",
        params={"table": "token_efficiency_ci", "condition": "C2"},
    )
    assert res.status_code == 200
    assert res.json()["rows"]
    fac = client.get(f"/experiments/{exp_id}/factorial")
    assert fac.status_code == 200
    drift = client.get(f"/experiments/{exp_id}/drift")
    assert drift.status_code == 200
    report = client.get(f"/experiments/{exp_id}/report", params={"format": "md"})
    assert report.status_code == 200
    assert "# ok" in report.text
    html = client.get(f"/experiments/{exp_id}/report", params={"format": "html"})
    assert html.status_code == 200
    assert "text/html" in html.headers["content-type"]


def test_run_409_busy(
    client: TestClient, mvp_paths: tuple[Path, Path]
) -> None:
    repos, manifest = mvp_paths
    runs = Path(client.app.state.store.runs_root)
    cfg = _mini_cfg(repos, manifest, runs)
    # Slow runner to hold the lock
    import threading
    import time

    from toklab.experiments.config import ExperimentConfig as EC

    release = threading.Event()

    def slow_runner(config: EC) -> dict[str, Any]:
        release.wait(timeout=5)
        return {
            "config_hash": compute_config_hash(config.canonical_dict()),
            "state": "complete",
            "name": config.name,
            "stages": {},
        }

    app = create_app(
        runs_root=runs,
        datasets_root=Path(__file__).resolve().parents[2] / "tests" / "fixtures",
        runner=slow_runner,
    )
    c = TestClient(app)
    exp_id = c.post("/experiments", json={"config": cfg}).json()["id"]
    assert c.post(f"/experiments/{exp_id}/run").status_code == 200
    busy = c.post(f"/experiments/{exp_id}/run")
    assert busy.status_code == 409
    release.set()
    time.sleep(0.1)


def test_factorial_409_missing_cell(
    client: TestClient, mvp_paths: tuple[Path, Path]
) -> None:
    repos, manifest = mvp_paths
    runs = Path(client.app.state.store.runs_root)
    cfg = _mini_cfg(repos, manifest, runs)
    exp_id = client.post("/experiments", json={"config": cfg}).json()["id"]
    store: RunStore = client.app.state.store
    # Incomplete factorial (missing c5 column values → write without c5)
    pd.DataFrame([{"k": 0, "c2": 1.0, "c3": 1.0, "c4": 1.0}]).to_parquet(
        store.run_dir(exp_id) / "factorial_effects.parquet", index=False
    )
    r = client.get(f"/experiments/{exp_id}/factorial")
    assert r.status_code == 409


def test_drift_409_not_ready(
    client: TestClient, mvp_paths: tuple[Path, Path]
) -> None:
    repos, manifest = mvp_paths
    runs = Path(client.app.state.store.runs_root)
    cfg = _mini_cfg(repos, manifest, runs)
    exp_id = client.post("/experiments", json={"config": cfg}).json()["id"]
    r = client.get(f"/experiments/{exp_id}/drift")
    assert r.status_code == 409


def test_report_409_not_ready(
    client: TestClient, mvp_paths: tuple[Path, Path]
) -> None:
    repos, manifest = mvp_paths
    runs = Path(client.app.state.store.runs_root)
    cfg = _mini_cfg(repos, manifest, runs)
    exp_id = client.post("/experiments", json={"config": cfg}).json()["id"]
    r = client.get(f"/experiments/{exp_id}/report")
    assert r.status_code == 409


def test_schema_tokenize_response(client: TestClient) -> None:
    r = client.post(
        "/tokenize/preview",
        json={"text": "ab", "conditions": ["C6"]},
    )
    data = r.json()
    assert set(data.keys()) >= {"text", "n_chars", "n_bytes", "conditions"}
    cond = data["conditions"][0]
    assert set(cond.keys()) >= {"condition", "n_tokens", "ids", "spans"}
    assert cond["spans"][0].keys() >= {"start", "end"}


def test_schema_experiment_status(
    client: TestClient, mvp_paths: tuple[Path, Path]
) -> None:
    repos, manifest = mvp_paths
    runs = Path(client.app.state.store.runs_root)
    cfg = _mini_cfg(repos, manifest, runs)
    exp_id = client.post("/experiments", json={"config": cfg}).json()["id"]
    data = client.get(f"/experiments/{exp_id}").json()
    assert set(data.keys()) >= {
        "id",
        "name",
        "status",
        "state",
        "stages",
        "run_dir",
    }
