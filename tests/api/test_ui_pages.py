"""Every UI page renders against a stored run (live API in a thread)."""

from __future__ import annotations

import importlib.util
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = [
    "Preview",
    "Comparison",
    "2x2 interaction",
    "Context utilisation",
    "Cost",
    "Quality",
    "Drift",
    "Report",
]


def _write_run():
    spec = importlib.util.spec_from_file_location(
        "_ksel_ui", ROOT / "tests" / "unit" / "test_k_selection.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._write_run


@pytest.fixture(scope="module")
def api_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    uvicorn = pytest.importorskip("uvicorn")
    from toklab.api.app import create_app
    from toklab.evaluation.report import render_report

    runs = tmp_path_factory.mktemp("runs")
    _write_run()(runs / "uirun", k_star=256)
    render_report(run_dir=runs / "uirun",
                  hypotheses_path=ROOT / "configs" / "hypotheses.yaml")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(runs_root=runs, datasets_root=runs),
                       host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.mark.parametrize("page", PAGES[1:])
def test_stored_result_pages_render(
    api_url: str, page: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("TOKLAB_API_URL", api_url)
    monkeypatch.setenv("TOKLAB_DEFAULT_RUN", "uirun")
    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=30)
    at.run()
    assert at.sidebar.text_input[1].value == "uirun"
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    if page == "Comparison":
        labels = {s.label: s.value for s in at.selectbox}
        assert labels == {"Eval set": "test", "K (adapted vocabulary size)": 256}
    if page in {"Quality", "Drift"}:
        assert any("not run" in w.value for w in at.warning)
    if page == "Cost":
        assert any("not run" in i.value for i in at.info)
