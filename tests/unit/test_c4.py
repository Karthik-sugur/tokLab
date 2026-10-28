"""Tests for structure-aware C4: segmenters, composed tokenizer, roles."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from toklab.core.interfaces import assert_lossless_segments
from toklab.parsing.roles import assign_ident_roles
from toklab.parsing.segmenters import NativeSegmenter, TreeSitterSegmenter
from toklab.tokenizers.composed import ComposedTokenizer, merge_leading_space_gaps
from toklab.tokenizers.native_hf import NativeHFTokenizer
from toklab.tokenizers.registry import get_tokenizer

FIXTURE_FILES = [
    "x = 1\n",
    "def foo(a, b):\n    return a + b\n",
    "class Widget:\n    def run(self):\n        return self.x\n",
    "print(foo())\n",
    "# comment\nvalue = 'café'\n",
    "for i in range(3):\n    pass\n",
]

PARSE_ERROR_FIXTURES = [
    "def foo(\n",  # truncated
    "class A:\n    def\n",
    "x = (\n",
    "if True\n    pass\n",  # missing colon — may still parse oddly
]


@pytest.fixture(scope="module")
def code_base() -> NativeHFTokenizer:
    return NativeHFTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", name="C2"
    )


@pytest.fixture(scope="module")
def ts_seg() -> TreeSitterSegmenter:
    return TreeSitterSegmenter()


def test_tree_sitter_lossless_on_fixtures(ts_seg: TreeSitterSegmenter) -> None:
    for text in FIXTURE_FILES:
        segs = ts_seg.segment(text)
        assert_lossless_segments(text, segs)
        segs_r = assign_ident_roles(text, segs)
        assert_lossless_segments(text, segs_r)


def test_native_segmenter_lossless(code_base: NativeHFTokenizer) -> None:
    seg = NativeSegmenter(code_base._tok)
    for text in FIXTURE_FILES:
        assert_lossless_segments(text, seg.segment(text))


def test_roles_on_defs_and_calls(ts_seg: TreeSitterSegmenter) -> None:
    text = "def foo(x):\n    return bar(x)\n"
    segs = assign_ident_roles(text, ts_seg.segment(text))
    by_text = {text[s.start : s.end]: s.role for s in segs if s.kind == "ident"}
    assert by_text.get("foo") == "def_name"
    assert by_text.get("x") in {"parameter", "other"}  # param and use
    assert by_text.get("bar") == "call"


@pytest.mark.parametrize("attach", [True, False])
def test_c4_roundtrip_both_leading_space_settings(
    code_base: NativeHFTokenizer,
    ts_seg: TreeSitterSegmenter,
    attach: bool,
) -> None:
    tok = ComposedTokenizer(
        ts_seg,
        code_base,
        attach_leading_space=attach,
        name="C4",
    )
    for text in FIXTURE_FILES:
        assert tok.decode(tok.encode(text).ids) == text


def test_parse_error_fixtures_still_roundtrip(
    code_base: NativeHFTokenizer,
    ts_seg: TreeSitterSegmenter,
) -> None:
    tok = ComposedTokenizer(ts_seg, code_base, attach_leading_space=True, name="C4")
    for text in PARSE_ERROR_FIXTURES:
        segs = ts_seg.segment(text)
        assert_lossless_segments(text, segs)
        assert tok.decode(tok.encode(text).ids) == text


_C4: ComposedTokenizer | None = None


def _c4() -> ComposedTokenizer:
    global _C4
    if _C4 is None:
        _C4 = get_tokenizer("C4", {"attach_leading_space": True})
    return _C4


@given(
    st.sampled_from(
        FIXTURE_FILES
        + [
            "a=b\n",
            "def g():\n  return 0\n",
            "import os\n",
            "",
            "x\n",
        ]
    )
)
@settings(max_examples=30, deadline=None)
def test_property_roundtrip_snippets(text: str) -> None:
    tok = _c4()
    assert tok.decode(tok.encode(text).ids) == text


def test_merge_leading_space_gaps() -> None:
    from toklab.core.types import Segment

    text = "x = foo"
    segs = [
        Segment(0, 1, "ident", "other"),
        Segment(1, 2, "gap", "other"),
        Segment(2, 3, "op", "other"),
        Segment(3, 4, "gap", "other"),
        Segment(4, 7, "ident", "other"),
    ]
    merged = merge_leading_space_gaps(segs, text)
    pieces = [text[s.start : s.end] for s in merged]
    assert pieces == ["x", " =", " foo"]


def test_registry_c4_default_attach_true(code_base: NativeHFTokenizer) -> None:
    tok = get_tokenizer("C4", {})
    assert tok.name == "C4"
    assert tok.attach_leading_space is True
    assert tok.decode(tok.encode("x = foo\n").ids) == "x = foo\n"


def test_report_c4_vs_c2_token_counts(
    code_base: NativeHFTokenizer,
    ts_seg: TreeSitterSegmenter,
    capsys: pytest.CaptureFixture[str],
) -> None:
    c2 = code_base
    c4 = ComposedTokenizer(ts_seg, code_base, attach_leading_space=True, name="C4")
    print("C4 vs C2 token counts on fixtures (informational, no direction assert):")
    for text in FIXTURE_FILES:
        n2 = len(c2.encode(text).ids)
        n4 = len(c4.encode(text).ids)
        preview = text.replace("\n", "\\n")[:40]
        print(f"  C2={n2:4d}  C4={n4:4d}  {preview!r}")
    captured = capsys.readouterr()
    assert "C4 vs C2" in captured.out
    print(captured.out)


def test_compare_cli_path(tmp_path: Path, code_base: NativeHFTokenizer) -> None:
    from toklab.data.splits import build_dataset
    from toklab.evaluation.compare import compare_conditions

    repos = tmp_path / "repos"
    for name, body in [
        ("repo_a", "def a():\n    return 1\n"),
        ("repo_b", "def b():\n    return 2\n"),
        ("repo_c", "class C:\n    pass\n"),
        ("repo_d", "x = 3\n"),
        ("repo_e", "print(4)\n"),
    ]:
        p = repos / name / "mod.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    man = tmp_path / "manifest"
    build_dataset(repos, man, seed=0)
    table = compare_conditions(
        manifest=man / "manifest.parquet",
        repos_dir=repos,
        conditions=["C2", "C4"],
        split="test",
    )
    assert "C2" in table and "C4" in table
    print(table)
