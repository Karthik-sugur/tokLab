"""Systems latency / memory harness (requires torch).

Only imported from the separate systems step (``toklab run systems``).
Intrinsic tokenization never imports this module.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

MIN_REPEATS = 10


@dataclass(frozen=True)
class SystemsTimingRow:
    condition: str
    seq_len: int
    device: str
    dtype: str
    n_warmup: int
    n_repeats: int
    latency_ms_median: float
    latency_ms_iqr: float
    peak_memory_bytes: int
    time_per_source_kb_ms: float
    quality_valid: bool
    notes: str


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _peak_memory_bytes(device: str) -> int:
    if device.startswith("cuda"):
        import torch

        return int(torch.cuda.max_memory_allocated())
    try:
        import resource

        # ru_maxrss is KB on Linux, bytes on macOS — normalize roughly.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys

        if sys.platform == "darwin":
            return int(rss)
        return int(rss * 1024)
    except Exception:  # noqa: BLE001
        return -1


def _iqr(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    s = sorted(xs)
    n = len(s)
    q1 = statistics.median(s[: n // 2])
    q3 = statistics.median(s[(n + 1) // 2 :])
    return float(q3 - q1)


def measure_forward_latency(
    model: Any,
    input_ids: Any,
    *,
    condition: str,
    seq_len: int,
    device: str,
    dtype: str,
    n_warmup: int = 5,
    n_repeats: int = 30,
    source_bytes: int = 0,
    quality_valid: bool = True,
    notes: str = "",
) -> SystemsTimingRow:
    """Warm-up then time ``n_repeats`` (>= 10) forward passes (batch size 1).

    Uses ``torch.cuda.synchronize`` around timers when ``device`` is CUDA.
    """
    import torch

    if n_repeats < MIN_REPEATS:
        raise ValueError(f"n_repeats must be >= {MIN_REPEATS}")
    if n_warmup < 1:
        raise ValueError("n_warmup must be >= 1")

    model.eval()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids=input_ids)
            _sync(device)

        times_ms: list[float] = []
        for _ in range(n_repeats):
            _sync(device)
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids)
            _sync(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    med = float(statistics.median(times_ms))
    peak = _peak_memory_bytes(device)
    if source_bytes > 0:
        kb = source_bytes / 1024.0
        tpkb = med / kb
    else:
        tpkb = float("nan")

    return SystemsTimingRow(
        condition=condition,
        seq_len=seq_len,
        device=device,
        dtype=dtype,
        n_warmup=n_warmup,
        n_repeats=n_repeats,
        latency_ms_median=med,
        latency_ms_iqr=_iqr(times_ms),
        peak_memory_bytes=peak,
        time_per_source_kb_ms=tpkb,
        quality_valid=quality_valid,
        notes=notes,
    )


@dataclass(frozen=True)
class TokenStream:
    """Token ids of an eval split (fixed file order) and the bytes they cover."""

    ids: list[int]
    file_ids: list[str]
    file_n_tokens: list[int]
    file_n_bytes: list[int]
    file_prefix_bytes: list[list[int]]

    def source_bytes(self, n_tokens: int) -> int:
        """Original UTF-8 bytes covered by the first ``n_tokens`` ids."""
        total = 0
        left = n_tokens
        for nt, nb, prefix in zip(
            self.file_n_tokens, self.file_n_bytes, self.file_prefix_bytes, strict=True
        ):
            if left >= nt:
                total += nb
                left -= nt
                continue
            if left > 0:
                total += prefix[left - 1]
            break
        return total


def build_token_stream(
    tokenizer: Any,
    files: Sequence[Any],
    texts: Mapping[str, str],
    *,
    max_tokens: int,
) -> TokenStream:
    """Concatenate encodings in ``files`` order until ``max_tokens`` ids exist."""
    ids: list[int] = []
    fids: list[str] = []
    ntoks: list[int] = []
    nbytes: list[int] = []
    prefixes: list[list[int]] = []
    for sf in files:
        if len(ids) >= max_tokens:
            break
        text = texts[sf.file_id]
        enc = tokenizer.encode(text)
        ids.extend(enc.ids)
        fids.append(sf.file_id)
        ntoks.append(len(enc.ids))
        nbytes.append(len(text.encode("utf-8")))
        prefixes.append([len(text[:end].encode("utf-8")) for _s, end in enc.spans])
    return TokenStream(ids, fids, ntoks, nbytes, prefixes)


def run_systems_benchmark(
    *,
    model_name: str,
    conditions: list[str],
    seq_lens: list[int],
    device: str | None = None,
    dtype: str = "float32",
    n_warmup: int = 5,
    n_repeats: int = 30,
    untrained_conditions: frozenset[str] | None = None,
    token_streams: Mapping[str, TokenStream] | None = None,
    embedding_rows: Mapping[str, int] | None = None,
    k: int | None = None,
) -> pd.DataFrame:
    """Time forward passes per condition / seq_len (batch size 1).

    ``token_streams`` supplies real input ids per condition (first ``seq_len``
    ids of the eval-split stream; ms per source KB uses the bytes they cover).
    Without it, random ids are used and the per-KB figure is a placeholder.

    ``embedding_rows`` resizes the embedding table for conditions whose
    adapted ids exceed the base table. Those rows are untrained: C3/C5 rows
    are tagged ``quality_valid=False`` regardless.
    """
    import torch
    from transformers import AutoModelForCausalLM

    untrained = untrained_conditions or frozenset({"C3", "C5"})
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, dtype)

    def load() -> Any:
        m = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True
        )
        m.to(device)
        m.eval()
        return m

    base_model = load()
    base_rows = int(getattr(base_model.config, "vocab_size", 0) or 0)

    rows: list[dict[str, Any]] = []
    for cond in conditions:
        qvalid = cond not in untrained
        want_rows = int((embedding_rows or {}).get(cond, 0))
        model = base_model
        resized = False
        if want_rows > base_rows:
            model = load()
            model.resize_token_embeddings(want_rows)
            model.to(device)
            resized = True
        stream = (token_streams or {}).get(cond)
        for t in seq_lens:
            notes = ["untrained_new_vocab_rows" if not qvalid else "base_weights"]
            if stream is not None:
                if len(stream.ids) < t:
                    raise ValueError(
                        f"{cond}: token stream has {len(stream.ids)} ids < seq_len {t}"
                    )
                input_ids = torch.tensor([stream.ids[:t]], device=device)
                source_bytes = stream.source_bytes(t)
                notes.append("real_inputs")
            else:
                input_ids = torch.randint(
                    0, min(1000, base_rows or 1000), (1, t), device=device
                )
                source_bytes = t
                notes.append("random_ids_bytes_placeholder")
            if resized:
                notes.append(f"embeddings_resized_to_{want_rows}")
            timing = measure_forward_latency(
                model,
                input_ids,
                condition=cond,
                seq_len=t,
                device=device,
                dtype=dtype,
                n_warmup=n_warmup,
                n_repeats=n_repeats,
                source_bytes=source_bytes,
                quality_valid=qvalid,
                notes=";".join(notes),
            )
            row = asdict(timing)
            row["k"] = k
            row["source_bytes"] = source_bytes
            row["torch_threads"] = int(torch.get_num_threads())
            rows.append(row)
        if resized:
            del model
    return pd.DataFrame(rows)


def run_systems_for_experiment_hash(
    config_hash: str, *, runs_root: Path | str = "runs"
) -> tuple[pd.DataFrame, Path]:
    """Separate pipeline step: systems benchmark for C2–C5 at the run's K*."""
    from toklab.evaluation.report import render_report
    from toklab.experiments.config import ExperimentConfig
    from toklab.experiments.runner import (
        _grid_tokenizers,
        _model_cfg,
        load_kstar_vocabs,
        open_experiment_corpus,
    )
    from toklab.experiments.store import RunStore
    from toklab.vocab.build import shared_base_from_cfg

    run_dir = Path(runs_root) / config_hash
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"no config.json in {run_dir}")
    config = ExperimentConfig.model_validate(
        json.loads(cfg_path.read_text(encoding="utf-8"))
    )
    sc = config.systems
    if not sc.enabled:
        raise ValueError("systems.enabled is false in this run's config")
    corpus = open_experiment_corpus(config)
    k_star, vocabs = load_kstar_vocabs(run_dir)
    model_cfg = _model_cfg(config)
    base = shared_base_from_cfg(model_cfg)
    toks = _grid_tokenizers(
        {**model_cfg, "base_tokenizer": base, "attach_leading_space": True}, vocabs
    )
    files = sorted(corpus.view(sc.input_split).files(), key=lambda f: f.file_id)
    texts = corpus.load_texts(files)
    max_t = max(sc.seq_lens)
    streams = {
        c: build_token_stream(toks[c], files, texts, max_tokens=max_t)
        for c in sc.conditions
    }
    rows_needed = {
        c: vocabs[mode].id_offset + len(vocabs[mode])
        for c, mode in (("C3", "native"), ("C5", "structural"))
    }
    df = run_systems_benchmark(
        model_name=config.model.model_name,
        conditions=list(sc.conditions),
        seq_lens=list(sc.seq_lens),
        device=sc.device,
        dtype=sc.dtype,
        n_warmup=sc.n_warmup,
        n_repeats=sc.n_repeats,
        token_streams=streams,
        embedding_rows=rows_needed,
        k=k_star,
    )
    df.insert(0, "input_split", sc.input_split)
    store = RunStore(runs_root)
    store.write_parquet(config_hash, "systems.parquet", df)
    status = store.load_status(config_hash) or {}
    status.setdefault("stages", {})["systems"] = "ok"
    store.write_status(config_hash, status)
    render_report(run_dir=run_dir, hypotheses_path=config.hypotheses_path)
    return df, run_dir
