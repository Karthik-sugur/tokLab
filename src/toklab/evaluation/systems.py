"""Systems latency / memory harness (requires torch).

Only imported from experiment stages that opt into systems metrics.
Intrinsic tokenization never imports this module.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


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
    """Warm-up then time >=30 forward passes (batch size 1).

    Uses ``torch.cuda.synchronize`` around timers when ``device`` is CUDA.
    """
    import torch

    if n_repeats < 30:
        raise ValueError("n_repeats must be >= 30")
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
) -> pd.DataFrame:
    """Load base causal LM once; time forward passes per condition/seq_len.

    C3/C5 (domain vocab) use untrained new rows conceptually — rows are tagged
    ``quality_valid=False`` for those conditions even when the underlying
    weights are the base model (systems latency approx. without resized rows).
    """
    import torch
    from transformers import AutoModelForCausalLM

    untrained = untrained_conditions or frozenset({"C3", "C5"})
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    for cond in conditions:
        qvalid = cond not in untrained
        note = (
            "untrained_new_vocab_rows"
            if not qvalid
            else "base_weights"
        )
        for t in seq_lens:
            input_ids = torch.randint(
                0,
                min(1000, getattr(model.config, "vocab_size", 1000)),
                (1, t),
                device=device,
            )
            # Approximate source bytes as UTF-8 of T ASCII chars for KB rate.
            source_bytes = t  # 1 byte/char placeholder for synthetic ids
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
                notes=note,
            )
            rows.append(asdict(timing))
    return pd.DataFrame(rows)
