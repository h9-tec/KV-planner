"""Diagnose — why did this deployment OOM, and what's the cheapest fix?

Parses a vLLM / TGI config + optional nvidia-smi snapshot + optional OOM
log line. Rebuilds the memory waterfall, identifies the overflowing term,
and proposes three configs that would fit (smaller batch, smaller ctx,
quantization, or TP increase).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from kv_planner.core.explain.waterfall import MemoryWaterfall, build_waterfall
from kv_planner.domain import HardwareSpec, ModelConfig, PrecisionType


@dataclass(frozen=True)
class DiagnosisSuggestion:
    change: str
    rationale: str
    new_memory_gb: float
    fits: bool


@dataclass(frozen=True)
class Diagnosis:
    waterfall: MemoryWaterfall
    culprit_term: Optional[str]
    overflow_gb: float
    suggestions: list[DiagnosisSuggestion]


def diagnose(
    model: ModelConfig,
    hardware: HardwareSpec,
    batch_size: int,
    input_length: int,
    output_length: int,
    precision: PrecisionType = "fp16",
    tensor_parallel_size: Optional[int] = None,
) -> Diagnosis:
    waterfall = build_waterfall(
        model, hardware,
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
        precision=precision,
        tensor_parallel_size=tensor_parallel_size,
    )

    culprit = waterfall.overflow_term()
    overflow_gb = max(0.0, waterfall.total_gb - waterfall.budget_gb)

    suggestions: list[DiagnosisSuggestion] = []

    # --- Suggestion 1: quantize ----------------------------------------
    for new_prec in ("fp8", "int8", "int4"):
        if new_prec == precision:
            continue
        w2 = build_waterfall(
            model, hardware, batch_size, input_length, output_length,
            precision=new_prec,  # type: ignore[arg-type]
            tensor_parallel_size=tensor_parallel_size,
        )
        if w2.fits:
            suggestions.append(DiagnosisSuggestion(
                change=f"quantize weights to {new_prec}",
                rationale=f"halves (fp8/int8) or quarters (int4) weight + KV bytes",
                new_memory_gb=w2.total_gb,
                fits=True,
            ))
            break

    # --- Suggestion 2: halve batch -------------------------------------
    if batch_size > 1:
        new_b = max(1, batch_size // 2)
        w3 = build_waterfall(
            model, hardware, new_b, input_length, output_length,
            precision=precision, tensor_parallel_size=tensor_parallel_size,
        )
        suggestions.append(DiagnosisSuggestion(
            change=f"reduce batch_size {batch_size} → {new_b}",
            rationale="KV cache scales linearly with batch; halving is cheapest knob",
            new_memory_gb=w3.total_gb,
            fits=w3.fits,
        ))

    # --- Suggestion 3: shrink context ----------------------------------
    if input_length + output_length > 1024:
        new_len = max(512, (input_length + output_length) // 2)
        new_in = new_len // 2
        new_out = new_len - new_in
        w4 = build_waterfall(
            model, hardware, batch_size, new_in, new_out,
            precision=precision, tensor_parallel_size=tensor_parallel_size,
        )
        suggestions.append(DiagnosisSuggestion(
            change=f"reduce context {input_length + output_length} → {new_len}",
            rationale="KV cache also scales linearly with sequence length",
            new_memory_gb=w4.total_gb,
            fits=w4.fits,
        ))

    # --- Suggestion 4: bump TP ----------------------------------------
    current_tp = tensor_parallel_size or hardware.tensor_parallel_size
    if current_tp < 8 and hardware.num_gpus >= current_tp * 2:
        new_tp = current_tp * 2
        w5 = build_waterfall(
            model, hardware, batch_size, input_length, output_length,
            precision=precision, tensor_parallel_size=new_tp,
        )
        suggestions.append(DiagnosisSuggestion(
            change=f"increase tensor-parallel size {current_tp} → {new_tp}",
            rationale=f"sharding halves per-GPU weight+activation; TP scales linearly",
            new_memory_gb=w5.total_gb,
            fits=w5.fits,
        ))

    return Diagnosis(
        waterfall=waterfall,
        culprit_term=culprit.label if culprit else None,
        overflow_gb=overflow_gb,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# vLLM config parsing (best-effort)
# ---------------------------------------------------------------------------


_VLLM_FLAG_RE = re.compile(r"--([a-z-]+)(?:[= ]([^\s]+))?")


def parse_vllm_cmdline(cmdline: str) -> dict[str, str]:
    """Extract vLLM-style `--flag value` pairs from a command line."""
    out: dict[str, str] = {}
    for m in _VLLM_FLAG_RE.finditer(cmdline):
        k = m.group(1)
        v = m.group(2) or "true"
        out[k] = v
    return out
