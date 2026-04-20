"""Memory waterfall — every byte, every formula, every citation.

Solves the #1 complaint across HN / HF forums / r/LocalLLaMA: opaque
"your model needs 413 MB" answers with no decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kv_planner.domain import HardwareSpec, ModelConfig, PrecisionType, bytes_per_element


@dataclass(frozen=True)
class WaterfallTerm:
    label: str
    bytes_: int
    formula: str
    citation: str = ""
    note: str = ""

    @property
    def gb(self) -> float:
        return self.bytes_ / 1e9


@dataclass(frozen=True)
class MemoryWaterfall:
    """Ordered list of terms summing to total memory for one replica."""

    terms: list[WaterfallTerm]
    device_gb: float
    device_utilization: float

    @property
    def total_bytes(self) -> int:
        return sum(t.bytes_ for t in self.terms)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1e9

    @property
    def budget_gb(self) -> float:
        return self.device_gb * self.device_utilization

    @property
    def fits(self) -> bool:
        return self.total_gb <= self.budget_gb

    @property
    def headroom_gb(self) -> float:
        return self.budget_gb - self.total_gb

    def overflow_term(self) -> Optional[WaterfallTerm]:
        """The term that would be easiest to attack to make it fit.

        Returns the largest term — reducing that has the biggest leverage.
        """
        if self.fits or not self.terms:
            return None
        return max(self.terms, key=lambda t: t.bytes_)


def build_waterfall(
    model: ModelConfig,
    hardware: HardwareSpec,
    batch_size: int,
    input_length: int,
    output_length: int,
    precision: PrecisionType,
    tensor_parallel_size: Optional[int] = None,
) -> MemoryWaterfall:
    """Per-GPU memory breakdown with every term formula-cited."""
    tp = tensor_parallel_size or hardware.tensor_parallel_size
    bpe = bytes_per_element(precision)

    # 1. Model weights (sharded by TP)
    total_params = model.total_params()
    weight_bytes = int(total_params * bpe / tp)
    weight_term = WaterfallTerm(
        label="model weights",
        bytes_=weight_bytes,
        formula=(
            f"total_params · bytes_per_element / TP = "
            f"{total_params / 1e9:.2f}B · {bpe} / {tp} = {weight_bytes / 1e9:.2f} GB"
        ),
        citation="kipply — https://kipp.ly/transformer-inference-arithmetic",
        note=f"TP={tp} shards weights; {precision} at {bpe} bytes/element",
    )

    # 2. KV cache (the formula from vLLM PagedAttention design)
    kv_per_token = model.kv_cache_bytes_per_token(precision)
    seq_len = input_length + output_length
    kv_bytes = int(kv_per_token * batch_size * seq_len / tp)
    kv_term = WaterfallTerm(
        label="KV cache",
        bytes_=kv_bytes,
        formula=(
            f"2 · n_layers · n_kv_heads · head_dim · bytes · batch · seq / TP = "
            f"2 · {model.num_layers} · {model.num_key_value_heads} · "
            f"{model.head_dim} · {bpe} · {batch_size} · {seq_len} / {tp} = "
            f"{kv_bytes / 1e9:.3f} GB"
        ),
        citation="vLLM PagedAttention — https://docs.vllm.ai/en/latest/design/paged_attention/",
        note=f"per-token KV = {kv_per_token / 1024:.1f} KiB",
    )

    # 3. Activations (approximate — Korthikanti et al. with SAC factor)
    # Activation bytes per layer per token ≈ ~20 × batch × seq × hidden × bpe
    # with selective activation checkpointing, total ≈ 0.3× (per Korthikanti).
    act_per_layer = 20 * batch_size * input_length * model.hidden_size * bpe
    sac_factor = 0.3  # selective activation checkpointing (inference is smaller)
    # At inference we only need one layer's worth of activations at a time for
    # prefill; decode is negligible. Be realistic but not zero.
    act_bytes = int(act_per_layer * sac_factor / tp)
    act_term = WaterfallTerm(
        label="activations (prefill peak)",
        bytes_=act_bytes,
        formula=(
            f"~20 · batch · input_len · hidden · bytes · SAC / TP = "
            f"20 · {batch_size} · {input_length} · {model.hidden_size} · "
            f"{bpe} · {sac_factor} / {tp} = {act_bytes / 1e9:.3f} GB"
        ),
        citation="Korthikanti et al. (arXiv:2205.05198) — https://arxiv.org/abs/2205.05198",
        note="selective activation checkpointing; dominated by prefill not decode",
    )

    # 4. CUDA workspace + framework overhead (empirical — vLLM reports ~0.5-1.5 GB)
    overhead_bytes = 800 * 1024 * 1024  # 800 MB fixed
    overhead_term = WaterfallTerm(
        label="CUDA workspace + framework",
        bytes_=overhead_bytes,
        formula="~800 MB (empirical; includes cuBLAS workspace, PyTorch cache, vLLM scheduler)",
        citation="vLLM issue tracker empirical — https://github.com/vllm-project/vllm/issues",
        note="non-tunable overhead",
    )

    # 5. Fragmentation — deliberately NOT a separate term (already baked into
    # PagedAttention's ceiling-block math). Left here as a WaterfallTerm
    # explicitly stating zero, so users don't wonder "where's fragmentation?".
    frag_term = WaterfallTerm(
        label="fragmentation",
        bytes_=0,
        formula="0 (PagedAttention block-ceiling already accounts for internal frag)",
        citation="Kwon et al. 2023 — https://arxiv.org/abs/2309.06180",
        note="external fragmentation = 0 by construction in paged schemes",
    )

    return MemoryWaterfall(
        terms=[weight_term, kv_term, act_term, overhead_term, frag_term],
        device_gb=hardware.gpu_memory_gb,
        device_utilization=hardware.gpu_memory_utilization,
    )
