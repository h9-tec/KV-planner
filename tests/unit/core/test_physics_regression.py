"""Pinned regression fixtures — sourced from the audit plan.

These tests anchor the physics engine against primary-source numbers
(HF config.json, NVIDIA datasheets, jax-ml scaling book, vLLM docs). If any
of them regress, the Roofline / memory / KV-cache math has drifted from
what the literature says.

No expensive setup; all numbers are derived from the test fixtures.
"""

from __future__ import annotations

import pytest

from kv_planner.core.memory import PagedMemoryCalculator
from kv_planner.core.performance import RooflineAnalyzer
from kv_planner.domain import HardwareSpec, ModelConfig
from kv_planner.infrastructure.hardware_db import GPUDatabase


# ---------------------------------------------------------------------------
# 1. KV-cache-per-token sanity
# ---------------------------------------------------------------------------


def test_llama3_8b_kv_cache_bytes_per_token_fp16(llama3_8b: ModelConfig) -> None:
    """Llama-3 8B fp16 KV cache = 131,072 bytes/token (128 KiB exact).

    Cross-check: 2 · 32 layers · 8 kv-heads · 128 head-dim · 2 bytes = 131,072.
    Reference: LMCache KV Cache Calculator (https://lmcache.ai/kv_cache_calculator.html)
    and vLLM PagedAttention design doc.
    """
    assert llama3_8b.kv_cache_bytes_per_token("fp16") == 131072


def test_kv_cache_scales_linearly_with_precision(llama3_8b: ModelConfig) -> None:
    """fp8 / int8 halves fp16; int4 quarters fp16."""
    fp16 = llama3_8b.kv_cache_bytes_per_token("fp16")
    assert llama3_8b.kv_cache_bytes_per_token("fp8") == fp16 // 2
    assert llama3_8b.kv_cache_bytes_per_token("int8") == fp16 // 2
    assert llama3_8b.kv_cache_bytes_per_token("int4") == fp16 // 4


# ---------------------------------------------------------------------------
# 2. Param count sanity
# ---------------------------------------------------------------------------


def test_llama3_8b_parameter_count_within_5pct(llama3_8b: ModelConfig) -> None:
    """Llama-3 8B total params ≈ 8.03e9 (Meta's published spec)."""
    params = llama3_8b.total_params()
    assert 7.6e9 < params < 8.5e9, f"Expected ~8.03B, got {params / 1e9:.2f}B"


def test_llama3_70b_parameter_count_within_5pct(llama3_70b: ModelConfig) -> None:
    """Llama-3 70B total params ≈ 70.6e9 (Meta's published spec)."""
    params = llama3_70b.total_params()
    assert 67e9 < params < 74e9, f"Expected ~70.6B, got {params / 1e9:.2f}B"


# ---------------------------------------------------------------------------
# 3. End-to-end Roofline sanity on H100
# ---------------------------------------------------------------------------


def test_h100_balance_point(h100_single: HardwareSpec) -> None:
    """Ridge = peak TFLOPS / BW ≈ 295 FLOPs/byte for H100 SXM5 @ fp16.

    Source: jax-ml scaling book, https://jax-ml.github.io/scaling-book/roofline/
    """
    ra = RooflineAnalyzer()
    bp = ra.get_hardware_balance_point(h100_single, "fp16")
    assert 280 < bp < 310, f"Expected ~295 FLOPs/byte, got {bp:.1f}"


def test_llama3_8b_h100_throughput_sanity(
    llama3_8b: ModelConfig, h100_single: HardwareSpec
) -> None:
    """Llama-3 8B on a single H100 SXM5 at B=32, in=2048, out=512, fp16.

    Published vLLM numbers for this config are in the low-thousands of
    tok/s. We accept any prediction in [500, 40_000] — a very wide window
    because the roofline is a theoretical ceiling not a measured throughput.
    The point of the assertion is to catch order-of-magnitude regressions
    (e.g., 10 tok/s or 1M tok/s would both be bugs).
    """
    ra = RooflineAnalyzer()
    m = ra.predict_latency(
        llama3_8b, h100_single, batch_size=32, input_length=2048, output_length=512
    )
    assert 500 < m.throughput_tokens_per_sec < 40_000


# ---------------------------------------------------------------------------
# 4. Paged memory fragmentation
# ---------------------------------------------------------------------------


def test_paged_fragmentation_exact_at_partial_block(
    llama3_8b: ModelConfig,
) -> None:
    """At seq_len=17, block_size=16 we expect 2 blocks, waste = 15/32 ≈ 46.9%.

    This is the *pure* internal fragmentation — the only fragmentation source
    in PagedAttention. If a `(1 + 0.04)` multiplier creeps back into the
    math, this number will shift and the assertion will fail.
    """
    calc = PagedMemoryCalculator(block_size=16)
    bk = calc.memory_breakdown(1, 17, llama3_8b, "fp16")
    assert bk["blocks"] == 2
    assert bk["tokens_allocated"] == 32
    assert bk["tokens_wasted"] == 15
    assert abs(bk["fragmentation_pct"] - 15 / 32 * 100) < 0.01


def test_paged_fragmentation_zero_at_exact_multiple(
    llama3_8b: ModelConfig,
) -> None:
    """Exact multiple of block_size → fragmentation_pct = 0."""
    calc = PagedMemoryCalculator(block_size=16)
    bk = calc.memory_breakdown(1, 2048, llama3_8b, "fp16")
    assert bk["fragmentation_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Decode latency grows with context (the #1 fixed bug)
# ---------------------------------------------------------------------------


def test_decode_latency_scales_with_kv_cache(
    llama3_8b: ModelConfig, h100_single: HardwareSpec
) -> None:
    """Per-step decode latency must increase with context length because the
    KV cache being read grows. The pre-fix code ignored this entirely."""
    ra = RooflineAnalyzer()
    short, _, _ = ra.predict_decode_latency(
        llama3_8b, h100_single, batch_size=32, sequence_length=128
    )
    long, _, _ = ra.predict_decode_latency(
        llama3_8b, h100_single, batch_size=32, sequence_length=32768
    )
    # Very long context must be at least a few × slower.
    assert long > short * 1.5


# ---------------------------------------------------------------------------
# 6. TP AllReduce sanity
# ---------------------------------------------------------------------------


def test_tp_allreduce_sublinear(llama3_8b: ModelConfig) -> None:
    """Comm overhead for TP=8 must be less than 8× the TP=2 number.

    Pre-fix the formula scaled as O(TP); correct ring-AllReduce scales as
    (TP-1)/TP which asymptotes. TP=8 / TP=2 cost ratio should be < 4.
    """
    ra = RooflineAnalyzer()
    # Call the internal helper directly for a focused unit test.
    h100_tp2 = GPUDatabase.to_hardware_spec(
        "H100-SXM-80GB", num_gpus=2, tensor_parallel_size=2
    )
    h100_tp8 = GPUDatabase.to_hardware_spec(
        "H100-SXM-80GB", num_gpus=8, tensor_parallel_size=8
    )
    t2 = ra._allreduce_time(h100_tp2, llama3_8b, 16, 2048, "fp16")
    t8 = ra._allreduce_time(h100_tp8, llama3_8b, 16, 2048, "fp16")
    assert t8 > t2, "more peers should cost more"
    assert t8 / t2 < 4.0, (
        f"ring-AllReduce scaling regression: TP=8 / TP=2 = {t8/t2:.2f}, "
        f"should be sub-linear (< 4×)"
    )


# ---------------------------------------------------------------------------
# 7. Super-linear scaling method is GONE
# ---------------------------------------------------------------------------


def test_super_linear_scaling_method_removed(h100_single: HardwareSpec) -> None:
    """The physically-impossible `kv_cache_super_linear_scaling_factor` must
    not exist any more — it was returning 13.9× for TP=2 and silently causing
    OOMs in production.
    """
    assert not hasattr(h100_single, "kv_cache_super_linear_scaling_factor")
