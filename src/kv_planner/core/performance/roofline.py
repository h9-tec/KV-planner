"""
Roofline-model performance predictor for LLM inference.

Separates **prefill** (compute-bound, matmul-dense) from **decode**
(memory-bound, weight-streaming) and models each against a Williams-style
roofline ceiling defined by the hardware's peak TFLOPS and memory bandwidth.

Correctness sources — every formula below has a citation:

* FLOPs per token: kipply, "Transformer Inference Arithmetic" —
  https://kipp.ly/transformer-inference-arithmetic
  Chowdhery et al., PaLM paper — https://arxiv.org/abs/2204.02311
  EleutherAI cookbook ``calc_transformer_flops.py`` —
  https://github.com/EleutherAI/cookbook
* KV-cache bytes per token: vLLM PagedAttention design —
  https://docs.vllm.ai/en/latest/design/paged_attention/
* Ring-AllReduce cost: Patarasuk & Yuan, JPDC 2009 —
  https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf
  Megatron-LM, Shoeybi et al. — https://arxiv.org/abs/1909.08053
* Roofline: Williams, Waterman, Patterson CACM 2009 —
  https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf
* FlashAttention (decode activation streaming): Dao et al. —
  https://arxiv.org/abs/2205.14135
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kv_planner.domain import HardwareSpec, ModelConfig, PrecisionType, bytes_per_element


@dataclass(frozen=True)
class PerformanceMetrics:
    """Roofline prediction output."""

    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    prefill_tflops: float
    decode_tflops: float
    mfu: float
    mbu: float
    throughput_tokens_per_sec: float
    is_prefill_compute_bound: bool
    is_decode_memory_bound: bool
    arithmetic_intensity: float


@dataclass(frozen=True)
class RooflineConfig:
    """Tunable knobs, every default cited.

    * ``compute_efficiency``: observed MFU for well-tuned prefill on H100 is
      40–60 %; we take 50 % as a realistic default (Databricks MBU blog
      https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices,
      PaLM MFU ~46 % https://arxiv.org/abs/2204.02311).
    * ``memory_efficiency``: measured decode MBU on H100 Llama-2-70B B=1
      is ~60–85 %; 75 % is a reasonable mid-point (Databricks, same ref).
    * ``effective_nvlink_bandwidth_gb_s``: ring-AllReduce sustained on
      H100-HGX is ~400 GB/s of the advertised 900 GB/s per-direction peak
      (NCCL tests — https://github.com/NVIDIA/nccl-tests/issues/212).
    * ``allreduce_per_layer``: 2 AllReduces per layer in Megatron-style TP
      (post-attention row-parallel + post-MLP row-parallel).
    """

    compute_efficiency: float = 0.50
    memory_efficiency: float = 0.75
    allreduce_per_layer: int = 2

    def __post_init__(self) -> None:
        for name in ("compute_efficiency", "memory_efficiency"):
            val = getattr(self, name)
            if not 0.0 < val <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {val}")
        if self.allreduce_per_layer < 0:
            raise ValueError(
                f"allreduce_per_layer must be non-negative, got {self.allreduce_per_layer}"
            )


def flops_per_token_per_layer(model: ModelConfig, sequence_length: int) -> int:
    """FLOPs per token per layer, including the attention term.

    Breakdown (matmul of ``[m,k]·[k,n]`` = 2·m·k·n FLOPs):

    * Q projection: ``2·d·d``
    * K, V projections (GQA-aware): ``2·d·kv_hidden`` each
    * Attention scores Q·Kᵀ: ``2·d·s`` (one new query token vs s past tokens,
      GQA replicates K across the group so the query-head count applies)
    * Attention Aᵀ·V: ``2·d·s``
    * Output projection: ``2·d·d``
    * FFN: ``n_matmuls · 2·d·d_ff``  (n_matmuls = 3 for SwiGLU, else 2)

    ``sequence_length`` is the **current position** — pass the accumulated
    context length for a decode step, or the prompt length for prefill.

    For prefill of a whole prompt of length ``s``, sum from 1 to s; the
    closed-form is the per-token formula with ``s/2`` in the attention term
    (arithmetic mean of positions). We approximate by evaluating at ``s``
    and noting that for compute-bound prefill the attention term is
    negligible relative to the 24·d² matmul bulk.
    """
    d = model.hidden_size
    d_kv = model.kv_hidden_size
    d_ff = model._ffn_intermediate
    n_matmuls = model.ffn_num_matmuls

    q_o = 2 * (2 * d * d)
    kv_proj = 2 * (2 * d * d_kv)
    attn_qk = 2 * d * sequence_length
    attn_v = 2 * d * sequence_length
    ffn = n_matmuls * 2 * d * d_ff

    return q_o + kv_proj + attn_qk + attn_v + ffn


class RooflineAnalyzer:
    """
    Predicts prefill / decode latency, MFU, MBU, throughput.

    Prefill: matmul-dense, treated as compute-bound once arithmetic intensity
    exceeds the hardware ridge; otherwise memory-bound by weight-read time.

    Decode: always memory-bound by weight + growing KV cache. Latency per
    generated token is ``(weights + kv_for_current_length) / bandwidth``.
    """

    def __init__(
        self,
        compute_efficiency: float = RooflineConfig.compute_efficiency,
        memory_efficiency: float = RooflineConfig.memory_efficiency,
        config: Optional[RooflineConfig] = None,
    ) -> None:
        if config is None:
            config = RooflineConfig(
                compute_efficiency=compute_efficiency,
                memory_efficiency=memory_efficiency,
            )
        self._config = config
        # Keep attributes for backward-compatible introspection.
        self.compute_efficiency = config.compute_efficiency
        self.memory_efficiency = config.memory_efficiency

    # ---------- FLOPs ----------

    def calculate_flops_per_token(
        self, model: ModelConfig, sequence_length: int = 0
    ) -> int:
        """Total forward-pass FLOPs for a single token at position ``sequence_length``.

        ``sequence_length=0`` removes the attention term — useful for the
        bulk-matmul portion only. Most callers should pass the real context.
        """
        return model.num_layers * flops_per_token_per_layer(model, sequence_length)

    # ---------- Roofline geometry ----------

    def get_hardware_balance_point(
        self, hardware: HardwareSpec, precision: PrecisionType = "fp16"
    ) -> float:
        """Ridge point in FLOPs/byte — precision-aware."""
        peak_flops = hardware.peak_tflops_for(precision) * 1e12
        peak_bw = hardware.memory_bandwidth_gb_s * 1e9
        return peak_flops / peak_bw

    def calculate_arithmetic_intensity_prefill(
        self,
        model: ModelConfig,
        batch_size: int,
        sequence_length: int,
        precision: PrecisionType = "fp16",
    ) -> float:
        """AI for prefill: weights streamed once, work scales with batch · seq."""
        bpe = bytes_per_element(precision)
        total_flops = self.calculate_flops_per_token(model, sequence_length) * batch_size * sequence_length
        # In prefill, weights are read essentially once. Activation streaming
        # is modeled by FlashAttention, so we do not add the attention-matrix
        # bytes. Include a small activation term for Q/K/V/hidden-state
        # checkpoints which scale with batch·seq·d.
        param_bytes = model.total_params() * bpe
        activation_bytes = batch_size * sequence_length * model.hidden_size * bpe * 4
        return total_flops / (param_bytes + activation_bytes)

    def calculate_arithmetic_intensity_decode(
        self,
        model: ModelConfig,
        batch_size: int,
        sequence_length: int,
        precision: PrecisionType = "fp16",
    ) -> float:
        """AI for one decode step.

        Per step: ``batch·(2N + attention)`` FLOPs, with memory reads of
        ``weights + batch · seq · kv_bytes_per_token``.
        """
        bpe = bytes_per_element(precision)
        flops = self.calculate_flops_per_token(model, sequence_length) * batch_size
        param_bytes = model.total_params() * bpe
        kv_bytes = model.kv_cache_bytes_per_token(precision) * batch_size * sequence_length
        return flops / (param_bytes + kv_bytes)

    # ---------- Back-compat shim (tests / external callers) ----------

    def calculate_arithmetic_intensity(
        self,
        model: ModelConfig,
        batch_size: int,
        sequence_length: int,
        precision: PrecisionType = "fp16",
    ) -> float:
        """Default to prefill AI — the AI that made sense in pre-rewrite callers."""
        return self.calculate_arithmetic_intensity_prefill(
            model, batch_size, sequence_length, precision
        )

    # ---------- Prefill ----------

    def predict_prefill_latency(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        batch_size: int,
        input_length: int,
        precision: PrecisionType = "fp16",
    ) -> tuple[float, float, bool]:
        """Return (latency_ms, achieved_tflops, is_compute_bound)."""
        bpe = bytes_per_element(precision)
        total_flops = (
            self.calculate_flops_per_token(model, input_length) * batch_size * input_length
        )

        ai = self.calculate_arithmetic_intensity_prefill(
            model, batch_size, input_length, precision
        )
        ridge = self.get_hardware_balance_point(hardware, precision)
        is_compute_bound = ai > ridge

        compute_time = total_flops / (
            hardware.peak_tflops_for(precision) * 1e12 * self._config.compute_efficiency
        )

        # Memory-bound time: reading weights once + writing KV (~proportional
        # to batch·seq). KV write is a small fraction of weight read except
        # for tiny models, but include it for correctness.
        param_bytes = model.total_params() * bpe
        kv_written = (
            model.kv_cache_bytes_per_token(precision) * batch_size * input_length
        )
        bw_time = (param_bytes + kv_written) / (
            hardware.memory_bandwidth_gb_s * 1e9 * self._config.memory_efficiency
        )

        # Actual wallclock is the max (the other resource is idle waiting).
        latency_sec = max(compute_time, bw_time)
        latency_sec += self._allreduce_time(
            hardware, model, batch_size, input_length, precision
        )

        achieved_tflops = (total_flops / latency_sec) / 1e12 if latency_sec > 0 else 0.0
        return latency_sec * 1000.0, achieved_tflops, is_compute_bound

    # ---------- Decode ----------

    def predict_decode_latency(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        batch_size: int,
        sequence_length: int,
        precision: PrecisionType = "fp16",
    ) -> tuple[float, float, bool]:
        """Return (per-token latency_ms, achieved_tflops, is_memory_bound).

        ``sequence_length`` is the current context length at this decode step
        (i.e., input_length + number_of_tokens_already_generated). Callers
        that want an average across an output run should integrate across
        ``range(input_length, input_length + output_length)``.
        """
        bpe = bytes_per_element(precision)

        flops_per_step = self.calculate_flops_per_token(model, sequence_length) * batch_size

        param_bytes = model.total_params() * bpe
        # **Bug-fix anchor**: reads the FULL growing KV cache, not one token's worth.
        kv_bytes = model.kv_cache_bytes_per_token(precision) * batch_size * sequence_length
        total_bytes = param_bytes + kv_bytes

        bw_time = total_bytes / (
            hardware.memory_bandwidth_gb_s * 1e9 * self._config.memory_efficiency
        )
        # Decode is essentially always memory-bound in modern LLMs — only
        # cross the ridge with enormous batches.
        latency_sec = bw_time
        latency_sec += self._allreduce_time(
            hardware, model, batch_size, sequence_length=1, precision=precision
        )

        achieved_tflops = (flops_per_step / latency_sec) / 1e12 if latency_sec > 0 else 0.0
        return latency_sec * 1000.0, achieved_tflops, True

    # ---------- AllReduce (ring) ----------

    def _allreduce_time(
        self,
        hardware: HardwareSpec,
        model: ModelConfig,
        batch_size: int,
        sequence_length: int,
        precision: PrecisionType,
    ) -> float:
        """Total AllReduce time per layer-group, summed across all layers.

        Ring-AllReduce: ``T ≈ 2·(N-1)/N · M/B  +  2·(N-1)·α``, with two such
        operations per transformer layer (post-attention, post-MLP).
        Returns zero for TP=1.
        """
        tp = hardware.tensor_parallel_size
        if tp <= 1 or self._config.allreduce_per_layer == 0:
            return 0.0

        bpe = bytes_per_element(precision)
        # Message size = full hidden-state tensor replicated per node before
        # reduction = batch · seq · d · bpe bytes.
        message_bytes = batch_size * sequence_length * model.hidden_size * bpe
        bw = hardware.interconnect_bandwidth_gb_s * 1e9
        alpha = hardware.interconnect_latency_us * 1e-6

        per_allreduce = 2 * (tp - 1) / tp * message_bytes / bw + 2 * (tp - 1) * alpha
        total = per_allreduce * self._config.allreduce_per_layer * model.num_layers
        return total

    # ---------- Utilization ----------

    def calculate_mfu(
        self, achieved_tflops: float, hardware: HardwareSpec, precision: PrecisionType = "fp16"
    ) -> float:
        """MFU against the precision-matched peak."""
        peak = hardware.peak_tflops_for(precision)
        if peak == 0:
            return 0.0
        return min(1.0, achieved_tflops / peak)

    def calculate_mbu(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        batch_size: int,
        latency_sec: float,
        precision: PrecisionType = "fp16",
        sequence_length: int = 1,
    ) -> float:
        """MBU = achieved memory bandwidth / peak bandwidth.

        ``sequence_length`` should reflect the context read during the step
        being measured (decode: current position; prefill: input length).
        """
        if latency_sec <= 0 or hardware.memory_bandwidth_gb_s <= 0:
            return 0.0

        bpe = bytes_per_element(precision)
        param_bytes = model.total_params() * bpe
        kv_bytes = model.kv_cache_bytes_per_token(precision) * batch_size * sequence_length
        total_bytes = param_bytes + kv_bytes

        achieved_gb_s = (total_bytes / latency_sec) / 1e9
        return min(1.0, achieved_gb_s / hardware.memory_bandwidth_gb_s)

    # ---------- End-to-end ----------

    def predict_latency(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        batch_size: int,
        input_length: int,
        output_length: int,
        precision: PrecisionType = "fp16",
    ) -> PerformanceMetrics:
        """End-to-end: prefill + total decode time."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        if output_length < 0:
            raise ValueError(f"output_length must be non-negative, got {output_length}")

        prefill_ms, prefill_tflops, is_compute_bound = self.predict_prefill_latency(
            model, hardware, batch_size, input_length, precision
        )

        # Decode cost grows as KV cache grows. Integrate step-by-step, but
        # the cost is linear in seq_len so the average is taken at
        # ``input_length + output_length/2``.
        avg_seq_during_decode = input_length + max(0, output_length // 2)
        per_token_ms, decode_tflops, is_memory_bound = self.predict_decode_latency(
            model, hardware, batch_size, avg_seq_during_decode, precision
        )
        decode_ms = per_token_ms * output_length
        total_ms = prefill_ms + decode_ms

        mfu = self.calculate_mfu(prefill_tflops, hardware, precision)
        mbu = self.calculate_mbu(
            model, hardware, batch_size, per_token_ms / 1000.0, precision, avg_seq_during_decode
        )

        total_tokens = batch_size * (input_length + output_length)
        throughput = total_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

        ai = self.calculate_arithmetic_intensity_prefill(
            model, batch_size, input_length, precision
        )

        return PerformanceMetrics(
            prefill_latency_ms=prefill_ms,
            decode_latency_ms=decode_ms,
            total_latency_ms=total_ms,
            prefill_tflops=prefill_tflops,
            decode_tflops=decode_tflops,
            mfu=mfu,
            mbu=mbu,
            throughput_tokens_per_sec=throughput,
            is_prefill_compute_bound=is_compute_bound,
            is_decode_memory_bound=is_memory_bound,
            arithmetic_intensity=ai,
        )

    def predict_throughput(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        batch_size: int,
        input_length: int,
        output_length: int,
        precision: PrecisionType = "fp16",
    ) -> float:
        return self.predict_latency(
            model, hardware, batch_size, input_length, output_length, precision
        ).throughput_tokens_per_sec
