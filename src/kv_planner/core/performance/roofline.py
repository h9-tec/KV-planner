"""
Roofline analysis for LLM inference performance prediction.

Implements the roofline model to predict:
- Prefill latency (compute-bound)
- Decode latency (memory-bound)
- MFU (Model FLOPS Utilization)
- MBU (Model Bandwidth Utilization)
- Throughput (tokens/second)

Based on research from:
- "LLM Inference Unveiled: Survey and Roofline Model Insights" (arXiv 2024)
- "Transformer Inference Arithmetic" (kipply's blog)
- vLLM performance analysis
"""

from dataclasses import dataclass
from typing import Literal, Optional

from kv_planner.domain import HardwareSpec, ModelConfig, InsufficientMemoryError


PrecisionType = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Performance prediction results from roofline analysis.

    Attributes:
        prefill_latency_ms: Time to process input prompt (milliseconds)
        decode_latency_ms: Time per output token generation (milliseconds)
        total_latency_ms: Total inference time for given sequence
        prefill_tflops: Achieved TFLOPS during prefill
        decode_tflops: Achieved TFLOPS during decode
        mfu: Model FLOPS Utilization (0.0-1.0)
        mbu: Model Bandwidth Utilization (0.0-1.0)
        throughput_tokens_per_sec: Total throughput (tokens/second)
        is_prefill_compute_bound: Whether prefill is compute-bound
        is_decode_memory_bound: Whether decode is memory-bound
        arithmetic_intensity: Operations per byte ratio
    """

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


class RooflineAnalyzer:
    """
    Roofline model analyzer for LLM inference performance prediction.

    The roofline model determines performance based on:
    1. Arithmetic Intensity (OPs/byte) vs Hardware Balance Point
    2. Prefill: Usually compute-bound (large matrix ops)
    3. Decode: Usually memory-bound (loading weights for single token)

    Example:
        >>> analyzer = RooflineAnalyzer()
        >>> metrics = analyzer.predict_latency(
        ...     model=llama3_8b,
        ...     hardware=rtx_5090,
        ...     batch_size=32,
        ...     input_length=2048,
        ...     output_length=512,
        ...     precision="fp16"
        ... )
        >>> print(f"Prefill: {metrics.prefill_latency_ms:.1f}ms")
        >>> print(f"Decode: {metrics.decode_latency_ms:.1f}ms")
        >>> print(f"Throughput: {metrics.throughput_tokens_per_sec:.0f} tokens/s")
    """

    # Constants for transformer FLOPs calculation
    FLOPS_PER_TOKEN_MULTIPLIER = 24  # For full forward pass (QKV + attn + MLP)

    # Communication overhead per layer (tensor parallelism)
    COMMUNICATION_LATENCY_US = 8.0  # microseconds per layer

    # Efficiency factors (conservative estimates for production systems)
    COMPUTE_EFFICIENCY = 0.65  # 65% of peak TFLOPS (accounts for kernel overhead)
    MEMORY_EFFICIENCY = 0.80  # 80% of peak bandwidth (realistic for inference)

    def __init__(
        self,
        compute_efficiency: float = COMPUTE_EFFICIENCY,
        memory_efficiency: float = MEMORY_EFFICIENCY,
    ) -> None:
        """
        Initialize roofline analyzer.

        Args:
            compute_efficiency: Fraction of peak TFLOPS achievable (0.0-1.0)
            memory_efficiency: Fraction of peak bandwidth achievable (0.0-1.0)
        """
        if not 0.0 < compute_efficiency <= 1.0:
            raise ValueError(
                f"compute_efficiency must be in (0, 1], got {compute_efficiency}"
            )
        if not 0.0 < memory_efficiency <= 1.0:
            raise ValueError(
                f"memory_efficiency must be in (0, 1], got {memory_efficiency}"
            )

        self.compute_efficiency = compute_efficiency
        self.memory_efficiency = memory_efficiency

    def calculate_flops_per_token(
        self,
        model: ModelConfig,
    ) -> int:
        """
        Calculate FLOPs required for one forward pass per token.

        Formula: F = n_layers × 24 × d_model²

        Breakdown per layer:
        - QKV projection: 2 × 3 × d_model² = 6 × d_model²
        - Attention output: 2 × d_model² = 2 × d_model²
        - MLP (feed-forward): 2 × 4 × d_model² × 2 = 16 × d_model²
        - Total: 24 × d_model²

        Args:
            model: Model configuration

        Returns:
            FLOPs per token (integer)
        """
        d_model = model.hidden_size
        n_layers = model.num_layers

        flops_per_layer = self.FLOPS_PER_TOKEN_MULTIPLIER * d_model * d_model
        total_flops = n_layers * flops_per_layer

        return total_flops

    def calculate_arithmetic_intensity(
        self,
        model: ModelConfig,
        batch_size: int,
        sequence_length: int,
        precision: PrecisionType = "fp16",
    ) -> float:
        """
        Calculate arithmetic intensity (FLOPs/byte).

        Arithmetic intensity determines if workload is:
        - Memory-bound: AI < hardware balance point
        - Compute-bound: AI > hardware balance point

        Args:
            model: Model configuration
            batch_size: Number of sequences
            sequence_length: Tokens per sequence
            precision: Numerical precision

        Returns:
            Arithmetic intensity (FLOPs/byte)
        """
        # FLOPs for processing sequence
        flops_per_token = self.calculate_flops_per_token(model)
        total_flops = flops_per_token * batch_size * sequence_length

        # Bytes transferred (model weights + activations)
        precision_bytes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "fp8": 1,
            "int8": 1,
            "int4": 0.5,
        }
        bytes_per_param = precision_bytes[precision]

        # Model parameters loaded from memory
        param_bytes = model.total_params() * bytes_per_param

        # Activation memory (approximate, depends on batch size)
        # Activations: batch × seq × hidden_size per layer
        activation_bytes = (
            batch_size * sequence_length * model.hidden_size * bytes_per_param * 2
        )  # Factor of 2 for intermediate activations

        total_bytes = param_bytes + activation_bytes

        # Arithmetic intensity
        arithmetic_intensity = total_flops / total_bytes

        return arithmetic_intensity

    def get_hardware_balance_point(self, hardware: HardwareSpec) -> float:
        """
        Calculate hardware's arithmetic intensity balance point.

        Balance point = Peak TFLOPS / Peak Bandwidth (GB/s)

        Workloads with AI above this are compute-bound.
        Workloads with AI below this are memory-bound.

        Args:
            hardware: Hardware specifications

        Returns:
            Balance point (FLOPs/byte)
        """
        # Convert TFLOPS to FLOPS and GB/s to bytes/s
        peak_flops = hardware.peak_tflops * 1e12
        peak_bandwidth_bytes_per_sec = hardware.hbm_bandwidth_gb_s * 1e9

        balance_point = peak_flops / peak_bandwidth_bytes_per_sec

        return balance_point

    def predict_prefill_latency(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        batch_size: int,
        input_length: int,
        precision: PrecisionType = "fp16",
    ) -> tuple[float, float, bool]:
        """
        Predict prefill (prompt processing) latency.

        Prefill is usually compute-bound for larger batches and sequences.

        Args:
            model: Model configuration
            hardware: Hardware specifications
            batch_size: Number of sequences
            input_length: Input prompt length (tokens)
            precision: Numerical precision

        Returns:
            Tuple of (latency_ms, achieved_tflops, is_compute_bound)
        """
        # Calculate FLOPs for prefill
        flops_per_token = self.calculate_flops_per_token(model)
        total_flops = flops_per_token * batch_size * input_length

        # Calculate arithmetic intensity
        ai = self.calculate_arithmetic_intensity(
            model, batch_size, input_length, precision
        )
        balance_point = self.get_hardware_balance_point(hardware)

        is_compute_bound = ai > balance_point

        if is_compute_bound:
            # Compute-bound: limited by TFLOPS
            effective_tflops = hardware.peak_tflops * self.compute_efficiency
            latency_sec = total_flops / (effective_tflops * 1e12)
        else:
            # Memory-bound: limited by bandwidth
            precision_bytes = {
                "fp32": 4,
                "fp16": 2,
                "bf16": 2,
                "fp8": 1,
                "int8": 1,
                "int4": 0.5,
            }
            bytes_per_param = precision_bytes[precision]
            param_bytes = model.total_params() * bytes_per_param

            effective_bandwidth = (
                hardware.hbm_bandwidth_gb_s * 1e9 * self.memory_efficiency
            )
            latency_sec = param_bytes / effective_bandwidth

        # Add communication overhead (tensor parallelism)
        if hardware.tensor_parallel_size > 1:
            comm_overhead_sec = (
                model.num_layers
                * self.COMMUNICATION_LATENCY_US
                * 1e-6
                * hardware.tensor_parallel_size
            )
            latency_sec += comm_overhead_sec

        # Calculate achieved TFLOPS
        achieved_tflops = (total_flops / latency_sec) / 1e12 if latency_sec > 0 else 0

        latency_ms = latency_sec * 1000

        return latency_ms, achieved_tflops, is_compute_bound

    def predict_decode_latency(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        batch_size: int,
        precision: PrecisionType = "fp16",
    ) -> tuple[float, float, bool]:
        """
        Predict decode (token generation) latency per token.

        Decode is usually memory-bound (loading weights for single token).

        Args:
            model: Model configuration
            hardware: Hardware specifications
            batch_size: Number of sequences
            precision: Numerical precision

        Returns:
            Tuple of (latency_ms_per_token, achieved_tflops, is_memory_bound)
        """
        # Decode processes one token at a time (autoregressive)
        flops_per_token = self.calculate_flops_per_token(model)
        total_flops = flops_per_token * batch_size

        # Decode is typically memory-bound (loading model weights)
        precision_bytes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "fp8": 1,
            "int8": 1,
            "int4": 0.5,
        }
        bytes_per_param = precision_bytes[precision]

        # Model parameters + KV cache reads
        param_bytes = model.total_params() * bytes_per_param

        # KV cache bytes per token (2 for K and V)
        kv_bytes_per_token = model.kv_cache_bytes_per_token(precision)

        total_bytes = param_bytes + (kv_bytes_per_token * batch_size)

        # Decode is memory-bound for typical batch sizes
        effective_bandwidth = hardware.hbm_bandwidth_gb_s * 1e9 * self.memory_efficiency
        latency_sec = total_bytes / effective_bandwidth

        # Add communication overhead
        if hardware.tensor_parallel_size > 1:
            comm_overhead_sec = (
                model.num_layers
                * self.COMMUNICATION_LATENCY_US
                * 1e-6
                * hardware.tensor_parallel_size
            )
            latency_sec += comm_overhead_sec

        # Calculate achieved TFLOPS
        achieved_tflops = (total_flops / latency_sec) / 1e12 if latency_sec > 0 else 0

        latency_ms = latency_sec * 1000

        is_memory_bound = True  # Decode is almost always memory-bound

        return latency_ms, achieved_tflops, is_memory_bound

    def calculate_mfu(
        self, achieved_tflops: float, hardware: HardwareSpec
    ) -> float:
        """
        Calculate Model FLOPS Utilization (MFU).

        MFU = achieved_tflops / peak_tflops

        Args:
            achieved_tflops: Measured TFLOPS
            hardware: Hardware specifications

        Returns:
            MFU (0.0-1.0)
        """
        if hardware.peak_tflops == 0:
            return 0.0

        mfu = achieved_tflops / hardware.peak_tflops
        return min(mfu, 1.0)  # Cap at 100%

    def calculate_mbu(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        batch_size: int,
        latency_sec: float,
        precision: PrecisionType = "fp16",
    ) -> float:
        """
        Calculate Model Bandwidth Utilization (MBU).

        MBU = achieved_bandwidth / peak_bandwidth
        where achieved_bandwidth = total_bytes / latency

        Args:
            model: Model configuration
            hardware: Hardware specifications
            batch_size: Number of sequences
            latency_sec: Measured latency (seconds)
            precision: Numerical precision

        Returns:
            MBU (0.0-1.0)
        """
        if latency_sec == 0 or hardware.hbm_bandwidth_gb_s == 0:
            return 0.0

        # Total bytes transferred (params + KV cache)
        precision_bytes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "fp8": 1,
            "int8": 1,
            "int4": 0.5,
        }
        bytes_per_param = precision_bytes[precision]
        param_bytes = model.total_params() * bytes_per_param
        kv_bytes = model.kv_cache_bytes_per_token(precision) * batch_size

        total_bytes = param_bytes + kv_bytes

        # Achieved bandwidth
        achieved_bandwidth_gb_s = (total_bytes / latency_sec) / 1e9

        # MBU
        mbu = achieved_bandwidth_gb_s / hardware.hbm_bandwidth_gb_s
        return min(mbu, 1.0)  # Cap at 100%

    def predict_latency(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        batch_size: int,
        input_length: int,
        output_length: int,
        precision: PrecisionType = "fp16",
    ) -> PerformanceMetrics:
        """
        Predict end-to-end inference latency and performance metrics.

        Args:
            model: Model configuration
            hardware: Hardware specifications
            batch_size: Number of sequences
            input_length: Input prompt length (tokens)
            output_length: Output generation length (tokens)
            precision: Numerical precision

        Returns:
            PerformanceMetrics with detailed predictions

        Example:
            >>> metrics = analyzer.predict_latency(
            ...     model=llama3_8b,
            ...     hardware=rtx_5090,
            ...     batch_size=32,
            ...     input_length=2048,
            ...     output_length=512,
            ...     precision="fp16"
            ... )
            >>> print(f"Total latency: {metrics.total_latency_ms:.0f}ms")
            >>> print(f"Throughput: {metrics.throughput_tokens_per_sec:.0f} tok/s")
        """
        # Validate inputs
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        if output_length < 0:
            raise ValueError(f"output_length must be non-negative, got {output_length}")

        # Predict prefill latency
        prefill_latency_ms, prefill_tflops, is_compute_bound = (
            self.predict_prefill_latency(
                model, hardware, batch_size, input_length, precision
            )
        )

        # Predict decode latency (per token)
        decode_latency_per_token_ms, decode_tflops, is_memory_bound = (
            self.predict_decode_latency(model, hardware, batch_size, precision)
        )

        # Total decode latency
        decode_latency_ms = decode_latency_per_token_ms * output_length

        # Total latency
        total_latency_ms = prefill_latency_ms + decode_latency_ms

        # Calculate MFU (use prefill TFLOPS as it's typically higher)
        mfu = self.calculate_mfu(prefill_tflops, hardware)

        # Calculate MBU (use decode as it's memory-bound)
        mbu = self.calculate_mbu(
            model,
            hardware,
            batch_size,
            decode_latency_per_token_ms / 1000,  # Convert to seconds
            precision,
        )

        # Calculate throughput (tokens/second)
        total_tokens = batch_size * (input_length + output_length)
        total_latency_sec = total_latency_ms / 1000
        throughput = total_tokens / total_latency_sec if total_latency_sec > 0 else 0

        # Arithmetic intensity
        ai = self.calculate_arithmetic_intensity(
            model, batch_size, input_length, precision
        )

        return PerformanceMetrics(
            prefill_latency_ms=prefill_latency_ms,
            decode_latency_ms=decode_latency_ms,
            total_latency_ms=total_latency_ms,
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
        """
        Predict throughput (tokens/second).

        Convenience method that wraps predict_latency().

        Args:
            model: Model configuration
            hardware: Hardware specifications
            batch_size: Number of sequences
            input_length: Input prompt length
            output_length: Output generation length
            precision: Numerical precision

        Returns:
            Throughput in tokens/second
        """
        metrics = self.predict_latency(
            model, hardware, batch_size, input_length, output_length, precision
        )
        return metrics.throughput_tokens_per_sec
