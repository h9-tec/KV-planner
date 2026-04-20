"""Hardware specification value object."""

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

from kv_planner.domain.exceptions import InvalidConfigurationError
from kv_planner.domain.precision import PrecisionType


InterconnectType = Literal["NVLink", "NVLink-C2C", "InfiniBand", "PCIe"]


@dataclass(frozen=True)
class HardwareSpec:
    """
    Immutable hardware specification value object.

    ``memory_bandwidth_gb_s`` is renamed from the legacy ``hbm_bandwidth_gb_s``
    because several supported GPUs (L40S, consumer RTX) use GDDR, not HBM.
    ``peak_tflops`` is the dense FP16 tensor-core number used as the default
    Roofline compute ceiling; for other precisions, populate
    ``peak_tflops_by_precision``.

    Attributes:
        gpu_model: GPU model identifier (e.g., "H100-SXM-80GB")
        num_gpus: Number of GPUs
        gpu_memory_gb: Memory per GPU in gigabytes
        peak_tflops: Peak dense FP16 TFLOPS (tensor core; NOT the CUDA FP32
            number and NOT the sparse number)
        peak_tflops_by_precision: Optional per-precision overrides. If a
            precision is missing, callers fall back to ``peak_tflops``.
        memory_bandwidth_gb_s: HBM or GDDR memory bandwidth (GB/s)
        l2_cache_mb: L2 cache size in megabytes
        interconnect_type: Type of inter-GPU interconnect
        interconnect_bandwidth_gb_s: Per-direction NVLink / PCIe bandwidth (GB/s)
        interconnect_latency_us: Per-hop link latency (microseconds)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of pipeline stages
        gpu_memory_utilization: Fraction of GPU memory to use (0-1)
        block_size: KV cache block size in tokens (vLLM PagedAttention)
        gpu_hourly_cost: On-demand GPU cost per hour (optional, for TCO)
    """

    gpu_model: str
    num_gpus: int
    gpu_memory_gb: float

    peak_tflops: float
    memory_bandwidth_gb_s: float
    peak_tflops_by_precision: Mapping[PrecisionType, float] = field(default_factory=dict)

    l2_cache_mb: float = 60.0

    interconnect_type: InterconnectType = "NVLink"
    interconnect_bandwidth_gb_s: float = 400.0  # measured NVLink-4 on H100-HGX
    interconnect_latency_us: float = 1.5  # typical NVLink per-hop latency

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    gpu_memory_utilization: float = 0.9
    block_size: int = 16

    gpu_hourly_cost: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.num_gpus <= 0:
            raise InvalidConfigurationError(f"num_gpus must be positive, got {self.num_gpus}")

        if self.gpu_memory_gb <= 0:
            raise InvalidConfigurationError(
                f"gpu_memory_gb must be positive, got {self.gpu_memory_gb}"
            )

        if self.peak_tflops <= 0:
            raise InvalidConfigurationError(
                f"peak_tflops must be positive, got {self.peak_tflops}"
            )

        if self.memory_bandwidth_gb_s <= 0:
            raise InvalidConfigurationError(
                f"memory_bandwidth_gb_s must be positive, got {self.memory_bandwidth_gb_s}"
            )

        if not 0 < self.gpu_memory_utilization <= 1:
            raise InvalidConfigurationError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )

        if self.tensor_parallel_size <= 0:
            raise InvalidConfigurationError(
                f"tensor_parallel_size must be positive, got {self.tensor_parallel_size}"
            )

        if self.pipeline_parallel_size <= 0:
            raise InvalidConfigurationError(
                f"pipeline_parallel_size must be positive, got {self.pipeline_parallel_size}"
            )

        total_parallelism = self.tensor_parallel_size * self.pipeline_parallel_size
        if total_parallelism > self.num_gpus:
            raise InvalidConfigurationError(
                f"tensor_parallel_size × pipeline_parallel_size ({total_parallelism}) "
                f"exceeds num_gpus ({self.num_gpus})"
            )

        if self.block_size <= 0:
            raise InvalidConfigurationError(f"block_size must be positive, got {self.block_size}")

    @property
    def total_memory_gb(self) -> float:
        """Total GPU memory across all GPUs."""
        return self.num_gpus * self.gpu_memory_gb

    @property
    def total_tflops(self) -> float:
        """Total FP16 compute capacity across all GPUs."""
        return self.num_gpus * self.peak_tflops

    @property
    def memory_bandwidth_tb_s(self) -> float:
        """Memory bandwidth in TB/s."""
        return self.memory_bandwidth_gb_s / 1000

    @property
    def effective_memory_gb(self) -> float:
        """Effective GPU memory after applying utilization factor."""
        return self.total_memory_gb * self.gpu_memory_utilization

    def peak_tflops_for(self, precision: PrecisionType) -> float:
        """Return the peak TFLOPS for a given precision.

        Looks up ``peak_tflops_by_precision[precision]`` first; falls back to
        a heuristic scaling of :attr:`peak_tflops` (FP16) for missing entries:
        FP8 = 2× FP16 (Hopper+ transformer engine), INT8 = 2× FP16,
        INT4 = 4× FP16. These heuristics are rough — callers that care about
        accuracy should populate ``peak_tflops_by_precision`` explicitly.
        """
        if precision in self.peak_tflops_by_precision:
            return self.peak_tflops_by_precision[precision]
        heuristic = {
            "fp32": 0.5,
            "fp16": 1.0,
            "bf16": 1.0,
            "fp8": 2.0,
            "int8": 2.0,
            "int4": 4.0,
        }
        return self.peak_tflops * heuristic.get(precision, 1.0)

    def available_kv_cache_memory_gb(self, model_size_gb: float) -> float:
        """
        Memory available for the KV cache after model weights load.

        Weights are sharded across the TP group (1/TP per GPU), so the
        per-GPU budget is ``gpu_memory_gb · utilization − model_size_gb/TP``.

        No "PagedAttention efficiency" factor is applied here — internal
        fragmentation is already accounted for by :class:`PagedMemoryCalculator`'s
        block-ceiling allocation (see its docstring for the correctness proof
        against vLLM's measured <4 % figure).
        """
        model_per_gpu_gb = model_size_gb / self.tensor_parallel_size
        available_per_gpu = (self.gpu_memory_gb * self.gpu_memory_utilization) - model_per_gpu_gb

        if available_per_gpu < 0:
            return 0.0

        return available_per_gpu * self.num_gpus

    @property
    def supports_disaggregation(self) -> bool:
        """Hardware supports disaggregated prefill-decode (≥2 GPUs and fast link)."""
        return self.num_gpus >= 2 and self.interconnect_bandwidth_gb_s >= 100

    def __repr__(self) -> str:
        parallelism = ""
        if self.tensor_parallel_size > 1 or self.pipeline_parallel_size > 1:
            parallelism = f", TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}"

        return (
            f"HardwareSpec({self.num_gpus}× {self.gpu_model}, "
            f"{self.total_memory_gb:.0f}GB total"
            f"{parallelism})"
        )
