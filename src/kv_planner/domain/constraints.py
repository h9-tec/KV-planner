"""Deployment constraints value object."""

from dataclasses import dataclass
from typing import Literal, Optional


FairnessPolicy = Literal["fcfs", "fair_share", "locality_aware"]


@dataclass(frozen=True)
class DeploymentConstraints:
    """
    Deployment constraints and preferences.

    Represents user-defined limits and optimization preferences.

    Attributes:
        max_sequence_length: Maximum allowed sequence length
        max_batch_size: Maximum allowed batch size
        min_throughput_tokens_per_sec: Minimum required throughput
        max_gpu_count: Maximum number of GPUs allowed
        max_monthly_cost_usd: Maximum monthly budget
        target_cost_per_million_tokens: Target cost per 1M tokens
        allow_quantization: Whether to consider quantization strategies
        allow_fp8_kv_cache: Whether to allow FP8 KV cache quantization
        allow_int8_kv_cache: Whether to allow INT8 KV cache quantization
        allow_int4_kv_cache: Whether to allow INT4 KV cache quantization
        allow_offloading: Whether to allow KV cache offloading
        allow_cpu_offload: Whether to allow CPU offloading
        allow_disk_offload: Whether to allow disk offloading
        allow_disaggregated_architecture: Whether to allow separate prefill/decode
        allow_prefix_caching: Whether to allow prefix caching
        allow_speculative_decoding: Whether to allow speculative decoding
        enable_multi_tenancy: Whether multi-tenancy is required
        fairness_policy: Multi-tenancy fairness policy
        target_gpu_utilization: Target GPU utilization (headroom for spikes)
        require_graceful_degradation: Require graceful degradation on OOM
        oom_safety_margin_gb: Safety margin for OOM prevention
    """

    # Hard limits
    max_sequence_length: Optional[int] = None
    max_batch_size: Optional[int] = None
    min_throughput_tokens_per_sec: Optional[float] = None

    # Budget
    max_gpu_count: Optional[int] = None
    max_monthly_cost_usd: Optional[float] = None
    target_cost_per_million_tokens: Optional[float] = None

    # Optimization preferences
    allow_quantization: bool = True
    allow_fp8_kv_cache: bool = True
    allow_int8_kv_cache: bool = False
    allow_int4_kv_cache: bool = False

    allow_offloading: bool = True
    allow_cpu_offload: bool = True
    allow_disk_offload: bool = False

    allow_disaggregated_architecture: bool = False
    allow_prefix_caching: bool = True
    allow_speculative_decoding: bool = False

    # Multi-tenancy
    enable_multi_tenancy: bool = False
    fairness_policy: FairnessPolicy = "fair_share"

    # Reliability
    target_gpu_utilization: float = 0.75
    require_graceful_degradation: bool = True
    oom_safety_margin_gb: float = 5.0

    def __post_init__(self) -> None:
        """Validate constraints."""
        # Use object.__setattr__ for frozen dataclass validation
        if self.max_sequence_length is not None and self.max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be positive, got {self.max_sequence_length}")

        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size}")

        if not 0 < self.target_gpu_utilization <= 1:
            raise ValueError(
                f"target_gpu_utilization must be in (0, 1], got {self.target_gpu_utilization}"
            )

    @property
    def allows_any_quantization(self) -> bool:
        """Check if any form of quantization is allowed."""
        return self.allow_quantization and (
            self.allow_fp8_kv_cache or self.allow_int8_kv_cache or self.allow_int4_kv_cache
        )

    @property
    def allows_any_offloading(self) -> bool:
        """Check if any form of offloading is allowed."""
        return self.allow_offloading and (self.allow_cpu_offload or self.allow_disk_offload)

    @property
    def is_cost_constrained(self) -> bool:
        """Check if deployment has cost constraints."""
        return (
            self.max_monthly_cost_usd is not None or self.target_cost_per_million_tokens is not None
        )

    @property
    def is_performance_constrained(self) -> bool:
        """Check if deployment has performance constraints."""
        return self.min_throughput_tokens_per_sec is not None
