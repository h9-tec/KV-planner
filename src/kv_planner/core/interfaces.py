"""
Protocol interfaces for core components.

Using Protocol (PEP 544) for structural subtyping, allowing flexible
implementations without forcing inheritance.
"""

from typing import Literal, Protocol, runtime_checkable

from kv_planner.domain import ModelConfig, HardwareSpec

PrecisionType = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]


@runtime_checkable
class MemoryCalculator(Protocol):
    """
    Protocol for KV cache memory calculations.

    Implementations must provide methods to calculate KV cache size
    and determine optimal batch sizes for given memory constraints.
    """

    def calculate_kv_cache_size(
        self,
        batch_size: int,
        sequence_length: int,
        model: ModelConfig,
        precision: PrecisionType,
    ) -> int:
        """
        Calculate total KV cache size in bytes.

        Args:
            batch_size: Number of sequences in batch
            sequence_length: Length of each sequence in tokens
            model: Model configuration
            precision: Precision type for KV cache

        Returns:
            Total KV cache size in bytes
        """
        ...

    def max_batch_size(
        self,
        available_memory_gb: float,
        sequence_length: int,
        model: ModelConfig,
        precision: PrecisionType,
    ) -> int:
        """
        Calculate maximum batch size for given memory constraint.

        Args:
            available_memory_gb: Available GPU memory in gigabytes
            sequence_length: Target sequence length
            model: Model configuration
            precision: Precision type for KV cache

        Returns:
            Maximum batch size that fits in memory
        """
        ...

    def max_sequence_length(
        self,
        available_memory_gb: float,
        batch_size: int,
        model: ModelConfig,
        precision: PrecisionType,
    ) -> int:
        """
        Calculate maximum sequence length for given memory constraint.

        Args:
            available_memory_gb: Available GPU memory in gigabytes
            batch_size: Target batch size
            model: Model configuration
            precision: Precision type for KV cache

        Returns:
            Maximum sequence length that fits in memory
        """
        ...


@runtime_checkable
class PerformanceAnalyzer(Protocol):
    """
    Protocol for performance analysis and prediction.

    Implementations should use roofline models or similar techniques
    to predict latency and throughput characteristics.
    """

    def predict_latency_ms(
        self,
        operation: str,  # "prefill" or "decode"
        batch_size: int,
        sequence_length: int,
        model: ModelConfig,
    ) -> float:
        """
        Predict operation latency in milliseconds.

        Args:
            operation: Operation type ("prefill" or "decode")
            batch_size: Number of sequences
            sequence_length: Sequence length
            model: Model configuration

        Returns:
            Predicted latency in milliseconds
        """
        ...

    def predict_throughput_tokens_per_sec(
        self,
        batch_size: int,
        sequence_length: int,
        model: ModelConfig,
    ) -> float:
        """
        Predict throughput in tokens per second.

        Args:
            batch_size: Number of sequences
            sequence_length: Sequence length
            model: Model configuration

        Returns:
            Predicted throughput in tokens/second
        """
        ...


@runtime_checkable
class CostAnalyzer(Protocol):
    """
    Protocol for cost analysis and TCO calculations.

    Implementations should calculate cost per token, TCO,
    and provide efficiency metrics.
    """

    def cost_per_million_tokens(
        self,
        throughput_tokens_per_sec: float,
        gpu_utilization: float,
        hardware: HardwareSpec,
    ) -> float:
        """
        Calculate cost per million tokens.

        Args:
            throughput_tokens_per_sec: Achieved throughput
            gpu_utilization: GPU utilization fraction (0-1)
            hardware: Hardware specification with cost info

        Returns:
            Cost per million tokens in USD
        """
        ...

    def monthly_tco(
        self,
        throughput_tokens_per_sec: float,
        gpu_utilization: float,
        hardware: HardwareSpec,
        staff_cost_monthly: float,
    ) -> dict[str, float]:
        """
        Calculate monthly Total Cost of Ownership.

        Args:
            throughput_tokens_per_sec: Achieved throughput
            gpu_utilization: GPU utilization fraction
            hardware: Hardware specification
            staff_cost_monthly: Monthly staff costs

        Returns:
            Dictionary with TCO breakdown
        """
        ...
