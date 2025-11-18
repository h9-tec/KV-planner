"""
Naive memory calculator for comparison.

This implements simple memory calculations without PagedAttention,
useful for baseline comparisons and fallback scenarios.
"""

import logging
from typing import Literal

from kv_planner.domain import ModelConfig
from kv_planner.domain.exceptions import InsufficientMemoryError

PrecisionType = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]


class NaiveMemoryCalculator:
    """
    Naive KV cache memory calculator (no PagedAttention).

    Uses simple contiguous allocation, which results in 60-80% fragmentation
    in production scenarios. Useful for comparison with PagedAttention.
    """

    FRAGMENTATION_OVERHEAD = 0.70  # 60-80% waste (typical without paging)

    def __init__(self):
        """Initialize NaiveMemoryCalculator."""
        self._logger = logging.getLogger(__name__)

    def calculate_kv_cache_size(
        self,
        batch_size: int,
        sequence_length: int,
        model: ModelConfig,
        precision: PrecisionType = "fp16",
    ) -> int:
        """
        Calculate KV cache size with naive contiguous allocation.

        Formula:
            bytes = batch_size × sequence_length × kv_bytes_per_token
            effective_bytes = bytes × (1 + fragmentation)

        Args:
            batch_size: Number of sequences
            sequence_length: Length per sequence
            model: Model configuration
            precision: KV cache precision

        Returns:
            Total KV cache size in bytes
        """
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")

        # Simple multiplication (no block granularity)
        bytes_per_token = model.kv_cache_bytes_per_token(precision)
        total_bytes = batch_size * sequence_length * bytes_per_token

        # Add fragmentation overhead (pre-allocation, alignment, etc.)
        effective_bytes = int(total_bytes * (1 + self.FRAGMENTATION_OVERHEAD))

        self._logger.debug(
            f"KV cache (naive): {batch_size}×{sequence_length} = "
            f"{effective_bytes / 1e9:.3f} GB (includes {self.FRAGMENTATION_OVERHEAD:.0%} fragmentation)"
        )

        return effective_bytes

    def max_batch_size(
        self,
        available_memory_gb: float,
        sequence_length: int,
        model: ModelConfig,
        precision: PrecisionType = "fp16",
    ) -> int:
        """Calculate maximum batch size (naive allocation)."""
        if available_memory_gb <= 0 or sequence_length <= 0:
            raise ValueError("available_memory_gb and sequence_length must be positive")

        memory_per_seq = self.calculate_kv_cache_size(1, sequence_length, model, precision)
        max_batch = int(available_memory_gb * 1e9 / memory_per_seq)

        if max_batch < 1:
            raise InsufficientMemoryError(
                f"Cannot fit 1 sequence of length {sequence_length} in {available_memory_gb:.2f} GB"
            )

        return max_batch

    def max_sequence_length(
        self,
        available_memory_gb: float,
        batch_size: int,
        model: ModelConfig,
        precision: PrecisionType = "fp16",
    ) -> int:
        """Calculate maximum sequence length (naive allocation)."""
        if available_memory_gb <= 0 or batch_size <= 0:
            raise ValueError("available_memory_gb and batch_size must be positive")

        # With naive allocation, simple division
        bytes_per_token = model.kv_cache_bytes_per_token(precision)
        effective_bytes_per_token = int(bytes_per_token * (1 + self.FRAGMENTATION_OVERHEAD))

        max_seq = int(available_memory_gb * 1e9 / (batch_size * effective_bytes_per_token))

        if max_seq < 1:
            raise InsufficientMemoryError(
                f"Cannot fit batch_size={batch_size} in {available_memory_gb:.2f} GB"
            )

        return max_seq
