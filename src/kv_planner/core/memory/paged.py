"""
PagedAttention-based memory calculator.

Implements vLLM's PagedAttention algorithm for KV cache management,
achieving <4% memory fragmentation vs 60-80% in naive allocation.

Reference:
    Efficient Memory Management for Large Language Model Serving with PagedAttention
    https://arxiv.org/abs/2309.06180
"""

import logging
from typing import Literal, Optional

from kv_planner.domain import ModelConfig, HardwareSpec
from kv_planner.domain.exceptions import InsufficientMemoryError

PrecisionType = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]


class PagedMemoryCalculator:
    """
    PagedAttention-based memory calculator for KV cache.

    Uses block-based allocation (16 tokens per block by default)
    to minimize fragmentation and enable near-optimal memory utilization.

    Attributes:
        block_size: Number of tokens per block (default: 16, vLLM standard)
        fragmentation_overhead: Memory overhead from block granularity (<4%)
    """

    DEFAULT_BLOCK_SIZE = 16  # vLLM default
    FRAGMENTATION_OVERHEAD = 0.04  # <4% waste (measured from vLLM)

    def __init__(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
        hardware: Optional[HardwareSpec] = None,
    ):
        """
        Initialize PagedMemoryCalculator.

        Args:
            block_size: Tokens per block (default: 16)
            hardware: Optional hardware spec for super-linear scaling

        Raises:
            ValueError: If block_size is not positive
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self._block_size = block_size
        self._hardware = hardware
        self._logger = logging.getLogger(__name__)

    def calculate_kv_cache_size(
        self,
        batch_size: int,
        sequence_length: int,
        model: ModelConfig,
        precision: PrecisionType = "fp16",
    ) -> int:
        """
        Calculate total KV cache size with block-level granularity.

        Formula:
            blocks_per_seq = ceil(sequence_length / block_size)
            total_blocks = batch_size × blocks_per_seq
            block_bytes = block_size × model.kv_cache_bytes_per_token(precision)
            total_bytes = total_blocks × block_bytes × (1 + fragmentation)

        Args:
            batch_size: Number of sequences in batch
            sequence_length: Length of each sequence
            model: Model configuration
            precision: KV cache precision

        Returns:
            Total KV cache size in bytes

        Raises:
            ValueError: If batch_size or sequence_length invalid
        """
        self._validate_inputs(batch_size, sequence_length)

        # Calculate blocks needed per sequence (ceiling division)
        blocks_per_seq = self._blocks_needed(sequence_length)

        # Total blocks across batch
        total_blocks = batch_size * blocks_per_seq

        # Bytes per block
        block_bytes = self._block_size * model.kv_cache_bytes_per_token(precision)

        # Total with fragmentation overhead
        total_bytes = total_blocks * block_bytes
        effective_bytes = int(total_bytes * (1 + self.FRAGMENTATION_OVERHEAD))

        self._logger.debug(
            f"KV cache: {batch_size}×{sequence_length} tokens = "
            f"{total_blocks} blocks ({blocks_per_seq} per seq) = "
            f"{effective_bytes / 1e9:.3f} GB"
        )

        return effective_bytes

    def max_batch_size(
        self,
        available_memory_gb: float,
        sequence_length: int,
        model: ModelConfig,
        precision: PrecisionType = "fp16",
    ) -> int:
        """
        Calculate maximum batch size for given memory constraint.

        Accounts for:
        - Block-level allocation granularity
        - Fragmentation overhead
        - Super-linear scaling with tensor parallelism (if hardware provided)

        Args:
            available_memory_gb: Available memory in GB
            sequence_length: Target sequence length
            model: Model configuration
            precision: KV cache precision

        Returns:
            Maximum batch size that fits in memory

        Raises:
            ValueError: If inputs invalid
            InsufficientMemoryError: If cannot fit even 1 sequence
        """
        self._validate_inputs(1, sequence_length)

        if available_memory_gb <= 0:
            raise ValueError(f"available_memory_gb must be positive, got {available_memory_gb}")

        # Apply super-linear scaling if hardware available
        effective_memory_gb = available_memory_gb
        if self._hardware is not None:
            scaling_factor = self._hardware.kv_cache_super_linear_scaling_factor()
            effective_memory_gb *= scaling_factor
            self._logger.debug(
                f"Applied super-linear scaling: {scaling_factor:.1f}× "
                f"({available_memory_gb:.1f} GB → {effective_memory_gb:.1f} GB effective)"
            )

        # Calculate memory per sequence
        memory_per_seq_bytes = self.calculate_kv_cache_size(
            batch_size=1,
            sequence_length=sequence_length,
            model=model,
            precision=precision,
        )

        # Maximum batch size
        max_batch = int(effective_memory_gb * 1e9 / memory_per_seq_bytes)

        if max_batch < 1:
            raise InsufficientMemoryError(
                f"Cannot fit even 1 sequence of length {sequence_length} "
                f"in {available_memory_gb:.2f} GB memory. "
                f"Required: {memory_per_seq_bytes / 1e9:.2f} GB"
            )

        self._logger.debug(
            f"Max batch size: {max_batch} "
            f"(seq_len={sequence_length}, mem={available_memory_gb:.1f} GB)"
        )

        return max_batch

    def max_sequence_length(
        self,
        available_memory_gb: float,
        batch_size: int,
        model: ModelConfig,
        precision: PrecisionType = "fp16",
    ) -> int:
        """
        Calculate maximum sequence length for given memory constraint.

        Uses binary search to find the largest sequence length that fits.

        Args:
            available_memory_gb: Available memory in GB
            batch_size: Target batch size
            model: Model configuration
            precision: KV cache precision

        Returns:
            Maximum sequence length in tokens

        Raises:
            ValueError: If inputs invalid
            InsufficientMemoryError: If cannot fit batch_size with any length
        """
        self._validate_inputs(batch_size, 1)

        if available_memory_gb <= 0:
            raise ValueError(f"available_memory_gb must be positive, got {available_memory_gb}")

        # Apply super-linear scaling
        effective_memory_gb = available_memory_gb
        if self._hardware is not None:
            scaling_factor = self._hardware.kv_cache_super_linear_scaling_factor()
            effective_memory_gb *= scaling_factor

        # Binary search for maximum sequence length
        # Upper bound: assume 1 byte per token (very conservative)
        left, right = self._block_size, int(effective_memory_gb * 1e9 / batch_size)

        max_seq_length = 0

        while left <= right:
            mid = (left + right) // 2

            try:
                kv_bytes = self.calculate_kv_cache_size(
                    batch_size=batch_size,
                    sequence_length=mid,
                    model=model,
                    precision=precision,
                )

                if kv_bytes <= effective_memory_gb * 1e9:
                    max_seq_length = mid
                    left = mid + 1  # Try longer
                else:
                    right = mid - 1  # Try shorter

            except Exception:
                right = mid - 1

        if max_seq_length < self._block_size:
            raise InsufficientMemoryError(
                f"Cannot fit batch_size={batch_size} in {available_memory_gb:.2f} GB"
            )

        self._logger.debug(
            f"Max sequence length: {max_seq_length} "
            f"(batch={batch_size}, mem={available_memory_gb:.1f} GB)"
        )

        return max_seq_length

    def memory_breakdown(
        self,
        batch_size: int,
        sequence_length: int,
        model: ModelConfig,
        precision: PrecisionType = "fp16",
    ) -> dict[str, float]:
        """
        Get detailed memory breakdown.

        Returns:
            Dictionary with memory usage breakdown in GB
        """
        kv_bytes = self.calculate_kv_cache_size(batch_size, sequence_length, model, precision)
        blocks_per_seq = self._blocks_needed(sequence_length)
        total_blocks = batch_size * blocks_per_seq

        # Actual tokens stored
        actual_tokens = batch_size * sequence_length
        # Allocated tokens (due to block granularity)
        allocated_tokens = total_blocks * self._block_size
        # Wasted tokens
        wasted_tokens = allocated_tokens - actual_tokens

        return {
            "kv_cache_gb": kv_bytes / 1e9,
            "blocks": total_blocks,
            "blocks_per_sequence": blocks_per_seq,
            "tokens_actual": actual_tokens,
            "tokens_allocated": allocated_tokens,
            "tokens_wasted": wasted_tokens,
            "fragmentation_pct": (wasted_tokens / allocated_tokens * 100) if allocated_tokens > 0 else 0,
        }

    def _blocks_needed(self, sequence_length: int) -> int:
        """
        Calculate number of blocks needed for sequence (ceiling division).

        Args:
            sequence_length: Sequence length in tokens

        Returns:
            Number of blocks (ceil(sequence_length / block_size))
        """
        return (sequence_length + self._block_size - 1) // self._block_size

    def _validate_inputs(self, batch_size: int, sequence_length: int) -> None:
        """
        Validate input parameters.

        Args:
            batch_size: Batch size to validate
            sequence_length: Sequence length to validate

        Raises:
            ValueError: If inputs are invalid
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    @property
    def block_size(self) -> int:
        """Get block size."""
        return self._block_size

    @property
    def fragmentation_overhead(self) -> float:
        """Get fragmentation overhead fraction."""
        return self.FRAGMENTATION_OVERHEAD
