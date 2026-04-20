"""
PagedAttention-based memory calculator.

Models vLLM's PagedAttention block-based KV cache allocation. The only
fragmentation source is **internal**: the last (potentially partially-used)
block of each sequence. External fragmentation is zero by construction
(all blocks are the same fixed size). vLLM reports <4 % waste in practice,
and that figure is the *total* — no additional overhead is layered on.

Reference:
    Kwon et al., 2023 — "Efficient Memory Management for Large Language Model
    Serving with PagedAttention". https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from kv_planner.domain import HardwareSpec, ModelConfig, PrecisionType
from kv_planner.domain.exceptions import InsufficientMemoryError


class PagedMemoryCalculator:
    """
    Block-granularity KV cache size calculator.

    Internal fragmentation = (blocks · block_size − actual_tokens) per
    sequence. At long context and the vLLM default block_size=16, this is
    ``< block_size / avg_seq_len`` in the limit (≤4 % for avg_seq_len ≥ 400).
    """

    DEFAULT_BLOCK_SIZE = 16  # vLLM platform-default on CUDA

    def __init__(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
        hardware: Optional[HardwareSpec] = None,
    ) -> None:
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
        """Return total KV cache bytes for a batch at a given sequence length.

        Formula::

            blocks_per_seq = ceil(seq_len / block_size)
            total_bytes    = batch · blocks_per_seq · block_size · kv_bytes_per_token

        That is the only fragmentation term — we do NOT multiply by
        (1 + 0.04): the ceil already accounts for the partial last block and
        that matches what vLLM reports.
        """
        self._validate_inputs(batch_size, sequence_length)

        blocks_per_seq = self._blocks_needed(sequence_length)
        total_blocks = batch_size * blocks_per_seq
        bytes_per_token = model.kv_cache_bytes_per_token(precision)
        total_bytes = total_blocks * self._block_size * bytes_per_token

        self._logger.debug(
            "KV cache: %d×%d tokens = %d blocks (%d per seq) = %.3f GB",
            batch_size,
            sequence_length,
            total_blocks,
            blocks_per_seq,
            total_bytes / 1e9,
        )
        return int(total_bytes)

    def max_batch_size(
        self,
        available_memory_gb: float,
        sequence_length: int,
        model: ModelConfig,
        precision: PrecisionType = "fp16",
    ) -> int:
        """Largest batch size that fits in ``available_memory_gb``.

        No super-linear scaling is applied — memory scales linearly with
        tensor-parallel size, and the correct way to express "more room for
        KV cache because weights are sharded" is to pass the already-reduced
        ``available_memory_gb`` the caller computed from
        :meth:`HardwareSpec.available_kv_cache_memory_gb`.
        """
        self._validate_inputs(1, sequence_length)

        if available_memory_gb <= 0:
            raise ValueError(
                f"available_memory_gb must be positive, got {available_memory_gb}"
            )

        memory_per_seq_bytes = self.calculate_kv_cache_size(
            batch_size=1,
            sequence_length=sequence_length,
            model=model,
            precision=precision,
        )

        max_batch = int(available_memory_gb * 1e9 / memory_per_seq_bytes)

        if max_batch < 1:
            raise InsufficientMemoryError(
                f"Cannot fit even 1 sequence of length {sequence_length} "
                f"in {available_memory_gb:.2f} GB memory. "
                f"Required: {memory_per_seq_bytes / 1e9:.2f} GB"
            )

        return max_batch

    def max_sequence_length(
        self,
        available_memory_gb: float,
        batch_size: int,
        model: ModelConfig,
        precision: PrecisionType = "fp16",
    ) -> int:
        """Largest sequence length that fits for a given batch size.

        Closed form (no binary search needed): invert the block-granularity
        formula to ``floor(mem / (batch · bytes_per_token)) `` rounded down
        to a block multiple.
        """
        self._validate_inputs(batch_size, 1)

        if available_memory_gb <= 0:
            raise ValueError(
                f"available_memory_gb must be positive, got {available_memory_gb}"
            )

        bytes_per_token = model.kv_cache_bytes_per_token(precision)
        budget_bytes = available_memory_gb * 1e9

        # Each added block costs batch · block_size · bytes_per_token
        bytes_per_block = batch_size * self._block_size * bytes_per_token
        max_blocks_per_seq = int(budget_bytes // bytes_per_block)
        max_seq_length = max_blocks_per_seq * self._block_size

        if max_seq_length < self._block_size:
            raise InsufficientMemoryError(
                f"Cannot fit batch_size={batch_size} at block size "
                f"{self._block_size} in {available_memory_gb:.2f} GB"
            )

        return max_seq_length

    def memory_breakdown(
        self,
        batch_size: int,
        sequence_length: int,
        model: ModelConfig,
        precision: PrecisionType = "fp16",
    ) -> dict[str, float]:
        """Detailed memory accounting for debugging / reporting."""
        kv_bytes = self.calculate_kv_cache_size(batch_size, sequence_length, model, precision)
        blocks_per_seq = self._blocks_needed(sequence_length)
        total_blocks = batch_size * blocks_per_seq

        actual_tokens = batch_size * sequence_length
        allocated_tokens = total_blocks * self._block_size
        wasted_tokens = allocated_tokens - actual_tokens

        return {
            "kv_cache_gb": kv_bytes / 1e9,
            "blocks": total_blocks,
            "blocks_per_sequence": blocks_per_seq,
            "tokens_actual": actual_tokens,
            "tokens_allocated": allocated_tokens,
            "tokens_wasted": wasted_tokens,
            "fragmentation_pct": (
                (wasted_tokens / allocated_tokens * 100) if allocated_tokens > 0 else 0.0
            ),
        }

    def _blocks_needed(self, sequence_length: int) -> int:
        return math.ceil(sequence_length / self._block_size)

    def _validate_inputs(self, batch_size: int, sequence_length: int) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    @property
    def block_size(self) -> int:
        return self._block_size
