"""
Prefix caching strategy analyzer.

Evaluates benefits of prefix caching (system prompts, repeated contexts)
for reducing latency and memory usage in LLM inference.

Based on research:
- vLLM Automatic Prefix Caching: 90% cost savings, 85% latency reduction
- ChunkAttention: 3.2-4.8× speedup for prefix-aware attention
- Learned Prefix Caching: 43% lower cache size, 11% throughput increase
"""

import logging
from dataclasses import dataclass
from typing import Literal

from kv_planner.core.memory import PagedMemoryCalculator
from kv_planner.domain import ModelConfig, HardwareSpec

PrecisionType = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]


@dataclass(frozen=True)
class PrefixCachingMetrics:
    """
    Prefix caching evaluation metrics.

    Attributes:
        prefix_length: Length of cached prefix in tokens
        hit_rate: Cache hit rate (0.0-1.0)
        memory_savings_pct: Memory saved by caching (0-100)
        latency_reduction_pct: Latency reduction from avoiding prefill (0-100)
        throughput_improvement: Throughput multiplier (>1.0 is better)
        effective_batch_size: Effective batch size with caching
        cost_savings_pct: Cost savings from reduced compute (0-100)
    """

    prefix_length: int
    hit_rate: float
    memory_savings_pct: float
    latency_reduction_pct: float
    throughput_improvement: float
    effective_batch_size: int
    cost_savings_pct: float

    # Detailed metrics
    memory_without_caching_gb: float
    memory_with_caching_gb: float
    prefill_tokens_saved: int
    requests_per_second_baseline: float
    requests_per_second_with_caching: float


class PrefixCachingAnalyzer:
    """
    Analyzes benefits of prefix caching for LLM inference.

    Prefix caching reuses KV cache for repeated prompt prefixes
    (e.g., system prompts, few-shot examples), reducing:
    - Memory footprint
    - Prefill latency
    - Compute cost

    Attributes:
        memory_calculator: KV cache memory calculator
    """

    # Research-validated benefits (from vLLM, ChunkAttention, Claude)
    MAX_LATENCY_REDUCTION = 0.85  # Up to 85% latency reduction (Anthropic Claude)
    MAX_COST_SAVINGS = 0.90       # Up to 90% cost savings (Anthropic Claude)
    SPEEDUP_FACTOR = 4.0          # 3.2-4.8× speedup (ChunkAttention)
    THROUGHPUT_INCREASE = 1.11    # 11% throughput increase (Learned Prefix Caching)

    def __init__(
        self,
        memory_calculator: PagedMemoryCalculator | None = None,
    ):
        """
        Initialize PrefixCachingAnalyzer.

        Args:
            memory_calculator: Memory calculator (creates default if None)
        """
        self._memory_calc = memory_calculator or PagedMemoryCalculator()
        self._logger = logging.getLogger(__name__)

    def evaluate_caching(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        prefix_length: int,
        total_length: int,
        batch_size: int,
        hit_rate: float,
        precision: PrecisionType = "fp16",
    ) -> PrefixCachingMetrics:
        """
        Evaluate prefix caching benefits.

        Args:
            model: Model configuration
            hardware: Hardware specification
            prefix_length: Length of cached prefix (tokens)
            total_length: Total sequence length (tokens)
            batch_size: Number of concurrent requests
            hit_rate: Cache hit rate (0.0-1.0, e.g., 0.8 = 80% hits)
            precision: KV cache precision

        Returns:
            PrefixCachingMetrics with detailed analysis

        Raises:
            ValueError: If parameters are invalid
        """
        if prefix_length <= 0:
            raise ValueError(f"prefix_length must be positive, got {prefix_length}")
        if prefix_length > total_length:
            raise ValueError(
                f"prefix_length ({prefix_length}) cannot exceed total_length ({total_length})"
            )
        if not 0.0 <= hit_rate <= 1.0:
            raise ValueError(f"hit_rate must be in [0, 1], got {hit_rate}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Calculate memory without caching
        # Each request stores full KV cache
        memory_without_caching = self._memory_calc.calculate_kv_cache_size(
            batch_size=batch_size,
            sequence_length=total_length,
            model=model,
            precision=precision,
        )

        # Calculate memory with caching
        # Prefix is stored once, unique suffixes stored per request
        prefix_memory = self._memory_calc.calculate_kv_cache_size(
            batch_size=1,
            sequence_length=prefix_length,
            model=model,
            precision=precision,
        )

        suffix_length = total_length - prefix_length
        suffix_memory = self._memory_calc.calculate_kv_cache_size(
            batch_size=batch_size,
            sequence_length=suffix_length,
            model=model,
            precision=precision,
        )

        # With hit_rate, some requests get cache hits (reuse prefix)
        # Others miss and need full prefill
        effective_prefix_memory = prefix_memory * (1 + (1 - hit_rate) * (batch_size - 1))
        memory_with_caching = effective_prefix_memory + suffix_memory

        # Memory savings
        memory_savings = memory_without_caching - memory_with_caching
        memory_savings_pct = (memory_savings / memory_without_caching) * 100

        # Latency reduction
        # Prefill latency is proportional to sequence length
        # With caching, only suffix needs prefill for cache hits
        prefill_tokens_saved = int(prefix_length * batch_size * hit_rate)
        total_prefill_tokens = total_length * batch_size
        latency_reduction_pct = (prefill_tokens_saved / total_prefill_tokens) * 100

        # Cap at research-validated maximum
        latency_reduction_pct = min(latency_reduction_pct, self.MAX_LATENCY_REDUCTION * 100)

        # Throughput improvement
        # More memory → higher batch size → higher throughput
        max_batch_without = self._memory_calc.max_batch_size(
            available_memory_gb=hardware.gpu_memory_gb,
            sequence_length=total_length,
            model=model,
            precision=precision,
        )

        memory_freed = memory_without_caching - memory_with_caching
        available_with_caching = hardware.gpu_memory_gb + (memory_freed / 1e9)

        max_batch_with = self._memory_calc.max_batch_size(
            available_memory_gb=available_with_caching,
            sequence_length=total_length,
            model=model,
            precision=precision,
        )

        throughput_improvement = max_batch_with / max_batch_without

        # Cost savings
        # Proportional to compute saved (prefill tokens avoided)
        cost_savings_pct = (prefill_tokens_saved / total_prefill_tokens) * 100
        cost_savings_pct = min(cost_savings_pct, self.MAX_COST_SAVINGS * 100)

        # Baseline throughput (requests/sec)
        # Simplified: assume decode dominates, ~100 tokens/sec per request
        baseline_rps = 100.0 / total_length
        cached_rps = baseline_rps * throughput_improvement

        self._logger.info(
            f"Prefix caching: {prefix_length}/{total_length} tokens, "
            f"hit_rate={hit_rate:.1%} → "
            f"{memory_savings_pct:.1f}% memory savings, "
            f"{latency_reduction_pct:.1f}% latency reduction"
        )

        return PrefixCachingMetrics(
            prefix_length=prefix_length,
            hit_rate=hit_rate,
            memory_savings_pct=memory_savings_pct,
            latency_reduction_pct=latency_reduction_pct,
            throughput_improvement=throughput_improvement,
            effective_batch_size=max_batch_with,
            cost_savings_pct=cost_savings_pct,
            memory_without_caching_gb=memory_without_caching / 1e9,
            memory_with_caching_gb=memory_with_caching / 1e9,
            prefill_tokens_saved=prefill_tokens_saved,
            requests_per_second_baseline=baseline_rps,
            requests_per_second_with_caching=cached_rps,
        )

    def estimate_hit_rate(
        self,
        num_unique_prefixes: int,
        total_requests: int,
        cache_size: int,
    ) -> float:
        """
        Estimate cache hit rate based on workload characteristics.

        Uses simple LRU-like model:
        - If num_unique_prefixes <= cache_size: hit_rate = (total - num_unique) / total
        - Otherwise: hit_rate decreases with more unique prefixes

        Args:
            num_unique_prefixes: Number of distinct prefixes in workload
            total_requests: Total number of requests
            cache_size: Maximum number of prefixes that can be cached

        Returns:
            Estimated hit rate (0.0-1.0)

        Raises:
            ValueError: If parameters are invalid
        """
        if num_unique_prefixes <= 0:
            raise ValueError(f"num_unique_prefixes must be positive, got {num_unique_prefixes}")
        if total_requests <= 0:
            raise ValueError(f"total_requests must be positive, got {total_requests}")
        if cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {cache_size}")

        # All prefixes fit in cache
        if num_unique_prefixes <= cache_size:
            # After initial cold misses, everything is a hit
            cold_misses = num_unique_prefixes
            hits = max(0, total_requests - cold_misses)
            hit_rate = hits / total_requests
        else:
            # Cache thrashing: some prefixes evicted
            # Simplified model: hit_rate ≈ cache_size / num_unique_prefixes
            hit_rate = min(1.0, cache_size / num_unique_prefixes)

        self._logger.debug(
            f"Estimated hit rate: {hit_rate:.2%} "
            f"(unique={num_unique_prefixes}, cache_size={cache_size}, requests={total_requests})"
        )

        return hit_rate

    def recommend_prefix_length(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        system_prompt_length: int,
        few_shot_examples_length: int,
        user_input_length: int,
        batch_size: int,
        hit_rate: float,
        precision: PrecisionType = "fp16",
    ) -> int:
        """
        Recommend optimal prefix length to cache.

        Evaluates different prefix lengths to maximize benefit.

        Args:
            model: Model configuration
            hardware: Hardware specification
            system_prompt_length: Length of system prompt
            few_shot_examples_length: Length of few-shot examples
            user_input_length: Average user input length
            batch_size: Concurrent requests
            hit_rate: Expected cache hit rate
            precision: KV cache precision

        Returns:
            Recommended prefix length (tokens)
        """
        total_length = system_prompt_length + few_shot_examples_length + user_input_length

        # Evaluate different prefix lengths
        candidates = [
            system_prompt_length,  # Just system prompt
            system_prompt_length + few_shot_examples_length,  # System + examples
        ]

        best_length = system_prompt_length
        best_benefit = 0.0

        for prefix_length in candidates:
            if prefix_length >= total_length:
                continue  # Can't cache entire sequence

            try:
                metrics = self.evaluate_caching(
                    model=model,
                    hardware=hardware,
                    prefix_length=prefix_length,
                    total_length=total_length,
                    batch_size=batch_size,
                    hit_rate=hit_rate,
                    precision=precision,
                )

                # Benefit = weighted sum of savings
                benefit = (
                    metrics.memory_savings_pct * 0.3 +
                    metrics.latency_reduction_pct * 0.5 +
                    metrics.cost_savings_pct * 0.2
                )

                if benefit > best_benefit:
                    best_benefit = benefit
                    best_length = prefix_length

            except Exception as e:
                self._logger.warning(f"Failed to evaluate prefix_length={prefix_length}: {e}")
                continue

        self._logger.info(
            f"Recommended prefix length: {best_length} tokens "
            f"(benefit score: {best_benefit:.1f})"
        )

        return best_length

    def compare_hit_rates(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        prefix_length: int,
        total_length: int,
        batch_size: int,
        hit_rates: list[float],
        precision: PrecisionType = "fp16",
    ) -> list[PrefixCachingMetrics]:
        """
        Compare benefits across different cache hit rates.

        Useful for sensitivity analysis.

        Args:
            model: Model configuration
            hardware: Hardware specification
            prefix_length: Cached prefix length
            total_length: Total sequence length
            batch_size: Concurrent requests
            hit_rates: List of hit rates to evaluate (e.g., [0.5, 0.7, 0.9])
            precision: KV cache precision

        Returns:
            List of PrefixCachingMetrics for each hit rate
        """
        results = []

        for hit_rate in hit_rates:
            metrics = self.evaluate_caching(
                model=model,
                hardware=hardware,
                prefix_length=prefix_length,
                total_length=total_length,
                batch_size=batch_size,
                hit_rate=hit_rate,
                precision=precision,
            )
            results.append(metrics)

        return results
