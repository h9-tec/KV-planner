"""Traffic model for request patterns."""

from dataclasses import dataclass
from typing import Literal, Optional

from kv_planner.domain.exceptions import InvalidConfigurationError


@dataclass(frozen=True)
class Distribution:
    """
    Statistical distribution for token counts.

    Attributes:
        mean: Average value
        p50: 50th percentile (median)
        p95: 95th percentile
        p99: 99th percentile
        std: Standard deviation
    """

    mean: float
    p50: float
    p95: float
    p99: float
    std: float

    def __post_init__(self) -> None:
        """Validate distribution."""
        if self.mean <= 0:
            raise InvalidConfigurationError(f"mean must be positive, got {self.mean}")

        if self.p50 <= 0:
            raise InvalidConfigurationError(f"p50 must be positive, got {self.p50}")

        if self.std < 0:
            raise InvalidConfigurationError(f"std must be non-negative, got {self.std}")

        # Validate percentile ordering
        if not (self.p50 <= self.p95 <= self.p99):
            raise InvalidConfigurationError(
                f"Percentiles must be ordered: p50 <= p95 <= p99, "
                f"got p50={self.p50}, p95={self.p95}, p99={self.p99}"
            )


WorkloadProfile = Literal["chatbot", "completion", "rag", "coding", "multimodal"]


@dataclass(frozen=True)
class TrafficModel:
    """
    Traffic model representing request patterns.

    Attributes:
        requests_per_second: Average request rate
        input_tokens: Distribution of input token counts
        output_tokens: Distribution of output token counts
        peak_multiplier: Peak vs average traffic ratio
        prefix_sharing_ratio: Fraction of requests sharing prefixes (0-1)
        avg_shared_prefix_length: Average length of shared prefix in tokens
        images_per_request: Average images per request (for multimodal)
        target_ttft_ms: Target time-to-first-token in milliseconds (SLA)
        target_tpot_ms: Target time-per-output-token in milliseconds (SLA)
        target_p95_latency_ms: Target p95 latency in milliseconds (SLA)
    """

    # Request rate
    requests_per_second: float

    # Token distributions
    input_tokens: Distribution
    output_tokens: Distribution

    # Traffic patterns
    peak_multiplier: float = 2.0

    # Prefix caching potential
    prefix_sharing_ratio: float = 0.0
    avg_shared_prefix_length: int = 0

    # Multi-modal
    images_per_request: float = 0.0

    # SLA requirements (optional)
    target_ttft_ms: Optional[float] = None
    target_tpot_ms: Optional[float] = None
    target_p95_latency_ms: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.requests_per_second <= 0:
            raise InvalidConfigurationError(
                f"requests_per_second must be positive, got {self.requests_per_second}"
            )

        if self.peak_multiplier < 1:
            raise InvalidConfigurationError(
                f"peak_multiplier must be >= 1, got {self.peak_multiplier}"
            )

        if not 0 <= self.prefix_sharing_ratio <= 1:
            raise InvalidConfigurationError(
                f"prefix_sharing_ratio must be in [0, 1], got {self.prefix_sharing_ratio}"
            )

        if self.avg_shared_prefix_length < 0:
            raise InvalidConfigurationError(
                f"avg_shared_prefix_length must be non-negative, "
                f"got {self.avg_shared_prefix_length}"
            )

        if self.images_per_request < 0:
            raise InvalidConfigurationError(
                f"images_per_request must be non-negative, got {self.images_per_request}"
            )

    @property
    def peak_requests_per_second(self) -> float:
        """Peak request rate."""
        return self.requests_per_second * self.peak_multiplier

    @property
    def avg_total_tokens(self) -> float:
        """Average total tokens per request (input + output)."""
        return self.input_tokens.mean + self.output_tokens.mean

    @property
    def p95_total_tokens(self) -> float:
        """P95 total tokens per request."""
        return self.input_tokens.p95 + self.output_tokens.p95

    def concurrent_requests_littles_law(self, avg_latency_sec: float) -> int:
        """
        Estimate concurrent requests using Little's Law.

        Little's Law: L = λ × W
        where L = number in system, λ = arrival rate, W = time in system

        Args:
            avg_latency_sec: Average request latency in seconds

        Returns:
            Estimated concurrent requests
        """
        return int(self.requests_per_second * avg_latency_sec)

    @property
    def workload_profile(self) -> WorkloadProfile:
        """
        Auto-detect workload type based on traffic patterns.

        Returns:
            Workload profile type
        """
        # RAG: High prefix sharing
        if self.prefix_sharing_ratio > 0.5:
            return "rag"

        # Completion: Output >> Input
        if self.output_tokens.mean > self.input_tokens.mean * 3:
            return "completion"

        # Multimodal: Has images
        if self.images_per_request > 0:
            return "multimodal"

        # Coding: Long input context
        if self.input_tokens.p95 > 4096:
            return "coding"

        # Default: Chatbot
        return "chatbot"

    @property
    def benefits_from_prefix_caching(self) -> bool:
        """Check if workload would benefit from prefix caching."""
        return self.prefix_sharing_ratio > 0.1 and self.avg_shared_prefix_length > 100

    @property
    def benefits_from_chunked_prefill(self) -> bool:
        """Check if workload would benefit from chunked prefill."""
        # Long inputs benefit from chunking (reduces head-of-line blocking)
        return self.input_tokens.p95 > 2048

    @property
    def is_latency_sensitive(self) -> bool:
        """Check if workload has strict latency requirements."""
        return (
            self.target_ttft_ms is not None
            or self.target_tpot_ms is not None
            or self.target_p95_latency_ms is not None
        )

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"TrafficModel({self.requests_per_second:.1f} RPS, "
            f"input={self.input_tokens.mean:.0f}±{self.input_tokens.std:.0f} tokens, "
            f"output={self.output_tokens.mean:.0f}±{self.output_tokens.std:.0f} tokens, "
            f"profile={self.workload_profile})"
        )
