"""
Cost analyzer for LLM deployment.

Calculates Total Cost of Ownership (TCO), $/million tokens,
and provides break-even analysis for different deployment strategies.

Based on 2025 pricing:
- H100: $3.00-$6.98/hr (AWS/Azure/GCP)
- A100: $0.70-$3.67/hr
- RTX 5090: $0.27-$0.47/hr
- RTX 4090: $0.18-$0.23/hr
- RTX 3090: $0.11-$0.12/hr
"""

import logging
from dataclasses import dataclass
from typing import Literal

from kv_planner.core.performance import RooflineAnalyzer
from kv_planner.domain import ModelConfig, HardwareSpec

PrecisionType = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]


@dataclass(frozen=True)
class GPUPricing:
    """
    GPU pricing information.

    Attributes:
        gpu_model: GPU model name
        cost_per_hour_usd: Cost per GPU per hour (USD)
        source: Pricing source (e.g., "AWS", "Azure", "Runpod")
        pricing_type: On-demand or spot/preemptible
    """

    gpu_model: str
    cost_per_hour_usd: float
    source: str = "market_average"
    pricing_type: Literal["on_demand", "spot"] = "on_demand"


@dataclass(frozen=True)
class CostMetrics:
    """
    Cost analysis metrics.

    Attributes:
        cost_per_hour: Total cost per hour (USD)
        cost_per_million_tokens: Cost per million tokens (USD)
        cost_per_request: Average cost per request (USD)
        monthly_cost_usd: Monthly cost at target load (USD)
        annual_cost_usd: Annual cost (USD)
        utilization_pct: GPU utilization percentage
        tokens_per_dollar: Tokens per dollar (inverse of cost)
        break_even_tokens_per_day: Tokens/day to match API pricing
    """

    cost_per_hour: float
    cost_per_million_tokens: float
    cost_per_request: float
    monthly_cost_usd: float
    annual_cost_usd: float
    utilization_pct: float
    tokens_per_dollar: float
    break_even_tokens_per_day: int

    # Detailed breakdown
    throughput_tokens_per_sec: float
    requests_per_second: float
    gpu_pricing: GPUPricing


class CostAnalyzer:
    """
    Analyzes deployment costs for LLM inference.

    Provides TCO calculations, $/million tokens, and
    cost comparisons between self-hosted and API options.

    Attributes:
        roofline_analyzer: Performance analyzer for throughput calculations
    """

    # Market average GPU pricing (2025, on-demand, USD/hour)
    DEFAULT_PRICING = {
        # Datacenter GPUs
        "H100-80GB": 4.50,   # Mid-range (AWS ~$3.90, Azure ~$6.98)
        "H100-96GB": 5.00,   # HBM3e variant
        "A100-40GB": 2.00,   # AWS ~$1.50-3.67
        "A100-80GB": 2.50,
        "MI300X": 5.50,      # AMD flagship
        "GB200": 10.00,      # Estimated (not yet available)

        # RTX 50 series (Blackwell)
        "RTX-5090": 0.35,    # $0.27-0.47/hr
        "RTX-5080": 0.25,    # Estimated
        "RTX-5070-Ti": 0.20, # Estimated
        "RTX-5070": 0.15,    # Estimated

        # RTX 40 series (Ada Lovelace)
        "RTX-4090": 0.21,    # $0.18-0.23/hr
        "RTX-4080-Super": 0.18,
        "RTX-4080": 0.17,
        "RTX-4070-Ti-Super": 0.15,
        "RTX-4070-Ti": 0.14,
        "RTX-4070-Super": 0.13,
        "RTX-4070": 0.12,

        # RTX 30 series (Ampere)
        "RTX-3090-Ti": 0.13,
        "RTX-3090": 0.12,    # $0.11-0.12/hr
        "RTX-3080-Ti": 0.11,
        "RTX-3080": 0.10,
        "RTX-3070-Ti": 0.09,
        "RTX-3070": 0.08,
        "RTX-3060-Ti": 0.07,
    }

    # Typical API pricing for comparison (USD per million tokens)
    # Based on 2025 market rates
    API_PRICING_PER_MILLION_TOKENS = {
        "gpt-4": 30.0,           # OpenAI GPT-4 (input)
        "gpt-3.5-turbo": 0.50,   # OpenAI GPT-3.5
        "claude-3-opus": 15.0,   # Anthropic Claude 3 Opus
        "claude-3-sonnet": 3.0,  # Anthropic Claude 3 Sonnet (uncached)
        "gemini-pro": 0.50,      # Google Gemini Pro
        "gemini-flash": 0.075,   # Google Gemini Flash (cheapest)
        "llama-3-8b": 0.20,      # Hosted Llama 3 8B (average)
        "llama-3-70b": 0.80,     # Hosted Llama 3 70B (average)
    }

    def __init__(
        self,
        roofline_analyzer: RooflineAnalyzer | None = None,
        pricing: dict[str, float] | None = None,
    ):
        """
        Initialize CostAnalyzer.

        Args:
            roofline_analyzer: Performance analyzer (creates default if None)
            pricing: Custom GPU pricing (uses defaults if None)
        """
        self._roofline = roofline_analyzer or RooflineAnalyzer()
        self._pricing = pricing or self.DEFAULT_PRICING.copy()
        self._logger = logging.getLogger(__name__)

    def analyze_cost(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        batch_size: int,
        input_length: int,
        output_length: int,
        requests_per_second: float,
        precision: PrecisionType = "fp16",
        custom_pricing: GPUPricing | None = None,
    ) -> CostMetrics:
        """
        Analyze deployment costs.

        Args:
            model: Model configuration
            hardware: Hardware specification
            batch_size: Batch size
            input_length: Average input length
            output_length: Average output length
            requests_per_second: Target requests per second
            precision: KV cache precision
            custom_pricing: Custom pricing (uses default if None)

        Returns:
            CostMetrics with detailed cost analysis

        Raises:
            ValueError: If parameters are invalid
        """
        if requests_per_second <= 0:
            raise ValueError(f"requests_per_second must be positive, got {requests_per_second}")

        # Get GPU pricing
        if custom_pricing:
            gpu_pricing = custom_pricing
        else:
            cost_per_gpu_hour = self._pricing.get(hardware.gpu_model, 1.0)
            gpu_pricing = GPUPricing(
                gpu_model=hardware.gpu_model,
                cost_per_hour_usd=cost_per_gpu_hour,
                source="default",
                pricing_type="on_demand",
            )

        # Calculate performance
        perf_metrics = self._roofline.predict_latency(
            model=model,
            hardware=hardware,
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            precision=precision,
        )

        # Total tokens per request
        tokens_per_request = input_length + output_length

        # Throughput (tokens/sec)
        throughput_tokens_per_sec = perf_metrics.throughput_tokens_per_sec

        # How many requests can we serve per second with this batch size?
        # Each batch takes total_latency_ms
        batches_per_second = 1000.0 / perf_metrics.total_latency_ms
        max_requests_per_second = batches_per_second * batch_size

        # Utilization: what fraction of capacity are we using?
        utilization_pct = min(100.0, (requests_per_second / max_requests_per_second) * 100)

        # Cost calculations
        cost_per_hour = gpu_pricing.cost_per_hour_usd * hardware.num_gpus

        # Tokens per hour at target load
        tokens_per_hour = throughput_tokens_per_sec * 3600 * (utilization_pct / 100)

        # Cost per million tokens
        if tokens_per_hour > 0:
            cost_per_million_tokens = (cost_per_hour / tokens_per_hour) * 1e6
        else:
            cost_per_million_tokens = float('inf')

        # Cost per request
        requests_per_hour = requests_per_second * 3600
        if requests_per_hour > 0:
            cost_per_request = cost_per_hour / requests_per_hour
        else:
            cost_per_request = float('inf')

        # Monthly and annual costs
        monthly_cost_usd = cost_per_hour * 24 * 30  # 30 days
        annual_cost_usd = cost_per_hour * 24 * 365  # 365 days

        # Tokens per dollar (inverse of cost)
        tokens_per_dollar = 1e6 / cost_per_million_tokens if cost_per_million_tokens > 0 else 0

        # Break-even: tokens/day to match typical API pricing
        # Use median API price (Claude Sonnet: $3/M tokens)
        api_price_per_million = 3.0
        daily_cost = cost_per_hour * 24
        break_even_tokens_per_day = int((daily_cost / api_price_per_million) * 1e6)

        self._logger.info(
            f"Cost analysis: {hardware.gpu_model} @ ${gpu_pricing.cost_per_hour_usd:.2f}/hr × {hardware.num_gpus} GPUs = "
            f"${cost_per_million_tokens:.2f}/M tokens "
            f"({utilization_pct:.1f}% utilization)"
        )

        return CostMetrics(
            cost_per_hour=cost_per_hour,
            cost_per_million_tokens=cost_per_million_tokens,
            cost_per_request=cost_per_request,
            monthly_cost_usd=monthly_cost_usd,
            annual_cost_usd=annual_cost_usd,
            utilization_pct=utilization_pct,
            tokens_per_dollar=tokens_per_dollar,
            break_even_tokens_per_day=break_even_tokens_per_day,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            requests_per_second=max_requests_per_second,
            gpu_pricing=gpu_pricing,
        )

    def compare_deployments(
        self,
        model: ModelConfig,
        hardware_options: list[HardwareSpec],
        batch_size: int,
        input_length: int,
        output_length: int,
        requests_per_second: float,
        precision: PrecisionType = "fp16",
    ) -> list[CostMetrics]:
        """
        Compare costs across different hardware options.

        Args:
            model: Model configuration
            hardware_options: List of hardware specs to compare
            batch_size: Batch size
            input_length: Average input length
            output_length: Average output length
            requests_per_second: Target requests per second
            precision: KV cache precision

        Returns:
            List of CostMetrics sorted by cost_per_million_tokens
        """
        results = []

        for hardware in hardware_options:
            try:
                metrics = self.analyze_cost(
                    model=model,
                    hardware=hardware,
                    batch_size=batch_size,
                    input_length=input_length,
                    output_length=output_length,
                    requests_per_second=requests_per_second,
                    precision=precision,
                )
                results.append(metrics)
            except Exception as e:
                self._logger.warning(f"Failed to analyze {hardware.gpu_model}: {e}")
                continue

        # Sort by cost per million tokens (ascending)
        results.sort(key=lambda m: m.cost_per_million_tokens)

        return results

    def break_even_analysis(
        self,
        self_hosted_cost_per_million: float,
        api_provider: str = "claude-3-sonnet",
    ) -> dict[str, float]:
        """
        Calculate break-even point vs API pricing.

        Args:
            self_hosted_cost_per_million: Self-hosted cost per million tokens
            api_provider: API provider to compare against

        Returns:
            Dictionary with break-even metrics
        """
        api_cost = self.API_PRICING_PER_MILLION_TOKENS.get(api_provider, 3.0)

        # Savings per million tokens
        savings_per_million = api_cost - self_hosted_cost_per_million
        savings_pct = (savings_per_million / api_cost) * 100 if api_cost > 0 else 0

        # Break-even: how many tokens to recover initial costs?
        # Assuming minimal setup cost (cloud GPUs), break-even is immediate
        # For on-prem, would need to factor in hardware purchase

        return {
            "api_cost_per_million": api_cost,
            "self_hosted_cost_per_million": self_hosted_cost_per_million,
            "savings_per_million": savings_per_million,
            "savings_pct": savings_pct,
            "break_even_tokens": 0,  # Immediate for cloud
        }

    def utilization_curve(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        batch_size: int,
        input_length: int,
        output_length: int,
        precision: PrecisionType = "fp16",
        utilization_points: list[float] | None = None,
    ) -> list[tuple[float, float]]:
        """
        Generate cost curve across different utilization levels.

        Args:
            model: Model configuration
            hardware: Hardware specification
            batch_size: Batch size
            input_length: Average input length
            output_length: Average output length
            precision: KV cache precision
            utilization_points: List of utilization percentages (default: 10%, 25%, 50%, 75%, 100%)

        Returns:
            List of (utilization_pct, cost_per_million_tokens) tuples
        """
        utilization_points = utilization_points or [10.0, 25.0, 50.0, 75.0, 100.0]

        # Calculate max throughput
        perf_metrics = self._roofline.predict_latency(
            model=model,
            hardware=hardware,
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            precision=precision,
        )

        batches_per_second = 1000.0 / perf_metrics.total_latency_ms
        max_requests_per_second = batches_per_second * batch_size

        curve = []

        for util_pct in utilization_points:
            target_rps = max_requests_per_second * (util_pct / 100)

            metrics = self.analyze_cost(
                model=model,
                hardware=hardware,
                batch_size=batch_size,
                input_length=input_length,
                output_length=output_length,
                requests_per_second=target_rps,
                precision=precision,
            )

            curve.append((util_pct, metrics.cost_per_million_tokens))

        return curve

    def set_pricing(self, gpu_model: str, cost_per_hour: float) -> None:
        """
        Set custom pricing for a GPU model.

        Args:
            gpu_model: GPU model name
            cost_per_hour: Cost per hour (USD)
        """
        self._pricing[gpu_model] = cost_per_hour
        self._logger.info(f"Updated pricing: {gpu_model} = ${cost_per_hour:.2f}/hr")

    def get_pricing(self, gpu_model: str) -> float:
        """
        Get pricing for a GPU model.

        Args:
            gpu_model: GPU model name

        Returns:
            Cost per hour (USD), or 1.0 if not found
        """
        return self._pricing.get(gpu_model, 1.0)
