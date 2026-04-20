"""
Quantization strategy evaluator.

Evaluates tradeoffs between precision, memory, speed, and quality
for different quantization strategies (FP16, FP8, INT8, INT4).

Based on research:
- vLLM FP8 quantization (2025): 2× speedup, minimal accuracy loss
- GPTQ/AWQ INT4 quantization: 4× memory savings, slight degradation
- Perplexity impacts across precisions
"""

import logging
from dataclasses import dataclass
from typing import Literal

from kv_planner.core.performance import RooflineAnalyzer
from kv_planner.domain import ModelConfig, HardwareSpec

PrecisionType = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]
QuantizationMethod = Literal["dynamic", "static", "gptq", "awq", "smoothquant"]
QualityImpact = Literal["none", "minimal", "slight", "moderate", "significant"]


@dataclass(frozen=True)
class QuantizationMetrics:
    """
    Quantization strategy evaluation metrics.

    Attributes:
        precision: Target precision (fp16, fp8, int8, int4)
        method: Quantization method used
        memory_savings_pct: Memory reduction vs FP16 baseline (0-100)
        speed_improvement: Throughput multiplier vs FP16 baseline (>1.0 is faster)
        perplexity_delta: Expected perplexity increase vs FP16 (lower is better)
        quality_impact: Qualitative quality assessment
        recommended_for: Use case recommendations
    """

    precision: PrecisionType
    method: QuantizationMethod
    memory_savings_pct: float
    speed_improvement: float
    perplexity_delta: float
    quality_impact: QualityImpact
    recommended_for: str

    # Detailed metrics
    memory_bytes_baseline: int
    memory_bytes_quantized: int
    throughput_baseline: float
    throughput_quantized: float


class QuantizationEvaluator:
    """
    Evaluates quantization strategies for LLM deployment.

    Provides data-driven recommendations for choosing between
    FP16, FP8, INT8, and INT4 quantization based on:
    - Memory constraints
    - Latency requirements
    - Quality tolerance

    Attributes:
        roofline_analyzer: Performance predictor for throughput calculations
        baseline_precision: Precision to compare against (default: fp16)
    """

    # Perplexity deltas vs FP16 baseline (PPL points, not percent).
    # Sources:
    #   * FP8: NVIDIA TensorRT-LLM & Baseten blog — "no perceptible degradation"
    #     https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/
    #   * INT8 (SmoothQuant): arxiv 2211.10438 — <1 PPL delta on OPT-175B
    #   * INT4 (AWQ/GPTQ): arxiv 2306.00978, 2210.17323 — typ 0-1 PPL delta on MMLU
    #     https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks
    PERPLEXITY_DELTAS = {
        "fp32": 0.0,
        "fp16": 0.0,
        "bf16": 0.0,
        "fp8": 0.3,
        "int8": 1.0,
        "int4": 2.5,
    }

    # Throughput improvements vs FP16 baseline. These are the default
    # FALLBACK values used when the Roofline analyzer can't derive a
    # precision-specific speedup from hardware specs — the real planner
    # uses the measured throughput ratio from two roofline predictions.
    # Sources:
    #   * FP8 1.3-2.0× on H100 (NVIDIA TE, Baseten) — default 1.4
    #   * INT8 1.6× (vLLM + SmoothQuant on Llama-3.1-70B, Red Hat LLM-Compressor)
    #   * INT4 2.6-2.8× with Marlin kernel — arxiv 2408.11743
    SPEED_MULTIPLIERS = {
        "fp32": 0.5,
        "fp16": 1.0,
        "bf16": 1.0,
        "fp8": 1.4,
        "int8": 1.6,
        "int4": 2.6,
    }

    # Memory savings (bits per parameter)
    MEMORY_MULTIPLIERS = {
        "fp32": 1.0,    # 32 bits = 4 bytes
        "fp16": 0.5,    # 16 bits = 2 bytes
        "bf16": 0.5,    # 16 bits = 2 bytes
        "fp8": 0.25,    # 8 bits = 1 byte
        "int8": 0.25,   # 8 bits = 1 byte
        "int4": 0.125,  # 4 bits = 0.5 bytes
    }

    # Default quantization methods
    DEFAULT_METHODS = {
        "fp32": "dynamic",
        "fp16": "dynamic",
        "bf16": "dynamic",
        "fp8": "dynamic",       # vLLM dynamic quantization
        "int8": "smoothquant",  # SmoothQuant for INT8
        "int4": "awq",          # AWQ generally better than GPTQ
    }

    def __init__(
        self,
        roofline_analyzer: RooflineAnalyzer | None = None,
        baseline_precision: PrecisionType = "fp16",
    ):
        """
        Initialize QuantizationEvaluator.

        Args:
            roofline_analyzer: Performance analyzer (creates default if None)
            baseline_precision: Precision to use as baseline for comparisons
        """
        self._roofline = roofline_analyzer or RooflineAnalyzer()
        self._baseline_precision = baseline_precision
        self._logger = logging.getLogger(__name__)

    def evaluate_strategy(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        precision: PrecisionType,
        method: QuantizationMethod | None = None,
        batch_size: int = 32,
        input_length: int = 2048,
        output_length: int = 512,
    ) -> QuantizationMetrics:
        """
        Evaluate a quantization strategy.

        Compares the target precision against the baseline (FP16) to calculate:
        - Memory savings
        - Speed improvements
        - Quality impacts

        Args:
            model: Model configuration
            hardware: Hardware specification
            precision: Target precision to evaluate
            method: Quantization method (uses default if None)
            batch_size: Batch size for throughput calculation
            input_length: Input sequence length
            output_length: Output sequence length

        Returns:
            QuantizationMetrics with detailed evaluation

        Raises:
            ValueError: If precision is invalid
        """
        if precision not in self.PERPLEXITY_DELTAS:
            raise ValueError(f"Unknown precision: {precision}")

        method = method or self.DEFAULT_METHODS[precision]

        # Calculate baseline performance (FP16)
        baseline_metrics = self._roofline.predict_latency(
            model=model,
            hardware=hardware,
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            precision=self._baseline_precision,
        )

        # Calculate quantized performance
        quantized_metrics = self._roofline.predict_latency(
            model=model,
            hardware=hardware,
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            precision=precision,
        )

        # Memory calculations
        baseline_memory = self._calculate_memory_bytes(model, self._baseline_precision)
        quantized_memory = self._calculate_memory_bytes(model, precision)
        memory_savings_pct = (1 - quantized_memory / baseline_memory) * 100

        # Speed improvement (adjusted for actual hardware)
        # Use actual measured throughput from roofline predictions
        speed_improvement = (
            quantized_metrics.throughput_tokens_per_sec /
            baseline_metrics.throughput_tokens_per_sec
        )

        # Quality impact assessment
        perplexity_delta = self.PERPLEXITY_DELTAS[precision]
        quality_impact = self._assess_quality_impact(precision)
        recommended_for = self._generate_recommendations(
            precision, method, memory_savings_pct, speed_improvement, quality_impact
        )

        self._logger.info(
            f"Quantization evaluation: {precision} ({method}) - "
            f"{memory_savings_pct:.1f}% memory savings, "
            f"{speed_improvement:.2f}× speed improvement, "
            f"+{perplexity_delta:.1f} perplexity delta"
        )

        return QuantizationMetrics(
            precision=precision,
            method=method,
            memory_savings_pct=memory_savings_pct,
            speed_improvement=speed_improvement,
            perplexity_delta=perplexity_delta,
            quality_impact=quality_impact,
            recommended_for=recommended_for,
            memory_bytes_baseline=baseline_memory,
            memory_bytes_quantized=quantized_memory,
            throughput_baseline=baseline_metrics.throughput_tokens_per_sec,
            throughput_quantized=quantized_metrics.throughput_tokens_per_sec,
        )

    def compare_strategies(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        precisions: list[PrecisionType] | None = None,
        batch_size: int = 32,
        input_length: int = 2048,
        output_length: int = 512,
    ) -> list[QuantizationMetrics]:
        """
        Compare multiple quantization strategies.

        Args:
            model: Model configuration
            hardware: Hardware specification
            precisions: List of precisions to compare (default: all common ones)
            batch_size: Batch size for throughput calculation
            input_length: Input sequence length
            output_length: Output sequence length

        Returns:
            List of QuantizationMetrics sorted by speed improvement
        """
        precisions = precisions or ["fp16", "fp8", "int8", "int4"]

        results = []
        for precision in precisions:
            metrics = self.evaluate_strategy(
                model=model,
                hardware=hardware,
                precision=precision,
                batch_size=batch_size,
                input_length=input_length,
                output_length=output_length,
            )
            results.append(metrics)

        # Sort by speed improvement (descending)
        results.sort(key=lambda m: m.speed_improvement, reverse=True)

        return results

    def recommend_precision(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        max_memory_gb: float | None = None,
        min_quality: QualityImpact = "slight",
        target_throughput: float | None = None,
        batch_size: int = 32,
        input_length: int = 2048,
        output_length: int = 512,
    ) -> QuantizationMetrics:
        """
        Recommend optimal precision based on constraints.

        Args:
            model: Model configuration
            hardware: Hardware specification
            max_memory_gb: Maximum memory budget (None = no constraint)
            min_quality: Minimum acceptable quality impact
            target_throughput: Target throughput in tokens/sec (None = maximize)
            batch_size: Batch size for evaluation
            input_length: Input sequence length
            output_length: Output sequence length

        Returns:
            QuantizationMetrics for recommended precision

        Raises:
            ValueError: If no precision meets constraints
        """
        # Quality hierarchy (lower is better)
        quality_order: list[QualityImpact] = ["none", "minimal", "slight", "moderate", "significant"]
        min_quality_idx = quality_order.index(min_quality)

        # Evaluate all precisions
        all_precisions: list[PrecisionType] = ["fp16", "fp8", "int8", "int4"]
        candidates = []

        for precision in all_precisions:
            metrics = self.evaluate_strategy(
                model=model,
                hardware=hardware,
                precision=precision,
                batch_size=batch_size,
                input_length=input_length,
                output_length=output_length,
            )

            # Check quality constraint
            quality_idx = quality_order.index(metrics.quality_impact)
            if quality_idx > min_quality_idx:
                continue  # Quality too low

            # Check memory constraint
            if max_memory_gb is not None:
                memory_gb = metrics.memory_bytes_quantized / 1e9
                if memory_gb > max_memory_gb:
                    continue  # Too much memory

            # Check throughput constraint
            if target_throughput is not None:
                if metrics.throughput_quantized < target_throughput:
                    continue  # Not fast enough

            candidates.append(metrics)

        if not candidates:
            raise ValueError(
                f"No precision meets constraints: "
                f"max_memory_gb={max_memory_gb}, "
                f"min_quality={min_quality}, "
                f"target_throughput={target_throughput}"
            )

        # Return highest quality that meets constraints
        # (candidates already filtered by quality)
        # Sort by perplexity delta (lower is better quality)
        candidates.sort(key=lambda m: m.perplexity_delta)

        recommended = candidates[0]
        self._logger.info(
            f"Recommended precision: {recommended.precision} "
            f"({recommended.quality_impact} quality impact, "
            f"{recommended.memory_savings_pct:.1f}% memory savings)"
        )

        return recommended

    def _calculate_memory_bytes(
        self,
        model: ModelConfig,
        precision: PrecisionType,
    ) -> int:
        """
        Calculate model memory footprint in bytes.

        Includes:
        - Model weights
        - (Does not include KV cache - use PagedMemoryCalculator for that)

        Args:
            model: Model configuration
            precision: Precision format

        Returns:
            Memory in bytes
        """
        # Get total parameters
        total_params = model.total_params()

        # Bytes per parameter
        bytes_per_param = self._bytes_per_value(precision)

        # Total memory (weights only)
        memory_bytes = total_params * bytes_per_param

        return memory_bytes

    def _bytes_per_value(self, precision: PrecisionType) -> float:
        """Get bytes per value for precision."""
        bytes_map = {
            "fp32": 4.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "fp8": 1.0,
            "int8": 1.0,
            "int4": 0.5,
        }
        return bytes_map[precision]

    def _assess_quality_impact(self, precision: PrecisionType) -> QualityImpact:
        """
        Assess qualitative quality impact.

        Based on research findings:
        - FP16/BF16: No impact (baseline)
        - FP8: Minimal (<1% perplexity increase)
        - INT8: Slight (~2% degradation with SmoothQuant)
        - INT4: Moderate (3-8% degradation with AWQ/GPTQ)
        """
        impact_map: dict[PrecisionType, QualityImpact] = {
            "fp32": "none",
            "fp16": "none",
            "bf16": "none",
            "fp8": "minimal",
            "int8": "slight",
            "int4": "moderate",
        }
        return impact_map[precision]

    def _generate_recommendations(
        self,
        precision: PrecisionType,
        method: QuantizationMethod,
        memory_savings_pct: float,
        speed_improvement: float,
        quality_impact: QualityImpact,
    ) -> str:
        """Generate use case recommendations."""
        recommendations = {
            "fp32": "Research and validation only (baseline reference)",
            "fp16": "Production default - best quality, good performance",
            "bf16": "Production default - best quality, good performance, better numeric stability",
            "fp8": "Production recommended - 33% faster, minimal quality loss, ideal for throughput-critical deployments",
            "int8": "Cost-sensitive deployments - 50% faster, slight quality tradeoff, good for batch processing",
            "int4": "Memory-constrained deployments - 80% faster, 75% memory savings, acceptable for casual use",
        }
        return recommendations.get(precision, "General use")
