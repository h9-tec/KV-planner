"""
Unified deployment planner.

Orchestrates all Phase 1-2 analyzers to generate complete
deployment plans with memory, performance, cost, and strategy recommendations.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from kv_planner.core.memory import PagedMemoryCalculator
from kv_planner.core.performance import RooflineAnalyzer, PerformanceMetrics
from kv_planner.core.strategies import (
    QuantizationEvaluator,
    QuantizationMetrics,
    PrefixCachingAnalyzer,
    PrefixCachingMetrics,
)
from kv_planner.core.cost import CostAnalyzer, CostMetrics
from kv_planner.domain import ModelConfig, HardwareSpec
from kv_planner.infrastructure.hardware_db import GPUDatabase
from kv_planner.infrastructure.hardware_db.laptop_adjustments import (
    is_laptop_gpu,
    adjust_performance_metrics,
    get_laptop_info,
)

PrecisionType = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]
OptimizationGoal = Literal["cost", "latency", "throughput", "quality", "balanced"]


@dataclass(frozen=True)
class DeploymentPlan:
    """
    Complete deployment plan with recommendations.

    Attributes:
        model: Model configuration
        hardware: Hardware specification
        optimization_goal: Optimization objective

        # Recommendations
        recommended_precision: Best precision for goal
        recommended_batch_size: Optimal batch size
        enable_prefix_caching: Whether to enable caching
        prefix_cache_length: Length of prefix to cache

        # Performance predictions
        performance: PerformanceMetrics

        # Cost analysis
        cost: CostMetrics

        # Strategy details
        quantization: QuantizationMetrics
        caching: PrefixCachingMetrics | None

        # Memory breakdown
        model_memory_gb: float
        kv_cache_memory_gb: float
        total_memory_gb: float
        memory_utilization_pct: float

        # Configuration
        vllm_config: dict[str, any]

        # Summary
        summary: str
    """

    model: ModelConfig
    hardware: HardwareSpec
    optimization_goal: OptimizationGoal

    # Recommendations
    recommended_precision: PrecisionType
    recommended_batch_size: int
    enable_prefix_caching: bool
    prefix_cache_length: int

    # Performance
    performance: PerformanceMetrics

    # Cost
    cost: CostMetrics

    # Strategies
    quantization: QuantizationMetrics
    caching: PrefixCachingMetrics | None

    # Memory
    model_memory_gb: float
    kv_cache_memory_gb: float
    total_memory_gb: float
    memory_utilization_pct: float

    # Configuration
    vllm_config: dict = field(default_factory=dict)

    # Summary
    summary: str = ""


class DeploymentPlanner:
    """
    Unified deployment planner.

    Orchestrates all analyzers to create complete deployment plans:
    - Memory analysis (PagedMemoryCalculator)
    - Performance prediction (RooflineAnalyzer)
    - Quantization optimization (QuantizationEvaluator)
    - Prefix caching (PrefixCachingAnalyzer)
    - Cost analysis (CostAnalyzer)

    Usage:
        planner = DeploymentPlanner()
        plan = planner.create_plan(
            model="meta-llama/Llama-3-8b-hf",
            hardware="RTX-5090",
            target_rps=10.0,
            optimization_goal="balanced",
        )
        print(plan.summary)
    """

    def __init__(
        self,
        memory_calculator: PagedMemoryCalculator | None = None,
        roofline_analyzer: RooflineAnalyzer | None = None,
        quantization_evaluator: QuantizationEvaluator | None = None,
        caching_analyzer: PrefixCachingAnalyzer | None = None,
        cost_analyzer: CostAnalyzer | None = None,
    ):
        """
        Initialize DeploymentPlanner.

        Args:
            memory_calculator: Memory calculator (creates default if None)
            roofline_analyzer: Performance analyzer (creates default if None)
            quantization_evaluator: Quantization evaluator (creates default if None)
            caching_analyzer: Caching analyzer (creates default if None)
            cost_analyzer: Cost analyzer (creates default if None)
        """
        self._memory_calc = memory_calculator or PagedMemoryCalculator()
        self._roofline = roofline_analyzer or RooflineAnalyzer()
        self._quantization = quantization_evaluator or QuantizationEvaluator(
            roofline_analyzer=self._roofline
        )
        self._caching = caching_analyzer or PrefixCachingAnalyzer(
            memory_calculator=self._memory_calc
        )
        self._cost = cost_analyzer or CostAnalyzer(roofline_analyzer=self._roofline)
        self._logger = logging.getLogger(__name__)

    def create_plan(
        self,
        model: ModelConfig | str,
        hardware: HardwareSpec | str,
        target_rps: float,
        input_length: int = 2048,
        output_length: int = 512,
        optimization_goal: OptimizationGoal = "balanced",
        max_memory_budget_gb: float | None = None,
        min_quality: Literal["none", "minimal", "slight", "moderate"] = "minimal",
        enable_caching: bool = True,
        system_prompt_length: int = 512,
    ) -> DeploymentPlan:
        """
        Create complete deployment plan.

        Args:
            model: Model config or HuggingFace model ID
            hardware: Hardware spec or GPU model name
            target_rps: Target requests per second
            input_length: Average input length
            output_length: Average output length
            optimization_goal: Optimization objective
            max_memory_budget_gb: Maximum memory budget (None = use available)
            min_quality: Minimum acceptable quality
            enable_caching: Whether to consider prefix caching
            system_prompt_length: System prompt length for caching

        Returns:
            DeploymentPlan with complete recommendations

        Raises:
            ValueError: If parameters are invalid
        """
        # Parse model and hardware
        if isinstance(model, str):
            model = self._parse_model(model)
        if isinstance(hardware, str):
            hardware = GPUDatabase.to_hardware_spec(hardware, num_gpus=1)

        self._logger.info(
            f"Creating deployment plan: {model.name} on {hardware.gpu_model}, "
            f"target {target_rps} RPS, goal={optimization_goal}"
        )

        # Step 1: Select optimal precision based on goal and constraints
        recommended_precision = self._select_precision(
            model=model,
            hardware=hardware,
            optimization_goal=optimization_goal,
            max_memory_budget_gb=max_memory_budget_gb or hardware.gpu_memory_gb,
            min_quality=min_quality,
            input_length=input_length,
            output_length=output_length,
        )

        self._logger.info(f"Selected precision: {recommended_precision}")

        # Step 2: Calculate optimal batch size
        recommended_batch_size = self._calculate_batch_size(
            model=model,
            hardware=hardware,
            precision=recommended_precision,
            target_rps=target_rps,
            input_length=input_length,
            output_length=output_length,
        )

        self._logger.info(f"Calculated batch size: {recommended_batch_size}")

        # Step 3: Evaluate prefix caching
        caching_metrics = None
        enable_prefix_caching = False
        prefix_cache_length = 0

        if enable_caching and system_prompt_length > 0:
            caching_metrics, enable_prefix_caching, prefix_cache_length = self._evaluate_caching(
                model=model,
                hardware=hardware,
                precision=recommended_precision,
                batch_size=recommended_batch_size,
                system_prompt_length=system_prompt_length,
                total_length=input_length,
            )

        self._logger.info(
            f"Prefix caching: {'enabled' if enable_prefix_caching else 'disabled'}"
        )

        # Step 4: Get performance metrics
        performance = self._roofline.predict_latency(
            model=model,
            hardware=hardware,
            batch_size=recommended_batch_size,
            input_length=input_length,
            output_length=output_length,
            precision=recommended_precision,
        )

        # Step 4.5: Apply laptop GPU adjustments if needed
        if is_laptop_gpu(hardware.gpu_model):
            adjusted_throughput, adjusted_total_latency = adjust_performance_metrics(
                gpu_model=hardware.gpu_model,
                throughput_tokens_per_sec=performance.throughput_tokens_per_sec,
                latency_ms=performance.total_latency_ms,
                profile="balanced",  # Use balanced profile by default
            )

            # Reconstruct PerformanceMetrics with adjusted values
            # Latency components scale inversely with throughput factor
            factor = adjusted_throughput / performance.throughput_tokens_per_sec
            adjusted_prefill_latency = performance.prefill_latency_ms / factor if factor > 0 else performance.prefill_latency_ms * 10
            adjusted_decode_latency = performance.decode_latency_ms / factor if factor > 0 else performance.decode_latency_ms * 10

            performance = PerformanceMetrics(
                prefill_latency_ms=adjusted_prefill_latency,
                decode_latency_ms=adjusted_decode_latency,
                total_latency_ms=adjusted_total_latency,
                prefill_tflops=performance.prefill_tflops * factor,  # TFLOPS scales with factor
                decode_tflops=performance.decode_tflops * factor,
                mfu=performance.mfu * factor,  # MFU scales with throughput
                mbu=performance.mbu * factor,  # MBU scales with throughput
                throughput_tokens_per_sec=adjusted_throughput,
                is_prefill_compute_bound=performance.is_prefill_compute_bound,
                is_decode_memory_bound=performance.is_decode_memory_bound,
                arithmetic_intensity=performance.arithmetic_intensity,
            )

            self._logger.info(
                f"Applied laptop GPU adjustment: {factor:.1%} performance retention"
            )

        # Step 5: Get quantization metrics
        quantization = self._quantization.evaluate_strategy(
            model=model,
            hardware=hardware,
            precision=recommended_precision,
            batch_size=recommended_batch_size,
            input_length=input_length,
            output_length=output_length,
        )

        # Step 6: Calculate costs
        cost = self._cost.analyze_cost(
            model=model,
            hardware=hardware,
            batch_size=recommended_batch_size,
            input_length=input_length,
            output_length=output_length,
            requests_per_second=target_rps,
            precision=recommended_precision,
        )

        # Step 7: Memory breakdown
        model_memory_gb = quantization.memory_bytes_quantized / 1e9

        kv_cache_memory_gb = self._memory_calc.calculate_kv_cache_size(
            batch_size=recommended_batch_size,
            sequence_length=input_length + output_length,
            model=model,
            precision=recommended_precision,
        ) / 1e9

        total_memory_gb = model_memory_gb + kv_cache_memory_gb
        memory_utilization_pct = (total_memory_gb / hardware.gpu_memory_gb) * 100

        # Step 8: Generate vLLM config
        vllm_config = self._generate_vllm_config(
            model=model,
            precision=recommended_precision,
            batch_size=recommended_batch_size,
            enable_prefix_caching=enable_prefix_caching,
            hardware=hardware,
        )

        # Step 9: Generate summary
        summary = self._generate_summary(
            model=model,
            hardware=hardware,
            precision=recommended_precision,
            batch_size=recommended_batch_size,
            performance=performance,
            cost=cost,
            quantization=quantization,
            caching_metrics=caching_metrics,
            enable_prefix_caching=enable_prefix_caching,
            optimization_goal=optimization_goal,
        )

        return DeploymentPlan(
            model=model,
            hardware=hardware,
            optimization_goal=optimization_goal,
            recommended_precision=recommended_precision,
            recommended_batch_size=recommended_batch_size,
            enable_prefix_caching=enable_prefix_caching,
            prefix_cache_length=prefix_cache_length,
            performance=performance,
            cost=cost,
            quantization=quantization,
            caching=caching_metrics,
            model_memory_gb=model_memory_gb,
            kv_cache_memory_gb=kv_cache_memory_gb,
            total_memory_gb=total_memory_gb,
            memory_utilization_pct=memory_utilization_pct,
            vllm_config=vllm_config,
            summary=summary,
        )

    def _select_precision(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        optimization_goal: OptimizationGoal,
        max_memory_budget_gb: float,
        min_quality: str,
        input_length: int,
        output_length: int,
    ) -> PrecisionType:
        """Select optimal precision based on goal."""
        # Goal-based precision selection
        if optimization_goal == "quality":
            # Prefer highest quality
            return "fp16"

        elif optimization_goal == "cost":
            # Prefer cheapest (INT4 if quality allows)
            try:
                recommended = self._quantization.recommend_precision(
                    model=model,
                    hardware=hardware,
                    max_memory_gb=max_memory_budget_gb,
                    min_quality=min_quality,  # type: ignore
                    batch_size=32,
                    input_length=input_length,
                    output_length=output_length,
                )
                return recommended.precision
            except Exception:
                return "fp8"  # Fallback

        elif optimization_goal == "latency":
            # Prefer fastest (FP8 good balance)
            return "fp8"

        elif optimization_goal == "throughput":
            # Prefer highest throughput (FP8 or INT4)
            return "fp8"

        else:  # balanced
            # Prefer FP8 (good balance of all metrics)
            return "fp8"

    def _calculate_batch_size(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        precision: PrecisionType,
        target_rps: float,
        input_length: int,
        output_length: int,
    ) -> int:
        """Calculate optimal batch size for target RPS."""
        # Start with max batch size that fits in memory
        max_batch = self._memory_calc.max_batch_size(
            available_memory_gb=hardware.gpu_memory_gb * 0.9,  # Leave 10% headroom
            sequence_length=input_length + output_length,
            model=model,
            precision=precision,
        )

        # Try different batch sizes and find one that meets target RPS
        candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        candidates = [b for b in candidates if b <= max_batch]

        if not candidates:
            return 1

        best_batch = candidates[0]

        for batch_size in candidates:
            perf = self._roofline.predict_latency(
                model=model,
                hardware=hardware,
                batch_size=batch_size,
                input_length=input_length,
                output_length=output_length,
                precision=precision,
            )

            # Requests per second this batch can handle
            batches_per_sec = 1000.0 / perf.total_latency_ms
            rps = batches_per_sec * batch_size

            if rps >= target_rps:
                best_batch = batch_size
                break
            else:
                best_batch = batch_size  # Keep trying larger

        return min(best_batch, max_batch)

    def _evaluate_caching(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        precision: PrecisionType,
        batch_size: int,
        system_prompt_length: int,
        total_length: int,
    ) -> tuple[PrefixCachingMetrics, bool, int]:
        """Evaluate if prefix caching is beneficial."""
        # Assume high hit rate for system prompts
        hit_rate = 0.9

        metrics = self._caching.evaluate_caching(
            model=model,
            hardware=hardware,
            prefix_length=system_prompt_length,
            total_length=total_length,
            batch_size=batch_size,
            hit_rate=hit_rate,
            precision=precision,
        )

        # Enable caching if savings > 20%
        enable = metrics.latency_reduction_pct > 20.0

        return metrics, enable, system_prompt_length if enable else 0

    def _generate_vllm_config(
        self,
        model: ModelConfig,
        precision: PrecisionType,
        batch_size: int,
        enable_prefix_caching: bool,
        hardware: HardwareSpec,
    ) -> dict:
        """Generate vLLM configuration."""
        # Map precision to vLLM dtype
        dtype_map = {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp8": "float8_e4m3fn",
            "int8": "int8",
            "int4": "int4",
        }

        config = {
            "model": model.name,
            "dtype": dtype_map.get(precision, "float16"),
            "max_model_len": model.max_position_embeddings,
            "max_num_seqs": batch_size,
            "gpu_memory_utilization": 0.9,
            "enable_prefix_caching": enable_prefix_caching,
            "tensor_parallel_size": hardware.num_gpus,
        }

        # Add quantization config if needed
        if precision in ["fp8", "int8", "int4"]:
            config["quantization"] = precision.upper()

        return config

    def _generate_summary(
        self,
        model: ModelConfig,
        hardware: HardwareSpec,
        precision: PrecisionType,
        batch_size: int,
        performance: PerformanceMetrics,
        cost: CostMetrics,
        quantization: QuantizationMetrics,
        caching_metrics: PrefixCachingMetrics | None,
        enable_prefix_caching: bool,
        optimization_goal: OptimizationGoal,
    ) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 80,
            "DEPLOYMENT PLAN",
            "=" * 80,
            "",
            f"Model: {model.name}",
            f"Hardware: {hardware.num_gpus}× {hardware.gpu_model}",
            f"Optimization Goal: {optimization_goal}",
            "",
            "RECOMMENDATIONS",
            "-" * 80,
            f"  Precision: {precision.upper()} ({quantization.quality_impact} quality impact)",
            f"  Batch Size: {batch_size}",
            f"  Prefix Caching: {'Enabled' if enable_prefix_caching else 'Disabled'}",
            "",
            "PERFORMANCE",
            "-" * 80,
            f"  Throughput: {performance.throughput_tokens_per_sec:,.0f} tokens/sec",
            f"  Latency: {performance.total_latency_ms:.0f} ms (prefill: {performance.prefill_latency_ms:.0f} ms, decode: {performance.decode_latency_ms:.0f} ms)",
            f"  MFU: {performance.mfu*100:.1f}%, MBU: {performance.mbu*100:.1f}%",
            "",
            "COST",
            "-" * 80,
            f"  Cost per hour: ${cost.cost_per_hour:.2f}",
            f"  Cost per million tokens: ${cost.cost_per_million_tokens:.2f}",
            f"  Monthly cost: ${cost.monthly_cost_usd:,.2f}",
            f"  GPU utilization: {cost.utilization_pct:.1f}%",
            "",
            "SAVINGS",
            "-" * 80,
            f"  Memory savings: {quantization.memory_savings_pct:.0f}% (vs FP16)",
            f"  Speed improvement: {quantization.speed_improvement:.2f}× (vs FP16)",
        ]

        if caching_metrics and enable_prefix_caching:
            lines.extend([
                f"  Caching latency reduction: {caching_metrics.latency_reduction_pct:.0f}%",
                f"  Caching memory savings: {caching_metrics.memory_savings_pct:.0f}%",
            ])

        lines.extend([
            "",
            "=" * 80,
        ])

        return "\n".join(lines)

    def _parse_model(self, model_id: str) -> ModelConfig:
        """Parse model ID to ModelConfig (simplified)."""
        # For now, just support Llama 3 8B
        # In production, would query HuggingFace API
        if "llama-3-8b" in model_id.lower() or "llama3-8b" in model_id.lower():
            return ModelConfig(
                name=model_id,
                num_layers=32,
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                vocab_size=128256,
                max_position_embeddings=8192,
            )
        else:
            raise ValueError(
                f"Model {model_id} not recognized. Please provide ModelConfig directly."
            )

    def compare_options(
        self,
        model: ModelConfig | str,
        hardware_options: list[str],
        target_rps: float,
        input_length: int = 2048,
        output_length: int = 512,
        optimization_goal: OptimizationGoal = "balanced",
    ) -> list[DeploymentPlan]:
        """
        Compare deployment plans across different hardware options.

        Args:
            model: Model config or ID
            hardware_options: List of GPU model names
            target_rps: Target requests per second
            input_length: Average input length
            output_length: Average output length
            optimization_goal: Optimization objective

        Returns:
            List of DeploymentPlan sorted by cost_per_million_tokens
        """
        plans = []

        for gpu_model in hardware_options:
            try:
                plan = self.create_plan(
                    model=model,
                    hardware=gpu_model,
                    target_rps=target_rps,
                    input_length=input_length,
                    output_length=output_length,
                    optimization_goal=optimization_goal,
                )
                plans.append(plan)
            except Exception as e:
                self._logger.warning(f"Failed to create plan for {gpu_model}: {e}")
                continue

        # Sort by cost per million tokens
        plans.sort(key=lambda p: p.cost.cost_per_million_tokens)

        return plans
