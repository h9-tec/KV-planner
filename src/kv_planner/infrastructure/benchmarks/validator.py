"""
Validator for comparing kv-planner predictions against vLLM benchmark results.

This module provides tools to:
- Compare predicted vs actual performance
- Calculate prediction accuracy
- Identify systematic biases
- Suggest model parameter tuning
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

from kv_planner.application import DeploymentPlan
from kv_planner.infrastructure.benchmarks.runner import BenchmarkResults


@dataclass
class MetricComparison:
    """
    Comparison of predicted vs actual metric.

    Attributes:
        metric_name: Name of the metric
        predicted: Predicted value
        actual: Actual measured value
        error: Absolute error (actual - predicted)
        error_pct: Error as percentage of actual
        within_tolerance: Whether error is within acceptable tolerance
    """

    metric_name: str
    predicted: float
    actual: float
    error: float
    error_pct: float
    within_tolerance: bool

    @property
    def is_overestimate(self) -> bool:
        """Whether prediction overestimates actual value."""
        return self.predicted > self.actual

    @property
    def is_underestimate(self) -> bool:
        """Whether prediction underestimates actual value."""
        return self.predicted < self.actual


@dataclass
class ValidationResults:
    """
    Results from validating predictions against benchmarks.

    Attributes:
        plan: Deployment plan being validated
        benchmark: Benchmark results
        comparisons: Individual metric comparisons
        overall_accuracy: Overall prediction accuracy (0-100%)
        passed: Whether validation passed (all metrics within tolerance)

        # Summary statistics
        mean_error_pct: Mean absolute error percentage
        max_error_pct: Maximum error percentage
        systematic_bias: "overestimate", "underestimate", or "balanced"

        # Tuning suggestions
        tuning_suggestions: List of suggested parameter adjustments
    """

    plan: DeploymentPlan
    benchmark: BenchmarkResults
    comparisons: list[MetricComparison] = field(default_factory=list)
    overall_accuracy: float = 0.0
    passed: bool = False

    mean_error_pct: float = 0.0
    max_error_pct: float = 0.0
    systematic_bias: Literal["overestimate", "underestimate", "balanced"] = "balanced"

    tuning_suggestions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 80,
            "VALIDATION RESULTS",
            "=" * 80,
            "",
            f"Model: {self.plan.model.name}",
            f"Hardware: {self.plan.hardware.gpu_model}",
            f"Precision: {self.plan.recommended_precision}",
            f"Batch Size: {self.plan.recommended_batch_size}",
            "",
            "ACCURACY SUMMARY",
            "-" * 80,
            f"Overall Accuracy: {self.overall_accuracy:.1f}%",
            f"Mean Error: {self.mean_error_pct:.1f}%",
            f"Max Error: {self.max_error_pct:.1f}%",
            f"Systematic Bias: {self.systematic_bias}",
            f"Validation: {'✅ PASSED' if self.passed else '❌ FAILED'}",
            "",
            "METRIC COMPARISONS",
            "-" * 80,
        ]

        for comp in self.comparisons:
            status = "✅" if comp.within_tolerance else "❌"
            direction = "↑" if comp.is_overestimate else "↓"
            lines.append(
                f"{status} {comp.metric_name:<25} "
                f"Predicted: {comp.predicted:>12,.1f}  "
                f"Actual: {comp.actual:>12,.1f}  "
                f"Error: {direction}{abs(comp.error_pct):>6.1f}%"
            )

        if self.tuning_suggestions:
            lines.extend([
                "",
                "TUNING SUGGESTIONS",
                "-" * 80,
            ])
            for i, suggestion in enumerate(self.tuning_suggestions, 1):
                lines.append(f"{i}. {suggestion}")

        lines.append("=" * 80)

        return "\n".join(lines)


class PredictionValidator:
    """
    Validator for comparing predictions against benchmark results.
    """

    def __init__(
        self,
        tolerance_pct: float = 20.0,
        strict_tolerance_pct: float = 10.0,
    ) -> None:
        """
        Initialize validator.

        Args:
            tolerance_pct: Acceptable error tolerance (default: 20%)
            strict_tolerance_pct: Strict tolerance for critical metrics (default: 10%)
        """
        self.tolerance_pct = tolerance_pct
        self.strict_tolerance_pct = strict_tolerance_pct

    def validate(
        self,
        plan: DeploymentPlan,
        benchmark: BenchmarkResults,
    ) -> ValidationResults:
        """
        Validate deployment plan against benchmark results.

        Args:
            plan: Deployment plan with predictions
            benchmark: Benchmark results

        Returns:
            ValidationResults
        """
        results = ValidationResults(plan=plan, benchmark=benchmark)

        # Compare metrics
        if benchmark.throughput_tokens_per_sec is not None:
            results.comparisons.append(
                self._compare_metric(
                    "Throughput (tok/s)",
                    plan.performance.throughput_tokens_per_sec,
                    benchmark.throughput_tokens_per_sec,
                    tolerance_pct=self.strict_tolerance_pct,
                )
            )

        if benchmark.mean_latency_ms is not None:
            # Convert plan latency to ms
            plan_latency_ms = plan.performance.total_latency_ms
            results.comparisons.append(
                self._compare_metric(
                    "Mean Latency (ms)",
                    plan_latency_ms,
                    benchmark.mean_latency_ms,
                    tolerance_pct=self.tolerance_pct,
                )
            )

        if benchmark.time_to_first_token_ms is not None:
            # Use prefill latency as TTFT proxy
            plan_ttft_ms = plan.performance.prefill_latency_ms
            results.comparisons.append(
                self._compare_metric(
                    "TTFT (ms)",
                    plan_ttft_ms,
                    benchmark.time_to_first_token_ms,
                    tolerance_pct=self.tolerance_pct,
                )
            )

        if benchmark.time_per_output_token_ms is not None:
            # Use decode latency per token as TPOT proxy
            plan_tpot_ms = plan.performance.decode_latency_ms / max(1, plan.recommended_batch_size)
            results.comparisons.append(
                self._compare_metric(
                    "TPOT (ms)",
                    plan_tpot_ms,
                    benchmark.time_per_output_token_ms,
                    tolerance_pct=self.tolerance_pct,
                )
            )

        if benchmark.peak_gpu_memory_gb is not None:
            results.comparisons.append(
                self._compare_metric(
                    "Peak Memory (GB)",
                    plan.memory.total_memory_gb,
                    benchmark.peak_gpu_memory_gb,
                    tolerance_pct=self.tolerance_pct,
                )
            )

        # Calculate summary statistics
        if results.comparisons:
            error_pcts = [abs(c.error_pct) for c in results.comparisons]
            results.mean_error_pct = sum(error_pcts) / len(error_pcts)
            results.max_error_pct = max(error_pcts)
            results.overall_accuracy = 100.0 - results.mean_error_pct

            # Check if passed
            results.passed = all(c.within_tolerance for c in results.comparisons)

            # Determine systematic bias
            overestimates = sum(1 for c in results.comparisons if c.is_overestimate)
            underestimates = sum(1 for c in results.comparisons if c.is_underestimate)

            if overestimates > underestimates * 1.5:
                results.systematic_bias = "overestimate"
            elif underestimates > overestimates * 1.5:
                results.systematic_bias = "underestimate"
            else:
                results.systematic_bias = "balanced"

            # Generate tuning suggestions
            results.tuning_suggestions = self._generate_tuning_suggestions(results)

        return results

    def _compare_metric(
        self,
        metric_name: str,
        predicted: float,
        actual: float,
        tolerance_pct: float,
    ) -> MetricComparison:
        """
        Compare predicted vs actual metric.

        Args:
            metric_name: Name of metric
            predicted: Predicted value
            actual: Actual measured value
            tolerance_pct: Acceptable error tolerance

        Returns:
            MetricComparison
        """
        error = actual - predicted
        error_pct = (error / actual) * 100.0 if actual != 0 else 0.0
        within_tolerance = abs(error_pct) <= tolerance_pct

        return MetricComparison(
            metric_name=metric_name,
            predicted=predicted,
            actual=actual,
            error=error,
            error_pct=error_pct,
            within_tolerance=within_tolerance,
        )

    def _generate_tuning_suggestions(
        self,
        results: ValidationResults,
    ) -> list[str]:
        """
        Generate tuning suggestions based on validation results.

        Args:
            results: Validation results

        Returns:
            List of tuning suggestions
        """
        suggestions = []

        # Find specific metric issues
        throughput_comp = next(
            (c for c in results.comparisons if "Throughput" in c.metric_name), None
        )
        latency_comp = next(
            (c for c in results.comparisons if "Latency" in c.metric_name), None
        )
        memory_comp = next(
            (c for c in results.comparisons if "Memory" in c.metric_name), None
        )

        # Throughput suggestions
        if throughput_comp and not throughput_comp.within_tolerance:
            if throughput_comp.is_overestimate:
                suggestions.append(
                    f"Throughput overestimated by {abs(throughput_comp.error_pct):.1f}%. "
                    "Consider reducing MFU/MBU estimates in roofline model."
                )
            else:
                suggestions.append(
                    f"Throughput underestimated by {abs(throughput_comp.error_pct):.1f}%. "
                    "Consider increasing MFU/MBU estimates or checking for bottlenecks."
                )

        # Latency suggestions
        if latency_comp and not latency_comp.within_tolerance:
            if latency_comp.is_overestimate:
                suggestions.append(
                    f"Latency overestimated by {abs(latency_comp.error_pct):.1f}%. "
                    "Roofline model may be too conservative. Check arithmetic intensity."
                )
            else:
                suggestions.append(
                    f"Latency underestimated by {abs(latency_comp.error_pct):.1f}%. "
                    "Consider accounting for scheduling overhead or memory bottlenecks."
                )

        # Memory suggestions
        if memory_comp and not memory_comp.within_tolerance:
            if memory_comp.is_overestimate:
                suggestions.append(
                    f"Memory overestimated by {abs(memory_comp.error_pct):.1f}%. "
                    "PagedAttention may be more efficient than modeled."
                )
            else:
                suggestions.append(
                    f"Memory underestimated by {abs(memory_comp.error_pct):.1f}%. "
                    "Check for unaccounted overhead (activations, temp buffers)."
                )

        # Systematic bias suggestions
        if results.systematic_bias == "overestimate":
            suggestions.append(
                "Systematic overestimation detected across metrics. "
                "Consider adjusting baseline efficiency assumptions upward."
            )
        elif results.systematic_bias == "underestimate":
            suggestions.append(
                "Systematic underestimation detected across metrics. "
                "Consider adding safety margins or accounting for real-world overhead."
            )

        # High error suggestions
        if results.max_error_pct > 50.0:
            suggestions.append(
                f"Maximum error is {results.max_error_pct:.1f}%, indicating model limitations. "
                "Consider validating hardware specs and model architecture assumptions."
            )

        return suggestions

    def validate_multiple(
        self,
        plans_and_benchmarks: list[tuple[DeploymentPlan, BenchmarkResults]],
    ) -> list[ValidationResults]:
        """
        Validate multiple plan-benchmark pairs.

        Args:
            plans_and_benchmarks: List of (plan, benchmark) tuples

        Returns:
            List of ValidationResults
        """
        return [
            self.validate(plan, benchmark)
            for plan, benchmark in plans_and_benchmarks
        ]

    def aggregate_results(
        self,
        validation_results: list[ValidationResults],
    ) -> dict:
        """
        Aggregate results across multiple validations.

        Args:
            validation_results: List of validation results

        Returns:
            Aggregated statistics
        """
        if not validation_results:
            return {}

        total_comparisons = sum(len(r.comparisons) for r in validation_results)
        passed_validations = sum(1 for r in validation_results if r.passed)

        all_errors = [
            abs(c.error_pct)
            for r in validation_results
            for c in r.comparisons
        ]

        return {
            "num_validations": len(validation_results),
            "num_comparisons": total_comparisons,
            "passed_validations": passed_validations,
            "pass_rate": passed_validations / len(validation_results) * 100.0,
            "mean_error_pct": sum(all_errors) / len(all_errors) if all_errors else 0.0,
            "max_error_pct": max(all_errors) if all_errors else 0.0,
            "min_error_pct": min(all_errors) if all_errors else 0.0,
            "overall_accuracy": 100.0 - (sum(all_errors) / len(all_errors) if all_errors else 0.0),
        }
