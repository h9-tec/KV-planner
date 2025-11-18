"""Unit tests for QuantizationEvaluator."""

import pytest

from kv_planner.core.strategies import QuantizationEvaluator, QuantizationMetrics
from kv_planner.core.performance import RooflineAnalyzer
from kv_planner.domain import ModelConfig, HardwareSpec


class TestQuantizationEvaluator:
    """Test suite for QuantizationEvaluator."""

    @pytest.fixture
    def evaluator(self) -> QuantizationEvaluator:
        """Create quantization evaluator with default settings."""
        return QuantizationEvaluator()

    @pytest.fixture
    def evaluator_with_analyzer(self) -> QuantizationEvaluator:
        """Create quantization evaluator with custom analyzer."""
        analyzer = RooflineAnalyzer(compute_efficiency=0.7, memory_efficiency=0.85)
        return QuantizationEvaluator(roofline_analyzer=analyzer)

    def test_evaluator_initialization(self) -> None:
        """Test evaluator initialization."""
        evaluator = QuantizationEvaluator(baseline_precision="fp16")
        assert evaluator._baseline_precision == "fp16"
        assert evaluator._roofline is not None

    def test_evaluator_with_custom_baseline(self) -> None:
        """Test evaluator with custom baseline precision."""
        evaluator = QuantizationEvaluator(baseline_precision="bf16")
        assert evaluator._baseline_precision == "bf16"

    def test_evaluate_fp16_baseline(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test evaluating FP16 (should be baseline)."""
        metrics = evaluator.evaluate_strategy(
            model=llama3_8b,
            hardware=h100_single,
            precision="fp16",
            batch_size=32,
            input_length=2048,
            output_length=512,
        )

        # FP16 vs FP16 should have no savings, 1.0× speed
        assert metrics.precision == "fp16"
        assert metrics.memory_savings_pct == pytest.approx(0.0, abs=0.1)
        assert metrics.speed_improvement == pytest.approx(1.0, abs=0.01)
        assert metrics.perplexity_delta == 0.0
        assert metrics.quality_impact == "none"

    def test_evaluate_fp8_strategy(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test FP8 quantization strategy."""
        metrics = evaluator.evaluate_strategy(
            model=llama3_8b,
            hardware=h100_single,
            precision="fp8",
            batch_size=32,
            input_length=2048,
            output_length=512,
        )

        # FP8 should save ~50% memory (2 bytes -> 1 byte)
        assert metrics.precision == "fp8"
        assert 40 <= metrics.memory_savings_pct <= 60  # ~50%
        assert metrics.speed_improvement > 1.0  # Faster
        assert metrics.perplexity_delta == 0.5  # Minimal
        assert metrics.quality_impact == "minimal"
        assert "throughput-critical" in metrics.recommended_for.lower()

    def test_evaluate_int4_strategy(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test INT4 quantization strategy."""
        metrics = evaluator.evaluate_strategy(
            model=llama3_8b,
            hardware=h100_single,
            precision="int4",
            batch_size=32,
            input_length=2048,
            output_length=512,
        )

        # INT4 should save ~75% memory (2 bytes -> 0.5 bytes)
        assert metrics.precision == "int4"
        assert 70 <= metrics.memory_savings_pct <= 80  # ~75%
        assert metrics.speed_improvement > 1.0  # Faster
        assert metrics.perplexity_delta == 5.0  # Moderate
        assert metrics.quality_impact == "moderate"
        assert "memory-constrained" in metrics.recommended_for.lower()

    def test_compare_strategies(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test comparing multiple strategies."""
        results = evaluator.compare_strategies(
            model=llama3_8b,
            hardware=h100_single,
            precisions=["fp16", "fp8", "int4"],
            batch_size=32,
            input_length=2048,
            output_length=512,
        )

        # Should have 3 results
        assert len(results) == 3

        # Should be sorted by speed improvement (descending)
        for i in range(len(results) - 1):
            assert results[i].speed_improvement >= results[i + 1].speed_improvement

        # INT4 should have highest memory savings
        int4_metrics = next(m for m in results if m.precision == "int4")
        fp16_metrics = next(m for m in results if m.precision == "fp16")
        assert int4_metrics.memory_savings_pct > fp16_metrics.memory_savings_pct

    def test_recommend_precision_no_constraints(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test precision recommendation with no constraints."""
        recommended = evaluator.recommend_precision(
            model=llama3_8b,
            hardware=h100_single,
            min_quality="moderate",  # Allow all precisions
            batch_size=32,
            input_length=2048,
            output_length=512,
        )

        # Should recommend FP16 (best quality)
        assert recommended.precision in ["fp16", "bf16"]
        assert recommended.quality_impact in ["none", "minimal"]

    def test_recommend_precision_memory_constrained(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test precision recommendation with memory constraint."""
        # Get FP16 memory usage
        fp16_metrics = evaluator.evaluate_strategy(
            model=llama3_8b,
            hardware=h100_single,
            precision="fp16",
        )

        # Set constraint to 60% of FP16 memory (should force INT4)
        max_memory = fp16_metrics.memory_bytes_baseline * 0.6 / 1e9

        recommended = evaluator.recommend_precision(
            model=llama3_8b,
            hardware=h100_single,
            max_memory_gb=max_memory,
            min_quality="moderate",  # Allow INT4
            batch_size=32,
            input_length=2048,
            output_length=512,
        )

        # Should recommend INT4 or INT8 (more aggressive quantization)
        assert recommended.precision in ["int4", "int8", "fp8"]
        assert recommended.memory_bytes_quantized / 1e9 <= max_memory

    def test_recommend_precision_quality_constrained(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test precision recommendation with quality constraint."""
        recommended = evaluator.recommend_precision(
            model=llama3_8b,
            hardware=h100_single,
            min_quality="minimal",  # Only FP16, FP8
            batch_size=32,
            input_length=2048,
            output_length=512,
        )

        # Should recommend FP16 or FP8 only
        assert recommended.precision in ["fp16", "bf16", "fp8"]
        assert recommended.quality_impact in ["none", "minimal"]

    def test_recommend_precision_impossible_constraints(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test that impossible constraints raise error."""
        with pytest.raises(ValueError, match="No precision meets constraints"):
            evaluator.recommend_precision(
                model=llama3_8b,
                hardware=h100_single,
                max_memory_gb=0.1,  # Impossibly low
                min_quality="none",
                batch_size=32,
                input_length=2048,
                output_length=512,
            )

    def test_invalid_precision(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test that invalid precision is rejected."""
        with pytest.raises(ValueError, match="Unknown precision"):
            evaluator.evaluate_strategy(
                model=llama3_8b,
                hardware=h100_single,
                precision="fp64",  # type: ignore
            )

    def test_memory_calculation(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig
    ) -> None:
        """Test memory calculation for different precisions."""
        # FP16: 2 bytes per parameter
        fp16_memory = evaluator._calculate_memory_bytes(llama3_8b, "fp16")

        # FP8: 1 byte per parameter (should be ~50% of FP16)
        fp8_memory = evaluator._calculate_memory_bytes(llama3_8b, "fp8")
        assert fp8_memory == pytest.approx(fp16_memory * 0.5, rel=0.01)

        # INT4: 0.5 bytes per parameter (should be ~25% of FP16)
        int4_memory = evaluator._calculate_memory_bytes(llama3_8b, "int4")
        assert int4_memory == pytest.approx(fp16_memory * 0.25, rel=0.01)

        # FP32: 4 bytes per parameter (should be 2× FP16)
        fp32_memory = evaluator._calculate_memory_bytes(llama3_8b, "fp32")
        assert fp32_memory == pytest.approx(fp16_memory * 2.0, rel=0.01)

    def test_quality_impact_assessment(self, evaluator: QuantizationEvaluator) -> None:
        """Test quality impact assessment."""
        assert evaluator._assess_quality_impact("fp16") == "none"
        assert evaluator._assess_quality_impact("fp8") == "minimal"
        assert evaluator._assess_quality_impact("int8") == "slight"
        assert evaluator._assess_quality_impact("int4") == "moderate"

    def test_bytes_per_value(self, evaluator: QuantizationEvaluator) -> None:
        """Test bytes per value for each precision."""
        assert evaluator._bytes_per_value("fp32") == 4.0
        assert evaluator._bytes_per_value("fp16") == 2.0
        assert evaluator._bytes_per_value("bf16") == 2.0
        assert evaluator._bytes_per_value("fp8") == 1.0
        assert evaluator._bytes_per_value("int8") == 1.0
        assert evaluator._bytes_per_value("int4") == 0.5

    def test_perplexity_deltas(self, evaluator: QuantizationEvaluator) -> None:
        """Test perplexity delta values are reasonable."""
        # FP16 baseline
        assert evaluator.PERPLEXITY_DELTAS["fp16"] == 0.0

        # FP8 should have minimal impact
        assert 0 < evaluator.PERPLEXITY_DELTAS["fp8"] < 1.0

        # INT4 should have moderate impact
        assert evaluator.PERPLEXITY_DELTAS["int4"] > evaluator.PERPLEXITY_DELTAS["int8"]
        assert evaluator.PERPLEXITY_DELTAS["int8"] > evaluator.PERPLEXITY_DELTAS["fp8"]

    def test_speed_multipliers(self, evaluator: QuantizationEvaluator) -> None:
        """Test speed multiplier values are reasonable."""
        # FP16 baseline
        assert evaluator.SPEED_MULTIPLIERS["fp16"] == 1.0

        # Lower precision should be faster
        assert evaluator.SPEED_MULTIPLIERS["fp8"] > 1.0
        assert evaluator.SPEED_MULTIPLIERS["int4"] > evaluator.SPEED_MULTIPLIERS["int8"]
        assert evaluator.SPEED_MULTIPLIERS["int8"] > evaluator.SPEED_MULTIPLIERS["fp16"]

    def test_rtx_gpu_quantization(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig
    ) -> None:
        """Test quantization evaluation on RTX GPUs."""
        from kv_planner.infrastructure.hardware_db import GPUDatabase

        rtx_5090 = GPUDatabase.to_hardware_spec("RTX-5090", num_gpus=1)

        # Evaluate FP16 vs FP8
        fp16_metrics = evaluator.evaluate_strategy(
            model=llama3_8b,
            hardware=rtx_5090,
            precision="fp16",
            batch_size=32,
            input_length=2048,
            output_length=512,
        )

        fp8_metrics = evaluator.evaluate_strategy(
            model=llama3_8b,
            hardware=rtx_5090,
            precision="fp8",
            batch_size=32,
            input_length=2048,
            output_length=512,
        )

        # FP8 should be faster and use less memory
        assert fp8_metrics.throughput_quantized > fp16_metrics.throughput_quantized
        assert fp8_metrics.memory_bytes_quantized < fp16_metrics.memory_bytes_quantized
        assert fp8_metrics.memory_savings_pct > 0

    def test_quantization_metrics_immutable(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test that QuantizationMetrics is immutable."""
        metrics = evaluator.evaluate_strategy(
            model=llama3_8b,
            hardware=h100_single,
            precision="fp8",
        )

        # Should not be able to modify
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            metrics.precision = "int4"  # type: ignore

    def test_custom_quantization_method(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test specifying custom quantization method."""
        metrics = evaluator.evaluate_strategy(
            model=llama3_8b,
            hardware=h100_single,
            precision="int4",
            method="gptq",  # Override default (awq)
        )

        assert metrics.method == "gptq"
        assert metrics.precision == "int4"

    def test_batch_size_effects_on_speed(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test that batch size affects speed improvement calculation."""
        # Small batch (more memory-bound)
        metrics_small = evaluator.evaluate_strategy(
            model=llama3_8b,
            hardware=h100_single,
            precision="fp8",
            batch_size=1,
            input_length=2048,
            output_length=512,
        )

        # Large batch (more compute-bound)
        metrics_large = evaluator.evaluate_strategy(
            model=llama3_8b,
            hardware=h100_single,
            precision="fp8",
            batch_size=64,
            input_length=2048,
            output_length=512,
        )

        # Both should show improvement, but may vary with batch size
        assert metrics_small.speed_improvement > 1.0
        assert metrics_large.speed_improvement > 1.0

        # Memory savings should be the same regardless of batch size
        assert metrics_small.memory_savings_pct == pytest.approx(
            metrics_large.memory_savings_pct, abs=0.1
        )

    def test_all_precisions_comparable(
        self, evaluator: QuantizationEvaluator, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test that all precisions can be evaluated."""
        precisions = ["fp32", "fp16", "bf16", "fp8", "int8", "int4"]

        for precision in precisions:
            metrics = evaluator.evaluate_strategy(
                model=llama3_8b,
                hardware=h100_single,
                precision=precision,  # type: ignore
                batch_size=32,
                input_length=2048,
                output_length=512,
            )

            assert metrics.precision == precision
            assert metrics.memory_bytes_quantized > 0
            assert metrics.throughput_quantized > 0
            assert metrics.quality_impact in ["none", "minimal", "slight", "moderate", "significant"]
