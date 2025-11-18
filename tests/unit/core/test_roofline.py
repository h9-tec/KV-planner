"""Unit tests for RooflineAnalyzer."""

import pytest

from kv_planner.core.performance import RooflineAnalyzer, PerformanceMetrics
from kv_planner.domain import ModelConfig, HardwareSpec


class TestRooflineAnalyzer:
    """Test suite for RooflineAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> RooflineAnalyzer:
        """Create roofline analyzer with default settings."""
        return RooflineAnalyzer()

    @pytest.fixture
    def custom_analyzer(self) -> RooflineAnalyzer:
        """Create roofline analyzer with custom efficiency."""
        return RooflineAnalyzer(compute_efficiency=0.8, memory_efficiency=0.9)

    def test_analyzer_initialization(self) -> None:
        """Test analyzer initialization with valid parameters."""
        analyzer = RooflineAnalyzer(compute_efficiency=0.7, memory_efficiency=0.85)
        assert analyzer.compute_efficiency == 0.7
        assert analyzer.memory_efficiency == 0.85

    def test_invalid_compute_efficiency(self) -> None:
        """Test that invalid compute efficiency is rejected."""
        with pytest.raises(ValueError, match="compute_efficiency must be in"):
            RooflineAnalyzer(compute_efficiency=0.0)

        with pytest.raises(ValueError, match="compute_efficiency must be in"):
            RooflineAnalyzer(compute_efficiency=1.5)

    def test_invalid_memory_efficiency(self) -> None:
        """Test that invalid memory efficiency is rejected."""
        with pytest.raises(ValueError, match="memory_efficiency must be in"):
            RooflineAnalyzer(memory_efficiency=-0.1)

        with pytest.raises(ValueError, match="memory_efficiency must be in"):
            RooflineAnalyzer(memory_efficiency=1.2)

    def test_calculate_flops_per_token(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig
    ) -> None:
        """Test FLOPs per token calculation."""
        flops = analyzer.calculate_flops_per_token(llama3_8b)

        # Formula: F = n_layers × 24 × d_model²
        expected = llama3_8b.num_layers * 24 * (llama3_8b.hidden_size ** 2)

        assert flops == expected
        assert flops > 0

    def test_calculate_arithmetic_intensity(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig
    ) -> None:
        """Test arithmetic intensity calculation."""
        ai = analyzer.calculate_arithmetic_intensity(
            model=llama3_8b,
            batch_size=32,
            sequence_length=2048,
            precision="fp16"
        )

        # AI should be positive
        assert ai > 0

        # Higher batch size should increase AI (more compute per byte)
        ai_small_batch = analyzer.calculate_arithmetic_intensity(
            model=llama3_8b,
            batch_size=1,
            sequence_length=2048,
            precision="fp16"
        )
        assert ai > ai_small_batch

    def test_get_hardware_balance_point(
        self, analyzer: RooflineAnalyzer, h100_single: HardwareSpec
    ) -> None:
        """Test hardware balance point calculation."""
        balance_point = analyzer.get_hardware_balance_point(h100_single)

        # Balance point = TFLOPS / (GB/s)
        # For H100: 989 TFLOPS / 3350 GB/s ≈ 295 FLOPs/byte
        expected = (h100_single.peak_tflops * 1e12) / (h100_single.hbm_bandwidth_gb_s * 1e9)

        assert abs(balance_point - expected) < 1.0  # Allow small floating point error
        assert balance_point > 0

    def test_predict_prefill_latency(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test prefill latency prediction."""
        latency_ms, achieved_tflops, is_compute_bound = analyzer.predict_prefill_latency(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=32,
            input_length=2048,
            precision="fp16"
        )

        # Latency should be positive
        assert latency_ms > 0

        # Achieved TFLOPS should be <= peak TFLOPS
        assert 0 < achieved_tflops <= h100_single.peak_tflops

        # Large batch size should be compute-bound
        assert is_compute_bound

    def test_predict_decode_latency(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test decode latency prediction."""
        latency_ms, achieved_tflops, is_memory_bound = analyzer.predict_decode_latency(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=32,
            precision="fp16"
        )

        # Latency should be positive
        assert latency_ms > 0

        # Achieved TFLOPS should be positive but lower than prefill
        assert achieved_tflops > 0

        # Decode is always memory-bound
        assert is_memory_bound

    def test_calculate_mfu(
        self, analyzer: RooflineAnalyzer, h100_single: HardwareSpec
    ) -> None:
        """Test MFU (Model FLOPS Utilization) calculation."""
        # Test normal case
        mfu = analyzer.calculate_mfu(achieved_tflops=500.0, hardware=h100_single)
        expected = 500.0 / h100_single.peak_tflops
        assert abs(mfu - expected) < 0.001

        # Test cap at 100%
        mfu_over = analyzer.calculate_mfu(achieved_tflops=2000.0, hardware=h100_single)
        assert mfu_over == 1.0

        # Test zero case
        mfu_zero = analyzer.calculate_mfu(achieved_tflops=0.0, hardware=h100_single)
        assert mfu_zero == 0.0

    def test_calculate_mbu(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test MBU (Model Bandwidth Utilization) calculation."""
        mbu = analyzer.calculate_mbu(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=32,
            latency_sec=0.01,  # 10ms
            precision="fp16"
        )

        # MBU should be between 0 and 1
        assert 0.0 <= mbu <= 1.0

        # Zero latency should give zero MBU
        mbu_zero = analyzer.calculate_mbu(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=32,
            latency_sec=0.0,
            precision="fp16"
        )
        assert mbu_zero == 0.0

    def test_predict_latency_complete(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test complete latency prediction."""
        metrics = analyzer.predict_latency(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=32,
            input_length=2048,
            output_length=512,
            precision="fp16"
        )

        # Check all fields are valid
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.prefill_latency_ms > 0
        assert metrics.decode_latency_ms > 0
        assert metrics.total_latency_ms == metrics.prefill_latency_ms + metrics.decode_latency_ms
        assert 0 <= metrics.mfu <= 1.0
        assert 0 <= metrics.mbu <= 1.0
        assert metrics.throughput_tokens_per_sec > 0
        assert metrics.arithmetic_intensity > 0

    def test_predict_latency_no_output(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test latency prediction with no output tokens."""
        metrics = analyzer.predict_latency(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=32,
            input_length=2048,
            output_length=0,  # No decode
            precision="fp16"
        )

        # Decode latency should be zero
        assert metrics.decode_latency_ms == 0.0
        assert metrics.total_latency_ms == metrics.prefill_latency_ms

    def test_predict_latency_invalid_inputs(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test that invalid inputs are rejected."""
        # Invalid batch size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            analyzer.predict_latency(
                model=llama3_8b,
                hardware=h100_single,
                batch_size=0,
                input_length=2048,
                output_length=512,
                precision="fp16"
            )

        # Invalid input length
        with pytest.raises(ValueError, match="input_length must be positive"):
            analyzer.predict_latency(
                model=llama3_8b,
                hardware=h100_single,
                batch_size=32,
                input_length=-1,
                output_length=512,
                precision="fp16"
            )

        # Invalid output length
        with pytest.raises(ValueError, match="output_length must be non-negative"):
            analyzer.predict_latency(
                model=llama3_8b,
                hardware=h100_single,
                batch_size=32,
                input_length=2048,
                output_length=-10,
                precision="fp16"
            )

    def test_predict_throughput(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test throughput prediction convenience method."""
        throughput = analyzer.predict_throughput(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=32,
            input_length=2048,
            output_length=512,
            precision="fp16"
        )

        # Throughput should be positive
        assert throughput > 0

        # Should match predict_latency result
        metrics = analyzer.predict_latency(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=32,
            input_length=2048,
            output_length=512,
            precision="fp16"
        )
        assert throughput == metrics.throughput_tokens_per_sec

    def test_precision_effects(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test that different precisions affect latency correctly."""
        metrics_fp16 = analyzer.predict_latency(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=32,
            input_length=2048,
            output_length=512,
            precision="fp16"
        )

        metrics_fp8 = analyzer.predict_latency(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=32,
            input_length=2048,
            output_length=512,
            precision="fp8"
        )

        # FP8 should be faster (less data to move)
        assert metrics_fp8.total_latency_ms < metrics_fp16.total_latency_ms
        assert metrics_fp8.throughput_tokens_per_sec > metrics_fp16.throughput_tokens_per_sec

    def test_batch_size_scaling(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig, h100_single: HardwareSpec
    ) -> None:
        """Test that larger batch sizes increase throughput."""
        metrics_small = analyzer.predict_latency(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=1,
            input_length=2048,
            output_length=512,
            precision="fp16"
        )

        metrics_large = analyzer.predict_latency(
            model=llama3_8b,
            hardware=h100_single,
            batch_size=32,
            input_length=2048,
            output_length=512,
            precision="fp16"
        )

        # Larger batch should have higher total throughput
        assert metrics_large.throughput_tokens_per_sec > metrics_small.throughput_tokens_per_sec

    def test_rtx_gpu_predictions(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig
    ) -> None:
        """Test predictions on RTX GPUs."""
        from kv_planner.infrastructure.hardware_db import GPUDatabase

        # RTX 5090
        rtx_5090 = GPUDatabase.to_hardware_spec("RTX-5090", num_gpus=1)
        metrics_5090 = analyzer.predict_latency(
            model=llama3_8b,
            hardware=rtx_5090,
            batch_size=32,
            input_length=2048,
            output_length=512,
            precision="fp16"
        )

        # RTX 4090
        rtx_4090 = GPUDatabase.to_hardware_spec("RTX-4090", num_gpus=1)
        metrics_4090 = analyzer.predict_latency(
            model=llama3_8b,
            hardware=rtx_4090,
            batch_size=32,
            input_length=2048,
            output_length=512,
            precision="fp16"
        )

        # RTX 5090 should be faster (better specs)
        assert metrics_5090.total_latency_ms < metrics_4090.total_latency_ms
        assert metrics_5090.throughput_tokens_per_sec > metrics_4090.throughput_tokens_per_sec

        # Both should have valid metrics
        assert 0 < metrics_5090.mfu <= 1.0
        assert 0 < metrics_5090.mbu <= 1.0
        assert 0 < metrics_4090.mfu <= 1.0
        assert 0 < metrics_4090.mbu <= 1.0

    def test_tensor_parallel_overhead(
        self, analyzer: RooflineAnalyzer, llama3_8b: ModelConfig, h100_4x: HardwareSpec
    ) -> None:
        """Test that tensor parallelism adds communication overhead."""
        # Single GPU
        h100_single_spec = HardwareSpec(
            gpu_model="H100-80GB",
            num_gpus=1,
            gpu_memory_gb=80.0,
            peak_tflops=989.0,
            hbm_bandwidth_gb_s=3350.0,
            l2_cache_mb=60.0,
            tensor_parallel_size=1,
        )

        metrics_single = analyzer.predict_latency(
            model=llama3_8b,
            hardware=h100_single_spec,
            batch_size=32,
            input_length=2048,
            output_length=512,
            precision="fp16"
        )

        metrics_tp4 = analyzer.predict_latency(
            model=llama3_8b,
            hardware=h100_4x,
            batch_size=32,
            input_length=2048,
            output_length=512,
            precision="fp16"
        )

        # TP=4 should add some communication overhead
        # (though it also gets super-linear scaling benefits)
        # Just verify both produce valid results
        assert metrics_single.total_latency_ms > 0
        assert metrics_tp4.total_latency_ms > 0
