"""Unit tests for PagedMemoryCalculator."""

import pytest

from kv_planner.core.memory import PagedMemoryCalculator, NaiveMemoryCalculator
from kv_planner.domain import ModelConfig, HardwareSpec, InsufficientMemoryError


class TestPagedMemoryCalculator:
    """Test suite for PagedMemoryCalculator."""

    @pytest.fixture
    def calculator(self) -> PagedMemoryCalculator:
        """Basic calculator without hardware."""
        return PagedMemoryCalculator(block_size=16)

    @pytest.fixture
    def calculator_with_hardware(self, h100_4x: HardwareSpec) -> PagedMemoryCalculator:
        """Calculator with hardware context (TP shard math is handled by caller)."""
        return PagedMemoryCalculator(block_size=16, hardware=h100_4x)

    def test_block_size_validation(self) -> None:
        """Test that invalid block sizes are rejected."""
        with pytest.raises(ValueError, match="block_size must be positive"):
            PagedMemoryCalculator(block_size=0)

        with pytest.raises(ValueError, match="block_size must be positive"):
            PagedMemoryCalculator(block_size=-1)

    def test_calculate_kv_cache_exact_blocks(
        self,
        calculator: PagedMemoryCalculator,
        llama3_8b: ModelConfig,
    ) -> None:
        """Exact-multiple seq length: no waste, no bogus 4% multiplier."""
        kv_bytes = calculator.calculate_kv_cache_size(
            batch_size=1,
            sequence_length=2048,
            model=llama3_8b,
            precision="fp16",
        )

        # 2048 tokens ÷ 16 tokens/block = 128 blocks, exact fit.
        # Per vLLM: 131,072 bytes/token fp16 for Llama-3 8B.
        expected = 2048 * llama3_8b.kv_cache_bytes_per_token("fp16")
        assert kv_bytes == expected

    def test_calculate_kv_cache_partial_block(
        self,
        calculator: PagedMemoryCalculator,
        llama3_8b: ModelConfig,
    ) -> None:
        """Partial last block: exactly one extra block allocated, no 4% on top."""
        kv_bytes = calculator.calculate_kv_cache_size(
            batch_size=1,
            sequence_length=17,
            model=llama3_8b,
            precision="fp16",
        )
        # 17 tokens -> ceil(17/16) = 2 blocks = 32 token-slots allocated.
        # The "fragmentation" is purely the 15 unused token-slots in the last block.
        expected = 2 * 16 * llama3_8b.kv_cache_bytes_per_token("fp16")
        assert kv_bytes == expected

    def test_calculate_kv_cache_batch(
        self,
        calculator: PagedMemoryCalculator,
        llama3_8b: ModelConfig,
    ) -> None:
        """Batch allocation is strictly linear in batch size (no frag multiplier)."""
        kv_bytes = calculator.calculate_kv_cache_size(
            batch_size=32,
            sequence_length=2048,
            model=llama3_8b,
            precision="fp16",
        )
        expected = 32 * 2048 * llama3_8b.kv_cache_bytes_per_token("fp16")
        assert kv_bytes == expected

    def test_precision_scaling(
        self,
        calculator: PagedMemoryCalculator,
        llama3_8b: ModelConfig,
    ) -> None:
        """Test that different precisions scale correctly."""
        base_fp16 = calculator.calculate_kv_cache_size(
            batch_size=1, sequence_length=1024, model=llama3_8b, precision="fp16"
        )

        # FP8 should be ~half of FP16
        base_fp8 = calculator.calculate_kv_cache_size(
            batch_size=1, sequence_length=1024, model=llama3_8b, precision="fp8"
        )
        assert abs(base_fp8 / base_fp16 - 0.5) < 0.01

        # FP32 should be ~double of FP16
        base_fp32 = calculator.calculate_kv_cache_size(
            batch_size=1, sequence_length=1024, model=llama3_8b, precision="fp32"
        )
        assert abs(base_fp32 / base_fp16 - 2.0) < 0.01

    def test_max_batch_size(
        self,
        calculator: PagedMemoryCalculator,
        llama3_8b: ModelConfig,
    ) -> None:
        """Test maximum batch size calculation."""
        available_memory_gb = 40.0  # 40 GB available

        max_batch = calculator.max_batch_size(
            available_memory_gb=available_memory_gb,
            sequence_length=2048,
            model=llama3_8b,
            precision="fp16",
        )

        assert max_batch > 0

        # Verify the calculated batch size actually fits
        kv_bytes = calculator.calculate_kv_cache_size(
            batch_size=max_batch,
            sequence_length=2048,
            model=llama3_8b,
            precision="fp16",
        )
        assert kv_bytes <= available_memory_gb * 1e9

    def test_max_batch_size_is_linear_in_memory(
        self,
        calculator: PagedMemoryCalculator,
        llama3_8b: ModelConfig,
    ) -> None:
        """Memory → batch is strictly linear. Supplying hardware context does
        not magically grant extra capacity (the old "super-linear" shortcut
        that returned 13.9× for TP=2 was physically bogus and has been removed)."""
        mb1 = calculator.max_batch_size(40.0, 2048, llama3_8b, "fp16")
        mb2 = calculator.max_batch_size(80.0, 2048, llama3_8b, "fp16")
        # Exactly 2× more memory → exactly 2× more batch capacity.
        assert mb2 == 2 * mb1

    def test_max_sequence_length(
        self,
        calculator: PagedMemoryCalculator,
        llama3_8b: ModelConfig,
    ) -> None:
        """Test maximum sequence length calculation."""
        available_memory_gb = 40.0
        batch_size = 8

        max_seq = calculator.max_sequence_length(
            available_memory_gb=available_memory_gb,
            batch_size=batch_size,
            model=llama3_8b,
            precision="fp16",
        )

        assert max_seq >= 16  # At least one block

        # Verify it actually fits
        kv_bytes = calculator.calculate_kv_cache_size(
            batch_size=batch_size,
            sequence_length=max_seq,
            model=llama3_8b,
            precision="fp16",
        )
        assert kv_bytes <= available_memory_gb * 1e9

    def test_insufficient_memory_error(
        self,
        calculator: PagedMemoryCalculator,
        llama3_8b: ModelConfig,
    ) -> None:
        """Test that insufficient memory raises appropriate error."""
        # Try to fit a huge sequence in tiny memory
        with pytest.raises(InsufficientMemoryError, match="Cannot fit"):
            calculator.max_batch_size(
                available_memory_gb=0.001,  # 1 MB
                sequence_length=100000,
                model=llama3_8b,
                precision="fp16",
            )

    def test_memory_breakdown(
        self,
        calculator: PagedMemoryCalculator,
        llama3_8b: ModelConfig,
    ) -> None:
        """Test detailed memory breakdown."""
        breakdown = calculator.memory_breakdown(
            batch_size=1,
            sequence_length=17,  # Partial block
            model=llama3_8b,
            precision="fp16",
        )

        assert breakdown["blocks"] == 2  # ceil(17/16)
        assert breakdown["blocks_per_sequence"] == 2
        assert breakdown["tokens_actual"] == 17
        assert breakdown["tokens_allocated"] == 32  # 2 blocks × 16 tokens
        assert breakdown["tokens_wasted"] == 15  # 32 - 17
        assert 0 < breakdown["fragmentation_pct"] < 50  # 15/32 ≈ 46.9%

    def test_invalid_inputs(
        self,
        calculator: PagedMemoryCalculator,
        llama3_8b: ModelConfig,
    ) -> None:
        """Test that invalid inputs are rejected."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculator.calculate_kv_cache_size(
                batch_size=0, sequence_length=100, model=llama3_8b, precision="fp16"
            )

        with pytest.raises(ValueError, match="sequence_length must be positive"):
            calculator.calculate_kv_cache_size(
                batch_size=1, sequence_length=-1, model=llama3_8b, precision="fp16"
            )


class TestNaiveMemoryCalculator:
    """Test suite for NaiveMemoryCalculator (baseline comparison)."""

    @pytest.fixture
    def naive_calculator(self) -> NaiveMemoryCalculator:
        """Naive calculator for comparison."""
        return NaiveMemoryCalculator()

    def test_naive_vs_paged_fragmentation(
        self,
        naive_calculator: NaiveMemoryCalculator,
        llama3_8b: ModelConfig,
    ) -> None:
        """Test that naive allocation has higher fragmentation than paged."""
        paged = PagedMemoryCalculator(block_size=16)

        # Same configuration
        batch_size, seq_length = 8, 2048

        naive_bytes = naive_calculator.calculate_kv_cache_size(
            batch_size=batch_size,
            sequence_length=seq_length,
            model=llama3_8b,
            precision="fp16",
        )

        paged_bytes = paged.calculate_kv_cache_size(
            batch_size=batch_size,
            sequence_length=seq_length,
            model=llama3_8b,
            precision="fp16",
        )

        # Naive should use significantly more memory (60-80% overhead vs 4%)
        assert naive_bytes > paged_bytes
        # Naive overhead should be ~17× worse (1.70 / 1.04)
        ratio = naive_bytes / paged_bytes
        assert 1.5 < ratio < 1.8  # Allow some variance


class TestProtocolCompliance:
    """Test that calculators conform to MemoryCalculator protocol."""

    def test_paged_calculator_protocol(self, llama3_8b: ModelConfig) -> None:
        """Test PagedMemoryCalculator implements protocol methods."""
        from kv_planner.core.interfaces import MemoryCalculator

        calc = PagedMemoryCalculator()

        # Check it has all required methods (structural subtyping)
        assert hasattr(calc, "calculate_kv_cache_size")
        assert hasattr(calc, "max_batch_size")
        assert hasattr(calc, "max_sequence_length")

        # Verify they work
        kv_size = calc.calculate_kv_cache_size(
            batch_size=1, sequence_length=1024, model=llama3_8b, precision="fp16"
        )
        assert kv_size > 0

        # Runtime check (with @runtime_checkable)
        assert isinstance(calc, MemoryCalculator)
