"""Unit tests for ModelConfig."""

import pytest

from kv_planner.domain import InvalidConfigurationError, ModelConfig


class TestModelConfig:
    """Test suite for ModelConfig value object."""

    def test_create_valid_gqa_model(self, llama3_8b: ModelConfig) -> None:
        """Test creating valid GQA model configuration."""
        assert llama3_8b.name == "meta-llama/Llama-3-8b-hf"
        assert llama3_8b.num_layers == 32
        assert llama3_8b.hidden_size == 4096
        assert llama3_8b.num_attention_heads == 32
        assert llama3_8b.num_key_value_heads == 8
        assert llama3_8b.is_gqa is True
        assert llama3_8b.is_mha is False
        assert llama3_8b.is_mqa is False

    def test_gqa_reduction_factor(self, llama3_8b: ModelConfig, llama3_70b: ModelConfig) -> None:
        """Test GQA reduction factor calculation."""
        assert llama3_8b.gqa_reduction_factor == 4.0  # 32 heads / 8 KV heads
        assert llama3_70b.gqa_reduction_factor == 8.0  # 64 heads / 8 KV heads

    def test_kv_cache_bytes_per_token(self, llama3_8b: ModelConfig) -> None:
        """Test KV cache bytes per token calculation."""
        # Formula: 2 × num_layers × num_kv_heads × head_dim × precision_bytes
        # 2 × 32 × 8 × 128 × 2 (fp16) = 131,072 bytes
        bytes_fp16 = llama3_8b.kv_cache_bytes_per_token("fp16")
        assert bytes_fp16 == 131_072

        # FP8 should be half of FP16
        bytes_fp8 = llama3_8b.kv_cache_bytes_per_token("fp8")
        assert bytes_fp8 == 65_536

        # FP32 should be double of FP16
        bytes_fp32 = llama3_8b.kv_cache_bytes_per_token("fp32")
        assert bytes_fp32 == 262_144

    def test_invalid_num_layers(self) -> None:
        """Test validation of num_layers."""
        with pytest.raises(InvalidConfigurationError, match="num_layers must be positive"):
            ModelConfig(
                name="invalid",
                num_layers=0,  # Invalid
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                vocab_size=50000,
                max_position_embeddings=2048,
            )

    def test_invalid_kv_heads_exceeds_attention_heads(self) -> None:
        """Test validation that KV heads cannot exceed attention heads."""
        with pytest.raises(InvalidConfigurationError, match="cannot exceed num_attention_heads"):
            ModelConfig(
                name="invalid",
                num_layers=32,
                hidden_size=4096,
                num_attention_heads=8,
                num_key_value_heads=32,  # Invalid: more than attention heads
                head_dim=128,
                vocab_size=50000,
                max_position_embeddings=2048,
            )

    def test_invalid_attention_heads_not_divisible(self) -> None:
        """Test validation that attention heads must be divisible by KV heads."""
        with pytest.raises(InvalidConfigurationError, match="must be divisible"):
            ModelConfig(
                name="invalid",
                num_layers=32,
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=7,  # Invalid: 32 not divisible by 7
                head_dim=128,
                vocab_size=50000,
                max_position_embeddings=2048,
            )

    def test_invalid_hidden_size_mismatch(self) -> None:
        """Test validation that hidden_size must equal heads × head_dim."""
        with pytest.raises(InvalidConfigurationError, match="must equal"):
            ModelConfig(
                name="invalid",
                num_layers=32,
                hidden_size=4000,  # Invalid: should be 32 × 128 = 4096
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                vocab_size=50000,
                max_position_embeddings=2048,
            )

    def test_mha_model(self) -> None:
        """Test Multi-Head Attention (MHA) model."""
        mha_model = ModelConfig(
            name="gpt2",
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=12,  # Same as attention heads = MHA
            head_dim=64,
            vocab_size=50257,
            max_position_embeddings=1024,
            attention_type="MHA",
        )

        assert mha_model.is_mha is True
        assert mha_model.is_gqa is False
        assert mha_model.is_mqa is False
        assert mha_model.gqa_reduction_factor == 1.0

    def test_mqa_model(self) -> None:
        """Test Multi-Query Attention (MQA) model."""
        mqa_model = ModelConfig(
            name="falcon-7b",
            num_layers=32,
            hidden_size=4544,
            num_attention_heads=71,
            num_key_value_heads=1,  # Single KV head = MQA
            head_dim=64,
            vocab_size=65024,
            max_position_embeddings=2048,
            attention_type="MQA",
        )

        assert mqa_model.is_mqa is True
        assert mqa_model.is_gqa is False
        assert mqa_model.is_mha is False
        assert mqa_model.gqa_reduction_factor == 71.0

    def test_immutability(self, llama3_8b: ModelConfig) -> None:
        """Test that ModelConfig is immutable."""
        with pytest.raises(AttributeError):
            llama3_8b.num_layers = 64  # type: ignore

    def test_repr(self, llama3_8b: ModelConfig) -> None:
        """Test string representation."""
        repr_str = repr(llama3_8b)
        assert "meta-llama/Llama-3-8b-hf" in repr_str
        assert "layers=32" in repr_str
        assert "GQA" in repr_str
