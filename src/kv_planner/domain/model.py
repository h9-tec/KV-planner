"""Model configuration value object."""

from dataclasses import dataclass
from typing import Literal, Optional

from kv_planner.domain.exceptions import InvalidConfigurationError


# Type aliases for semantic clarity
PrecisionType = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]
AttentionType = Literal["MHA", "GQA", "MQA", "MLA"]


@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable model configuration value object.

    Represents the architecture of a transformer-based language model,
    following the value object pattern from Domain-Driven Design.

    Attributes:
        name: Model identifier (e.g., "meta-llama/Llama-3-70b-hf")
        num_layers: Number of transformer layers
        hidden_size: Model dimension (d_model)
        num_attention_heads: Total number of attention heads
        num_key_value_heads: Number of KV heads (for GQA/MQA support)
        head_dim: Dimension per attention head
        vocab_size: Vocabulary size
        max_position_embeddings: Maximum sequence length supported
        attention_type: Type of attention mechanism
        sliding_window_size: Window size for sliding window attention (e.g., Mistral)
        attention_sink_tokens: Number of attention sink tokens (StreamingLLM)
        is_moe: Whether model uses Mixture of Experts
        num_experts: Total number of experts (for MoE models)
        num_experts_per_token: Experts activated per token (for MoE)
        is_multimodal: Whether model is multimodal (vision + text)
        default_dtype: Default precision for weights
    """

    # Core architecture
    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int

    # Advanced attention
    attention_type: AttentionType = "GQA"
    sliding_window_size: Optional[int] = None
    attention_sink_tokens: int = 4

    # MoE support
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_experts_per_token: Optional[int] = None

    # Multi-modal
    is_multimodal: bool = False

    # Precision
    default_dtype: PrecisionType = "fp16"

    def __post_init__(self) -> None:
        """Validate invariants (fail fast)."""
        # Validate positive values
        if self.num_layers <= 0:
            raise InvalidConfigurationError(
                f"num_layers must be positive, got {self.num_layers}"
            )
        if self.hidden_size <= 0:
            raise InvalidConfigurationError(
                f"hidden_size must be positive, got {self.hidden_size}"
            )
        if self.num_attention_heads <= 0:
            raise InvalidConfigurationError(
                f"num_attention_heads must be positive, got {self.num_attention_heads}"
            )
        if self.num_key_value_heads <= 0:
            raise InvalidConfigurationError(
                f"num_key_value_heads must be positive, got {self.num_key_value_heads}"
            )
        if self.head_dim <= 0:
            raise InvalidConfigurationError(f"head_dim must be positive, got {self.head_dim}")

        # Validate GQA/MQA relationship
        if self.num_key_value_heads > self.num_attention_heads:
            raise InvalidConfigurationError(
                f"num_key_value_heads ({self.num_key_value_heads}) cannot exceed "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        # Validate attention heads divide evenly
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise InvalidConfigurationError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads}) for GQA"
            )

        # Validate hidden_size relationship
        expected_hidden = self.num_attention_heads * self.head_dim
        if self.hidden_size != expected_hidden:
            raise InvalidConfigurationError(
                f"hidden_size ({self.hidden_size}) must equal "
                f"num_attention_heads × head_dim ({expected_hidden})"
            )

        # Validate MoE configuration
        if self.is_moe:
            if self.num_experts is None or self.num_experts <= 0:
                raise InvalidConfigurationError("MoE models must specify num_experts > 0")
            if self.num_experts_per_token is None or self.num_experts_per_token <= 0:
                raise InvalidConfigurationError(
                    "MoE models must specify num_experts_per_token > 0"
                )
            if self.num_experts_per_token > self.num_experts:
                raise InvalidConfigurationError(
                    f"num_experts_per_token ({self.num_experts_per_token}) cannot exceed "
                    f"num_experts ({self.num_experts})"
                )

    @property
    def gqa_reduction_factor(self) -> float:
        """
        Calculate KV cache reduction factor from Grouped Query Attention.

        Returns:
            Reduction factor (e.g., 8.0 for Llama 3 70B with 8 heads per KV head)
        """
        return self.num_attention_heads / self.num_key_value_heads

    @property
    def is_gqa(self) -> bool:
        """Check if model uses Grouped Query Attention."""
        return self.num_key_value_heads < self.num_attention_heads and self.num_key_value_heads > 1

    @property
    def is_mqa(self) -> bool:
        """Check if model uses Multi-Query Attention (single KV head)."""
        return self.num_key_value_heads == 1

    @property
    def is_mha(self) -> bool:
        """Check if model uses standard Multi-Head Attention."""
        return self.num_key_value_heads == self.num_attention_heads

    def kv_cache_bytes_per_token(self, precision: PrecisionType = "fp16") -> int:
        """
        Calculate KV cache size per token in bytes.

        Formula: 2 × num_layers × num_kv_heads × head_dim × precision_bytes
        The factor of 2 accounts for both Key and Value matrices.

        Args:
            precision: Precision type for KV cache

        Returns:
            Bytes per token for KV cache
        """
        precision_bytes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "fp8": 1,
            "int8": 1,
            "int4": 0.5,  # Theoretical 4 bits; actual storage often 8-bit padded
        }

        bytes_per_element = precision_bytes[precision]

        # 2× for K and V
        return int(2 * self.num_layers * self.num_key_value_heads * self.head_dim * bytes_per_element)

    def supports_precision(self, precision: PrecisionType) -> bool:
        """Check if model supports the given precision for KV cache."""
        # Most models support fp32, fp16, bf16, fp8
        # int8 and int4 require quantization support
        if precision in ("fp32", "fp16", "bf16"):
            return True
        if precision == "fp8":
            # FP8 support varies by model and hardware
            return True  # Assume supported, can be overridden
        return False  # int8/int4 require explicit support

    def __repr__(self) -> str:
        """Human-readable representation."""
        attention_info = f"{self.attention_type}"
        if self.is_gqa:
            attention_info += f" (GQA: {self.gqa_reduction_factor:.1f}x reduction)"
        elif self.is_mqa:
            attention_info += " (MQA)"

        return (
            f"ModelConfig(name='{self.name}', "
            f"layers={self.num_layers}, "
            f"hidden={self.hidden_size}, "
            f"attention={attention_info})"
        )

    def total_params(self) -> int:
        """
        Estimate total model parameters.

        Approximate calculation for transformer models:
        - Attention: num_layers × 4 × hidden² (Q, K, V, O projections)
        - MLP: num_layers × 8 × hidden² (up, down, gate for  SwiGLU)
        - Embeddings: vocab_size × hidden
        - Layer norms: negligible

        Returns:
            Estimated total parameters (integer)
        """
        hidden = self.hidden_size
        layers = self.num_layers
        vocab = self.vocab_size

        # Attention parameters (Q, K, V, O projections)
        attn_params = layers * 4 * hidden * hidden

        # MLP parameters (up, down, gate for SwiGLU/FFN)
        # Standard transformer: 4 × hidden intermediate size
        # Total: up_proj + down_proj + gate_proj ≈ 8 × hidden²
        mlp_params = layers * 8 * hidden * hidden

        # Embedding parameters
        embed_params = vocab * hidden

        # Total
        total = attn_params + mlp_params + embed_params

        return total
