"""Model configuration value object."""

from dataclasses import dataclass
from typing import Literal, Optional

from kv_planner.domain.exceptions import InvalidConfigurationError
from kv_planner.domain.precision import PrecisionType, bytes_per_element


AttentionType = Literal["MHA", "GQA", "MQA", "MLA"]
FFNType = Literal["swiglu", "standard"]


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

    # FFN shape. Modern Llama/Mistral/Qwen use SwiGLU (3 matrices, intermediate
    # size typically ~8/3·d so total FFN params ≈ 8·d²). GPT-2-era models use
    # standard 2-matrix FFN (intermediate = 4·d, total ≈ 8·d²). Coefficient
    # happens to be identical; the number of matmuls differs.
    ffn_type: FFNType = "swiglu"
    ffn_intermediate_size: Optional[int] = None  # if None, derived per ffn_type

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

        Formula::

            2 · num_layers · num_key_value_heads · head_dim · bytes_per_element

        The factor of 2 accounts for both K and V. GQA is handled naturally by
        ``num_key_value_heads`` being smaller than ``num_attention_heads``.

        Source: vLLM PagedAttention design doc
        (https://docs.vllm.ai/en/latest/design/paged_attention/) — for Llama-3
        8B in fp16 this returns exactly 131 072 bytes (128 KiB) per token.

        Args:
            precision: Precision type for KV cache

        Returns:
            Bytes per token for KV cache (integer; may round down for int4)
        """
        return int(
            2
            * self.num_layers
            * self.num_key_value_heads
            * self.head_dim
            * bytes_per_element(precision)
        )

    @property
    def kv_hidden_size(self) -> int:
        """Total K/V projection width = num_kv_heads · head_dim."""
        return self.num_key_value_heads * self.head_dim

    @property
    def ffn_num_matmuls(self) -> int:
        """How many matmuls the FFN block performs per token (2 standard, 3 SwiGLU)."""
        return 3 if self.ffn_type == "swiglu" else 2

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

    @property
    def _ffn_intermediate(self) -> int:
        """FFN intermediate dimension (d_ff), honouring ``ffn_intermediate_size``.

        Default for SwiGLU is (8/3)·d rounded up to a multiple of 256 (Llama
        convention); default for standard FFN is 4·d. Override with
        ``ffn_intermediate_size`` in :class:`ModelConfig` for exact param counts.
        """
        if self.ffn_intermediate_size is not None:
            return self.ffn_intermediate_size
        if self.ffn_type == "swiglu":
            return int(round(self.hidden_size * 8 / 3 / 256) * 256)
        return 4 * self.hidden_size

    def total_params(self) -> int:
        """
        Estimate total model parameters — GQA-aware.

        Breakdown per layer (decoder-only transformer):

        * Attention
          * Q projection: ``d · d`` parameters
          * K, V projections (GQA): ``d · (num_kv_heads · head_dim)`` each
          * Output projection: ``d · d`` parameters
        * FFN
          * SwiGLU (3 matmuls, gate + up + down): ``3 · d · d_ff``
          * Standard (2 matmuls, up + down): ``2 · d · d_ff``
        * Layer-norms: negligible — excluded.

        Plus:

        * Token embedding: ``vocab_size · d``
        * LM head: ``vocab_size · d`` (we assume untied — Llama-3 does not tie)

        Reference: kipply, "Transformer Inference Arithmetic" and
        EleutherAI cookbook ``calc_transformer_params.py``
        (https://github.com/EleutherAI/cookbook).
        """
        hidden = self.hidden_size
        layers = self.num_layers
        vocab = self.vocab_size
        kv_hidden = self.num_key_value_heads * self.head_dim

        q_and_o = 2 * hidden * hidden
        kv = 2 * hidden * kv_hidden
        attn_params_per_layer = q_and_o + kv

        d_ff = self._ffn_intermediate
        if self.ffn_type == "swiglu":
            ffn_params_per_layer = 3 * hidden * d_ff
        else:
            ffn_params_per_layer = 2 * hidden * d_ff

        per_layer = attn_params_per_layer + ffn_params_per_layer

        # MoE: only active experts contribute at inference time for FLOPs, but
        # ALL experts sit in memory. total_params reports stored params, so
        # multiply FFN portion by num_experts.
        if self.is_moe and self.num_experts is not None:
            per_layer = attn_params_per_layer + ffn_params_per_layer * self.num_experts

        dense = layers * per_layer

        # Untied embeddings: in + out. Set to single vocab·hidden if
        # you model a tied-embedding variant.
        embed_params = 2 * vocab * hidden

        return dense + embed_params
