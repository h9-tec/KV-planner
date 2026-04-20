"""Canonical model architectures — verified against HF config.json."""

from __future__ import annotations

from dataclasses import dataclass

from kv_planner.domain import ModelConfig


@dataclass(frozen=True)
class ModelPreset:
    """A named ModelConfig with a short description for the UI."""

    label: str
    description: str
    config: ModelConfig


# Every architecture field below is cross-checked against the model's
# Hugging Face config.json at time of writing.
PRESETS: dict[str, ModelPreset] = {
    "llama-3.2-1b": ModelPreset(
        label="Llama 3.2 1B",
        description="Meta, 16 layers, GQA 4×",
        config=ModelConfig(
            name="meta-llama/Llama-3.2-1B-Instruct",
            num_layers=16, hidden_size=2048,
            num_attention_heads=32, num_key_value_heads=8, head_dim=64,
            vocab_size=128256, max_position_embeddings=131072,
            attention_type="GQA", ffn_type="swiglu", ffn_intermediate_size=8192,
        ),
    ),
    "llama-3.2-3b": ModelPreset(
        label="Llama 3.2 3B",
        description="Meta, 28 layers, GQA 3×",
        config=ModelConfig(
            name="meta-llama/Llama-3.2-3B-Instruct",
            num_layers=28, hidden_size=3072,
            num_attention_heads=24, num_key_value_heads=8, head_dim=128,
            vocab_size=128256, max_position_embeddings=131072,
            attention_type="GQA", ffn_type="swiglu", ffn_intermediate_size=8192,
        ),
    ),
    "llama-3-8b": ModelPreset(
        label="Llama 3 8B",
        description="Meta, 32 layers, GQA 4×",
        config=ModelConfig(
            name="meta-llama/Meta-Llama-3-8B-Instruct",
            num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_key_value_heads=8, head_dim=128,
            vocab_size=128256, max_position_embeddings=8192,
            attention_type="GQA", ffn_type="swiglu", ffn_intermediate_size=14336,
        ),
    ),
    "llama-3-70b": ModelPreset(
        label="Llama 3 70B",
        description="Meta, 80 layers, GQA 8×",
        config=ModelConfig(
            name="meta-llama/Meta-Llama-3-70B-Instruct",
            num_layers=80, hidden_size=8192,
            num_attention_heads=64, num_key_value_heads=8, head_dim=128,
            vocab_size=128256, max_position_embeddings=8192,
            attention_type="GQA", ffn_type="swiglu", ffn_intermediate_size=28672,
        ),
    ),
    "qwen2.5-7b": ModelPreset(
        label="Qwen 2.5 7B",
        description="Alibaba, 28 layers, GQA 7×",
        config=ModelConfig(
            name="Qwen/Qwen2.5-7B-Instruct",
            num_layers=28, hidden_size=3584,
            num_attention_heads=28, num_key_value_heads=4, head_dim=128,
            vocab_size=152064, max_position_embeddings=32768,
            attention_type="GQA", ffn_type="swiglu", ffn_intermediate_size=18944,
        ),
    ),
    "qwen2.5-14b": ModelPreset(
        label="Qwen 2.5 14B",
        description="Alibaba, 48 layers, GQA 5×",
        config=ModelConfig(
            name="Qwen/Qwen2.5-14B-Instruct",
            num_layers=48, hidden_size=5120,
            num_attention_heads=40, num_key_value_heads=8, head_dim=128,
            vocab_size=152064, max_position_embeddings=32768,
            attention_type="GQA", ffn_type="swiglu", ffn_intermediate_size=13824,
        ),
    ),
    "qwen2.5-32b": ModelPreset(
        label="Qwen 2.5 32B",
        description="Alibaba, 64 layers, GQA 5×",
        config=ModelConfig(
            name="Qwen/Qwen2.5-32B-Instruct",
            num_layers=64, hidden_size=5120,
            num_attention_heads=40, num_key_value_heads=8, head_dim=128,
            vocab_size=152064, max_position_embeddings=32768,
            attention_type="GQA", ffn_type="swiglu", ffn_intermediate_size=27648,
        ),
    ),
    "qwen2.5-72b": ModelPreset(
        label="Qwen 2.5 72B",
        description="Alibaba, 80 layers, GQA 7×",
        config=ModelConfig(
            name="Qwen/Qwen2.5-72B-Instruct",
            num_layers=80, hidden_size=8192,
            num_attention_heads=64, num_key_value_heads=8, head_dim=128,
            vocab_size=152064, max_position_embeddings=32768,
            attention_type="GQA", ffn_type="swiglu", ffn_intermediate_size=29568,
        ),
    ),
    "mistral-7b": ModelPreset(
        label="Mistral 7B v0.3",
        description="Mistral AI, 32 layers, GQA 4×",
        config=ModelConfig(
            name="mistralai/Mistral-7B-Instruct-v0.3",
            num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_key_value_heads=8, head_dim=128,
            vocab_size=32768, max_position_embeddings=32768,
            attention_type="GQA", ffn_type="swiglu", ffn_intermediate_size=14336,
        ),
    ),
    "phi-4": ModelPreset(
        label="Phi-4 14B",
        description="Microsoft, 40 layers, GQA 4×",
        config=ModelConfig(
            name="microsoft/phi-4",
            num_layers=40, hidden_size=5120,
            num_attention_heads=40, num_key_value_heads=10, head_dim=128,
            vocab_size=100352, max_position_embeddings=16384,
            attention_type="GQA", ffn_type="swiglu", ffn_intermediate_size=17920,
        ),
    ),
    "deepseek-r1-distill-7b": ModelPreset(
        label="DeepSeek-R1-Distill 7B",
        description="Qwen-based distill, 28 layers, GQA 7×",
        config=ModelConfig(
            name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            num_layers=28, hidden_size=3584,
            num_attention_heads=28, num_key_value_heads=4, head_dim=128,
            vocab_size=152064, max_position_embeddings=131072,
            attention_type="GQA", ffn_type="swiglu", ffn_intermediate_size=18944,
        ),
    ),
    # Note: Gemma 2 uses non-standard attention geometry
    # (hidden_size != num_attention_heads × head_dim), which our ModelConfig
    # invariant check rejects. Add after relaxing that constraint.
}


def preset_keys() -> list[str]:
    """Keys in display-sensible order (smallest → largest)."""
    return list(PRESETS.keys())
