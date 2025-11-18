"""Pytest configuration and fixtures."""

import pytest

from kv_planner.domain import (
    DeploymentConstraints,
    Distribution,
    HardwareSpec,
    ModelConfig,
    TrafficModel,
)


@pytest.fixture
def llama3_8b() -> ModelConfig:
    """Llama 3 8B model configuration."""
    return ModelConfig(
        name="meta-llama/Llama-3-8b-hf",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA: 4× reduction
        head_dim=128,
        vocab_size=128256,
        max_position_embeddings=8192,
        attention_type="GQA",
    )


@pytest.fixture
def llama3_70b() -> ModelConfig:
    """Llama 3 70B model configuration."""
    return ModelConfig(
        name="meta-llama/Llama-3-70b-hf",
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA: 8× reduction
        head_dim=128,
        vocab_size=128256,
        max_position_embeddings=8192,
        attention_type="GQA",
    )


@pytest.fixture
def h100_single() -> HardwareSpec:
    """Single H100 80GB GPU."""
    return HardwareSpec(
        gpu_model="H100-80GB",
        num_gpus=1,
        gpu_memory_gb=80.0,
        peak_tflops=989.0,  # FP16
        hbm_bandwidth_gb_s=3350.0,  # ~3.35 TB/s
        l2_cache_mb=60.0,
    )


@pytest.fixture
def h100_4x() -> HardwareSpec:
    """4× H100 80GB GPUs with tensor parallelism."""
    return HardwareSpec(
        gpu_model="H100-80GB",
        num_gpus=4,
        gpu_memory_gb=80.0,
        peak_tflops=989.0,
        hbm_bandwidth_gb_s=3350.0,
        tensor_parallel_size=4,
        interconnect_type="NVLink",
        interconnect_bandwidth_gb_s=900.0,
    )


@pytest.fixture
def a100_single() -> HardwareSpec:
    """Single A100 80GB GPU."""
    return HardwareSpec(
        gpu_model="A100-80GB",
        num_gpus=1,
        gpu_memory_gb=80.0,
        peak_tflops=312.0,  # FP16
        hbm_bandwidth_gb_s=2039.0,  # ~2 TB/s
        l2_cache_mb=40.0,
    )


@pytest.fixture
def chatbot_traffic() -> TrafficModel:
    """Chatbot traffic pattern."""
    return TrafficModel(
        requests_per_second=10.0,
        input_tokens=Distribution(mean=512, p50=400, p95=1024, p99=2048, std=200),
        output_tokens=Distribution(mean=256, p50=200, p95=512, p99=1024, std=100),
    )


@pytest.fixture
def rag_traffic() -> TrafficModel:
    """RAG workload with prefix sharing."""
    return TrafficModel(
        requests_per_second=20.0,
        input_tokens=Distribution(mean=2048, p50=2000, p95=4096, p99=8192, std=500),
        output_tokens=Distribution(mean=512, p50=400, p95=1024, p99=2048, std=200),
        prefix_sharing_ratio=0.6,  # High prefix sharing
        avg_shared_prefix_length=1500,
    )


@pytest.fixture
def production_traffic() -> TrafficModel:
    """Production-scale traffic."""
    return TrafficModel(
        requests_per_second=50.0,
        input_tokens=Distribution(mean=1024, p50=800, p95=4096, p99=8192, std=500),
        output_tokens=Distribution(mean=512, p50=400, p95=2048, p99=4096, std=300),
        peak_multiplier=3.0,
        target_p95_latency_ms=2000.0,
    )


@pytest.fixture
def default_constraints() -> DeploymentConstraints:
    """Default deployment constraints."""
    return DeploymentConstraints()


@pytest.fixture
def strict_constraints() -> DeploymentConstraints:
    """Strict constraints with cost limits."""
    return DeploymentConstraints(
        max_sequence_length=4096,
        max_batch_size=64,
        target_cost_per_million_tokens=0.10,
        target_gpu_utilization=0.80,
    )
