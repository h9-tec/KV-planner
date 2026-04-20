"""Regression tests for TrainingPlanner.

These pin the memory/compute math against literature-backed expectations
(16 B/param for full-FT mixed-precision, NF4 ≈ 0.6 B/param for QLoRA,
LoRA trainable set is <1 % of total for rank 16).
"""

from __future__ import annotations

import pytest

from kv_planner.core.training import TrainingPlanner
from kv_planner.domain import HardwareSpec, ModelConfig
from kv_planner.infrastructure.hardware_db import GPUDatabase


def test_full_ft_llama3_8b_memory_dominated_by_optimizer(llama3_8b: ModelConfig) -> None:
    """Full FT of Llama-3 8B needs ~16 B/param ≈ 128 GB weights+grads+opt."""
    h = GPUDatabase.to_hardware_spec("H100-SXM-80GB")
    plan = TrainingPlanner().plan(
        model=llama3_8b, hardware=h, method="full_ft",
        precision="bf16", batch_size=1, sequence_length=2048,
    )
    # 8.03 B params × 16 B ≈ 128.5 GB — way over 80 GB H100 budget.
    assert 120 < plan.model_weight_gb < 140
    assert plan.fits_per_gpu is False


def test_qlora_shrinks_weights_to_under_5gb(llama3_8b: ModelConfig) -> None:
    """QLoRA via NF4 should fit Llama-3 8B weights in ~5 GB."""
    h = GPUDatabase.to_hardware_spec("H100-SXM-80GB")
    plan = TrainingPlanner().plan(
        model=llama3_8b, hardware=h, method="qlora",
        batch_size=1, sequence_length=1024,
    )
    assert 3 < plan.model_weight_gb < 6
    assert plan.fits_per_gpu is True


def test_lora_trainable_params_scale_with_rank(llama3_8b: ModelConfig) -> None:
    """Trainable params should grow roughly linearly with rank."""
    h = GPUDatabase.to_hardware_spec("H100-SXM-80GB")
    p = TrainingPlanner()
    r8 = p.plan(llama3_8b, h, method="lora", lora_rank=8).trainable_params
    r32 = p.plan(llama3_8b, h, method="lora", lora_rank=32).trainable_params
    ratio = r32 / r8
    assert 3.5 < ratio < 4.5  # should be very close to 4


def test_llama3_70b_qlora_fits_single_h100(llama3_70b: ModelConfig) -> None:
    """Real-world capability: 70B QLoRA on a single 80 GB H100."""
    h = GPUDatabase.to_hardware_spec("H100-SXM-80GB")
    plan = TrainingPlanner().plan(
        model=llama3_70b, hardware=h, method="qlora",
        batch_size=1, sequence_length=2048,
    )
    assert plan.fits_per_gpu is True
    assert plan.total_memory_gb < 80.0


def test_training_hours_scales_with_dataset_size(llama3_8b: ModelConfig) -> None:
    h = GPUDatabase.to_hardware_spec("H100-SXM-80GB")
    p = TrainingPlanner()
    small = p.plan(llama3_8b, h, method="qlora",
                   dataset_tokens=100_000, num_epochs=1).est_training_hours
    big = p.plan(llama3_8b, h, method="qlora",
                 dataset_tokens=1_000_000, num_epochs=1).est_training_hours
    assert big / small == pytest.approx(10.0, rel=0.05)
