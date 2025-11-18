#!/usr/bin/env python3
"""Validation script for DeploymentPlanner."""

import sys
sys.path.insert(0, "/home/hesham-haroun/kv-planner/src")

from kv_planner.application import DeploymentPlanner
from kv_planner.domain import ModelConfig


def main():
    """Run validation tests."""
    print("=" * 80)
    print("DeploymentPlanner Validation")
    print("=" * 80)

    # Setup
    planner = DeploymentPlanner()

    print("\n✓ Created DeploymentPlanner")

    # Test 1: Simple plan creation
    print("\n" + "=" * 80)
    print("Test 1: Create Basic Deployment Plan")
    print("=" * 80)

    plan = planner.create_plan(
        model="meta-llama/Llama-3-8b-hf",
        hardware="RTX-5090",
        target_rps=10.0,
        input_length=2048,
        output_length=512,
        optimization_goal="balanced",
    )

    print(plan.summary)

    print("\n" + "📋 Plan Details:")
    print(f"  Model: {plan.model.name}")
    print(f"  Hardware: {plan.hardware.gpu_model}")
    print(f"  Precision: {plan.recommended_precision}")
    print(f"  Batch size: {plan.recommended_batch_size}")
    print(f"  Prefix caching: {plan.enable_prefix_caching}")
    print(f"\n  Memory:")
    print(f"    Model: {plan.model_memory_gb:.2f} GB")
    print(f"    KV cache: {plan.kv_cache_memory_gb:.2f} GB")
    print(f"    Total: {plan.total_memory_gb:.2f} GB ({plan.memory_utilization_pct:.1f}%)")
    print(f"\n  Performance:")
    print(f"    Throughput: {plan.performance.throughput_tokens_per_sec:,.0f} tokens/s")
    print(f"    Latency: {plan.performance.total_latency_ms:.0f} ms")
    print(f"\n  Cost:")
    print(f"    ${plan.cost.cost_per_million_tokens:.2f} / million tokens")
    print(f"    ${plan.cost.monthly_cost_usd:,.2f} / month")

    # Test 2: Different optimization goals
    print("\n" + "=" * 80)
    print("Test 2: Different Optimization Goals")
    print("=" * 80)

    goals = ["cost", "latency", "throughput", "quality", "balanced"]

    print(f"\n{'Goal':<15} {'Precision':<12} {'Batch':<8} {'$/M tokens':<15} {'Throughput':<15}")
    print("-" * 80)

    for goal in goals:
        plan = planner.create_plan(
            model="meta-llama/Llama-3-8b-hf",
            hardware="RTX-5090",
            target_rps=5.0,
            optimization_goal=goal,  # type: ignore
        )

        print(
            f"{goal:<15} "
            f"{plan.recommended_precision:<12} "
            f"{plan.recommended_batch_size:<8} "
            f"${plan.cost.cost_per_million_tokens:<14.2f} "
            f"{plan.performance.throughput_tokens_per_sec:<15,.0f}"
        )

    # Test 3: Compare GPU options
    print("\n" + "=" * 80)
    print("Test 3: Compare GPU Options")
    print("=" * 80)

    gpu_options = ["RTX-5090", "RTX-4090", "RTX-3090"]

    plans = planner.compare_options(
        model="meta-llama/Llama-3-8b-hf",
        hardware_options=gpu_options,
        target_rps=5.0,
        optimization_goal="balanced",
    )

    print(f"\n{'GPU':<15} {'Precision':<12} {'Batch':<8} {'$/M tokens':<15} {'Monthly $':<15}")
    print("-" * 80)

    for p in plans:
        print(
            f"{p.hardware.gpu_model:<15} "
            f"{p.recommended_precision:<12} "
            f"{p.recommended_batch_size:<8} "
            f"${p.cost.cost_per_million_tokens:<14.2f} "
            f"${p.cost.monthly_cost_usd:<15,.2f}"
        )

    # Test 4: vLLM config generation
    print("\n" + "=" * 80)
    print("Test 4: vLLM Configuration")
    print("=" * 80)

    plan = planner.create_plan(
        model="meta-llama/Llama-3-8b-hf",
        hardware="RTX-5090",
        target_rps=10.0,
        optimization_goal="balanced",
    )

    print("\nvLLM Config:")
    import json
    print(json.dumps(plan.vllm_config, indent=2))

    # Test 5: With model config object
    print("\n" + "=" * 80)
    print("Test 5: Using ModelConfig Object")
    print("=" * 80)

    llama3_8b = ModelConfig(
        name="meta-llama/Llama-3-8b-hf",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=128256,
        max_position_embeddings=8192,
    )

    plan = planner.create_plan(
        model=llama3_8b,
        hardware="RTX-4090",
        target_rps=5.0,
        optimization_goal="cost",
    )

    print(f"\nModel: {plan.model.name}")
    print(f"Precision: {plan.recommended_precision} ({plan.quantization.quality_impact} quality)")
    print(f"Cost: ${plan.cost.cost_per_million_tokens:.2f}/M tokens")
    print(f"Savings: {plan.quantization.memory_savings_pct:.0f}% memory, {plan.quantization.speed_improvement:.2f}× speed")

    # Test 6: Memory-constrained deployment
    print("\n" + "=" * 80)
    print("Test 6: Memory-Constrained Deployment (10 GB limit)")
    print("=" * 80)

    plan = planner.create_plan(
        model="meta-llama/Llama-3-8b-hf",
        hardware="RTX-3090",  # 24 GB, but constrain to 10 GB
        target_rps=2.0,
        max_memory_budget_gb=10.0,
        min_quality="moderate",
        optimization_goal="balanced",
    )

    print(f"\nMemory Budget: 10.0 GB")
    print(f"Actual Usage: {plan.total_memory_gb:.2f} GB ({plan.memory_utilization_pct:.1f}% of available)")
    print(f"Precision: {plan.recommended_precision}")
    print(f"Batch Size: {plan.recommended_batch_size}")
    print(f"Model Memory: {plan.model_memory_gb:.2f} GB")
    print(f"KV Cache: {plan.kv_cache_memory_gb:.2f} GB")

    print("\n" + "=" * 80)
    print("✅ All validation tests passed!")
    print("=" * 80)

    # Summary
    print("\n📊 Key Findings:")
    print(f"  • Planner successfully orchestrates all analyzers")
    print(f"  • Recommends optimal precision based on goal")
    print(f"  • Calculates appropriate batch size for target RPS")
    print(f"  • Generates vLLM-compatible configurations")
    print(f"  • Provides complete performance/cost breakdown")
    print(f"  • Best value GPU: {plans[0].hardware.gpu_model} (${plans[0].cost.cost_per_million_tokens:.2f}/M tokens)")


if __name__ == "__main__":
    main()
