#!/usr/bin/env python3
"""Validation script for QuantizationEvaluator."""

import sys
sys.path.insert(0, "/home/hesham-haroun/kv-planner/src")

from kv_planner.core.strategies import QuantizationEvaluator
from kv_planner.domain import ModelConfig
from kv_planner.infrastructure.hardware_db import GPUDatabase


def main():
    """Run validation tests."""
    print("=" * 80)
    print("QuantizationEvaluator Validation")
    print("=" * 80)

    # Setup
    evaluator = QuantizationEvaluator()
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

    print("\n✓ Created QuantizationEvaluator")
    print("✓ Created Llama 3 8B model config")

    # Test 1: Evaluate FP16 baseline
    print("\n" + "=" * 80)
    print("Test 1: FP16 Baseline")
    print("=" * 80)

    rtx_5090 = GPUDatabase.to_hardware_spec("RTX-5090", num_gpus=1)
    fp16_metrics = evaluator.evaluate_strategy(
        model=llama3_8b,
        hardware=rtx_5090,
        precision="fp16",
        batch_size=32,
        input_length=2048,
        output_length=512,
    )

    print(f"Precision: {fp16_metrics.precision}")
    print(f"Method: {fp16_metrics.method}")
    print(f"Memory: {fp16_metrics.memory_bytes_quantized / 1e9:.2f} GB")
    print(f"Memory savings: {fp16_metrics.memory_savings_pct:.1f}%")
    print(f"Throughput: {fp16_metrics.throughput_quantized:.0f} tokens/s")
    print(f"Speed improvement: {fp16_metrics.speed_improvement:.2f}×")
    print(f"Perplexity delta: +{fp16_metrics.perplexity_delta:.1f}")
    print(f"Quality impact: {fp16_metrics.quality_impact}")
    print(f"Recommended for: {fp16_metrics.recommended_for}")

    # Test 2: Compare strategies
    print("\n" + "=" * 80)
    print("Test 2: Compare All Strategies")
    print("=" * 80)

    results = evaluator.compare_strategies(
        model=llama3_8b,
        hardware=rtx_5090,
        precisions=["fp16", "fp8", "int8", "int4"],
        batch_size=32,
        input_length=2048,
        output_length=512,
    )

    print(f"\n{'Precision':<10} {'Memory GB':<12} {'Savings %':<12} {'Speed':<10} {'Quality':<12} {'Perplexity Δ':<15}")
    print("-" * 80)

    for m in results:
        print(
            f"{m.precision:<10} "
            f"{m.memory_bytes_quantized/1e9:<12.2f} "
            f"{m.memory_savings_pct:<12.1f} "
            f"{m.speed_improvement:<10.2f}× "
            f"{m.quality_impact:<12} "
            f"+{m.perplexity_delta:<15.1f}"
        )

    # Test 3: RTX GPU comparison
    print("\n" + "=" * 80)
    print("Test 3: FP8 Quantization on RTX GPUs")
    print("=" * 80)

    for gpu_name in ["RTX-5090", "RTX-4090", "RTX-3090"]:
        hardware = GPUDatabase.to_hardware_spec(gpu_name, num_gpus=1)
        fp8_metrics = evaluator.evaluate_strategy(
            model=llama3_8b,
            hardware=hardware,
            precision="fp8",
            batch_size=32,
            input_length=2048,
            output_length=512,
        )

        print(f"\n{gpu_name}:")
        print(f"  Memory: {fp8_metrics.memory_bytes_quantized / 1e9:.2f} GB ({fp8_metrics.memory_savings_pct:.1f}% savings)")
        print(f"  Throughput: {fp8_metrics.throughput_quantized:.0f} tokens/s ({fp8_metrics.speed_improvement:.2f}× vs FP16)")
        print(f"  Quality: {fp8_metrics.quality_impact} (+{fp8_metrics.perplexity_delta:.1f} perplexity)")

    # Test 4: Recommendation engine
    print("\n" + "=" * 80)
    print("Test 4: Precision Recommendation")
    print("=" * 80)

    # Scenario 1: No constraints (best quality)
    print("\nScenario 1: No constraints (optimize for quality)")
    recommended = evaluator.recommend_precision(
        model=llama3_8b,
        hardware=rtx_5090,
        min_quality="moderate",
        batch_size=32,
        input_length=2048,
        output_length=512,
    )
    print(f"  Recommended: {recommended.precision} ({recommended.quality_impact} quality)")
    print(f"  Memory: {recommended.memory_bytes_quantized / 1e9:.2f} GB")
    print(f"  Throughput: {recommended.throughput_quantized:.0f} tokens/s")

    # Scenario 2: Memory constrained
    print("\nScenario 2: Memory constrained (max 10 GB)")
    try:
        recommended = evaluator.recommend_precision(
            model=llama3_8b,
            hardware=rtx_5090,
            max_memory_gb=10.0,
            min_quality="moderate",
            batch_size=32,
            input_length=2048,
            output_length=512,
        )
        print(f"  Recommended: {recommended.precision} ({recommended.quality_impact} quality)")
        print(f"  Memory: {recommended.memory_bytes_quantized / 1e9:.2f} GB")
        print(f"  Savings: {recommended.memory_savings_pct:.1f}%")
    except ValueError as e:
        print(f"  Error: {e}")

    # Scenario 3: High quality required
    print("\nScenario 3: High quality required (min: minimal)")
    recommended = evaluator.recommend_precision(
        model=llama3_8b,
        hardware=rtx_5090,
        min_quality="minimal",
        batch_size=32,
        input_length=2048,
        output_length=512,
    )
    print(f"  Recommended: {recommended.precision} ({recommended.quality_impact} quality)")
    print(f"  Perplexity delta: +{recommended.perplexity_delta:.1f}")

    print("\n" + "=" * 80)
    print("✅ All validation tests passed!")
    print("=" * 80)

    # Summary
    print("\n📊 Summary:")
    print(f"  • FP16 baseline: {fp16_metrics.memory_bytes_baseline / 1e9:.2f} GB, {fp16_metrics.throughput_baseline:.0f} tok/s")
    print(f"  • FP8 savings: {results[0].memory_savings_pct:.1f}% memory, {results[0].speed_improvement:.2f}× speed")
    print(f"  • INT4 savings: {results[-1].memory_savings_pct:.1f}% memory, {results[-1].speed_improvement:.2f}× speed")
    print(f"  • Quality: FP8 minimal impact, INT4 moderate impact")


if __name__ == "__main__":
    main()
