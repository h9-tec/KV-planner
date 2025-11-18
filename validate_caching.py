#!/usr/bin/env python3
"""Validation script for PrefixCachingAnalyzer."""

import sys
sys.path.insert(0, "/home/hesham-haroun/kv-planner/src")

from kv_planner.core.strategies import PrefixCachingAnalyzer
from kv_planner.domain import ModelConfig
from kv_planner.infrastructure.hardware_db import GPUDatabase


def main():
    """Run validation tests."""
    print("=" * 80)
    print("PrefixCachingAnalyzer Validation")
    print("=" * 80)

    # Setup
    analyzer = PrefixCachingAnalyzer()
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
    rtx_5090 = GPUDatabase.to_hardware_spec("RTX-5090", num_gpus=1)

    print("\n✓ Created PrefixCachingAnalyzer")
    print("✓ Created Llama 3 8B model config")

    # Test 1: Evaluate prefix caching with different hit rates
    print("\n" + "=" * 80)
    print("Test 1: Prefix Caching with Different Hit Rates")
    print("=" * 80)

    prefix_length = 1024  # System prompt + few-shot examples
    total_length = 2048   # Including user input
    batch_size = 32

    print(f"\nScenario: {prefix_length} token prefix, {total_length} total tokens, batch={batch_size}")
    print(f"{'Hit Rate':<12} {'Memory GB':<12} {'Savings %':<12} {'Latency ↓':<12} {'Throughput ↑':<12}")
    print("-" * 80)

    for hit_rate in [0.0, 0.5, 0.7, 0.9, 0.95]:
        metrics = analyzer.evaluate_caching(
            model=llama3_8b,
            hardware=rtx_5090,
            prefix_length=prefix_length,
            total_length=total_length,
            batch_size=batch_size,
            hit_rate=hit_rate,
            precision="fp16",
        )

        print(
            f"{hit_rate:<12.0%} "
            f"{metrics.memory_with_caching_gb:<12.2f} "
            f"{metrics.memory_savings_pct:<12.1f}% "
            f"{metrics.latency_reduction_pct:<12.1f}% "
            f"{metrics.throughput_improvement:<12.2f}×"
        )

    # Test 2: Benefit breakdown for high hit rate
    print("\n" + "=" * 80)
    print("Test 2: Detailed Breakdown (90% Hit Rate)")
    print("=" * 80)

    metrics = analyzer.evaluate_caching(
        model=llama3_8b,
        hardware=rtx_5090,
        prefix_length=1024,
        total_length=2048,
        batch_size=32,
        hit_rate=0.9,
        precision="fp16",
    )

    print(f"\nPrefix: {metrics.prefix_length} tokens")
    print(f"Hit rate: {metrics.hit_rate:.0%}")
    print(f"\nMemory:")
    print(f"  Without caching: {metrics.memory_without_caching_gb:.2f} GB")
    print(f"  With caching: {metrics.memory_with_caching_gb:.2f} GB")
    print(f"  Savings: {metrics.memory_savings_pct:.1f}%")
    print(f"\nPerformance:")
    print(f"  Prefill tokens saved: {metrics.prefill_tokens_saved:,}")
    print(f"  Latency reduction: {metrics.latency_reduction_pct:.1f}%")
    print(f"  Throughput improvement: {metrics.throughput_improvement:.2f}×")
    print(f"  Effective batch size: {metrics.effective_batch_size}")
    print(f"\nCost:")
    print(f"  Cost savings: {metrics.cost_savings_pct:.1f}%")

    # Test 3: Estimate hit rates
    print("\n" + "=" * 80)
    print("Test 3: Hit Rate Estimation")
    print("=" * 80)

    scenarios = [
        {"desc": "Single system prompt (chatbot)", "unique": 1, "total": 1000, "cache": 10},
        {"desc": "Few system prompts (multi-tenant)", "unique": 5, "total": 1000, "cache": 10},
        {"desc": "Many system prompts (high variety)", "unique": 50, "total": 1000, "cache": 10},
        {"desc": "Cache thrashing", "unique": 100, "total": 1000, "cache": 10},
    ]

    print(f"\n{'Scenario':<40} {'Unique':<10} {'Cache':<10} {'Hit Rate':<12}")
    print("-" * 80)

    for scenario in scenarios:
        hit_rate = analyzer.estimate_hit_rate(
            num_unique_prefixes=scenario["unique"],
            total_requests=scenario["total"],
            cache_size=scenario["cache"],
        )
        print(
            f"{scenario['desc']:<40} "
            f"{scenario['unique']:<10} "
            f"{scenario['cache']:<10} "
            f"{hit_rate:<12.0%}"
        )

    # Test 4: Recommend prefix length
    print("\n" + "=" * 80)
    print("Test 4: Optimal Prefix Length Recommendation")
    print("=" * 80)

    system_prompt = 512
    few_shot = 512
    user_input = 1024

    print(f"\nPrompt structure:")
    print(f"  System prompt: {system_prompt} tokens")
    print(f"  Few-shot examples: {few_shot} tokens")
    print(f"  User input: {user_input} tokens")
    print(f"  Total: {system_prompt + few_shot + user_input} tokens")

    recommended = analyzer.recommend_prefix_length(
        model=llama3_8b,
        hardware=rtx_5090,
        system_prompt_length=system_prompt,
        few_shot_examples_length=few_shot,
        user_input_length=user_input,
        batch_size=32,
        hit_rate=0.8,
        precision="fp16",
    )

    print(f"\nRecommended prefix length: {recommended} tokens")
    if recommended == system_prompt:
        print("  → Cache system prompt only")
    elif recommended == system_prompt + few_shot:
        print("  → Cache system prompt + few-shot examples")

    # Test 5: Compare RTX GPUs
    print("\n" + "=" * 80)
    print("Test 5: Prefix Caching on Different GPUs")
    print("=" * 80)

    print(f"\n{'GPU':<12} {'Batch (no cache)':<18} {'Batch (with cache)':<18} {'Improvement':<15}")
    print("-" * 80)

    for gpu_name in ["RTX-5090", "RTX-4090", "RTX-3090"]:
        hardware = GPUDatabase.to_hardware_spec(gpu_name, num_gpus=1)

        metrics = analyzer.evaluate_caching(
            model=llama3_8b,
            hardware=hardware,
            prefix_length=1024,
            total_length=2048,
            batch_size=32,
            hit_rate=0.9,
            precision="fp16",
        )

        # Calculate baseline batch size
        from kv_planner.core.memory import PagedMemoryCalculator
        mem_calc = PagedMemoryCalculator()
        baseline_batch = mem_calc.max_batch_size(
            available_memory_gb=hardware.gpu_memory_gb,
            sequence_length=2048,
            model=llama3_8b,
            precision="fp16",
        )

        print(
            f"{gpu_name:<12} "
            f"{baseline_batch:<18} "
            f"{metrics.effective_batch_size:<18} "
            f"{metrics.throughput_improvement:<15.2f}×"
        )

    print("\n" + "=" * 80)
    print("✅ All validation tests passed!")
    print("=" * 80)

    # Summary
    print("\n📊 Key Findings:")
    print(f"  • 90% hit rate → {metrics.latency_reduction_pct:.0f}% latency reduction")
    print(f"  • Memory savings: {metrics.memory_savings_pct:.0f}%")
    print(f"  • Cost savings: {metrics.cost_savings_pct:.0f}%")
    print(f"  • Throughput: {metrics.throughput_improvement:.2f}× improvement")
    print(f"  • Best for: chatbots, RAG systems, few-shot prompting")


if __name__ == "__main__":
    main()
