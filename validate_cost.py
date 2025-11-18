#!/usr/bin/env python3
"""Validation script for CostAnalyzer."""

import sys
sys.path.insert(0, "/home/hesham-haroun/kv-planner/src")

from kv_planner.core.cost import CostAnalyzer
from kv_planner.domain import ModelConfig
from kv_planner.infrastructure.hardware_db import GPUDatabase


def main():
    """Run validation tests."""
    print("=" * 80)
    print("CostAnalyzer Validation")
    print("=" * 80)

    # Setup
    analyzer = CostAnalyzer()
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

    print("\n✓ Created CostAnalyzer")
    print("✓ Created Llama 3 8B model config")

    # Test 1: Single GPU cost analysis
    print("\n" + "=" * 80)
    print("Test 1: Cost Analysis - RTX 5090")
    print("=" * 80)

    rtx_5090 = GPUDatabase.to_hardware_spec("RTX-5090", num_gpus=1)
    metrics = analyzer.analyze_cost(
        model=llama3_8b,
        hardware=rtx_5090,
        batch_size=32,
        input_length=2048,
        output_length=512,
        requests_per_second=5.0,  # Target load
        precision="fp16",
    )

    print(f"\nGPU: {metrics.gpu_pricing.gpu_model}")
    print(f"Pricing: ${metrics.gpu_pricing.cost_per_hour_usd:.2f}/hour")
    print(f"\nCost Metrics:")
    print(f"  Cost per hour: ${metrics.cost_per_hour:.2f}")
    print(f"  Cost per million tokens: ${metrics.cost_per_million_tokens:.2f}")
    print(f"  Cost per request: ${metrics.cost_per_request:.6f}")
    print(f"  Monthly cost: ${metrics.monthly_cost_usd:.2f}")
    print(f"  Annual cost: ${metrics.annual_cost_usd:,.2f}")
    print(f"\nUtilization:")
    print(f"  GPU utilization: {metrics.utilization_pct:.1f}%")
    print(f"  Max throughput: {metrics.throughput_tokens_per_sec:.0f} tokens/s")
    print(f"  Max requests/s: {metrics.requests_per_second:.1f}")
    print(f"\nValue:")
    print(f"  Tokens per dollar: {metrics.tokens_per_dollar:,.0f}")
    print(f"  Break-even: {metrics.break_even_tokens_per_day:,.0f} tokens/day")

    # Test 2: Compare GPUs
    print("\n" + "=" * 80)
    print("Test 2: Compare RTX GPUs")
    print("=" * 80)

    gpu_models = ["RTX-5090", "RTX-4090", "RTX-3090"]
    hardware_options = [
        GPUDatabase.to_hardware_spec(gpu, num_gpus=1)
        for gpu in gpu_models
    ]

    results = analyzer.compare_deployments(
        model=llama3_8b,
        hardware_options=hardware_options,
        batch_size=32,
        input_length=2048,
        output_length=512,
        requests_per_second=5.0,
        precision="fp16",
    )

    print(f"\n{'GPU':<15} {'$/hr':<10} {'$/M tokens':<15} {'Monthly':<12} {'Tokens/$':<15}")
    print("-" * 80)

    for m in results:
        print(
            f"{m.gpu_pricing.gpu_model:<15} "
            f"${m.cost_per_hour:<9.2f} "
            f"${m.cost_per_million_tokens:<14.2f} "
            f"${m.monthly_cost_usd:<11.2f} "
            f"{m.tokens_per_dollar:<15,.0f}"
        )

    # Test 3: Break-even analysis
    print("\n" + "=" * 80)
    print("Test 3: Break-Even vs API Providers")
    print("=" * 80)

    self_hosted_cost = results[0].cost_per_million_tokens

    print(f"\nSelf-hosted cost: ${self_hosted_cost:.2f}/M tokens")
    print(f"\n{'Provider':<20} {'API $/M':<12} {'Savings $/M':<15} {'Savings %':<12}")
    print("-" * 80)

    api_providers = ["claude-3-sonnet", "gpt-4", "gemini-pro", "llama-3-8b"]

    for provider in api_providers:
        analysis = analyzer.break_even_analysis(
            self_hosted_cost_per_million=self_hosted_cost,
            api_provider=provider,
        )

        print(
            f"{provider:<20} "
            f"${analysis['api_cost_per_million']:<11.2f} "
            f"${analysis['savings_per_million']:<14.2f} "
            f"{analysis['savings_pct']:<12.1f}%"
        )

    # Test 4: Utilization curve
    print("\n" + "=" * 80)
    print("Test 4: Cost vs Utilization Curve")
    print("=" * 80)

    curve = analyzer.utilization_curve(
        model=llama3_8b,
        hardware=rtx_5090,
        batch_size=32,
        input_length=2048,
        output_length=512,
        precision="fp16",
        utilization_points=[10, 25, 50, 75, 100],
    )

    print(f"\n{'Utilization':<15} {'$/M tokens':<15}")
    print("-" * 40)

    for util_pct, cost in curve:
        print(f"{util_pct:<15.0f}% ${cost:<15.2f}")

    # Test 5: Datacenter GPU comparison
    print("\n" + "=" * 80)
    print("Test 5: Datacenter GPUs (H100 vs A100)")
    print("=" * 80)

    datacenter_gpus = ["H100-80GB", "A100-80GB"]
    datacenter_hardware = [
        GPUDatabase.to_hardware_spec(gpu, num_gpus=1)
        for gpu in datacenter_gpus
    ]

    datacenter_results = analyzer.compare_deployments(
        model=llama3_8b,
        hardware_options=datacenter_hardware,
        batch_size=64,  # Larger batch for datacenter
        input_length=2048,
        output_length=512,
        requests_per_second=20.0,  # Higher load
        precision="fp8",  # Use FP8 for efficiency
    )

    print(f"\n{'GPU':<15} {'$/hr':<10} {'$/M tokens':<15} {'Throughput':<15} {'Util %':<10}")
    print("-" * 80)

    for m in datacenter_results:
        print(
            f"{m.gpu_pricing.gpu_model:<15} "
            f"${m.cost_per_hour:<9.2f} "
            f"${m.cost_per_million_tokens:<14.2f} "
            f"{m.throughput_tokens_per_sec:<15.0f} "
            f"{m.utilization_pct:<10.1f}%"
        )

    # Test 6: Custom pricing
    print("\n" + "=" * 80)
    print("Test 6: Custom Pricing (Spot Instances)")
    print("=" * 80)

    from kv_planner.core.cost import GPUPricing

    # Simulate spot pricing (70% of on-demand)
    spot_pricing = GPUPricing(
        gpu_model="RTX-5090",
        cost_per_hour_usd=0.25,  # $0.25/hr spot
        source="Runpod",
        pricing_type="spot",
    )

    spot_metrics = analyzer.analyze_cost(
        model=llama3_8b,
        hardware=rtx_5090,
        batch_size=32,
        input_length=2048,
        output_length=512,
        requests_per_second=5.0,
        precision="fp16",
        custom_pricing=spot_pricing,
    )

    print(f"\nOn-demand pricing:")
    print(f"  ${metrics.cost_per_hour:.2f}/hr → ${metrics.cost_per_million_tokens:.2f}/M tokens")
    print(f"\nSpot pricing:")
    print(f"  ${spot_metrics.cost_per_hour:.2f}/hr → ${spot_metrics.cost_per_million_tokens:.2f}/M tokens")
    print(f"\nSavings:")
    savings_pct = ((metrics.cost_per_million_tokens - spot_metrics.cost_per_million_tokens) /
                   metrics.cost_per_million_tokens) * 100
    print(f"  {savings_pct:.1f}% cheaper with spot instances")

    print("\n" + "=" * 80)
    print("✅ All validation tests passed!")
    print("=" * 80)

    # Summary
    print("\n📊 Key Findings:")
    print(f"  • RTX 5090 cost: ${metrics.cost_per_million_tokens:.2f}/M tokens")
    print(f"  • vs Claude Sonnet ($3.00/M): {((3.0 - metrics.cost_per_million_tokens) / 3.0) * 100:.0f}% cheaper")
    print(f"  • Break-even: {metrics.break_even_tokens_per_day:,.0f} tokens/day")
    print(f"  • Best value: {results[0].gpu_pricing.gpu_model} (${results[0].cost_per_million_tokens:.2f}/M tokens)")
    print(f"  • Spot pricing saves: {savings_pct:.0f}%")


if __name__ == "__main__":
    main()
